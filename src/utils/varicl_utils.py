"""Shared helpers for the variable-ICL, train-pooled function-vector method.

This module provides thin variants of the canonical FV primitives for a regime where
each prompt draws a random 1-10 ICL demonstration count and every quantity (mean head
activations + CIE intervention/measurement) is read at the query predictive token --
the final/last non-pad token of the prompt (T=-1). Because the read position is a single,
length-independent slot (`attention_mask.sum(1)-1`), there is no per-role token-label
parsing and no dummy-label canonical grid: mean activations are stored as
(n_layers, n_heads, head_dim) instead of (n_layers, n_heads, n_tokens, head_dim).
"""
import numpy as np
import torch
from baukit import TraceDict

from src.compute_indirect_effect import _project_attention_inputs
from src.utils.eval_utils import compute_individual_token_rank, get_answer_id
from src.utils.prompt_utils import create_prompt, word_pairs_to_prompt_data


def sample_variable_icl_count(rng, lo=1, hi=10):
    """Draw a per-prompt ICL demonstration count in [lo, hi] (inclusive)."""
    return int(rng.integers(lo, hi + 1))


def split_activations_by_head(activations, model_config):
    """Reshape attention out_proj inputs to expose the per-head axis."""
    new_shape = activations.size()[:-1] + (
        model_config['n_heads'],
        model_config['resid_dim'] // model_config['n_heads'],
    )
    return activations.view(*new_shape)


def build_varicl_prompt_data(dataset, args, model_config, task_index, query_idx, shuffle_labels, seed_base):
    """Build a single variable-ICL prompt for a query example.

    Mirrors compute_multitask_top_aie_heads.build_prompt_data but the shot count is the
    FIRST draw from default_rng(seed_base + 100_000*task_index + query_idx) so each prompt's
    length + demonstrations are identical regardless of how tasks are sharded across workers
    (task_index is the GLOBAL position in the task list). The same int reseeds NumPy's global
    RNG before word_pairs_to_prompt_data so label shuffling stays reproducible.
    """
    # Local import to avoid a circular import at module load time.
    from src.eval_scripts.compute_multitask_top_aie_heads import sample_demo_indices

    seed = seed_base + 100_000 * task_index + query_idx
    rng = np.random.default_rng(seed)
    n_shots_i = sample_variable_icl_count(rng, args.min_shots, args.max_shots)

    query_data = dataset[args.query_split][query_idx]
    n_query = len(dataset[args.query_split])
    n_demo = len(dataset[args.demo_split])
    demo_indices = sample_demo_indices(
        query_idx=query_idx,
        query_split=args.query_split,
        demo_split=args.demo_split,
        n_query=n_query,
        n_demo=n_demo,
        n_shots=n_shots_i,
        rng=rng,
    )
    demo_pairs = dataset[args.demo_split][demo_indices]

    # word_pairs_to_prompt_data handles label shuffling through NumPy's global RNG.
    np.random.seed(seed)
    prepend_bos = False if model_config["prepend_bos"] else True
    return word_pairs_to_prompt_data(
        demo_pairs,
        query_target_pair=query_data,
        prepend_bos_token=prepend_bos,
        shuffle_labels=shuffle_labels,
        prefixes=args.prefixes,
        separators=args.separators,
    )


def get_last_token_mean_head_activations(dataset, args, model, model_config, tokenizer,
                                         task_index, query_indices, seed_base):
    """Mean per-head attention out_proj input at the LAST non-pad token of each prompt.

    Adapted from extract_utils.get_mean_head_activations but reads a single length-
    independent position (per-row prompt_lens = attention_mask.sum(1)-1) instead of a
    dummy-label canonical token grid. Returns (n_layers, n_heads, head_dim).
    """
    n_layers = model_config['n_layers']
    n_heads = model_config['n_heads']
    head_dim = model_config['resid_dim'] // n_heads

    activation_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64, device=model.device)
    count = 0
    batch_size = max(1, int(args.batch_size))
    query_indices = list(query_indices)

    old_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    try:
        for batch_start in range(0, len(query_indices), batch_size):
            batch_query_indices = query_indices[batch_start:batch_start + batch_size]
            sentences = []
            for query_idx in batch_query_indices:
                prompt_data = build_varicl_prompt_data(
                    dataset, args, model_config, task_index=task_index, query_idx=int(query_idx),
                    shuffle_labels=False, seed_base=seed_base,
                )
                sentences.append(create_prompt(prompt_data))

            inputs = tokenizer(sentences, return_tensors='pt', padding=True).to(model.device)
            prompt_lens = inputs.attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(sentences), device=model.device)

            with TraceDict(model, layers=model_config['attn_hook_names'], retain_input=True, retain_output=False) as td:
                model(**inputs)

            layer_inputs = []
            for layer in model_config['attn_hook_names']:
                layer_input = td[layer].input
                if isinstance(layer_input, tuple):
                    layer_input = layer_input[0]
                layer_inputs.append(split_activations_by_head(layer_input, model_config))

            # (batch, layers, tokens, heads, head_dim)
            stack_initial = torch.stack(layer_inputs).permute(1, 0, 2, 3, 4)
            # Select the last non-pad token per row -> (batch, layers, heads, head_dim)
            last_token = stack_initial[batch_indices, :, prompt_lens]
            activation_sum += last_token.sum(dim=0).double()
            count += len(sentences)
    finally:
        tokenizer.padding_side = old_padding_side

    if count == 0:
        raise RuntimeError("No prompts available for last-token mean head activations")
    mean_activations = (activation_sum / count).float()
    return mean_activations


def varicl_correctness_filter(dataset, args, model, model_config, tokenizer, task_index, seed_base):
    """Teacher-forced rank of the correct answer for variable-ICL, unshuffled prompts.

    Mirrors the rank logic of eval_utils.n_shot_eval_no_intervention (get_answer_id +
    compute_individual_token_rank at the last non-pad token) but builds each item with
    build_varicl_prompt_data (variable shots, shuffle_labels=False). Returns clean_rank_list;
    a query is "correct" when its rank is 0.
    """
    clean_rank_list = []
    batch_size = max(1, int(args.batch_size_filter_eval))
    split_len = len(dataset[args.query_split])

    old_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    try:
        for batch_start in range(0, split_len, batch_size):
            batch_query_indices = range(batch_start, min(batch_start + batch_size, split_len))
            sentences = []
            target_token_ids = []
            for query_idx in batch_query_indices:
                prompt_data = build_varicl_prompt_data(
                    dataset, args, model_config, task_index=task_index, query_idx=int(query_idx),
                    shuffle_labels=False, seed_base=seed_base,
                )
                target = prompt_data['query_target']['output']
                target = target[0] if isinstance(target, list) else target
                sentence = create_prompt(prompt_data)
                target_token_ids.append(get_answer_id(sentence, target, tokenizer))
                sentences.append(sentence)

            inputs = tokenizer(sentences, return_tensors='pt', padding=True).to(model.device)
            prompt_lens = inputs.attention_mask.sum(dim=1) - 1
            output = model(**inputs).logits
            clean_outputs = output[torch.arange(output.shape[0], device=model.device), prompt_lens]

            for clean_output, target_token_id in zip(clean_outputs, target_token_ids):
                clean_rank_list.append(compute_individual_token_rank(clean_output, target_token_id))
    finally:
        tokenizer.padding_side = old_padding_side

    return clean_rank_list


def batch_varicl_last_token_intervention(prompt_data_batch, avg_activations, model, model_config, tokenizer):
    """Last-token (query predictive) CIE intervention for a batch of variable-ICL prompts.

    Copy of compute_indirect_effect.batch_activation_replacement_last_token_intervention with
    the canonical-grid idx-map dropped: avg_activations is (n_layers, n_heads, head_dim), so the
    replacement reads avg_activations[layer, head_n] (a single position) rather than indexing a
    token axis. The intervention stays at the query token (prompt_lens). Reuses
    _project_attention_inputs unchanged.
    """
    device = model.device
    avg_activations = avg_activations.to(device)

    sentences = []
    target_token_ids = []

    for prompt_data in prompt_data_batch:
        query_target_pair = prompt_data['query_target']
        target = query_target_pair['output']
        if isinstance(target, list):
            target = target[0]
        prompt_string = create_prompt(prompt_data)
        token_id_of_interest = get_answer_id(prompt_string, target, tokenizer)
        if isinstance(token_id_of_interest, list):
            token_id_of_interest = token_id_of_interest[0]
        target_token_ids.append(token_id_of_interest)
        sentences.append(prompt_string)

    old_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    try:
        inputs = tokenizer(sentences, return_tensors='pt', padding=True).to(device)
    finally:
        tokenizer.padding_side = old_padding_side

    prompt_lens = inputs.attention_mask.sum(dim=1) - 1
    target_token_ids = torch.LongTensor(target_token_ids).to(device)
    batch_indices = torch.arange(len(sentences), device=device)

    clean_output = model(**inputs).logits[batch_indices, prompt_lens]
    clean_probs = torch.softmax(clean_output, dim=-1)

    indirect_effect_storage = torch.zeros(len(sentences), model_config['n_layers'], model_config['n_heads'], device=device)

    for layer in range(model_config['n_layers']):
        head_hook_layer = [model_config['attn_hook_names'][layer]]

        for head_n in range(model_config['n_heads']):
            def make_replace_batch_activation(layer, head_n):
                def replace_batch_activation(output, layer_name, inputs):
                    current_layer = int(layer_name.split('.')[2])
                    if current_layer != layer:
                        return output

                    if isinstance(inputs, tuple):
                        inputs = inputs[0]

                    original_shape = inputs.shape
                    head_dim = model_config['resid_dim'] // model_config['n_heads']
                    head_inputs = inputs.view(*inputs.size()[:-1], model_config['n_heads'], head_dim)
                    head_inputs[batch_indices, prompt_lens, head_n] = avg_activations[layer, head_n].to(head_inputs.dtype)
                    head_inputs = head_inputs.view(*original_shape)
                    return _project_attention_inputs(head_inputs, layer_name, model, model_config)
                return replace_batch_activation

            replace_batch_activation = make_replace_batch_activation(layer, head_n)
            with TraceDict(model, layers=head_hook_layer, edit_output=replace_batch_activation):
                output = model(**inputs).logits[batch_indices, prompt_lens]

            intervention_probs = torch.softmax(output, dim=-1)
            indirect_effect_storage[:, layer, head_n] = (
                intervention_probs[batch_indices, target_token_ids] - clean_probs[batch_indices, target_token_ids]
            )

    return indirect_effect_storage.cpu()
