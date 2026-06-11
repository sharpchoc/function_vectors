"""
Capture paired 1-shot ICL residual-stream activations for the function-geometry
experiment (GPT-J-6B; loads the model + baukit, GPU recommended).

For a shared single-token output word `w` (e.g. " down") that two functions can
both produce, build two prompts that are identical except for the demo input word:

    f1 (e.g. antonym):   Q: up\nA: down\n\nQ: <query>\nA:
    f2 (e.g. synonym):   Q: beneath\nA: down\n\nQ: <query>\nA:

The demo label token (" down") is the same token in both prompts; only the demo
input differs, so the activation difference at " down" isolates function/context,
not token identity. We capture activations at TWO positions per prompt:
    source = the demo label token (last_label_token, icl_example_index == 1)
    target = the final query token (last_prompt_token, icl_example_index is None)

Two task pairs are supported:
    antonym_synonym -> (antonym, synonym)        [~555 shared single-token outputs]
    landmark_park   -> (landmark-country, park-country)  [~84 shared]

Output: shards of activations [rows, 28, 4096] (fp32) + per-row metadata extended
with {function, task_pair, output_word, query_word, role} + an index.json, written
with the same flush_shard / index.json structure as extract_residual_stream_activations.py.

This script intentionally loads GPT-J via baukit (TraceDict) and is NOT import-safe
without those dependencies; the downstream analysis lives in
analyze_oneshot_geometry.py (pure numpy/torch, no model).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data

from extract_residual_stream_activations import (
    flush_shard,
    get_residual_stack,
    make_token_record,
    selected_token_records,
)


# task_pair -> (function-1 task name, function-2 task name)
TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "landmark_park": ("landmark-country", "park-country"),
}


# --- copied from extract_targeted_residual_stream_activations.py ---
def stable_seed(*parts):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_seed(*parts))
# -------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture paired 1-shot ICL residual-stream activations (source=demo label, target=final query token)."
    )
    parser.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    parser.add_argument("--max_words", type=int, default=None,
                        help="Cap the number of shared output words processed (None=all).")
    parser.add_argument("--store_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--save_path_root", type=str, default="results/oneshot_paired")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--shard_size", type=int, default=100,
                        help="Number of (output word) units per shard (each contributes 4 rows: 2 functions x 2 roles).")
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    return parser.parse_args()


def store_dtype_to_torch(store_dtype):
    return torch.float16 if store_dtype == "float16" else torch.float32


def load_task_json(root_data_dir, task):
    """Load an abstractive dataset JSON (list of {'input','output'} dicts)."""
    path = Path(root_data_dir) / "abstractive" / f"{task}.json"
    with open(path, "r") as f:
        records = json.load(f)
    return records


def build_output_to_inputs(records):
    """Map output_word -> sorted list of distinct input words that produce it."""
    out_to_in = {}
    for rec in records:
        out = str(rec["output"]).strip()
        inp = str(rec["input"]).strip()
        out_to_in.setdefault(out, set()).add(inp)
    return {out: sorted(inputs) for out, inputs in out_to_in.items()}


def is_single_space_token(tokenizer, word):
    """True iff ' '+word is a single token (the space-prefixed convention used by word_pairs_to_prompt_data)."""
    return len(tokenizer(" " + word).input_ids) == 1


def select_shared_words(tokenizer, out_to_in_f1, out_to_in_f2):
    """Output words producible under BOTH functions and single-token space-prefixed."""
    shared = sorted(set(out_to_in_f1).intersection(out_to_in_f2))
    kept = [w for w in shared if is_single_space_token(tokenizer, w)]
    return kept


def build_prompt_data(demo_input, demo_output, query_input, query_output, args):
    """One-demo ICL prompt: word_pairs is a column dict {'input':[...],'output':[...]}."""
    word_pairs = {"input": [demo_input], "output": [demo_output]}
    query_target_pair = {"input": query_input, "output": query_output}
    return word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair=query_target_pair,
        prepend_bos_token=False,  # GPT-J prepend_bos == False
        prefixes=args.prefixes,
        separators=args.separators,
        prepend_space=True,
    )


def extract_source_target_records(token_labels):
    """Re-derive the two positions of interest from selected_token_records (never hard-coded):
    source = last_label_token with icl_example_index == 1
    target = last_prompt_token with icl_example_index is None
    Returns dict {'source': record, 'target': record}.
    """
    records = selected_token_records(token_labels)
    source = None
    target = None
    for rec in records:
        if rec["token_role"] == "last_label_token" and rec["icl_example_index"] == 1:
            source = rec
        elif rec["token_role"] == "last_prompt_token" and rec["icl_example_index"] is None:
            target = rec
    if source is None or target is None:
        raise ValueError("Could not derive both source (last_label_token@1) and target (last_prompt_token) records.")
    return {"source": source, "target": target}


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    store_dtype = store_dtype_to_torch(args.store_dtype)

    task_f1, task_f2 = TASK_PAIRS[args.task_pair]
    print(f"task_pair={args.task_pair} -> f1={task_f1}, f2={task_f2}")

    records_f1 = load_task_json(args.root_data_dir, task_f1)
    records_f2 = load_task_json(args.root_data_dir, task_f2)
    out_to_in_f1 = build_output_to_inputs(records_f1)
    out_to_in_f2 = build_output_to_inputs(records_f2)

    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    shared_words = select_shared_words(tokenizer, out_to_in_f1, out_to_in_f2)
    print(f"shared single-token output words producible under both functions: {len(shared_words)}")
    if args.max_words is not None:
        shared_words = shared_words[: args.max_words]
        print(f"capped to {len(shared_words)} words (--max_words)")

    # Shared query pool: output words producible by both (a sensible, in-domain pool).
    query_pool = list(shared_words)

    output_root = Path(args.save_path_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_dir = output_root / args.task_pair

    # Clear any prior run for this task_pair.
    if output_dir.exists():
        for old_shard in output_dir.glob("shard_*.pt"):
            old_shard.unlink()
        old_index = output_dir / "index.json"
        if old_index.exists():
            old_index.unlink()

    config = {
        "task_pair": args.task_pair,
        "function_tasks": {"f1": task_f1, "f2": task_f2},
        "model_name": args.model_name,
        "model_config": model_config,
        "seed": args.seed,
        "store_dtype": str(store_dtype),
        "prefixes": args.prefixes,
        "separators": args.separators,
        "n_shots": 1,
        "roles": ["source", "target"],
        "role_definitions": {
            "source": "last_label_token, icl_example_index == 1 (demo label token)",
            "target": "last_prompt_token, icl_example_index is None (final query token)",
        },
        "n_shared_words": len(shared_words),
    }

    shard_activations = []
    shard_metadata = []
    shard_paths = []
    shard_index = 0
    words_in_shard = 0
    n_words_done = 0

    for w in shared_words:
        rng = stable_rng(args.seed, args.task_pair, w)

        demo_in_f1 = str(rng.choice(out_to_in_f1[w]))
        demo_in_f2 = str(rng.choice(out_to_in_f2[w]))

        # Query word: from the shared pool, excluding w and either demo input. Same q for f1 and f2.
        forbidden = {w, demo_in_f1, demo_in_f2}
        candidates = [q for q in query_pool if q not in forbidden]
        if not candidates:
            print(f"  skip w={w!r}: no valid query word")
            continue
        q = str(rng.choice(candidates))

        # Sanity target: the captured source row token id should equal tokenizer(' '+w).input_ids[0].
        expected_source_token_id = tokenizer(" " + w).input_ids[0]

        for function, demo_input, task_name in (("f1", demo_in_f1, task_f1), ("f2", demo_in_f2, task_f2)):
            # Demo output = the shared word w (space-prefixed by word_pairs_to_prompt_data).
            # Query output is unused for capture but kept consistent with the function's own mapping
            # is not required; the query word is identical across f1/f2 by construction.
            prompt_data = build_prompt_data(demo_input, w, q, w, args)
            residual_stack, token_labels, prompt_string = get_residual_stack(
                prompt_data, model, model_config, tokenizer, include_embeddings=False
            )

            role_records = extract_source_target_records(token_labels)

            # Verify the captured source row's token id equals tokenizer(' '+w).input_ids[0].
            # Re-derive from the actual prompt ids at the source position (never hard-coded).
            full_input_ids = tokenizer(prompt_string).input_ids
            src_pos = role_records["source"]["token_position"]
            captured_source_token_id = full_input_ids[src_pos]
            assert captured_source_token_id == expected_source_token_id, (
                f"source token id mismatch for w={w!r} ({function}): "
                f"captured={captured_source_token_id} expected(' '+w)={expected_source_token_id}"
            )

            for role in ("source", "target"):
                rec = role_records[role]
                pos = rec["token_position"]
                if pos >= residual_stack.shape[1]:
                    raise IndexError(
                        f"Token position {pos} exceeds residual sequence length {residual_stack.shape[1]}"
                    )
                act = residual_stack[:, pos, :].cpu().to(store_dtype)
                shard_activations.append(act)

                metadata = dict(make_token_record(rec["token_role"], rec["icl_example_index"],
                                                   (rec["token_position"], rec["token_text"], rec["token_label"])))
                metadata.update({
                    "task_pair": args.task_pair,
                    "function": function,
                    "function_task": task_name,
                    "output_word": w,
                    "query_word": q,
                    "demo_input": demo_input,
                    "role": role,
                })
                shard_metadata.append(metadata)

        n_words_done += 1
        words_in_shard += 1
        if words_in_shard >= args.shard_size:
            shard_path = flush_shard(shard_activations, shard_metadata, output_dir, shard_index, config)
            if shard_path is not None:
                shard_paths.append(str(shard_path))
            shard_activations = []
            shard_metadata = []
            shard_index += 1
            words_in_shard = 0

    shard_path = flush_shard(shard_activations, shard_metadata, output_dir, shard_index, config)
    if shard_path is not None:
        shard_paths.append(str(shard_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump({"config": config, "shards": shard_paths}, f, indent=2)

    print(f"processed {n_words_done} output words -> {len(shard_paths)} shards")
    print(f"source token id == tokenizer(' '+w).input_ids[0] verified for all words")
    print(index_path)


if __name__ == "__main__":
    main()
