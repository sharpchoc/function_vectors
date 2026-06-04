import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from baukit import TraceDict
from tqdm import tqdm

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import get_token_meta_labels, load_dataset, word_pairs_to_prompt_data


LABEL_TOKEN_RE = re.compile(r"^demonstration_(\d+)_label_token$")
DEFAULT_TOKEN_ROLES = [
    "pre_label_token",
    "first_label_token",
    "last_label_token",
    "last_prompt_token",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a targeted set of residual-stream activations for large task collections."
    )
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument(
        "--save_path_root",
        type=str,
        default="results/residual_activations/gptj_40tasks_170prompts_4tokens",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--max_train_prompts", type=int, default=130)
    parser.add_argument("--max_test_prompts", type=int, default=40)
    parser.add_argument("--max_valid_prompts", type=int, default=None)
    parser.add_argument("--target_icl_example_index", type=int, default=10)
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--shard_size", type=int, default=100, help="Number of prompts per shard.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--storage_dtype", choices=["model", "float16", "float32"], default="float16")
    parser.add_argument("--include_embeddings", action="store_true", default=True)
    parser.add_argument("--no_include_embeddings", dest="include_embeddings", action="store_false")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--no_overwrite_existing", dest="overwrite_existing", action="store_false")
    parser.set_defaults(overwrite_existing=True)
    return parser.parse_args()


def storage_dtype_to_torch(storage_dtype, model_dtype):
    if storage_dtype == "model":
        return model_dtype if model_dtype in (torch.float16, torch.float32, torch.bfloat16) else torch.float32
    if storage_dtype == "float16":
        return torch.float16
    if storage_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown storage dtype: {storage_dtype}")


def stable_seed(*parts):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_seed(*parts))


def get_split_prompt_cap(args, split):
    if split == "train":
        return args.max_train_prompts
    if split == "test":
        return args.max_test_prompts
    if split == "valid":
        return args.max_valid_prompts
    raise ValueError(f"Unknown split: {split}")


def sample_query_indices(task, split, split_size, args):
    cap = get_split_prompt_cap(args, split)
    n_prompts = split_size if cap is None else min(cap, split_size)
    rng = stable_rng(args.seed, task, split, "query_indices")
    return rng.choice(np.arange(split_size), size=n_prompts, replace=False).tolist()


def sample_demo_indices(task, split, query_idx, dataset, args):
    train_indices = np.arange(len(dataset["train"]))
    if split == "train":
        train_indices = train_indices[train_indices != query_idx]
    if len(train_indices) < args.n_shots:
        raise ValueError(f"Not enough train examples to sample {args.n_shots} demonstrations")
    rng = stable_rng(args.seed, task, split, int(query_idx), "demo_indices")
    return rng.choice(train_indices, args.n_shots, replace=False)


def make_prompt(dataset, split, query_idx, demo_indices, model_config, prefixes, separators):
    prepend_bos = False if model_config["prepend_bos"] else True
    word_pairs = dataset["train"][demo_indices]
    query_pair = dataset[split][query_idx]
    return word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair=query_pair,
        prepend_bos_token=prepend_bos,
        prefixes=prefixes,
        separators=separators,
    )


def get_embedding_layer_name(model, model_config):
    name_or_path = model_config["name_or_path"].lower()
    if "gpt-j" in name_or_path or "gpt2" in name_or_path:
        return "transformer.wte"
    if any(model_name in name_or_path for model_name in ["llama", "gemma", "olmo", "qwen"]):
        return "model.embed_tokens"
    if "gpt-neox" in name_or_path or "pythia" in name_or_path:
        return "gpt_neox.embed_in"

    for module_name in ["transformer.wte", "model.embed_tokens", "gpt_neox.embed_in"]:
        if any(name == module_name for name, _ in model.named_modules()):
            return module_name
    raise NotImplementedError(f"Embedding hook is not defined for {model_config['name_or_path']}")


def get_residual_stack(prompt_data, model, model_config, tokenizer, include_embeddings=False):
    query = prompt_data["query_target"]["input"]
    token_labels, prompt_string = get_token_meta_labels(
        prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"]
    )
    inputs = tokenizer([prompt_string], return_tensors="pt").to(model.device)

    layers = list(model_config["layer_hook_names"])
    embedding_layer_name = None
    if include_embeddings:
        embedding_layer_name = get_embedding_layer_name(model, model_config)
        layers = [embedding_layer_name] + layers

    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:
        model(**inputs)

    layer_outputs = []
    if include_embeddings:
        embedding_output = td[embedding_layer_name].output
        if isinstance(embedding_output, tuple):
            embedding_output = embedding_output[0]
        if embedding_output.dim() == 3:
            embedding_output = embedding_output[0]
        layer_outputs.append(embedding_output.detach())

    for layer_name in model_config["layer_hook_names"]:
        output = td[layer_name].output
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3:
            output = output[0]
        layer_outputs.append(output.detach())

    return torch.stack(layer_outputs, dim=0), token_labels, prompt_string


def make_token_record(token_role, icl_example_index, token):
    token_position, token_text, token_label = token
    return {
        "token_role": token_role,
        "icl_example_index": icl_example_index,
        "token_position": int(token_position),
        "token_text": token_text,
        "token_label": token_label,
    }


def selected_token_records(token_labels, target_icl_example_index, token_roles):
    allowed_roles = set(token_roles)
    tokens_by_position = {
        int(token_position): (token_position, token_text, token_label)
        for token_position, token_text, token_label in token_labels
    }
    label_groups = {}
    for token_position, token_text, token_label in token_labels:
        match = LABEL_TOKEN_RE.match(token_label)
        if match:
            icl_example_index = int(match.group(1))
            label_groups.setdefault(icl_example_index, []).append((token_position, token_text, token_label))

    if target_icl_example_index not in label_groups:
        raise ValueError(f"No label tokens found for ICL example {target_icl_example_index}")

    label_tokens = sorted(label_groups[target_icl_example_index], key=lambda x: x[0])
    first_label_token = label_tokens[0]
    last_label_token = label_tokens[-1]
    pre_label_position = int(first_label_token[0]) - 1
    if pre_label_position < 0 or pre_label_position not in tokens_by_position:
        raise ValueError(f"Could not find pre-label token for ICL example {target_icl_example_index}")
    pre_label_token = tokens_by_position[pre_label_position]

    final_candidates = [x for x in token_labels if x[2] == "query_predictive_token"]
    final_token = max(final_candidates, key=lambda x: x[0]) if final_candidates else token_labels[-1]

    candidate_records = {
        "pre_label_token": make_token_record("pre_label_token", target_icl_example_index, pre_label_token),
        "first_label_token": make_token_record("first_label_token", target_icl_example_index, first_label_token),
        "last_label_token": make_token_record("last_label_token", target_icl_example_index, last_label_token),
        "last_prompt_token": make_token_record("last_prompt_token", None, final_token),
    }
    unknown_roles = allowed_roles.difference(candidate_records)
    if unknown_roles:
        raise ValueError(f"Unsupported token roles: {sorted(unknown_roles)}")
    return [candidate_records[token_role] for token_role in token_roles]


def build_metadata(task, split, prompt_index, query_idx, demo_indices, prompt_data, token_record):
    metadata = {
        "task": task,
        "split": split,
        "prompt_index": int(prompt_index),
        "query_source_index": int(query_idx),
        "query_input": prompt_data["query_target"]["input"].strip(),
        "query_output": prompt_data["query_target"]["output"].strip(),
        **token_record,
    }
    if token_record["icl_example_index"] is not None:
        demo_pos = token_record["icl_example_index"] - 1
        demo = prompt_data["examples"][demo_pos]
        metadata.update(
            {
                "demo_source_index": int(demo_indices[demo_pos]),
                "demo_input": demo["input"].strip(),
                "demo_output": demo["output"].strip(),
            }
        )
    else:
        metadata.update({"demo_source_index": None, "demo_input": None, "demo_output": None})
    return metadata


def flush_shard(activations, metadata, output_dir, shard_index, config):
    if not activations:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{shard_index:05d}.pt"
    torch.save(
        {
            "activations": torch.stack(activations, dim=0),
            "metadata": metadata,
            "config": config,
        },
        output_path,
    )
    return output_path


def clear_existing_output(output_dir):
    if not output_dir.exists():
        return
    for old_shard in output_dir.glob("shard_*.pt"):
        old_shard.unlink()
    old_index = output_dir / "index.json"
    if old_index.exists():
        old_index.unlink()


def extract_split(task, split, dataset, model, model_config, tokenizer, args, output_root, store_dtype):
    query_indices = sample_query_indices(task, split, len(dataset[split]), args)
    output_dir = output_root / task / split
    if args.overwrite_existing:
        clear_existing_output(output_dir)

    config = {
        "task": task,
        "split": split,
        "model_name": args.model_name,
        "model_config": model_config,
        "seed": args.seed,
        "test_split": args.test_split,
        "n_shots": args.n_shots,
        "max_train_prompts": args.max_train_prompts,
        "max_test_prompts": args.max_test_prompts,
        "max_valid_prompts": args.max_valid_prompts,
        "sampled_prompt_count": len(query_indices),
        "available_prompt_count": len(dataset[split]),
        "query_indices": [int(x) for x in query_indices],
        "target_icl_example_index": args.target_icl_example_index,
        "token_roles": args.token_roles,
        "prefixes": args.prefixes,
        "separators": args.separators,
        "storage_dtype": str(store_dtype),
        "include_embeddings": args.include_embeddings,
        "overwrite_existing": args.overwrite_existing,
    }

    shard_activations = []
    shard_metadata = []
    shard_paths = []
    shard_index = 0
    prompts_in_shard = 0

    for prompt_index, query_idx in enumerate(tqdm(query_indices, desc=f"{task}/{split}")):
        demo_indices = sample_demo_indices(task, split, int(query_idx), dataset, args)
        prompt_data = make_prompt(dataset, split, int(query_idx), demo_indices, model_config, args.prefixes, args.separators)
        residual_stack, token_labels, _ = get_residual_stack(
            prompt_data, model, model_config, tokenizer, include_embeddings=args.include_embeddings
        )
        token_records = selected_token_records(
            token_labels, args.target_icl_example_index, args.token_roles
        )
        for token_record in token_records:
            token_position = token_record["token_position"]
            if token_position >= residual_stack.shape[1]:
                raise IndexError(
                    f"Token position {token_position} exceeds residual sequence length {residual_stack.shape[1]}"
                )
            shard_activations.append(residual_stack[:, token_position, :].cpu().to(store_dtype))
            shard_metadata.append(
                build_metadata(task, split, prompt_index, int(query_idx), demo_indices, prompt_data, token_record)
            )

        prompts_in_shard += 1
        if prompts_in_shard >= args.shard_size:
            shard_path = flush_shard(shard_activations, shard_metadata, output_dir, shard_index, config)
            shard_paths.append(str(shard_path))
            shard_activations = []
            shard_metadata = []
            shard_index += 1
            prompts_in_shard = 0

    shard_path = flush_shard(shard_activations, shard_metadata, output_dir, shard_index, config)
    if shard_path is not None:
        shard_paths.append(str(shard_path))

    index_path = output_dir / "index.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump({"config": config, "shards": shard_paths}, f, indent=2)
    return index_path


def main():
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be positive")
    if args.target_icl_example_index > args.n_shots:
        raise ValueError("--target_icl_example_index must be <= --n_shots")

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()
    store_dtype = storage_dtype_to_torch(args.storage_dtype, next(model.parameters()).dtype)

    output_root = Path(args.save_path_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    for task in args.dataset_names:
        print(f"Loading Dataset: {task}")
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
        for split in args.splits:
            index_path = extract_split(
                task, split, dataset, model, model_config, tokenizer, args, output_root, store_dtype
            )
            print(index_path)


if __name__ == "__main__":
    main()
