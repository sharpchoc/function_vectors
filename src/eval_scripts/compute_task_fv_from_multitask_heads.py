#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.utils.model_utils import load_gpt_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a task function vector using a saved multitask top-head set."
    )
    parser.add_argument("--task", required=True, help="Task name, e.g. antonym.")
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument(
        "--heads_path",
        type=Path,
        default=Path("results/multitask_aie_heads/multitask_top_aie_heads.pt"),
        help="Output .pt from compute_multitask_top_aie_heads.py.",
    )
    parser.add_argument("--mean_activations_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--n_top_heads", type=int, default=None, help="Optional prefix of the saved head ranking.")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_top_heads(path, n_top_heads):
    data = torch_load_trusted(path, map_location="cpu")
    if not isinstance(data, dict) or "top_heads" not in data:
        raise KeyError(f"{path} does not contain a 'top_heads' entry")

    top_heads = data["top_heads"]
    if n_top_heads is not None:
        top_heads = top_heads[:n_top_heads]
    return [(int(layer), int(head), float(score)) for layer, head, score in top_heads], data


def get_out_projection(model, model_config, layer):
    name = model_config["name_or_path"].lower()
    if "gpt2-xl" in name:
        return model.transformer.h[layer].attn.c_proj
    if "gpt-j" in name:
        return model.transformer.h[layer].attn.out_proj
    if "llama" in name or "gemma" in name or "olmo" in name or "qwen" in name:
        return model.model.layers[layer].self_attn.o_proj
    if "gpt-neox" in name or "pythia" in name:
        return model.gpt_neox.layers[layer].attention.dense
    raise NotImplementedError(f"Attention output projection is not defined for {model_config['name_or_path']}")


def compute_function_vector_from_heads(mean_activations, top_heads, model, model_config):
    model_resid_dim = model_config["resid_dim"]
    model_head_dim = model_resid_dim // model_config["n_heads"]
    function_vector = torch.zeros((1, 1, model_resid_dim), device=model.device)
    token_idx = -1

    for layer, head, _ in top_heads:
        out_proj = get_out_projection(model, model_config, layer)
        x = torch.zeros(model_resid_dim)
        x[head * model_head_dim : (head + 1) * model_head_dim] = mean_activations[layer, head, token_idx]
        d_out = out_proj(x.reshape(1, 1, model_resid_dim).to(model.device).to(model.dtype))
        function_vector += d_out

    return function_vector.to(model.dtype).reshape(1, model_resid_dim)


def main():
    args = parse_args()
    mean_path = args.mean_activations_path or args.fv_root / args.task / f"{args.task}_mean_head_activations.pt"
    if not mean_path.exists():
        raise FileNotFoundError(mean_path)

    output_dir = args.output_dir or args.fv_root / args.task
    output_name = args.output_name or f"{args.task}_function_vector_multitask_heads.pt"
    output_path = output_dir / output_name
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")

    top_heads, heads_data = load_top_heads(args.heads_path, args.n_top_heads)
    mean_activations = torch_load_trusted(mean_path, map_location="cpu")

    torch.set_grad_enabled(False)
    print("Loading model")
    model, _, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    fv = compute_function_vector_from_heads(mean_activations, top_heads, model, model_config)
    fv = fv.detach().float().cpu().reshape(-1)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "function_vector": fv,
            "top_heads": top_heads,
            "n_top_heads": len(top_heads),
            "task": args.task,
            "heads_path": str(args.heads_path),
            "mean_activations_path": str(mean_path),
            "model_name": args.model_name,
        },
        output_path,
    )

    metadata = {
        "task": args.task,
        "model_name": args.model_name,
        "model_config": model_config,
        "heads_path": str(args.heads_path),
        "heads_source_tasks": heads_data.get("tasks"),
        "heads_total_prompts": heads_data.get("total_prompts"),
        "mean_activations_path": str(mean_path),
        "output_path": str(output_path),
        "n_top_heads": len(top_heads),
        "top_heads": top_heads,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
