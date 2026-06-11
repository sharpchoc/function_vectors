#!/usr/bin/env python
"""Compute task function vectors from the multitask (general) top-head set, for all tasks.

For every task in the manifest (train + test), builds a function vector as the sum over the
top-N multitask-selected heads of that task's mean head activations, projected to the
residual stream through each head's attention output projection. This is the same
construction as compute_task_fv_from_multitask_heads.py, but loads the model once and loops
over all tasks, writing FVs under a parallel fv_root with the standard filename
({task}/{task}_function_vector.pt) so the regression/PCA pipeline can consume them directly
via --fv_root.
"""
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
    parser = argparse.ArgumentParser(description="Build FVs from the multitask top-head set for all tasks.")
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"),
                        help="Root holding each task's {task}_mean_head_activations.pt.")
    parser.add_argument("--heads_path", type=Path,
                        default=Path("results/multitask_aie_heads/multitask_top_aie_heads.pt"))
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of manifest tasks to build (for sharding across instances). "
                             "Default: all train+test tasks.")
    parser.add_argument("--manifest_name", type=str, default="fv_manifest.json",
                        help="Manifest filename under --output_root. Give each parallel shard a "
                             "distinct name (e.g. fv_manifest.part1.json) so they don't clobber.")
    parser.add_argument("--output_root", type=Path, default=Path("results/gptj_fv_multitask_top10"))
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
    manifest = json.loads(args.task_manifest.read_text())
    tasks = list(manifest["train_tasks"]) + list(manifest["test_tasks"])
    if args.tasks is not None:
        requested = set(args.tasks)
        unknown = requested - set(tasks)
        if unknown:
            raise ValueError(f"--tasks not in manifest: {sorted(unknown)}")
        tasks = [t for t in tasks if t in requested]  # keep manifest order
        print(f"Sharded build: {len(tasks)} of {len(manifest['train_tasks']) + len(manifest['test_tasks'])} tasks")

    top_heads, heads_data = load_top_heads(args.heads_path, args.n_top_heads)
    print(f"Using {len(top_heads)} multitask top heads: {[(l, h) for l, h, _ in top_heads]}")

    torch.set_grad_enabled(False)
    print("Loading model")
    model, _, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device, revision=args.revision)
    model.eval()

    args.output_root.mkdir(parents=True, exist_ok=True)
    produced = []
    for task in tasks:
        mean_path = args.fv_root / task / f"{task}_mean_head_activations.pt"
        if not mean_path.exists():
            raise FileNotFoundError(mean_path)
        out_dir = args.output_root / task
        out_path = out_dir / f"{task}_function_vector.pt"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")

        mean_activations = torch_load_trusted(mean_path, map_location="cpu")
        fv = compute_function_vector_from_heads(mean_activations, top_heads, model, model_config)
        fv = fv.detach().float().cpu().reshape(-1)

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "function_vector": fv,
                "top_heads": top_heads,
                "n_top_heads": len(top_heads),
                "task": task,
                "fv_source": "multitask_top_heads",
                "heads_path": str(args.heads_path),
                "mean_activations_path": str(mean_path),
                "model_name": args.model_name,
            },
            out_path,
        )
        split = "train" if task in manifest["train_tasks"] else "test"
        produced.append({"task": task, "split": split, "fv_path": str(out_path),
                         "fv_norm": float(fv.norm())})
        print(f"  {task:24s} [{split}] ||fv||={float(fv.norm()):.3f} -> {out_path}")

    summary = {
        "model_name": args.model_name,
        "heads_path": str(args.heads_path),
        "n_top_heads": len(top_heads),
        "top_heads": top_heads,
        "heads_source_tasks": heads_data.get("tasks"),
        "fv_root_input": str(args.fv_root),
        "output_root": str(args.output_root),
        "task_manifest": str(args.task_manifest),
        "n_tasks": len(produced),
        "produced": produced,
    }
    (args.output_root / args.manifest_name).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(produced)} function vectors to {args.output_root}")
    print(args.output_root / args.manifest_name)


if __name__ == "__main__":
    main()
