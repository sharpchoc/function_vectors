#!/usr/bin/env python
"""Stage-2: build variable-ICL function vectors from the train-pooled top-head set.

Mirror of compute_all_task_fvs_from_multitask_heads.py for the variable-ICL method. The key
difference is that the per-task mean head activations here are SINGLE-POSITION tensors of shape
(n_layers, n_heads, head_dim) -- read at the query predictive token -- so the function vector
construction indexes mean_activations[layer, head] (NOT mean_activations[L, H, -1], which would
wrongly grab a scalar off a 3-D tensor). For every task in the manifest (train + test) the FV is
the sum over the top-N train-pooled heads of that task's mean head activations, projected to the
residual stream through each head's attention output projection.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.eval_scripts.compute_all_task_fvs_from_multitask_heads import (
    get_out_projection,
    load_top_heads,
    torch_load_trusted,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Build variable-ICL FVs from the train-pooled top-head set for all tasks.")
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/multitask_aie_heads_varicl"),
                        help="Root holding each task's {task}_mean_head_activations_varicl.pt.")
    parser.add_argument("--heads_path", type=Path,
                        default=Path("results/multitask_aie_heads_varicl/multitask_top_aie_heads.pt"))
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of manifest tasks to build (for sharding across instances). "
                             "Default: all train+test tasks.")
    parser.add_argument("--manifest_name", type=str, default="fv_manifest.json",
                        help="Manifest filename under --output_root. Give each parallel shard a "
                             "distinct name (e.g. fv_manifest.part1.json) so they don't clobber.")
    parser.add_argument("--output_root", type=Path, default=Path("results/function_vectors/gpt-j/train_varicl"))
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compute_function_vector_from_heads(mean_activations, top_heads, model, model_config):
    """Sum the top-head contributions projected through each head's attention out_proj.

    mean_activations is (n_layers, n_heads, head_dim) -- a single (query-token) position --
    so the per-head slot is mean_activations[layer, head] (no token axis).
    """
    model_resid_dim = model_config["resid_dim"]
    model_head_dim = model_resid_dim // model_config["n_heads"]
    function_vector = torch.zeros((1, 1, model_resid_dim), device=model.device)
    for layer, head, _ in top_heads:
        out_proj = get_out_projection(model, model_config, layer)
        x = torch.zeros(model_resid_dim)
        x[head * model_head_dim : (head + 1) * model_head_dim] = mean_activations[layer, head]
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
    print(f"Using {len(top_heads)} train-pooled top heads: {[(l, h) for l, h, _ in top_heads]}")

    torch.set_grad_enabled(False)
    print("Loading model")
    model, _, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device, revision=args.revision)
    model.eval()

    args.output_root.mkdir(parents=True, exist_ok=True)
    # Copy the top-head artifact into output_root as heads.pt so write_fv_head_metadata.py can read it.
    heads_out = args.output_root / "heads.pt"
    if heads_out.exists() and not args.overwrite:
        raise FileExistsError(f"{heads_out} exists. Pass --overwrite to replace it.")
    shutil.copyfile(args.heads_path, heads_out)

    produced = []
    for task in tasks:
        mean_path = args.fv_root / task / f"{task}_mean_head_activations_varicl.pt"
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
                "fv_source": "multitask_top_heads_varicl",
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
        "heads_pt": str(heads_out),
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
