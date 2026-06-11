#!/usr/bin/env python
"""Select top-AIE heads for an arbitrary SUBSET of tasks from precomputed per-task CIE results.

This reuses the per-task <task>/<task>_cie_result.pt files written by
compute_multitask_top_aie_heads.py. Each file stores the summed causal indirect effect
(`indirect_effect_sum`) and the prompt count (`n_prompts`) for that task, so any subset of
tasks can be re-aggregated as a prompt-count-weighted mean without recomputing CIE:

    mean_subset = sum_t(indirect_effect_sum_t) / sum_t(n_prompts_t)

Examples:
  # heads for an explicit subset
  python src/eval_scripts/select_heads_from_cie_subset.py \
    --cie_root results/multitask_aie_heads_all_tasks \
    --tasks antonym synonym country-capital --n_top_heads 40 \
    --output results/multitask_aie_heads_all_tasks/subsets/lexical_top_heads.pt

  # heads for the held-out test_tasks split, using the all-tasks CIE cache
  python src/eval_scripts/select_heads_from_cie_subset.py \
    --cie_root results/multitask_aie_heads_all_tasks \
    --task_split_path task_splits/abstractive_train_test_tasks_29.json \
    --task_split_keys test_tasks
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cie_root",
        type=Path,
        required=True,
        help="Directory containing <task>/<task>_cie_result.pt files (a compute_multitask save_path_root).",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Explicit task subset.")
    parser.add_argument(
        "--task_split_path",
        type=Path,
        default=None,
        help="Optional split JSON to draw tasks from instead of --tasks.",
    )
    parser.add_argument(
        "--task_split_keys",
        nargs="+",
        default=["train_tasks"],
        help="Keys in --task_split_path to concatenate when --task_split_path is used.",
    )
    parser.add_argument("--n_top_heads", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None, help="Optional .pt path to save the selection.")
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def resolve_tasks(args):
    if args.tasks is not None:
        return list(args.tasks)
    if args.task_split_path is None:
        raise ValueError("Provide either --tasks or --task_split_path.")
    with open(args.task_split_path, "r") as f:
        split = json.load(f)
    tasks, seen = [], set()
    for key in args.task_split_keys:
        if key not in split:
            raise KeyError(f"{args.task_split_path} has no key '{key}'")
        for task in split[key]:
            if task not in seen:
                seen.add(task)
                tasks.append(task)
    return tasks


def cie_result_path(root, task):
    return root / task / f"{task}_cie_result.pt"


def top_heads_from_scores(scores, n_top_heads):
    k = min(int(n_top_heads), scores.numel())
    topk_vals, topk_inds = torch.topk(scores.reshape(-1), k=k, largest=True)
    layers, heads = np.unravel_index(topk_inds.cpu().numpy(), tuple(scores.shape))
    return [
        (int(layer), int(head), round(float(score), 6))
        for layer, head, score in zip(layers, heads, topk_vals.cpu())
    ]


def main():
    args = parse_args()
    tasks = resolve_tasks(args)
    print(f"Selecting heads from {len(tasks)} task(s): {tasks}")

    global_sum = None
    total_prompts = 0
    per_task = []
    missing = []
    for task in tasks:
        path = cie_result_path(args.cie_root, task)
        if not path.exists():
            missing.append(task)
            continue
        result = torch_load_trusted(path, map_location="cpu")
        ie_sum = result["indirect_effect_sum"].double()
        global_sum = ie_sum.clone() if global_sum is None else global_sum + ie_sum
        total_prompts += int(result["n_prompts"])
        per_task.append({"task": task, "n_prompts": int(result["n_prompts"]), "cie_result_path": str(path)})

    if missing:
        raise FileNotFoundError(f"No per-task CIE result found for: {missing} under {args.cie_root}")
    if global_sum is None or total_prompts == 0:
        raise RuntimeError("No prompts aggregated; nothing to select.")

    mean_indirect_effect = (global_sum / total_prompts).float()
    top_heads = top_heads_from_scores(mean_indirect_effect, args.n_top_heads)

    print(f"\nTotal prompts: {total_prompts}")
    print("Top heads:")
    for layer, head, score in top_heads:
        print(f"L{layer} H{head}: {score}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean_indirect_effect": mean_indirect_effect.cpu(),
                "top_heads": top_heads,
                "n_top_heads": int(args.n_top_heads),
                "tasks": tasks,
                "total_prompts": int(total_prompts),
                "per_task": per_task,
                "cie_root": str(args.cie_root),
            },
            args.output,
        )
        print(f"\nSaved selection: {args.output}")


if __name__ == "__main__":
    main()
