#!/usr/bin/env python
"""Add the variable-ICL (`train_varicl`) FV as a third series to the held-out per-layer
steering comparison.

The existing held-out eval (`evaluate_heldout_multitask_head_fvs.py`) wrote, for each of the
9 test tasks, per-layer zero-shot + 10-shot-shuffled intervention top-1 for two methods
(train-only multitask heads vs task-specific) into
`results/heldout_multitask_head_eval/<task>/comparison_summary.json`. Rather than recompute
those (a ~hours-long sweep), this script reuses them and ONLY evaluates the prebuilt
`train_varicl` FV, using the SAME filter set (clean_rank_list==0 from
`results/gptj_fv/<task>/fs_results_layer_sweep.json`) and the SAME seed/layer sweep, so the
new line is directly overlayable. It then writes a 3-series plot
`<task>_effectiveness_by_layer_with_varicl.png` and a `varicl_comparison_summary.json`.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_heldout_multitask_head_fvs import (
    evaluate_fv,
    get_filter_set,
    load_function_vector,
    summarize_results,
    torch_load_trusted,
    write_json,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Overlay the train_varicl FV on the held-out per-layer steering plots.")
    p.add_argument("--task_split_path", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    # Source of the existing two-series results + the canonical filter set.
    p.add_argument("--eval_root", type=Path, default=Path("results/heldout_multitask_head_eval"))
    p.add_argument("--filter_fv_root", type=Path, default=Path("results/gptj_fv"),
                   help="Where each task's fs_results_layer_sweep.json (filter source) lives.")
    # The variable-ICL FVs to evaluate.
    p.add_argument("--varicl_fv_root", type=Path, default=Path("results/function_vectors/gpt-j/train_varicl"))
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--edit_layer", type=int, default=-1)
    p.add_argument("--batch_size_baseline", type=int, default=1)
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--generate_str", action="store_true")
    p.add_argument("--metric", type=str, default="f1_score")
    p.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    p.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    p.set_defaults(filter_to_correct_icl=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_test_tasks(args):
    if args.tasks is not None:
        return args.tasks
    split = json.loads(args.task_split_path.read_text())
    return split[args.task_split_key]


def plot_three_series(task, existing, varicl_summary, plot_path):
    """existing = comparison_summary.json dict (multitask_heads + task_specific_heads)."""
    series = [("Zero-shot + FV", "zs_intervention_top1_by_layer"),
              ("10-shot shuffled + FV", "fs_shuffled_intervention_top1_by_layer")]
    methods = [
        ("Multitask heads (train, fixed-ICL)", existing["multitask_heads"], "o", "tab:blue"),
        ("Task-specific heads", existing["task_specific_heads"], "s", "tab:orange"),
        ("Variable-ICL multitask (train_varicl)", varicl_summary, "^", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, key) in zip(axes, series):
        for label, summ, marker, color in methods:
            by_layer = {int(l): float(v) for l, v in summ[key].items()}
            layers = sorted(by_layer)
            ax.plot(layers, [by_layer[l] for l in layers], marker=marker, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Edit layer")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Intervention top-1 accuracy")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(task)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()
    tasks = load_test_tasks(args)

    torch.set_grad_enabled(False)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    aggregate = {"tasks": tasks, "varicl_fv_root": str(args.varicl_fv_root), "comparisons": {}}

    for task in tasks:
        print(f"\n=== {task} ===")
        task_dir = args.eval_root / task
        existing_path = task_dir / "comparison_summary.json"
        if not existing_path.exists():
            raise FileNotFoundError(f"{existing_path} (run evaluate_heldout_multitask_head_fvs.py first)")
        existing = json.loads(existing_path.read_text())

        varicl_fv_path = args.varicl_fv_root / task / f"{task}_function_vector.pt"
        if not varicl_fv_path.exists():
            raise FileNotFoundError(varicl_fv_path)

        out_path = task_dir / "varicl_comparison_summary.json"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")

        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)

        # Reproduce the SAME filter set the original run used (reads the cached fs_results).
        filter_args = SimpleNamespace(
            filter_to_correct_icl=args.filter_to_correct_icl, generate_str=args.generate_str,
            fv_root=args.filter_fv_root, seed=args.seed, n_shots=args.n_shots, metric=args.metric,
            prefixes=args.prefixes, separators=args.separators, batch_size_baseline=args.batch_size_baseline,
        )
        filter_set, filter_source = get_filter_set(filter_args, task, dataset, model, model_config, tokenizer, task_dir)
        if filter_source != existing.get("filter_source"):
            print(f"  WARNING: filter_source mismatch: {filter_source} vs recorded {existing.get('filter_source')}")
        n_filt = None if filter_set is None else int(len(filter_set))
        print(f"  filter set: {n_filt} examples (recorded {existing.get('n_filtered_test_examples')})")

        varicl_fv, varicl_heads = load_function_vector(varicl_fv_path)

        eval_args = SimpleNamespace(
            edit_layer=args.edit_layer, seed=args.seed, n_shots=args.n_shots,
            prefixes=args.prefixes, separators=args.separators, generate_str=args.generate_str,
            metric=args.metric, filter_set=filter_set,
        )
        print("  Evaluating train_varicl FV across layers")
        varicl_zs, varicl_fs = evaluate_fv(eval_args, dataset, varicl_fv, model, model_config, tokenizer)

        write_json(task_dir / "varicl_heads_zs_results.json", varicl_zs)
        write_json(task_dir / "varicl_heads_fs_shuffled_results.json", varicl_fs)
        varicl_summary = {"top_heads": varicl_heads, **summarize_results(varicl_zs, varicl_fs)}

        plot_path = plot_three_series(
            task, existing, varicl_summary, task_dir / f"{task}_effectiveness_by_layer_with_varicl.png"
        )
        task_out = {
            "task": task,
            "varicl_fv_path": str(varicl_fv_path),
            "filter_source": filter_source,
            "n_filtered_test_examples": n_filt,
            "train_varicl": varicl_summary,
            "multitask_heads": existing["multitask_heads"],
            "task_specific_heads": existing["task_specific_heads"],
            "effectiveness_plot_path": str(plot_path),
        }
        write_json(out_path, task_out)
        aggregate["comparisons"][task] = {
            "best_zs": {
                "train_varicl": varicl_summary["best_zs_intervention_top1"],
                "multitask": existing["multitask_heads"]["best_zs_intervention_top1"],
                "task_specific": existing["task_specific_heads"]["best_zs_intervention_top1"],
            },
            "best_fs_shuffled": {
                "train_varicl": varicl_summary["best_fs_shuffled_intervention_top1"],
                "multitask": existing["multitask_heads"]["best_fs_shuffled_intervention_top1"],
                "task_specific": existing["task_specific_heads"]["best_fs_shuffled_intervention_top1"],
            },
        }
        print(f"  wrote {plot_path}")

    agg_path = args.eval_root / "heldout_varicl_vs_others_summary.json"
    write_json(agg_path, aggregate)
    print(f"\n{agg_path}")


if __name__ == "__main__":
    main()
