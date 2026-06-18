#!/usr/bin/env python
"""Held-out per-layer steering eval for the max_shots=4 variable-ICL FV (top-40 heads), plotted
against the existing max_shots=10 variable-ICL FV (top-40) and the task-specific FV reference.

Only the max-4 FV is evaluated here (one layer sweep per test task). The max-10 top-40 curve is read
from the prebuilt nheads sweep (`results/heldout_varicl_nheads_sweep/<task>/nheads_sweep_by_layer.json`,
N=40 entry) and the task-specific curve from the original held-out eval
(`results/heldout_multitask_head_eval/<task>/comparison_summary.json`, task_specific_heads) — both use
the SAME filter set / seed / layer sweep, so all three lines are directly overlayable. No recompute of
the max-10 or task-specific lines, and no no-FV-baseline recompute (the cached fs_results_layer_sweep
filter is reused).
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
    write_json,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset

SERIES = [
    ("Zero-shot + FV", "zs_intervention_top1_by_layer"),
    ("10-shot shuffled + FV", "fs_shuffled_intervention_top1_by_layer"),
]
METHODS = [  # (label, summary-dict-key, marker, color)
    ("Variable-ICL max=4 (top-40)", "train_varicl_max4", "^", "tab:red"),
    ("Variable-ICL max=10 (top-40)", "train_varicl_max10", "o", "tab:green"),
    ("Task-specific heads", "task_specific", "s", "tab:orange"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Steering eval for the max_shots=4 varicl FV (top-40).")
    p.add_argument("--task_split_path", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--max4_fv_root", type=Path, default=Path("results/function_vectors/gpt-j/train_varicl_max4_top40"))
    # Reference curves (read-only, no recompute).
    p.add_argument("--max10_sweep_root", type=Path, default=Path("results/heldout_varicl_nheads_sweep"),
                   help="Holds <task>/nheads_sweep_by_layer.json with the N=40 max-10 varicl curve.")
    p.add_argument("--max10_n_top_heads", type=str, default="40")
    p.add_argument("--baseline_eval_root", type=Path, default=Path("results/heldout_multitask_head_eval"),
                   help="Holds <task>/comparison_summary.json with the task_specific_heads curve.")
    # Filter-set source (method-agnostic cached baseline).
    p.add_argument("--filter_fv_root", type=Path, default=Path("results/gptj_fv"))
    p.add_argument("--output_root", type=Path, default=Path("results/heldout_varicl_max4_top40"))
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
    return json.loads(args.task_split_path.read_text())[args.task_split_key]


def read_max10_curve(args, task):
    """N=40 max-10 varicl by-layer curve from the prebuilt nheads sweep."""
    path = args.max10_sweep_root / task / "nheads_sweep_by_layer.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get(args.max10_n_top_heads)
    return entry  # dict with zs_/fs_..._by_layer keys, or None


def read_task_specific_curve(args, task):
    path = args.baseline_eval_root / task / "comparison_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("task_specific_heads")


def plot_task(task, curves, plot_path):
    """curves[method_key] = summary dict (or None) with the two by-layer keys."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, layer_key) in zip(axes, SERIES):
        for label, key, marker, color in METHODS:
            summ = curves.get(key)
            if summ is None or layer_key not in summ:
                continue
            by_layer = {int(l): float(v) for l, v in summ[layer_key].items() if v is not None}
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


def mean_by_layer(summaries, layer_key):
    """Mean across tasks of a by-layer curve; layers present in every summary."""
    per_layer = {}
    counts = {}
    for summ in summaries:
        if summ is None or layer_key not in summ:
            continue
        for l, v in summ[layer_key].items():
            if v is None:
                continue
            per_layer[l] = per_layer.get(l, 0.0) + float(v)
            counts[l] = counts.get(l, 0) + 1
    n = len(summaries)
    # keep only layers present for all contributing tasks
    return {l: per_layer[l] / counts[l] for l in per_layer if counts[l] == n}


def best_layer(summ):
    if summ is None:
        return None, None, None, None
    zs = {int(l): float(v) for l, v in summ.get("zs_intervention_top1_by_layer", {}).items() if v is not None}
    fs = {int(l): float(v) for l, v in summ.get("fs_shuffled_intervention_top1_by_layer", {}).items() if v is not None}
    bz = max(zs, key=zs.get) if zs else None
    bf = max(fs, key=fs.get) if fs else None
    return (None if bz is None else bz, None if bz is None else zs[bz],
            None if bf is None else bf, None if bf is None else fs[bf])


def main():
    args = parse_args()
    tasks = load_test_tasks(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    per_task_max4, per_task_max10, per_task_ts = [], [], []
    aggregate = {"tasks": tasks, "max4_fv_root": str(args.max4_fv_root), "per_task": []}

    for task in tasks:
        print(f"\n=== {task} ===")
        out_dir = args.output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)

        fv_path = args.max4_fv_root / task / f"{task}_function_vector.pt"
        if not fv_path.exists():
            raise FileNotFoundError(fv_path)

        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)

        filter_args = SimpleNamespace(
            filter_to_correct_icl=args.filter_to_correct_icl, generate_str=args.generate_str,
            fv_root=args.filter_fv_root, seed=args.seed, n_shots=args.n_shots, metric=args.metric,
            prefixes=args.prefixes, separators=args.separators, batch_size_baseline=args.batch_size_baseline,
        )
        filter_set, filter_source = get_filter_set(filter_args, task, dataset, model, model_config, tokenizer, out_dir)
        n_filt = None if filter_set is None else int(len(filter_set))
        print(f"  filter set: {n_filt} examples (source {filter_source})")

        max4_fv, max4_heads = load_function_vector(fv_path)
        eval_args = SimpleNamespace(
            edit_layer=args.edit_layer, seed=args.seed, n_shots=args.n_shots,
            prefixes=args.prefixes, separators=args.separators, generate_str=args.generate_str,
            metric=args.metric, filter_set=filter_set,
        )
        print("  Evaluating max_shots=4 varicl FV (top-40) across layers")
        zs, fs = evaluate_fv(eval_args, dataset, max4_fv, model, model_config, tokenizer)
        max4_summary = {"n_top_heads": None if max4_heads is None else len(max4_heads), **summarize_results(zs, fs)}

        max10_summary = read_max10_curve(args, task)
        ts_summary = read_task_specific_curve(args, task)
        if max10_summary is None:
            print(f"  WARNING: no max-10 N={args.max10_n_top_heads} curve for {task}")
        if ts_summary is None:
            print(f"  WARNING: no task-specific curve for {task}")

        per_task_max4.append(max4_summary)
        per_task_max10.append(max10_summary)
        per_task_ts.append(ts_summary)

        curves = {"train_varicl_max4": max4_summary, "train_varicl_max10": max10_summary,
                  "task_specific": ts_summary}
        plot_path = plot_task(task, curves, out_dir / f"{task}_effectiveness_by_layer_max4_vs_max10.png")

        bz4 = best_layer(max4_summary); bz10 = best_layer(max10_summary); bzts = best_layer(ts_summary)
        task_entry = {
            "task": task, "n_filtered_test_examples": n_filt, "filter_source": filter_source,
            "best_zs": {"max4": bz4[1], "max4_layer": bz4[0], "max10": bz10[1], "max10_layer": bz10[0],
                        "task_specific": bzts[1], "task_specific_layer": bzts[0]},
            "best_fs_shuffled": {"max4": bz4[3], "max4_layer": bz4[2], "max10": bz10[3], "max10_layer": bz10[2],
                                 "task_specific": bzts[3], "task_specific_layer": bzts[2]},
            "effectiveness_plot_path": str(plot_path),
        }
        write_json(out_dir / f"{task}_summary.json", {**task_entry, "train_varicl_max4": max4_summary})
        aggregate["per_task"].append(task_entry)
        print(f"  best zs: max4 {bz4[1]}@L{bz4[0]} | max10 {bz10[1]}@L{bz10[0]} | task-spec {bzts[1]}@L{bzts[0]}")

    # Aggregate (mean over tasks) plot + table.
    agg_curves = {}
    for key, summaries in (("train_varicl_max4", per_task_max4),
                           ("train_varicl_max10", per_task_max10),
                           ("task_specific", per_task_ts)):
        agg_curves[key] = {
            "zs_intervention_top1_by_layer": mean_by_layer(summaries, "zs_intervention_top1_by_layer"),
            "fs_shuffled_intervention_top1_by_layer": mean_by_layer(summaries, "fs_shuffled_intervention_top1_by_layer"),
        }
    plot_task("AGGREGATE (mean over test tasks)", agg_curves,
              args.output_root / "AGGREGATE_effectiveness_by_layer_max4_vs_max10.png")

    def mean_best(entries, cond, field):
        vals = [e[cond][field] for e in entries if e[cond][field] is not None]
        return float(np.mean(vals)) if vals else None

    aggregate["mean_best"] = {
        "zs": {m: mean_best(aggregate["per_task"], "best_zs", m) for m in ("max4", "max10", "task_specific")},
        "fs_shuffled": {m: mean_best(aggregate["per_task"], "best_fs_shuffled", m) for m in ("max4", "max10", "task_specific")},
    }
    write_json(args.output_root / "heldout_varicl_max4_summary.json", aggregate)
    print(f"\nmean best-layer top-1: {json.dumps(aggregate['mean_best'], indent=2)}")
    print(f"\nWrote {args.output_root}/heldout_varicl_max4_summary.json")


if __name__ == "__main__":
    main()
