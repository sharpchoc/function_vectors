#!/usr/bin/env python
"""SANDBOX: plot the matched per-prompt-FV steering arm against cached baselines.

Reads (no GPU): per-prompt summaries from evaluate_heldout_perprompt_fv.py, the cached
train_varicl_top40 curves (heldout_varicl_nheads_sweep/<task>/nheads_sweep_by_layer.json["40"])
and the cached task-specific curves (heldout_multitask_head_eval/<task>/comparison_summary.json).

Grid-only PNG policy: two figures + one CSV.
  * steering_grid_by_task.png  -- one row per task x 2 columns (zero-shot, 10-shot shuffled)
  * steering_aggregate.png     -- 1x2, unweighted mean over tasks
  * best_layer_summary.csv     -- best-layer top-1 per arm per task
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.paths import RESULTS_ROOT, STEERING_COMPARISON_DIR  # noqa: E402

CONDITIONS = [("Zero-shot + FV", "zs_intervention_top1_by_layer"),
              ("10-shot shuffled + FV", "fs_shuffled_intervention_top1_by_layer")]
ARMS = [  # (label, marker, color)
    ("Per-prompt FV (matched query)", "P", "tab:red"),
    ("Variable-ICL top-40 (canonical)", "^", "tab:green"),
    ("Task-specific heads", "s", "tab:orange"),
]


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: per-prompt steering comparison plots.")
    p.add_argument("--task_split_path", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--perprompt_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/steering_eval")
    p.add_argument("--nheads_sweep_root", type=Path,
                   default=STEERING_COMPARISON_DIR / "heldout_varicl_nheads_sweep")
    p.add_argument("--base_eval_root", type=Path,
                   default=STEERING_COMPARISON_DIR / "heldout_multitask_head_eval")
    p.add_argument("--output_dir", type=Path, default=None, help="Default: --perprompt_root")
    return p.parse_args()


def by_layer(summary, key):
    return {int(l): float(v) for l, v in summary[key].items()}


def load_task_curves(args, task):
    perprompt = json.loads((args.perprompt_root / task / "perprompt_summary.json").read_text())["perprompt"]
    varicl40 = json.loads((args.nheads_sweep_root / task / "nheads_sweep_by_layer.json").read_text())["40"]
    task_specific = json.loads((args.base_eval_root / task / "comparison_summary.json").read_text())["task_specific_heads"]
    return [perprompt, varicl40, task_specific]  # ARMS order


def draw_condition(ax, curves, key):
    for (label, marker, color), summ in zip(ARMS, curves):
        data = by_layer(summ, key)
        layers = sorted(data)
        ax.plot(layers, [data[l] for l in layers], marker=marker, markersize=3.5,
                color=color, label=label, lw=1.3)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    tasks = list(args.tasks) if args.tasks is not None else \
        json.loads(args.task_split_path.read_text())[args.task_split_key]
    out_dir = args.output_dir if args.output_dir is not None else args.perprompt_root
    out_dir.mkdir(parents=True, exist_ok=True)

    all_curves = {task: load_task_curves(args, task) for task in tasks}

    # Grid: rows = tasks, cols = conditions.
    fig, axes = plt.subplots(len(tasks), 2, figsize=(12.5, 2.9 * len(tasks)), squeeze=False,
                             sharex=True)
    for r, task in enumerate(tasks):
        for c, (title, key) in enumerate(CONDITIONS):
            ax = axes[r][c]
            draw_condition(ax, all_curves[task], key)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"{task}\ntop-1")
            if r == len(tasks) - 1:
                ax.set_xlabel("Edit layer")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle("SANDBOX: matched per-prompt FV steering vs baselines (intervention top-1)",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    grid_path = out_dir / "steering_grid_by_task.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Aggregate: unweighted mean over tasks (matching plot_nheads_steering_comparison).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, key) in zip(axes, CONDITIONS):
        for arm_idx, (label, marker, color) in enumerate(ARMS):
            per_task = [by_layer(all_curves[t][arm_idx], key) for t in tasks]
            layers = sorted(set.intersection(*(set(d) for d in per_task)))
            means = [float(np.mean([d[l] for d in per_task])) for l in layers]
            ax.plot(layers, means, marker=marker, markersize=4, color=color, label=label, lw=1.5)
        ax.set_title(f"{title} — mean over {len(tasks)} tasks")
        ax.set_xlabel("Edit layer")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Intervention top-1 accuracy")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("SANDBOX: matched per-prompt FV steering vs baselines (aggregate)")
    fig.tight_layout()
    agg_path = out_dir / "steering_aggregate.png"
    fig.savefig(agg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Best-layer CSV.
    csv_path = out_dir / "best_layer_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "arm", "best_zs_layer", "best_zs_top1",
                         "best_fs_shuffled_layer", "best_fs_shuffled_top1"])
        for task in tasks:
            for (label, _, _), summ in zip(ARMS, all_curves[task]):
                writer.writerow([task, label, summ["best_zs_layer"],
                                 f"{summ['best_zs_intervention_top1']:.4f}",
                                 summ["best_fs_shuffled_layer"],
                                 f"{summ['best_fs_shuffled_intervention_top1']:.4f}"])
    print(f"wrote {grid_path}\nwrote {agg_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
