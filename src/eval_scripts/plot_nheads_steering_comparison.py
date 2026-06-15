#!/usr/bin/env python
"""Overlay held-out steering effectiveness for the train-pooled multitask FV at several
n_top_heads (10/20/40) into a single PNG per task, with the task-specific FV as a fixed
reference line.

Reads each n's per-task comparison_summary.json (full per-layer curves) from
results/heldout_multitask_head_eval{,_top20,_top40}/ and writes one combined PNG per task
(+ an aggregate mean-over-tasks PNG) into a single output folder. Pure plotting; no model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]

# n_top_heads -> results root. n=10 is the pre-existing baseline folder; n=20/40 are
# consolidated under one parent (results/heldout_multitask_head_eval_nheads/top{20,40}).
NHEADS_DIR = REPO_ROOT / "results/heldout_multitask_head_eval_nheads"
DEFAULT_ROOTS = {
    10: REPO_ROOT / "results/heldout_multitask_head_eval",
    20: NHEADS_DIR / "top20",
    40: NHEADS_DIR / "top40",
}
CONDITIONS = [
    ("Zero-shot + FV", "zs_intervention_top1_by_layer"),
    ("10-shot shuffled + FV", "fs_shuffled_intervention_top1_by_layer"),
]
COLORS = {10: "tab:blue", 20: "tab:orange", 40: "tab:green"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, default=NHEADS_DIR)
    return p.parse_args()


def load_summaries(roots):
    """Return {n: {task: comparison_summary dict}} and the task list (from n=10)."""
    summaries = {}
    tasks = None
    for n, root in roots.items():
        agg = json.loads((root / "heldout_multitask_vs_task_specific_summary.json").read_text())
        per_task = {}
        for task in agg["tasks"]:
            per_task[task] = json.loads((root / task / "comparison_summary.json").read_text())
        summaries[n] = per_task
        if n == min(roots):
            tasks = agg["tasks"]
    return summaries, tasks


def curve(summary, group, key):
    """layer-sorted (layers, scores) for one condition of one FV group."""
    d = {int(layer): float(score) for layer, score in summary[group][key].items()}
    layers = sorted(d)
    return layers, [d[layer] for layer in layers]


def plot_task(task, summaries, ns, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, key) in zip(axes, CONDITIONS):
        for n in ns:
            layers, scores = curve(summaries[n][task], "multitask_heads", key)
            ax.plot(layers, scores, marker="o", ms=3, color=COLORS[n],
                    label=f"Multitask heads (n={n})")
        # task-specific FV is independent of n; take it from n=10
        layers, scores = curve(summaries[min(ns)][task], "task_specific_heads", key)
        ax.plot(layers, scores, marker="s", ms=3, color="black", ls="--",
                label="Task-specific heads")
        ax.set_title(title)
        ax.set_xlabel("Edit layer")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Intervention top-1 accuracy")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{task} — train-pooled FV at n_top_heads 10/20/40 vs task-specific")
    fig.tight_layout()
    out = output_dir / f"{task}_effectiveness_by_layer_nheads.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_aggregate(tasks, summaries, ns, output_dir):
    """Mean over tasks of the per-layer curve (only layers common to all tasks)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, key) in zip(axes, CONDITIONS):
        for n in ns:
            stacks = []
            for task in tasks:
                _, scores = curve(summaries[n][task], "multitask_heads", key)
                stacks.append(scores)
            m = np.mean(np.array(stacks), axis=0)
            ax.plot(range(len(m)), m, marker="o", ms=3, color=COLORS[n],
                    label=f"Multitask heads (n={n})")
        stacks = [curve(summaries[min(ns)][task], "task_specific_heads", key)[1] for task in tasks]
        m = np.mean(np.array(stacks), axis=0)
        ax.plot(range(len(m)), m, marker="s", ms=3, color="black", ls="--",
                label="Task-specific heads")
        ax.set_title(title)
        ax.set_xlabel("Edit layer")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean intervention top-1 accuracy")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Mean over {len(tasks)} held-out tasks — n_top_heads 10/20/40 vs task-specific")
    fig.tight_layout()
    out = output_dir / "AGGREGATE_effectiveness_by_layer_nheads.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    roots = {n: r for n, r in DEFAULT_ROOTS.items() if r.exists()}
    if len(roots) < 2:
        raise SystemExit(f"Need >=2 n-head result roots; found {sorted(roots)}")
    ns = sorted(roots)
    summaries, tasks = load_summaries(roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = [plot_task(task, summaries, ns, args.output_dir) for task in tasks]
    written.append(plot_aggregate(tasks, summaries, ns, args.output_dir))
    print(f"Wrote {len(written)} PNGs to {args.output_dir} (n_top_heads={ns})")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
