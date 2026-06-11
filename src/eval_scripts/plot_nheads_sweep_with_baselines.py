#!/usr/bin/env python
"""Replot the train_varicl top-N head sweep WITH the task-specific and train-selected
(fixed-ICL multitask) FV steering curves overlaid. Pure matplotlib from saved JSON -- no GPU.

Per task, reads:
  - varicl top-N per-layer: results/heldout_varicl_nheads_sweep/<task>/nheads_sweep_by_layer.json
  - baselines per-layer:    results/heldout_multitask_head_eval/<task>/comparison_summary.json
                            (multitask_heads = train-selected fixed-ICL; task_specific_heads)
Writes <task>_nheads_sweep_with_baselines.png (zero-shot + 10-shot-shuffled panels).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=Path, default=Path("results/heldout_varicl_nheads_sweep"))
    ap.add_argument("--eval_root", type=Path, default=Path("results/heldout_multitask_head_eval"))
    ap.add_argument("--n_values", nargs="+", type=int, default=[10, 20, 30, 40])
    ap.add_argument("--tasks", nargs="+", default=None)
    return ap.parse_args()


def as_layer_map(by_layer):
    return {int(l): float(v) for l, v in by_layer.items()}


def plot_task(task, varicl_by_n, baselines, n_values, out_path):
    series = [("Zero-shot + FV", "zs_intervention_top1_by_layer"),
              ("10-shot shuffled + FV", "fs_shuffled_intervention_top1_by_layer")]
    cmap = plt.get_cmap("viridis")
    ncolors = {n: cmap(i / max(1, len(n_values) - 1)) for i, n in enumerate(n_values)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, (title, key) in zip(axes, series):
        # varicl top-N curves
        for n in n_values:
            m = as_layer_map(varicl_by_n[str(n)][key])
            xs = sorted(m)
            ax.plot(xs, [m[x] for x in xs], marker="o", ms=3.5, color=ncolors[n], label=f"varicl top-{n}")
        # baselines (dashed, distinct)
        if baselines:
            mt = as_layer_map(baselines["multitask_heads"][key])
            ts = as_layer_map(baselines["task_specific_heads"][key])
            xs = sorted(mt)
            ax.plot(xs, [mt[x] for x in xs], color="black", lw=1.8, ls="--", marker="x", ms=4,
                    label="train-selected (fixed-ICL, top-10)")
            xs = sorted(ts)
            ax.plot(xs, [ts[x] for x in xs], color="crimson", lw=1.8, ls=":", marker="s", ms=3.5,
                    label="task-specific (top-10)")
        ax.set_title(title)
        ax.set_xlabel("Edit layer")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Intervention top-1 accuracy")
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle(f"{task} — train_varicl top-N sweep vs task-specific & train-selected FVs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    tasks = args.tasks or sorted(p.name for p in args.sweep_root.iterdir()
                                 if p.is_dir() and not p.name.startswith("_"))
    for task in tasks:
        sweep_f = args.sweep_root / task / "nheads_sweep_by_layer.json"
        comp_f = args.eval_root / task / "comparison_summary.json"
        if not sweep_f.exists():
            print(f"  skip {task}: no {sweep_f}")
            continue
        varicl_by_n = json.loads(sweep_f.read_text())
        baselines = json.loads(comp_f.read_text()) if comp_f.exists() else None
        if baselines is None:
            print(f"  WARNING {task}: no baseline comparison_summary.json")
        out = plot_task(task, varicl_by_n, baselines, args.n_values,
                        args.sweep_root / task / f"{task}_nheads_sweep_with_baselines.png")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
