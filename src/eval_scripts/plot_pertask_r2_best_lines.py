#!/usr/bin/env python
"""Line plot of per-task best-over-layers ridge R^2 vs token position.

Reads the per_task_r2.csv written by plot_fulldim_ridge_pertask_r2.py; for each task and each of
the 31 token positions (icl example x role, ordered icl01/pre .. icl10/finaltok) takes the MAX
test R^2 over the 29 layers. One line per task.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FV_FORMATION_DIR
from eval_scripts.merge_fulldim_ridge_results import position_key, position_label, run_title

# Fixed categorical hue order (validated palette; identity never color-alone: lines are also
# direct-labeled at their right end).
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]


def parse_args():
    p = argparse.ArgumentParser(description="Best-over-layers per-task R^2 vs token position.")
    p.add_argument("--input_csv", type=Path,
                   default=FV_FORMATION_DIR / "fulldim_ridge_activation_to_fv_varicl_top40_plus_numbers"
                   / "per_task_r2" / "per_task_r2.csv")
    p.add_argument("--tasks", nargs="+",
                   default=["antonym", "synonym", "prev_number", "next_number"],
                   help="Tasks to plot, in legend/color order.")
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <csv dir>/best_r2_by_position_lines.png")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.output if args.output is not None else args.input_csv.parent / "best_r2_by_position_lines.png"

    best = {}  # (task, (icl, role)) -> max r2 over layers
    with open(args.input_csv) as f:
        for r in csv.DictReader(f):
            if r["task"] not in args.tasks:
                continue
            key = (r["task"], (int(r["icl_example_index"]), r["token_role"]))
            v = float(r["test_r2"])
            if key not in best or v > best[key]:
                best[key] = v

    positions = sorted({k[1] for k in best}, key=lambda ir: position_key(*ir))
    labels = [position_label(*p) for p in positions]
    x = range(len(positions))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, task in enumerate(args.tasks):
        y = [best[(task, pos)] for pos in positions]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=task)
        ax.annotate(task, (len(positions) - 1 + 0.3, y[-1]), color=color,
                    fontsize=8, va="center", ha="left", annotation_clip=False)

    # Boundaries between ICL examples, so "which example" reads at a glance.
    for j in range(1, len(positions)):
        if positions[j][0] != positions[j - 1][0]:
            ax.axvline(j - 0.5, color="0.88", linewidth=0.8, zorder=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_xlim(-0.5, len(positions) + 2.2)  # right headroom for the direct labels
    ax.set_xlabel("token position (icl example / role)")
    ax.set_ylabel("best test R² over layers (train-mean baseline)")
    ax.yaxis.grid(True, color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Best-over-layers R² by token position, per held-out task")
    fig.suptitle(run_title(args.input_csv.parent.parent.name), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
