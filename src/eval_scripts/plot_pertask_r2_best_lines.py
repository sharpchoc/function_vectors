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
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/varicl_top40_plus_number_digits"
                   / "per_task_r2" / "per_task_r2.csv")
    p.add_argument("--tasks", nargs="+",
                   default=["antonym", "synonym", "prev_number_digits", "next_number_digits"],
                   help="Tasks to plot, in legend/color order. Number tasks default to the "
                        "_digits variants (project convention 2026-07-20); pass the word tasks "
                        "explicitly to plot the old study.")
    p.add_argument("--roles", nargs="+", default=None,
                   help="Only plot these token roles (e.g. pre_label_token). Default: all.")
    p.add_argument("--average_roles", action="store_true",
                   help="Collapse the selected roles into ONE point per ICL example: the mean of "
                        "the per-role best-over-layers R^2 values.")
    p.add_argument("--average_label", type=str, default="mean",
                   help="Role shorthand shown in x tick labels when --average_roles is set.")
    p.add_argument("--value_column", type=str, default="test_r2",
                   help="CSV column holding the per-cell value to maximize over layers.")
    p.add_argument("--ylabel", type=str, default="best test R² over layers (train-mean baseline)")
    p.add_argument("--title_metric", type=str, default="R²",
                   help="Metric name shown in the axes title.")
    p.add_argument("--suptitle", type=str, default=None,
                   help="Override the run_title(...) suptitle derived from the input dir name.")
    p.add_argument("--no_role_note", action="store_true",
                   help="Suppress the role-selection suffix in the axes title.")
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
            if args.roles is not None and r["token_role"] not in args.roles:
                continue
            key = (r["task"], (int(r["icl_example_index"]), r["token_role"]))
            v = float(r[args.value_column])
            if key not in best or v > best[key]:
                best[key] = v

    if args.average_roles:
        # One point per ICL example: mean of the per-role best-over-layers values.
        pooled = {}
        for (task, (icl, role)), v in best.items():
            pooled.setdefault((task, icl), []).append(v)
        n_roles = {len(v) for v in pooled.values()}
        if len(n_roles) != 1:
            raise ValueError(f"Uneven role counts per (task, icl): {sorted(n_roles)}")
        best = {(task, (icl, args.average_label)): sum(v) / len(v) for (task, icl), v in pooled.items()}
        positions = sorted({k[1] for k in best})
        labels = [f"icl{icl:02d}/{role}" for icl, role in positions]
    else:
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
    pad = max(1.2, len(positions) * 0.08)  # right headroom for the direct labels
    ax.set_xlim(-0.5, len(positions) - 1 + pad)
    ax.set_xlabel("token position (icl example / role)")
    ax.set_ylabel(args.ylabel)
    ax.yaxis.grid(True, color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    role_note = "" if args.roles is None else f" — {', '.join(args.roles)} only"
    if args.average_roles:
        role_note = f" — mean over {{{', '.join(args.roles or ['all roles'])}}} per example"
    if args.no_role_note:
        role_note = ""
    ax.set_title(f"Best-over-layers {args.title_metric} by token position, per held-out task{role_note}")
    fig.suptitle(args.suptitle if args.suptitle is not None
                 else run_title(args.input_csv.parent.parent.name), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
