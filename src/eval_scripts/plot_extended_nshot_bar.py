#!/usr/bin/env python
"""Ranked bar chart of extended_tasks n-shot accuracy (default n=6), ascending.

Reads results/general/extended_tasks_nshot_sweep/nshot_accuracy.csv; writes
nshot_bar_{n}shot.png next to it. Bars colored by task origin (validated 2-slot
categorical pair; identity also carried by the legend, not color alone).
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import GENERAL_DIR  # noqa: E402

# validated reference palette (dataviz skill): slot1 blue / slot2 orange, light surface
C_NEW, C_ORIG = "#2a78d6", "#eb6834"
C_PRUNED = "#c3c8ce"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
PRUNE_AT = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    root = GENERAL_DIR / "extended_tasks_nshot_sweep"
    rows = [r for r in csv.DictReader(open(root / "nshot_accuracy.csv"))
            if int(r["n_shots"]) == args.n]
    assert rows, f"no rows for n={args.n}"
    rows.sort(key=lambda r: float(r["accuracy"]))
    tasks = [r["task"] for r in rows]
    accs = [float(r["accuracy"]) for r in rows]
    # tasks under the 30% pruning threshold are shown as a neutral "pruned" class
    colors = [C_PRUNED if float(r["accuracy"]) < PRUNE_AT
              else (C_NEW if r["origin"] == "new" else C_ORIG) for r in rows]
    n_pruned = sum(1 for a in accs if a < PRUNE_AT)

    fig, ax = plt.subplots(figsize=(30, 6.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(range(len(tasks)), accs, color=colors, width=0.82)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=90, fontsize=5.2, color=INK2)
    ax.set_xlim(-0.8, len(tasks) - 0.2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"{args.n}-shot accuracy", fontsize=11, color=INK)
    ax.set_title(f"GPT-J {args.n}-shot accuracy by task — extended_tasks ({len(tasks)} tasks), ascending. "
                 f"T=1.0 sampled generation, full-label match, 50 prompts/task.",
                 fontsize=12, color=INK)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.axhline(PRUNE_AT, color=INK2, lw=1.1, ls="--", alpha=0.7)
    ax.text(1, PRUNE_AT + 0.012, "pruning threshold (30%)", fontsize=9, color=INK2)
    n_new = sum(1 for r in rows if r["origin"] == "new" and float(r["accuracy"]) >= PRUNE_AT)
    n_orig = sum(1 for r in rows if r["origin"] != "new" and float(r["accuracy"]) >= PRUNE_AT)
    ax.legend(handles=[Patch(color=C_NEW, label=f"new task ({n_new})"),
                       Patch(color=C_ORIG, label=f"original abstractive ({n_orig})"),
                       Patch(color=C_PRUNED, label=f"pruned tasks, <30% ({n_pruned})")],
              loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()
    out = root / f"nshot_bar_{args.n}shot.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
