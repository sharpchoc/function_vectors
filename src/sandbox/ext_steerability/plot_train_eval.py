#!/usr/bin/env python
"""SANDBOX: per-task bar charts for the ext-steerability train-task eval (CPU).

Two panels (test_zeroshot, test_mixedtask10): best-layer accuracy per task with the
pooled-selected 39-head FV (alpha=1), sorted ascending, unsteered baseline as black dashes.
Reads results/sandbox/ext_steerability/train_tasks_summary.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import RESULTS_ROOT

OUT_DIR = RESULTS_ROOT / "sandbox" / "ext_steerability"


def main():
    rows = list(csv.DictReader(open(OUT_DIR / "train_tasks_summary.csv")))
    fig, axes = plt.subplots(2, 1, figsize=(21, 11), dpi=140)
    for ax, setting, title in (
        (axes[0], "test_zeroshot", "zero-shot"),
        (axes[1], "test_mixedtask10", "mixed-task mixed-label 10-shot"),
    ):
        rs = sorted(rows, key=lambda r: float(r[f"{setting}_best"]))
        x = np.arange(len(rs))
        ax.bar(x, [float(r[f"{setting}_best"]) for r in rs], width=0.75, color="tab:blue",
               label="steered, best layer (39-head pooled FV)")
        ax.plot(x, [float(r[f"{setting}_base"]) for r in rs], "k_", markersize=8,
                markeredgewidth=1.4, linestyle="none", label="no steering")
        ax.set_xticks(x)
        ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=6.8)
        ax.set_ylabel("full-label acc")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(f"test setting: {title} — 72 train tasks, alpha=1, 50 queries/task", fontsize=11)
    fig.suptitle("Pooled sparse-selected head set (39 heads, lambda=0.005, train metric zero-shot) — "
                 "TRAIN-task steering (SANDBOX ext_steerability phase 1)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "train_tasks_bars.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
