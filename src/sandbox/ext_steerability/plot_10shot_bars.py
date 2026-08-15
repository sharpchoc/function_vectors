#!/usr/bin/env python
"""TRIAL figure: per-task bars (pooled 39-head FV / staged 111-head FV / no-steering
baseline) for the two corrupted 10-shot settings, tasks ascending by staged accuracy.
Usage: plot_10shot_bars.py <summary_csv> <out_png>"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
fig, axes = plt.subplots(2, 1, figsize=(21, 11), dpi=140)
for ax, s, title in (
    (axes[0], "test_sametask_shuffled10", "same-task shuffled-label 10-shot"),
    (axes[1], "test_mixedtask10", "mixed-task mixed-label 10-shot"),
):
    rs = sorted(rows, key=lambda r: float(r[f"{s}_staged_best"]))
    x = np.arange(len(rs))
    ax.bar(x - 0.27, [float(r[f"{s}_pooled_best"]) for r in rs], width=0.27,
           color="tab:blue", label="pooled 39-head FV")
    ax.bar(x, [float(r[f"{s}_staged_best"]) for r in rs], width=0.27,
           color="tab:orange", label="TRIAL staged 111-head FV")
    ax.bar(x + 0.27, [float(r[f"{s}_base"]) for r in rs], width=0.27,
           color="0.45", label="no-steering baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=6.8)
    ax.set_ylabel("best-layer full-label acc")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"test setting: {title} — 72 train tasks, alpha=1, ascending by staged FV",
                 fontsize=11)
fig.suptitle("Corrupted-context steering: pooled vs TRIAL-staged FV vs baseline "
             "(SANDBOX ext_steerability)", fontsize=12)
fig.tight_layout()
fig.savefig(dst, bbox_inches="tight")
print(f"wrote {dst}")
