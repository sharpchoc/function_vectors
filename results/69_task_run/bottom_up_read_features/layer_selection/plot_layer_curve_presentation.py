#!/usr/bin/env python3.12
"""Presentation cut of the injection-layer sweep (1-shot dummy-target scaffold).

Reads layer_summary.csv (best-alpha rows) and per_task_by_layer.csv (for the
real-1-shot baseline) from this directory; writes layer_curve_presentation.png
next to them. Vocabulary: target tokens (user decision 2026-08-31).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[3]   # results/69_task_run/bottom_up_read_features/layer_selection -> repo root
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from utils.paper_style import apply_paper_style, C, label_bars  # noqa: E402,F401

apply_paper_style()

OUT = HERE / "layer_curve_presentation.png"

L, acc = [], []
for r in csv.DictReader(open(HERE / "layer_summary.csv")):
    if r["alpha"] == "best":
        L.append(int(r["layer"])); acc.append(float(r["mean_acc_all"]))

r1 = np.mean([float(r["real_1shot"])
              for r in csv.DictReader(open(HERE / "per_task_by_layer.csv"))])

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)

ax.plot(L, acc, color=C.read, lw=2.2, marker="o", ms=5, mfc=C.read, mec="white",
        mew=0.9, zorder=3, label="steered dummy-target scaffold (best $\\alpha$)")
ax.axhline(r1, color=C.grey, lw=1.6, ls=(0, (5, 2.5)), zorder=2,
           label=f"real 1-shot demonstration = {r1:.3f}")

ax.set_xlim(-0.6, 27.6); ax.set_ylim(0, 0.245)
ax.set_xticks(range(0, 28, 3))
ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
ax.set_xlabel("injection layer", fontsize=11)
ax.set_ylabel("steered accuracy (mean, 69 tasks)", fontsize=11)
ax.set_title("Dummy target token steering (1-shot)", fontsize=13, pad=10)
ax.legend(loc="upper right", fontsize=9.5)
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
