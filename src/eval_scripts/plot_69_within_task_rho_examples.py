#!/usr/bin/env python
"""Illustrative per-task presence-vs-accuracy trajectories for Appendix H (CPU).

Three example tasks showing what the within-task Spearman looks like: the perfect case
(adjective_to_adverb, rho=1.000), the median case (english-french, rho=0.964 — also a
held-out task), and the worst case in the pool (french_noun_gender, rho=0.643). Each
panel: the task's seven (presence, accuracy) points for n = 0..6 demonstrations,
connected in n order. Data = diagnostics_per_task.csv (meanL9-20 presence variant).

Output: results/69_task_run/write_feature_and_model_accuracy/within_task_rho_examples.png
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import TASK69_RUN_DIR  # noqa: E402
from utils.paper_style import apply_paper_style, C, label_bars  # noqa: E402,F401

apply_paper_style()

SRC = TASK69_RUN_DIR / "write_feature_and_model_accuracy"
TASKS = ["adjective_to_adverb", "english-french", "french_noun_gender"]
TITLES = {"adjective_to_adverb": "perfect", "english-french": "median; a held-out task",
          "french_noun_gender": "worst of 69"}

# Qwen2.5 port (2026-09-22): --src / --tasks auto (max, median, min within-task rho)
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", type=Path, default=SRC)
_ap.add_argument("--tasks", nargs="+", default=TASKS, help="three task names, or 'auto'")
_ap.add_argument("--band_label", default="L9–20")
_args = _ap.parse_args()
SRC = _args.src
rows = {r["task"]: r for r in csv.DictReader(open(SRC / "diagnostics_per_task.csv"))}
if _args.tasks == ["auto"]:
    ordered = sorted(rows, key=lambda t: float(rows[t]["within_task_rho"]))
    TASKS = [ordered[-1], ordered[len(ordered) // 2], ordered[0]]
    TITLES = {TASKS[0]: "best", TASKS[1]: "median", TASKS[2]: f"worst of {len(rows)}"}
    for t in TASKS:
        if rows[t]["group"] == "heldout":
            TITLES[t] += "; a held-out task"
else:
    TASKS = _args.tasks

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
for ax, t in zip(axes, TASKS):
    r = rows[t]
    pres = [float(r[f"presence_n{n}"]) for n in range(7)]
    acc = [float(r[f"acc_n{n}"]) for n in range(7)]
    c = C.accent if r["group"] == "heldout" else C.write
    ax.plot(pres, acc, color=c, lw=1.4, alpha=0.6, zorder=2)
    ax.scatter(pres, acc, s=36, color=c, zorder=3, edgecolor="white", lw=0.8)
    for n in (0, 6):
        ax.annotate(f"n={n}", (pres[n], acc[n]), textcoords="offset points",
                    xytext=(7, -3), color=C.note)
    ax.set_title(f"{t}\n$\\rho$ = {float(r['within_task_rho']):.3f}  ({TITLES[t]})")
    ax.set_xlabel(f"FV presence (mean cos, {_args.band_label})")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, axis="both")
axes[0].set_ylabel("sampled exact-match accuracy")
fig.suptitle("Within-task presence vs accuracy: one task = seven points (n = 0…6 demonstrations)",
             x=0.01, ha="left", y=1.02)
fig.tight_layout()
out = SRC / "within_task_rho_examples.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
