#!/usr/bin/env python
"""SIMPLE Appendix-C figure: which read-feature construction steers the 1-shot dummy scaffold?

Reads bottom_up_read_features/head_selection/summary.csv (no recompute) and draws three bars at
each method's best matched layer — raw task mean (L7), mean-difference = task mean − shared mean
(L3), sparse-selected target-slot head sum (L3 selection, injected at L3) — with the real 1-shot
demonstration as a dashed reference. Writes head_selection/method_bars.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

D = TASK69_RUN_DIR / "bottom_up_read_features" / "head_selection"
acc = {r["condition"]: float(r["mean_acc"]) for r in csv.DictReader(open(D / "summary.csv")) if r["task_group"] == "all"}
bars = [("raw task mean\n$m_A$ (L7)", acc["rawmean@L7"], "#0e7c6b"),
        ("mean-difference\n$m_A - \\bar m$ (L3)", acc["meandiff@L3"], "#5fa8a0"),
        ("sparse-selected\nhead sum (L3)", acc["headsum_L3sel@L3"], "#b7d3d0")]
INK, MUTED = "#181c1e", "#5d6771"
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
xs = range(len(bars))
bb = ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62, zorder=3)
for b, (_, v, _) in zip(bb, bars):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=11.5, fontweight="bold", color=INK)
ax.axhline(acc["real_1shot"], color="#2e8b57", lw=1.6, ls=(0, (5, 2.5)), zorder=2)
ax.text(len(bars) - 0.55, acc["real_1shot"] + 0.005, f"real 1-shot demonstration {acc['real_1shot']:.3f}", ha="right", va="bottom", fontsize=9.5, color="#2e8b57")
ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=10.5)
ax.set_ylabel("steered accuracy (mean, 69 tasks)", fontsize=11, color=INK)
ax.set_ylim(0, 0.25)
ax.set_title("Which read-feature vector steers the 1-shot dummy scaffold", loc="left", fontsize=12, color=INK, pad=10)
ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color("#c9ccc7")
ax.tick_params(colors=MUTED, labelsize=10)
fig.tight_layout()
fig.savefig(D / "method_bars.png", facecolor="white")
print(f"wrote {D}/method_bars.png", {b[0].split(chr(10))[0]: round(b[1], 3) for b in bars})
