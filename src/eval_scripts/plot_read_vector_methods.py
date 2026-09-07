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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402
from utils.paper_style import apply_paper_style, C, label_bars  # noqa: E402

apply_paper_style()

D = TASK69_RUN_DIR / "bottom_up_read_features" / "head_selection"
acc = {r["condition"]: float(r["mean_acc"]) for r in csv.DictReader(open(D / "summary.csv")) if r["task_group"] == "all"}
READ_65 = mcolors.to_hex(tuple(0.65 * v + 0.35 for v in mcolors.to_rgb(C.read)))  # C.read at 65 %
bars = [("raw task mean\n$m_A$ (L7)", acc["rawmean@L7"], C.read),
        ("mean-difference\n$m_A - \\bar m$ (L3)", acc["meandiff@L3"], READ_65),
        ("sparse-selected\nhead sum (L3)", acc["headsum_L3sel@L3"], C.read_light)]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
xs = range(len(bars))
bb = ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62, zorder=3)
ax.set_ylim(0, 0.25)
label_bars(ax, bb, fmt="{:.3f}", pad=0.02)
ax.axhline(acc["real_1shot"], color=C.grey, lw=1.2, ls=(0, (5, 2.5)), zorder=2)
ax.text(len(bars) - 0.55, acc["real_1shot"] + 0.005, f"real 1-shot demonstration {acc['real_1shot']:.3f}", ha="right", va="bottom", color=C.note)
ax.set_xticks(list(xs), [b[0] for b in bars])
ax.set_ylabel("steered accuracy (mean, 69 tasks)")
ax.set_title("Which read-feature vector steers the 1-shot dummy scaffold")
fig.tight_layout()
fig.savefig(D / "method_bars.png")
print(f"wrote {D}/method_bars.png", {b[0].split(chr(10))[0]: round(b[1], 3) for b in bars})
