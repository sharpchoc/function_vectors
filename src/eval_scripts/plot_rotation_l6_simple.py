#!/usr/bin/env python
"""SIMPLE main-text figure: the read→write map at L6 is a rotation (+ one scalar).

Reads results/69_task_run/understanding_read_write_linear_map/rotation_vs_ridge_summary.csv
(the held-out fits block; no refitting) and draws one bar chart for read layer L6 only:
   train-mean baseline | rotation (Procrustes) | rotation + scale | unconstrained linear (ridge)
y = held-out R^2 (14 tasks, test-mean reference), i.e. how much of the held-out task FVs each
map class predicts from the read feature. The two mean shifts (\\bar m out, \\bar v in) are part
of every fit and belong in the caption, not the plot.

Writes understanding_read_write_linear_map/rotation_l6_simple.png (+ .csv).
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

D = TASK69_RUN_DIR / "understanding_read_write_linear_map"
LAYER = 6

fits = {}
block = None
for line in open(D / "rotation_vs_ridge_summary.csv"):
    if line.startswith("# held-out fits"):
        block = "fits"; continue
    if line.startswith("#"):
        block = None; continue
    if block == "fits" and not line.startswith("layer,"):
        L, method, lam, r2_test, r2_train, ccos, cos = line.strip().split(",")
        if int(L) == LAYER:
            fits[method] = float(r2_test)

order = [("trainmean_baseline", "mean shift only\n(no map)", "0.72"),
         ("rotation", "rotation", "#7c3aad"),
         ("rotation+scale", "rotation\n+ one scalar", "#7c3aad"),
         ("ridge", "unconstrained\nlinear map", "#2a78d6")]
vals = [fits[k] for k, _, _ in order]

with open(D / "rotation_l6_simple.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "heldout_r2_testmean_L6"])
    for (k, _, _), v in zip(order, vals): w.writerow([k, f"{v:.4f}"])

fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
xs = range(len(order))
for x, (k, lab, col), v in zip(xs, order, vals):
    ax.bar([x], [max(v, 0)], width=0.6, color=col, zorder=3)
    ax.text(x, max(v, 0) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5,
            fontweight="bold", color="#181c1e")
ax.set_xticks(list(xs), [lab for _, lab, _ in order], fontsize=10.5)
ax.set_ylim(0, 0.78)
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_ylabel("held-out $R^2$ (14 tasks)", fontsize=11)
ax.set_title("Predicting write features from L6 read features", fontsize=12.5, loc="left", pad=10)
ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(D / "rotation_l6_simple.png", facecolor="white")
print("L6 held-out R^2:", {k: round(v, 3) for (k, _, _), v in zip(order, vals)},
      f"| rotation+scale = {100 * fits['rotation+scale'] / fits['ridge']:.0f}% of ridge")
