#!/usr/bin/env python
"""Presentation cut of the bank-(a) ridge layer sweep: held-out per-prompt R^2 only.

Replaces taskfv_r2_all28_heldout_perprompt.png (bank-(b) X) as the Claim-6 figure.
Reads layer_sweep_bankA/taskfv_r2.csv; writes taskfv_r2_heldout_perprompt.png next to it.
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

D = TASK69_RUN_DIR / "FV_linear_decodability" / "labeltoken_fv_ridge" / "layer_sweep_bankA"

rows = list(csv.DictReader(open(D / "taskfv_r2.csv")))
L = [int(r["layer"]) for r in rows]
y = [float(r["test_perprompt"]) for r in rows]
yc = [float(r["test_centroid"]) for r in rows]

BLUE, INK, MUTED = "#2a78d6", "#181c1e", "#5d6771"
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.plot(L, y, color=BLUE, lw=2.2, marker="o", ms=5, mfc=BLUE, mec="white", mew=0.9,
        zorder=3, label="held-out $R^2$")
pk = max(range(len(L)), key=lambda i: y[i])
ax.set_xlim(-0.6, 27.6)
ax.set_ylim(0, 0.75)
ax.set_xticks(range(0, 28, 3))
ax.set_xlabel("layer of the target-token activation (X)", color=INK, fontsize=11)
ax.set_ylabel("held-out $R^2$ vs task FV", color=INK, fontsize=11)
ax.set_title("Where the write feature is linearly readable from the raw read feature $m_A(\\ell)$",
             fontsize=12.5, color=INK, loc="left", pad=10)
ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#c9ccc7")
ax.tick_params(colors=MUTED, labelsize=10)
fig.tight_layout()
fig.savefig(D / "taskfv_r2_heldout_perprompt.png", facecolor="white")
print(f"wrote {D}/taskfv_r2_heldout_perprompt.png  peak L{L[pk]} = {y[pk]:.3f} "
      f"(centroid peak {max(yc):.3f})")
