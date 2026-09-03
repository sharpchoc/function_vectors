#!/usr/bin/env python
"""Appendix figure: cross-family cosines at L6 only (matched cos(m_A, v_A) vs mismatched cos(m_A, v_B)).

Reads the arrays stored by understanding_read_write_linear_map/rotation_vs_ridge.py
(rotation_vs_ridge_spectra.npz); no recompute. Overwrites crossfamily_cos_hists.png with the
single L6 panel (the old two-panel L6/L13 version is regenerable from rotation_vs_ridge.py).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

D = TASK69_RUN_DIR / "understanding_read_write_linear_map"
z = np.load(D / "rotation_vs_ridge_spectra.npz")
bins = np.linspace(-0.15, 0.4, 56)
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for variant, color in (("uncentered", "0.55"), ("centered", "#a8742a")):
    ax.hist(z[f"L6_crossfam_mismatched_{variant}"], bins=bins, density=True, color=color, alpha=0.3,
            label=f"mismatched cos($m_A$, $v_B$), {variant}")
for variant, color in (("uncentered", "k"), ("centered", "#c0392b")):
    ax.hist(z[f"L6_crossfam_matched_{variant}"], bins=bins, density=True, color=color, alpha=0.75,
            histtype="step", lw=1.8, label=f"matched cos($m_A$, $v_A$), {variant}")
ax.axvline(0, color="0.6", lw=0.8)
ax.set_xlim(-0.15, 0.4)
ax.set_xlabel("cos(read $m_A$(L6), write FV)")
ax.set_ylabel("density")
ax.set_title("Each task's read feature vs its own / other tasks' FVs (L6)", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=8.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(D / "crossfamily_cos_hists.png", facecolor="white")
print(f"wrote {D}/crossfamily_cos_hists.png")
