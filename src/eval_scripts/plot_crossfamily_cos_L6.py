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
from utils.paper_style import apply_paper_style, C  # noqa: E402

apply_paper_style()

D = TASK69_RUN_DIR / "understanding_read_write_linear_map"
z = np.load(D / "rotation_vs_ridge_spectra.npz")
bins = np.linspace(-0.15, 0.4, 56)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for variant, color in (("uncentered", C.lightgrey), ("centered", C.accent_light)):
    ax.hist(z[f"L6_crossfam_mismatched_{variant}"], bins=bins, density=True, color=color, alpha=0.65,
            label=f"mismatched cos($m_A$, $v_B$), {variant}")
for variant, color in (("uncentered", C.text), ("centered", C.read)):
    ax.hist(z[f"L6_crossfam_matched_{variant}"], bins=bins, density=True, color=color,
            histtype="step", lw=1.4, label=f"matched cos($m_A$, $v_A$), {variant}")
ax.axvline(0, color=C.grey, lw=0.8)
ax.set_xlim(-0.15, 0.4)
ax.set_xlabel("cos(read $m_A$(L6), write FV)")
ax.set_ylabel("density")
ax.set_title("Each task's read feature vs its own / other tasks' FVs (L6)")
ax.legend()
fig.tight_layout()
fig.savefig(D / "crossfamily_cos_hists.png")
print(f"wrote {D}/crossfamily_cos_hists.png")
