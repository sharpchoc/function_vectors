#!/usr/bin/env python
"""Per-task bars for the top-22-PC alpha sweep: full FV vs projection at alpha=1 vs best alpha (CPU).

Two panels (train/heldout), three bars per task; the chosen alpha is printed above the
best-alpha bar wherever it isn't 1. Reads debugging/taskmean_k90_alpha_sweep.csv. Writes
FV_dimensionality_reduction/debugging/taskmean_k90_alpha_bars.png.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import TASK69_RUN_DIR  # noqa: E402

DBG = TASK69_RUN_DIR / "FV_dimensionality_reduction" / "debugging"
C_FULL, C_A1, C_BEST = "tab:orange", "#b34040", "#2e7d8a"


def main():
    rows = [{**r, "full": float(r["zs_full_fv"]), "a1": float(r["zs_alpha1"]),
             "best": float(r["zs_best_over_alphas"]), "ba": float(r["best_alpha"])}
            for r in csv.DictReader(open(DBG / "taskmean_k90_alpha_sweep.csv"))]
    fig, axes = plt.subplots(2, 1, figsize=(21, 11), dpi=140,
                             gridspec_kw={"height_ratios": [55, 30]})
    for ax, grp, title in ((axes[0], "train", "TRAIN tasks (n=55)"),
                           (axes[1], "heldout", "HELD-OUT tasks (n=14)")):
        rs = sorted([r for r in rows if r["group"] == grp], key=lambda r: r["full"])
        x = np.arange(len(rs))
        w = 0.27
        ax.bar(x - w, [r["full"] for r in rs], width=w, color=C_FULL,
               label=f"full 37-head FV (alpha=1) — mean {np.mean([r['full'] for r in rs]):.2f}")
        ax.bar(x, [r["a1"] for r in rs], width=w, color=C_A1,
               label=f"top-22 PCs, alpha=1 — mean {np.mean([r['a1'] for r in rs]):.2f}")
        ax.bar(x + w, [r["best"] for r in rs], width=w, color=C_BEST,
               label=f"top-22 PCs, per-task best alpha — mean {np.mean([r['best'] for r in rs]):.2f}")
        for xi, r in zip(x, rs):
            if r["ba"] != 1.0:
                ax.text(xi + w, r["best"] + 0.012, f"{r['ba']:g}", ha="center",
                        fontsize=5.6, color=C_BEST)
        ax.set_xticks(x)
        ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel("zero-shot full-label acc")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.8, len(rs) - 0.2)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Does rescaling fix the top-22-PC projection? Per-task zero-shot accuracy, best "
                 "injection layer; the chosen alpha (of 1/1.25/1.5/2) is printed above the teal bar "
                 "where it isn't 1 — ascending by full-FV accuracy", fontsize=12)
    fig.tight_layout()
    out = DBG / "taskmean_k90_alpha_bars.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
