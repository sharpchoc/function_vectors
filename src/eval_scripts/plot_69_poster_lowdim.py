#!/usr/bin/env python
"""Poster figure: aggregate zero-shot accuracy over all 69 tasks (CPU).

Bars: unsteered / full 37-head FV / 50-dim projection (variant B adds the 22-dim
truncation bar). Means over all 69 tasks, best injection layer per task, alpha=1.
Writes FV_dimensionality_reduction/poster_lowdim{,_with22}.png.
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

DR = TASK69_RUN_DIR / "FV_dimensionality_reduction"


def main():
    s50 = list(csv.DictReader(open(DR / "sparse_all69/pc_sparse_summary.csv")))
    a22 = {r["task"]: float(r["zs_best_over_alphas"]) for r in
           csv.DictReader(open(DR / "debugging/taskmean_k90_alpha_sweep.csv"))}
    base = np.mean([float(r["zs_base"]) for r in s50])
    full = np.mean([float(r["zs_full_best"]) for r in s50])
    p22 = np.mean([a22[r["task"]] for r in s50])

    variants = [("poster_lowdim.png",
                 [("no steering", base, "#b9bec6"),
                  ("full FV\n(4096-dim)", full, "#e8862e"),
                  ("FV projected onto\n22 directions", p22, "#2273b8")])]
    for fname, bars in variants:
        fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=220)
        x = np.arange(len(bars))
        for xi, (lbl, val, color) in zip(x, bars):
            ax.bar(xi, val, width=0.62, color=color)
            ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=17,
                    fontweight="bold", color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=13.5)
        ax.set_ylabel("zero-shot task accuracy", fontsize=14)
        ax.set_ylim(0, 0.88)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="y", labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Function-vector steering is low-dimensional\n"
                     "(mean over 69 tasks, GPT-J)", fontsize=14.5, pad=14)
        fig.tight_layout()
        fig.savefig(DR / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {DR / fname}")


if __name__ == "__main__":
    main()
