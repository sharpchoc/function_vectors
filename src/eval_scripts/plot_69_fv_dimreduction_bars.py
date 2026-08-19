#!/usr/bin/env python
"""Paired-bar comparison: full 37-head FV vs 46-PC projected FV, zero-shot steering (CPU).

Two panels (55 train / 14 held-out tasks), per task two bars (full, projected), ascending by
full-FV accuracy; unsteered baselines as black dashes. Reads
results/69_task_run/FV_dimensionality_analysis/pc_sparse_summary.csv. Writes
results/69_task_run/FV_dimensionality_reduction/zeroshot_full_vs_projected.png.
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

SRC = TASK69_RUN_DIR / "FV_dimensionality_reduction" / "train_test_split" / "pc_sparse_summary.csv"
OUT_DIR = TASK69_RUN_DIR / "FV_dimensionality_reduction" / "train_test_split"


def main():
    rows = list(csv.DictReader(open(SRC)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(20, 10.5), dpi=140,
                             gridspec_kw={"height_ratios": [55, 30]})
    for ax, grp, title in ((axes[0], "train", "TRAIN tasks (n=55) — PCs and c fit on these"),
                           (axes[1], "heldout",
                            "HELD-OUT tasks (n=14) — never seen by basis or selection")):
        rs = sorted([r for r in rows if r["group"] == grp],
                    key=lambda r: float(r["zs_full_best"]))
        x = np.arange(len(rs))
        full = [float(r["zs_full_best"]) for r in rs]
        proj = [float(r["zs_best"]) for r in rs]
        base = [float(r["zs_base"]) for r in rs]
        w = 0.4
        ax.bar(x - w / 2, full, width=w, color="tab:orange",
               label=f"full 37-head FV — mean {np.mean(full):.2f}")
        ax.bar(x + w / 2, proj, width=w, color="tab:blue",
               label=f"46-PC projected FV — mean {np.mean(proj):.2f}")
        ax.plot(x, base, "k_", markersize=7, markeredgewidth=1.2, linestyle="none",
                label=f"no steering — mean {np.mean(base):.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel("zero-shot full-label acc")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.7, len(rs) - 0.3)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Zero-shot steering: full 37-head FV vs the same FV projected onto 46 "
                 "sparse-selected uncentered PCs (pooled fit on the 55 train tasks) — "
                 "alpha=1, best layer, 50 queries/task, ascending by full-FV accuracy",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "zeroshot_full_vs_projected.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
