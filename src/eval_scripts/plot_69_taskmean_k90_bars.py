#!/usr/bin/env python
"""Per-task comparison bars: full FV vs 50 sparse-selected PCs vs top-22 task-mean PCs (CPU).

Two panels (55 train / 14 held-out), three bars per task, ascending by full-FV accuracy.
Reads debugging/taskmean_k90_summary.csv (top-22, full) and sparse_all69/pc_sparse_summary.csv
(50-PC). Writes FV_dimensionality_reduction/debugging/taskmean_k90_bars.png.
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
C_FULL, C_50, C_22 = "tab:orange", "tab:blue", "#b34040"


def main():
    k90 = {r["task"]: r for r in csv.DictReader(open(DR / "low_dim_22d/taskmean_k90_summary.csv"))}
    p50 = {r["task"]: float(r["zs_best"]) for r in
           csv.DictReader(open(DR / "train_test_together_50d/sparse_all69/pc_sparse_summary.csv"))}
    rows = [{"task": t, "group": r["group"], "full": float(r["zeroshot_acc_full_fv"]),
             "k22": float(r["zeroshot_acc_top22_taskmean_pcs"]), "k50": p50[t]}
            for t, r in k90.items()]
    fig, axes = plt.subplots(2, 1, figsize=(21, 11), dpi=140,
                             gridspec_kw={"height_ratios": [55, 30]})
    for ax, grp, title in ((axes[0], "train", "TRAIN tasks (n=55)"),
                           (axes[1], "heldout", "HELD-OUT tasks (n=14)")):
        rs = sorted([r for r in rows if r["group"] == grp], key=lambda r: r["full"])
        x = np.arange(len(rs))
        w = 0.27
        for off, key, color, lbl in ((-w, "full", C_FULL, "full 37-head FV"),
                                     (0, "k50", C_50, "50 sparse-selected PCs (all-task fit)"),
                                     (w, "k22", C_22, "top-22 task-mean PCs (90% energy, no optimiser)")):
            vals = [r[key] for r in rs]
            ax.bar(x + off, vals, width=w, color=color,
                   label=f"{lbl} — mean {np.mean(vals):.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel("zero-shot full-label acc")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.8, len(rs) - 0.2)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Zero-shot steering per task: full FV vs 50 steering-selected PCs vs plain top-22 "
                 "variance PCs of the task means — alpha=1, best layer, 50 queries/task, "
                 "ascending by full-FV accuracy", fontsize=12)
    fig.tight_layout()
    out = DR / "low_dim_22d/taskmean_k90_bars.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
