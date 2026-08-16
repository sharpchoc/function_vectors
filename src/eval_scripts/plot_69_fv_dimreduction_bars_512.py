#!/usr/bin/env python
"""Paired-bar comparison: full 37-head FV vs FV projected onto ALL 512 dictionary PCs (CPU).

Same layout as plot_69_fv_dimreduction_bars.py, but the projection uses the entire top-512
uncentered train-PC dictionary (oracle ceiling of any sparse selection) — evals from
artifacts/69_task_run/pc_sparse/all512_probe/evals/. Writes
results/69_task_run/FV_dimensionality_reduction/debugging/zeroshot_full_vs_projected512.png.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

EVALS = ARTIFACTS_ROOT / "69_task_run" / "pc_sparse" / "all512_probe" / "evals"
REF = TASK69_RUN_DIR / "FV_dimensionality_analysis" / "pc_sparse_summary.csv"
OUT = TASK69_RUN_DIR / "FV_dimensionality_reduction" / "debugging" / "zeroshot_full_vs_projected512.png"


def main():
    ref = {r["task"]: r for r in csv.DictReader(open(REF))}
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    fig, axes = plt.subplots(2, 1, figsize=(20, 10.5), dpi=140,
                             gridspec_kw={"height_ratios": [55, 30]})
    for ax, tasks, title in (
            (axes[0], split["train_tasks"],
             "TRAIN tasks (n=55) — FVs are 99.99% inside the 512-PC span"),
            (axes[1], split["heldout_tasks"],
             "HELD-OUT tasks (n=14) — never seen by the PC basis")):
        rows = []
        for t in tasks:
            s = json.load(open(EVALS / f"{t}.json"))["settings"]["test_zeroshot"]
            rows.append({"task": t, "proj": max(s["acc_by_layer"]), "base": s["baseline"],
                         "full": float(ref[t]["zs_full_best"])})
        rows.sort(key=lambda r: r["full"])
        x = np.arange(len(rows))
        w = 0.4
        ax.bar(x - w / 2, [r["full"] for r in rows], width=w, color="tab:orange",
               label=f"full 37-head FV — mean {np.mean([r['full'] for r in rows]):.2f}")
        ax.bar(x + w / 2, [r["proj"] for r in rows], width=w, color="tab:blue",
               label=f"FV projected onto ALL 512 PCs — mean {np.mean([r['proj'] for r in rows]):.2f}")
        ax.plot(x, [r["base"] for r in rows], "k_", markersize=7, markeredgewidth=1.2,
                linestyle="none", label=f"no steering — mean {np.mean([r['base'] for r in rows]):.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([r["task"] for r in rows], rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel("zero-shot full-label acc")
        ax.set_ylim(0, 1.02); ax.set_xlim(-0.7, len(rows) - 0.3)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Zero-shot steering: full 37-head FV vs the same FV projected onto the ENTIRE "
                 "512-PC dictionary (uncentered PCA of train per-prompt FVs; 96.3% of stack "
                 "energy) — oracle ceiling of any sparse PC selection; alpha=1, best layer, "
                 "50 queries/task", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
