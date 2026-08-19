#!/usr/bin/env python
"""Dimensionality analysis for one cell of the read-direction definition sweep (CPU).

Same protocol as the original Read_direction_geometry figures (centered PCA, float64 SVD,
55 train tasks): (A) task-level read dirs, (B) pooled per-prompt stack with stable rank,
(C) within-task curves. Reads the uniform sweep tree
artifacts/69_task_run/read_dir_sweep/<bracket>/<task>.pt
(compute_read_dir_sweep.py) and analyzes one (bracket, Lever-4 normalization) cell:

  --bracket {cosine_M, dot_M, cosine_perhead, dot_perhead}
  --norm    {unit, natural}   unit: rows r as stored; natural: r * norm (and r_task *
                              r_task_norm at task level)

Outputs in results/69_task_run/Read_direction_geometry/<bracket>__<norm>/:
  read_dir_dimensionality.png, spectra.npz, summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

SWEEP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")


def evr(mat):
    """Centered SVD -> (singular values, cumulative explained-variance ratio)."""
    x = mat.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    v = s ** 2
    return s, np.cumsum(v) / v.sum()


def n_at(cum, frac):
    return int(np.searchsorted(cum, frac) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket", required=True, choices=BRACKETS)
    ap.add_argument("--norm", required=True, choices=("unit", "natural"))
    args = ap.parse_args()
    cell = f"{args.bracket}__{args.norm}"
    OUT_DIR = TASK69_RUN_DIR / "top_down_read_features" / "dimensionality_analysis" / cell

    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train = split["train_tasks"]

    r_tasks, stacks, within, mean_r = [], [], [], []
    for t in train:
        d = torch.load(SWEEP_ROOT / args.bracket / f"{t}.pt", map_location="cpu", weights_only=False)
        assert d["group"] == "train" and d["bracket"] == args.bracket
        r = d["r"].numpy()
        assert r.shape == (150, 4096)
        assert np.abs(np.linalg.norm(r, axis=1) - 1).max() < 1e-5, f"{t}: non-unit stored rows"
        rt = d["r_task"].numpy()
        if args.norm == "natural":
            r = r * d["norm"].numpy()[:, None]
            rt = rt * d["r_task_norm"]
        r_tasks.append(rt)
        stacks.append(r)
        _, cum_w = evr(r)
        within.append(cum_w)
        mean_r.append(r.mean(axis=0))
    r_tasks = np.stack(r_tasks)
    stack = np.concatenate(stacks, 0)
    within = np.stack(within)
    mean_r = np.stack(mean_r)

    s_task, cum_task = evr(r_tasks)
    s_pool, cum_pool = evr(stack)
    _, cum_mean_r = evr(mean_r)
    x64 = stack.astype(np.float64)
    sr_raw = (x64 ** 2).sum() / np.linalg.svd(x64, compute_uv=False)[0] ** 2
    sr_cent = ((x64 - x64.mean(0, keepdims=True)) ** 2).sum() / (s_pool[0] ** 2)
    med = np.median(within, axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "spectra.npz",
                        **{f"{cell}_s_rtask": s_task, f"{cell}_s_pooled": s_pool,
                           f"{cell}_within_cum": within}, tasks=np.array(train))

    vlabel = f"{args.bracket}, {args.norm} norm"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=150)
    ax = axes[0]
    ax.plot(np.arange(1, len(cum_task) + 1), cum_task, "o-", ms=3, color="tab:blue")
    ax.set_title(f"[{vlabel}] (A) task-level read dirs r_task (n={len(train)}), centered PCA")
    ax = axes[1]
    ax.plot(np.arange(1, len(cum_pool) + 1), cum_pool, "-", color="tab:blue")
    ax.set_xscale("log")
    ax.set_title(f"[{vlabel}] (B) pooled per-prompt read dirs ({stack.shape[0]} rows)\n"
                 f"stable rank: {sr_raw:.1f} raw / {sr_cent:.1f} centered")
    ax = axes[2]
    for cum_w in within:
        ax.plot(np.arange(1, len(cum_w) + 1), cum_w, "-", lw=0.6, alpha=0.25, color="tab:blue")
    ax.plot(np.arange(1, len(med) + 1), med, "-", lw=2, color="tab:orange", label="median task")
    ax.legend(fontsize=8)
    ax.set_title(f"[{vlabel}] (C) within-task per-prompt read dirs (150/task), one curve per task")
    for ax, cum in zip(axes, (cum_task, cum_pool, med)):
        for frac, c in ((0.90, "#999999"), (0.95, "#cccccc")):
            ax.axhline(frac, ls=":", lw=0.8, color=c)
        ax.annotate(f"90%@{n_at(cum, .90)} PCs\n95%@{n_at(cum, .95)} PCs",
                    xy=(0.97, 0.08), xycoords="axes fraction", ha="right", fontsize=8.5)
        ax.set_xlabel("# principal components")
        ax.set_ylabel("cumulative variance explained")
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Read-direction dimensionality — sweep cell {cell} "
                 "(55 train tasks, 37-head pooled set, prunedfail_seed43, energy-90 "
                 "truncation), centered PCA, float64 SVD", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "read_dir_dimensionality.png", bbox_inches="tight")

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "n_rows", "n90_pcs", "n95_pcs", "stable_rank_raw", "stable_rank_centered"])
        w.writerow([f"{cell}_r_task", len(train), n_at(cum_task, .90), n_at(cum_task, .95), "", ""])
        w.writerow([f"{cell}_pooled_perprompt", stack.shape[0], n_at(cum_pool, .90), n_at(cum_pool, .95),
                    round(float(sr_raw), 2), round(float(sr_cent), 2)])
        w.writerow([f"{cell}_within_task_median", 150, n_at(med, .90), n_at(med, .95), "", ""])
        w.writerow([f"{cell}_task_mean_of_r", len(train), n_at(cum_mean_r, .90), n_at(cum_mean_r, .95), "", ""])
    print(f"[{cell}] r_task: 90%@{n_at(cum_task,.9)} 95%@{n_at(cum_task,.95)} | "
          f"pooled: 90%@{n_at(cum_pool,.9)} 95%@{n_at(cum_pool,.95)} "
          f"sr raw={sr_raw:.2f} cent={sr_cent:.2f} | "
          f"within med: 90%@{n_at(med,.9)} 95%@{n_at(med,.95)}", flush=True)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
