#!/usr/bin/env python
"""Dimensionality analysis of the DOT-PRODUCT per-prompt read directions (69-run, CPU).

Same protocol as plot_69_read_dir_dimensionality.py (which covers the cosine/pseudo-inverse
read dirs, results/.../Read_direction_geometry/cosine_similarity/), applied to the
dot-product read dirs r_dot = M^T v / ||M^T v|| from
src/sandbox/ext_steerability/compute_perprompt_dot_read_dirs_37.py. Single variant, so one
figure row:
  (A) centered PCA of the 55 task-level read directions r_task (unit vectors);
  (B) pooled per-prompt read dirs (55x150 = 8250 unit rows), stable rank annotated;
  (C) within-task centered PCA curves (one per task, 150 prompts each).
summary.csv also reports the mean-of-per-prompt-r PCA (row dot_task_mean_of_r).

Outputs in results/69_task_run/Read_direction_geometry/dot_product/:
  read_dir_dimensionality.png, spectra.npz, summary.csv
All SVDs in float64 on CPU.
"""
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

RD_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_dot_read_dirs"
OUT_DIR = TASK69_RUN_DIR / "Read_direction_geometry" / "dot_product"


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
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train = split["train_tasks"]

    r_tasks, stacks, within, mean_r = [], [], [], []
    for t in train:
        d = torch.load(RD_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
        assert d["group"] == "train"
        r = d["r"].numpy()  # (150, 4096) fp32 unit rows
        assert r.shape == (150, 4096)
        assert np.abs(np.linalg.norm(r, axis=1) - 1).max() < 1e-5, f"{t}: non-unit read dirs"
        r_tasks.append(d["r_task"].numpy())
        stacks.append(r)
        _, cum_w = evr(r)
        within.append(cum_w)
        mean_r.append(r.mean(axis=0))
    r_tasks = np.stack(r_tasks)            # (55, 4096)
    stack = np.concatenate(stacks, 0)      # (8250, 4096)
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
    np.savez_compressed(OUT_DIR / "spectra.npz", dot_s_rtask=s_task, dot_s_pooled=s_pool,
                        dot_within_cum=within, tasks=np.array(train))

    vlabel = "dot product"
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
    fig.suptitle("Read-direction dimensionality — dot-product read dirs r = M^T v / ||M^T v|| "
                 "(55 train tasks, 37-head pooled set, prunedfail_seed43), centered PCA, "
                 "float64 SVD", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "read_dir_dimensionality.png", bbox_inches="tight")

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "n_rows", "n90_pcs", "n95_pcs", "stable_rank_raw", "stable_rank_centered"])
        w.writerow(["dot_r_task", len(train), n_at(cum_task, .90), n_at(cum_task, .95), "", ""])
        w.writerow(["dot_pooled_perprompt", stack.shape[0], n_at(cum_pool, .90), n_at(cum_pool, .95),
                    round(float(sr_raw), 2), round(float(sr_cent), 2)])
        w.writerow(["dot_within_task_median", 150, n_at(med, .90), n_at(med, .95), "", ""])
        w.writerow(["dot_task_mean_of_r", len(train), n_at(cum_mean_r, .90), n_at(cum_mean_r, .95), "", ""])
    print(f"[dot] r_task: 90%@{n_at(cum_task,.9)} 95%@{n_at(cum_task,.95)} | "
          f"pooled: 90%@{n_at(cum_pool,.9)} 95%@{n_at(cum_pool,.95)} "
          f"sr raw={sr_raw:.2f} cent={sr_cent:.2f} | "
          f"within med: 90%@{n_at(med,.9)} 95%@{n_at(med,.95)}", flush=True)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
