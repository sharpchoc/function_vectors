#!/usr/bin/env python
"""FV dimensionality analysis on the 69-task-run TRAIN tasks (CPU).

Inputs: ARTIFACTS_ROOT/69_task_run/perprompt_fvs/<task>.pt (capture_69_perprompt_fvs.py).
Consistency gate: per-task mean of per-prompt FVs must match the FV built from means.pt
(cos > 0.999) — hard stop otherwise.

Outputs in results/69_task_run/FV_dimensionality_analysis/:
  fv_dimensionality.png  3 panels: (A) centered PCA of the 55 task-mean FVs, cumulative
                         variance vs #PCs; (B) same for the pooled per-prompt stack
                         (55x150 = 8250 rows), with stable rank annotated; (C) within-task
                         centered PCA curves (one per task, 150 prompts each).
  spectra.npz            singular values for all three analyses
  summary.csv            n90/n95 PCs + stable ranks
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

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

PP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
MEANS_ROOT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"
OUT_DIR = TASK69_RUN_DIR / "FV_dimensionality_analysis"


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
    sel = None
    task_fvs, stacks, within_curves = [], [], []
    for t in train:
        d = torch.load(PP_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
        fv = d["fv"].float().numpy()  # (150, 4096)
        assert fv.shape[0] == 150
        sel = d["sel_flat"] if sel is None else sel
        assert d["sel_flat"] == sel
        task_fvs.append(fv.mean(axis=0))
        stacks.append(fv)
        _, cum_w = evr(fv)
        within_curves.append(cum_w)
    task_fvs = np.stack(task_fvs)          # (55, 4096)
    stack = np.concatenate(stacks, axis=0)  # (8250, 4096)

    # consistency gate vs means.pt-built FVs (import here to keep deps local)
    from src.sandbox.isolation_upper_bound.run_task import build_contributions_single
    import types
    dummy = types.SimpleNamespace()
    worst = 1.0
    for i, t in enumerate(train):
        means = torch.load(MEANS_ROOT / t / "means.pt", map_location="cpu", weights_only=False)
        hm = means["head_means"]  # (28,16,256) fp32
        # cheap W_O-free check is impossible; approximate gate: cos between mean-of-perprompt
        # raw and head_means restricted to selected heads
        d = torch.load(PP_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
        raw_mean = d["raw"].float().mean(dim=0)  # (37, 256)
        hm_sel = torch.stack([hm[f // 16, f % 16] for f in sel])
        cos = torch.nn.functional.cosine_similarity(raw_mean.flatten(), hm_sel.flatten(), dim=0).item()
        worst = min(worst, cos)
        assert cos > 0.999, f"CONSISTENCY GATE FAILED for {t}: cos={cos:.6f} — HARD STOP, report to user"
    print(f"consistency gate passed (worst raw-mean cos vs means.pt: {worst:.6f})")

    s_task, cum_task = evr(task_fvs)
    s_pool, cum_pool = evr(stack)
    fro2 = (stack.astype(np.float64) ** 2).sum()
    sr_raw = fro2 / (np.linalg.svd(stack.astype(np.float64), compute_uv=False)[0] ** 2)
    centered = stack.astype(np.float64) - stack.astype(np.float64).mean(axis=0, keepdims=True)
    sr_cent = (centered ** 2).sum() / (s_pool[0] ** 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "spectra.npz",
                        s_taskmean=s_task, s_pooled_perprompt=s_pool,
                        within_task_cum=np.stack(within_curves), tasks=np.array(train),
                        sel_flat=np.array(sel))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=150)
    ax = axes[0]
    ax.plot(np.arange(1, len(cum_task) + 1), cum_task, "o-", ms=3, color="tab:blue")
    ax.set_title(f"(A) task-mean FVs (n={len(train)}), centered PCA")
    ax = axes[1]
    ax.plot(np.arange(1, len(cum_pool) + 1), cum_pool, "-", color="tab:blue")
    ax.set_xscale("log")
    ax.set_title(f"(B) pooled per-prompt FVs ({stack.shape[0]} rows)\n"
                 f"stable rank: {sr_raw:.1f} raw / {sr_cent:.1f} centered")
    ax = axes[2]
    for cum_w in within_curves:
        ax.plot(np.arange(1, len(cum_w) + 1), cum_w, "-", lw=0.6, alpha=0.25, color="tab:blue")
    med = np.median(np.stack(within_curves), axis=0)
    ax.plot(np.arange(1, len(med) + 1), med, "-", lw=2, color="tab:orange", label="median task")
    ax.legend(fontsize=8)
    ax.set_title("(C) within-task per-prompt FVs (150/task), one curve per task")
    for ax, cum in zip(axes, (cum_task, cum_pool, med)):
        for frac, c in ((0.90, "#999999"), (0.95, "#cccccc")):
            ax.axhline(frac, ls=":", lw=0.8, color=c)
        ax.annotate(f"90%@{n_at(cum, .90)} PCs\n95%@{n_at(cum, .95)} PCs",
                    xy=(0.97, 0.08), xycoords="axes fraction", ha="right", fontsize=8.5)
        ax.set_xlabel("# principal components")
        ax.set_ylabel("cumulative variance explained")
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.25)
    fig.suptitle("FV dimensionality — 55 train tasks, 37-head pooled FV "
                 "(prunedfail_seed43), centered PCA, float64 SVD", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fv_dimensionality.png", bbox_inches="tight")

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "n_rows", "n90_pcs", "n95_pcs", "stable_rank_raw", "stable_rank_centered"])
        w.writerow(["task_mean_fvs", len(train), n_at(cum_task, .90), n_at(cum_task, .95), "", ""])
        w.writerow(["pooled_perprompt", stack.shape[0], n_at(cum_pool, .90), n_at(cum_pool, .95),
                    round(float(sr_raw), 2), round(float(sr_cent), 2)])
        w.writerow(["within_task_median", 150, n_at(med, .90), n_at(med, .95), "", ""])
    print(f"task-mean: 90%@{n_at(cum_task,.9)} 95%@{n_at(cum_task,.95)} of {len(cum_task)}")
    print(f"pooled per-prompt: 90%@{n_at(cum_pool,.9)} 95%@{n_at(cum_pool,.95)}; "
          f"stable rank raw={sr_raw:.2f} centered={sr_cent:.2f}")
    print(f"within-task median: 90%@{n_at(med,.9)} 95%@{n_at(med,.95)}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
