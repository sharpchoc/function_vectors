#!/usr/bin/env python
"""Ablation debugging: cross-task pairwise cosine similarity of the two ablated directions.

Compares, over the 69-task pool, the geometry of
  read features : unit-normed L6 raw label-token means
                  (artifacts/69_task_run/label_resid_means/<task>.pt, row 6) — the
                  direction ablated in bottom_up_read_features/ablation, and
  task FVs      : unit-normed sums of the pooled 37-head means mapped through W_O
                  (identical construction to ablate_fv_cue6.unit_fv) — the direction
                  ablated in FV_ablation.
Hypothesis under test: read features are far more similar across tasks than FVs, so
own-vs-counterfactual read-feature ablation cannot separate, while FV ablation can.

CPU-only (GPT-J loaded fp32 on CPU just for out_proj weights). Outputs to
results/69_task_run/bottom_up_read_features/ablation/debugging/:
  cossim_hist.png    two-panel histogram (read features | FVs), means + cf-pair means
  cossim_summary.csv mean/median/p5/p95 of all 2346 pairs + the 69 assigned cf pairs
  pairwise_cos.npz   raw pair values + index arrays (regenerate any view)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR

SPLIT_PATH = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
READDIR_ROOT = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
HEADMEANS_ROOT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"
SELECTION_PATH = HEADMEANS_ROOT / "prunedfail_seed43" / "pooled_sparse" / "selection.json"
CF_PAIRS_PATH = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json"
FV_EVAL_ROOT = ARTIFACTS_ROOT / "69_task_run" / "FV_ablation" / "eval"
OUT_DIR = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation" / "debugging"
READ_LAYER = 6


def load_read_features(tasks):
    rows = []
    for t in tasks:
        d = torch.load(READDIR_ROOT / f"{t}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][READ_LAYER].float()
        rows.append(d / d.norm())
    M = torch.stack(rows)
    assert torch.isfinite(M).all()
    assert (M.norm(dim=1) - 1).abs().max() < 1e-5
    return M


def load_fvs(tasks, model_dir):
    from transformers import AutoModelForCausalLM
    from src.sandbox.isolation_upper_bound.run_task import build_contributions_single

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    cfg = {"n_layers": len(model.transformer.h), "n_heads": 16,
           "resid_dim": model.config.n_embd}
    sel_flat = torch.tensor(json.load(open(SELECTION_PATH))["selected_flat"])
    rows, norms = [], {}
    for t in tasks:
        means = torch.load(HEADMEANS_ROOT / t / "means.pt", map_location="cpu",
                           weights_only=False)
        C = build_contributions_single(means["head_means"], model, cfg)
        v = C[sel_flat].sum(dim=0).float()
        norms[t] = float(v.norm())
        rows.append(v / v.norm())
    M = torch.stack(rows)
    assert torch.isfinite(M).all()
    return M, norms


def crosscheck_fv_norms(norms):
    """Rebuilt FV norms must match the ones the peer's eval run recorded."""
    checked = 0
    for t, n in norms.items():
        f = FV_EVAL_ROOT / f"{t}.json"
        if not f.exists():
            continue
        ref = json.load(open(f)).get("norm_fv_own")
        if ref is None:
            continue
        assert abs(n - ref) < 1e-2 * max(1.0, ref), f"{t}: rebuilt {n:.3f} vs eval {ref:.3f}"
        checked += 1
    assert checked >= 3, f"only {checked} FV norms cross-checked"
    print(f"FV norm cross-check OK ({checked} tasks)")


def pair_stats(M, tasks, cf_pairs):
    G = (M @ M.T).numpy()
    iu = np.triu_indices(len(tasks), k=1)
    all_pairs = G[iu]
    idx = {t: i for i, t in enumerate(tasks)}
    cf_vals = np.array([G[idx[t], idx[cf_pairs[t]]] for t in tasks])
    return all_pairs, cf_vals, iu


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/gpt-j-6b"
    split = json.load(open(SPLIT_PATH))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    assert len(tasks) == 69
    cf_pairs = json.load(open(CF_PAIRS_PATH))["pairs"]

    R = load_read_features(tasks)
    F, norms = load_fvs(tasks, model_dir)
    crosscheck_fv_norms(norms)

    r_all, r_cf, iu = pair_stats(R, tasks, cf_pairs)
    f_all, f_cf, _ = pair_stats(F, tasks, cf_pairs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "pairwise_cos.npz",
             tasks=np.array(tasks), pair_i=iu[0], pair_j=iu[1],
             read_all=r_all, fv_all=f_all,
             cf_task=np.array([cf_pairs[t] for t in tasks]),
             read_cf=r_cf, fv_cf=f_cf)

    import csv
    with open(OUT_DIR / "cossim_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n_pairs", "mean", "median", "p5", "p95", "min", "max"])
        for name, v in (("read_feature_all", r_all), ("fv_all", f_all),
                        ("read_feature_cfpairs", r_cf), ("fv_cfpairs", f_cf)):
            w.writerow([name, len(v), *(round(float(x), 4) for x in (
                v.mean(), np.median(v), np.percentile(v, 5), np.percentile(v, 95),
                v.min(), v.max()))])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    bins = np.linspace(-0.2, 1.0, 61)
    for ax, vals, cfv, title, color in (
            (axes[0], r_all, r_cf, "read features (L6 target means)", "tab:blue"),
            (axes[1], f_all, f_cf, "task FVs (37-head sums)", "tab:orange")):
        ax.hist(vals, bins=bins, color=color, alpha=0.75,
                label=f"all pairs (mean {vals.mean():.3f})")
        ax.hist(cfv, bins=bins, color="k", histtype="step", lw=1.5,
                label=f"assigned cf pairs (mean {cfv.mean():.3f})")
        ax.axvline(vals.mean(), color=color, ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("pairwise cosine similarity")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pair count")
    fig.suptitle("Cross-task similarity of the ablated directions (69 tasks, 2346 pairs)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cossim_hist.png", dpi=150)

    print(f"read features: mean {r_all.mean():.4f} (cf pairs {r_cf.mean():.4f})")
    print(f"task FVs     : mean {f_all.mean():.4f} (cf pairs {f_cf.mean():.4f})")
    print(f"outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
