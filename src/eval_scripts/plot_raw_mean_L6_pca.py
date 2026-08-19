#!/usr/bin/env python
"""Centered PCA of the 69 per-task L6 label-token mean activations (CPU, float64).

The raw-mean layer sweep found L6 the best steering layer (results/69_task_run/
raw_mean_steering). Here: one vector per task = the task's mean block-6 output at the last
demo label token of the clean 10-shot prompts (capture_label_resid_means.py), 69 x 4096;
centered SVD; cumulative variance-explained curve — the same protocol as the
Read_direction_geometry dimensionality analyses (evr / n90 / n95 conventions).

Outputs in results/69_task_run/raw_mean_steering/dimensionality/:
  pca_curve.png, spectra.npz, summary.csv
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

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

RM_ROOT = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "dimensionality_analysis" / "dimensionality"
LAYER = 6


def n_at(cum, frac):
    return int(np.searchsorted(cum, frac) + 1)


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)
    X = np.stack([torch.load(RM_ROOT / f"{t}.pt", map_location="cpu",
                             weights_only=False)["resid_means"][LAYER].numpy()
                  for t in tasks]).astype(np.float64)          # (69, 4096)
    assert X.shape == (69, 4096)

    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    v = s ** 2
    cum = np.cumsum(v) / v.sum()
    n50, n80, n90, n95 = (n_at(cum, f) for f in (0.50, 0.80, 0.90, 0.95))
    sr = float(v.sum() / v[0])
    norms = np.linalg.norm(X, axis=1)
    print(f"L{LAYER} label-token task means: 50%@{n50} 80%@{n80} 90%@{n90} 95%@{n95} of "
          f"{len(cum)} | stable rank {sr:.2f} | mean-vector norms "
          f"{norms.min():.1f}-{norms.max():.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "spectra.npz", singular_values=s, cum_var=cum,
                        tasks=np.array(tasks), layer=LAYER)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    ax.plot(np.arange(1, len(cum) + 1), cum, "o-", ms=3.5, color="tab:red")
    for frac, c in ((0.90, "#999999"), (0.95, "#cccccc")):
        ax.axhline(frac, ls=":", lw=0.9, color=c)
    ax.annotate(f"50%@{n50} PCs\n80%@{n80} PCs\n90%@{n90} PCs\n95%@{n95} PCs\n"
                f"stable rank {sr:.1f}",
                xy=(0.97, 0.08), xycoords="axes fraction", ha="right", fontsize=9)
    ax.set_xlabel("# principal components")
    ax.set_ylabel("cumulative variance explained")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.set_title(f"Centered PCA of the 69 per-task L{LAYER} label-token mean activations\n"
                 "(the raw-mean steering vectors at the best layer; last-demo-label token, "
                 "clean 10-shot prompts)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "pca_curve.png", bbox_inches="tight")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "n_tasks", "n50_pcs", "n80_pcs", "n90_pcs", "n95_pcs",
                    "stable_rank", "mean_vec_norm_min", "mean_vec_norm_max"])
        w.writerow([LAYER, len(tasks), n50, n80, n90, n95, round(sr, 2),
                    round(float(norms.min()), 1), round(float(norms.max()), 1)])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
