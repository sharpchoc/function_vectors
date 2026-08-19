#!/usr/bin/env python
"""Pairwise overlap of the pooled-90% PC subspaces across the four UNIT-norm read-direction
brackets (cosine_M, dot_M, cosine_perhead, dot_perhead). CPU float64.

For each bracket: pooled per-prompt unit rows (55 train tasks x 150), centered SVD, keep the
top-k basis at cum sigma^2 >= 0.90 (the same k_unit as the dimensionality figures). For each
pair (a, b):
  - symmetric overlap  ||V_a^T V_b||_F^2 / min(k_a, k_b)   (mean squared principal cosine
    of the smaller subspace inside the larger; 1 = nested, chance ~= max(k)/4096)
  - variance-weighted containment both ways: sum_j s_j^2 ||P_b v_j^a||^2 / sum_j s_j^2
  - chance level max(k_a, k_b)/4096 for calibration.

Outputs in results/69_task_run/Read_direction_geometry/cross_bracket_overlap/:
  overlap_heatmap.png (symmetric overlap matrix, containments annotated),
  overlap.npz (all matrices + principal cosines per pair), overlap_summary.csv
"""
import csv
import json
import sys
from itertools import combinations
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

SWEEP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep"
OUT_DIR = TASK69_RUN_DIR / "top_down_read_features" / "definition_sweep" / "cross_bracket_overlap"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")
D_MODEL = 4096


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    bases, svals, ks = {}, {}, {}
    for b in BRACKETS:
        rows = []
        for t in split["train_tasks"]:
            d = torch.load(SWEEP_ROOT / b / f"{t}.pt", map_location="cpu", weights_only=False)
            rows.append(d["r"].numpy())
        x = np.concatenate(rows, 0).astype(np.float64)
        x -= x.mean(axis=0, keepdims=True)
        _, s, vh = np.linalg.svd(x, full_matrices=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        k = int(np.searchsorted(e, 0.90) + 1)
        bases[b], svals[b], ks[b] = vh[:k], s[:k], k
        print(f"{b}: k_unit={k}", flush=True)

    n = len(BRACKETS)
    sym = np.eye(n)
    cont = np.eye(n)  # cont[i, j] = variance-weighted containment of i's subspace in j's
    npz = {f"k_{b}": ks[b] for b in BRACKETS}
    csv_rows = []
    for (i, a), (j, b) in combinations(enumerate(BRACKETS), 2):
        C = bases[a] @ bases[b].T                     # (k_a, k_b)
        f2 = float((C ** 2).sum())
        sym[i, j] = sym[j, i] = f2 / min(ks[a], ks[b])
        w_a = svals[a] ** 2
        cont[i, j] = float((w_a * (C ** 2).sum(axis=1)).sum() / w_a.sum())
        w_b = svals[b] ** 2
        cont[j, i] = float((w_b * (C ** 2).sum(axis=0)).sum() / w_b.sum())
        pcos = np.linalg.svd(C, compute_uv=False)     # principal cosines
        npz[f"principal_cos_{a}__{b}"] = pcos
        chance = max(ks[a], ks[b]) / D_MODEL
        csv_rows.append([a, b, ks[a], ks[b], round(sym[i, j], 4),
                         round(cont[i, j], 4), round(cont[j, i], 4), round(chance, 4)])
        print(f"{a} vs {b}: sym={sym[i,j]:.4f} | {a}-in-{b}={cont[i,j]:.4f} "
              f"{b}-in-{a}={cont[j,i]:.4f} | chance~{chance:.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "overlap.npz", sym_overlap=sym, containment=cont,
                        brackets=np.array(BRACKETS), **npz)

    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=150)
    im = ax.imshow(sym, vmin=0, vmax=1, cmap="viridis")
    labels = [f"{b}\n(k={ks[b]})" for b in BRACKETS]
    ax.set_xticks(range(n), labels, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            txt = f"{sym[i, j]:.2f}" if i == j else f"{sym[i, j]:.2f}\nrow-in-col {cont[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if sym[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="symmetric overlap  ||Va' Vb||_F^2 / min(k)")
    ax.set_title("Pooled-90% PC subspace overlap across UNIT-norm brackets\n"
                 "(55 train tasks; random-subspace chance ~= max(k)/4096 <= 0.08)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlap_heatmap.png", bbox_inches="tight")

    with open(OUT_DIR / "overlap_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bracket_a", "bracket_b", "k_a", "k_b", "symmetric_overlap",
                    "weighted_containment_a_in_b", "weighted_containment_b_in_a", "chance"])
        w.writerows(csv_rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
