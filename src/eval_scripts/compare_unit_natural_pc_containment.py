#!/usr/bin/env python
"""Is the pooled-90% PC subspace of the NATURAL-norm read dirs contained in the pooled-90%
subspace of the UNIT-norm counterpart? (One bracket per invocation; CPU float64.)

For a sweep bracket (compute_read_dir_sweep.py tree), build the pooled per-prompt stacks
(55 train tasks x 150 rows) for both Lever-4 variants, center each, SVD with vectors, cut
each at its own cumulative-sigma^2 >= 0.90 rank (k_unit, k_nat). Containment of natural
PC j in the unit subspace is ||V_unit[:k_unit] v_j||^2 in [0, 1]; we report:
  - per-PC containment curve (j = 1..k_nat),
  - variance-weighted containment  sum_j s_j^2 c_j / sum_j s_j^2 (j <= k_nat),
  - the reverse direction (unit PCs in the natural subspace) for context.

Outputs per bracket in
results/69_task_run/Read_direction_geometry/unit_vs_natural_containment/:
  <bracket>_containment.png, <bracket>_containment.npz, <bracket>_summary.csv
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

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

SWEEP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep"
OUT_DIR = TASK69_RUN_DIR / "top_down_read_features" / "definition_sweep" / "unit_vs_natural_containment"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")


def topk_pcs(stack):
    """Centered SVD -> (Vh, singular values, k at cum sigma^2 >= 0.90)."""
    x = stack.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    e = np.cumsum(s ** 2) / (s ** 2).sum()
    return vh, s, int(np.searchsorted(e, 0.90) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket", required=True, choices=BRACKETS)
    args = ap.parse_args()

    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    unit_rows, nat_rows = [], []
    for t in split["train_tasks"]:
        d = torch.load(SWEEP_ROOT / args.bracket / f"{t}.pt", map_location="cpu", weights_only=False)
        r = d["r"].numpy()
        unit_rows.append(r)
        nat_rows.append(r * d["norm"].numpy()[:, None])
    unit = np.concatenate(unit_rows, 0)
    nat = np.concatenate(nat_rows, 0)

    vh_u, s_u, k_u = topk_pcs(unit)
    vh_n, s_n, k_n = topk_pcs(nat)
    print(f"[{args.bracket}] k_unit={k_u} k_nat={k_n}", flush=True)

    C = vh_u[:k_u] @ vh_n[:k_n].T                 # (k_u, k_n)
    cont_n_in_u = (C ** 2).sum(axis=0)            # per natural PC
    w = s_n[:k_n] ** 2
    weighted = float((w * cont_n_in_u).sum() / w.sum())
    rev = (C ** 2).sum(axis=1)                    # per unit PC, in natural subspace
    w_u = s_u[:k_u] ** 2
    weighted_rev = float((w_u * rev).sum() / w_u.sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{args.bracket}_containment.npz",
                        cont_nat_in_unit=cont_n_in_u, cont_unit_in_nat=rev,
                        s_unit=s_u, s_nat=s_n, k_unit=k_u, k_nat=k_n)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    ax.plot(np.arange(1, k_n + 1), cont_n_in_u, "o-", ms=2.5, lw=0.9, color="tab:blue",
            label=f"natural PC j in unit-90% subspace (k_nat={k_n})")
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("natural-variant PC index j (within its own 90% cut)")
    ax.set_ylabel("contained energy  ||P_unit v_j||^2")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title(f"[{args.bracket}] natural-90% PCs inside unit-90% subspace (k_unit={k_u})\n"
                 f"variance-weighted containment {weighted:.4f} | min {cont_n_in_u.min():.3f} | "
                 f"reverse (unit in natural) weighted {weighted_rev:.4f}", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{args.bracket}_containment.png", bbox_inches="tight")

    with open(OUT_DIR / f"{args.bracket}_summary.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["bracket", "k_unit", "k_nat", "weighted_containment_nat_in_unit",
                       "min_pc_containment", "median_pc_containment",
                       "weighted_containment_unit_in_nat"])
        wcsv.writerow([args.bracket, k_u, k_n, round(weighted, 4),
                       round(float(cont_n_in_u.min()), 4),
                       round(float(np.median(cont_n_in_u)), 4), round(weighted_rev, 4)])
    print(f"[{args.bracket}] weighted containment nat-in-unit={weighted:.4f} "
          f"min={cont_n_in_u.min():.3f} med={np.median(cont_n_in_u):.3f} | "
          f"reverse={weighted_rev:.4f}", flush=True)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
