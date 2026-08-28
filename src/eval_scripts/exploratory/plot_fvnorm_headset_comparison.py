#!/usr/bin/env python
"""SANDBOX: per-task median ||v^j_A|| under canonical top-40 vs vanilla_sparse_opt23 heads.

Scatter of the fixed10 per-task median per-prompt FV norms from the two head definitions
(same captured activations, same prompts -- only the head set differs), with task labels and
Spearman/Pearson correlations. Answers: is the part-14 task-norm ordering a property of the
tasks or an artifact of the CIE-selected canonical head set?
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from utils.paths import FV_FORMATION_DIR, RESULTS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top40_npz", type=Path,
                   default=FV_FORMATION_DIR / "attention_head_analysis" / "perprompt_fv_norms"
                   / "fvnorm_perprompt_fixed10.npz")
    p.add_argument("--sparse_npz", type=Path,
                   default=RESULTS_ROOT / "sandbox" / "perprompt_fv_norms_vanilla_sparse_opt23"
                   / "fvnorm_perprompt_fixed10.npz")
    p.add_argument("--out_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox" / "perprompt_fv_norms_vanilla_sparse_opt23")
    return p.parse_args()


def main():
    args = parse_args()
    a = np.load(args.top40_npz, allow_pickle=True)
    b = np.load(args.sparse_npz, allow_pickle=True)
    assert list(a["tasks"]) == list(b["tasks"]), "task orders must match"
    tasks = list(a["tasks"])
    med_a = np.median(a["norms"], axis=1)
    med_b = np.median(b["norms"], axis=1)
    rho, _ = spearmanr(med_a, med_b)
    r, _ = pearsonr(med_a, med_b)

    fig, ax = plt.subplots(figsize=(9.5, 9), dpi=150)
    ax.scatter(med_a, med_b, s=28, color="tab:blue", zorder=3)
    for t, x, y in zip(tasks, med_a, med_b):
        ax.annotate(t, (x, y), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    lo = min(med_a.min(), med_b.min()) - 3
    hi = max(med_a.max(), med_b.max()) + 3
    ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"median $\|v^j_A\|$ — canonical pooled top-40 heads (part 14)")
    ax.set_ylabel(r"median $\|v^j_A\|$ — SANDBOX vanilla_sparse_opt23 heads (23)")
    ax.set_title("Per-task median per-prompt FV norm under two head definitions (fixed10)\n"
                 f"Spearman ρ = {rho:.2f}, Pearson r = {r:.2f} — same prompts/activations, "
                 "only the head set differs", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "fvnorm_median_top40_vs_sparse23.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"Spearman rho {rho:.3f}, Pearson r {r:.3f}")
    dr = med_b - med_a
    order = np.argsort(dr)
    print("largest drops (top40 -> sparse23):",
          [(tasks[i], f"{dr[i]:+.1f}") for i in order[:4]])
    print("largest gains:", [(tasks[i], f"{dr[i]:+.1f}") for i in order[-4:]])


if __name__ == "__main__":
    main()
