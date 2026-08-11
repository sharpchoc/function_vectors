#!/usr/bin/env python
"""SANDBOX: horizontal per-task range plot of per-prompt FV norms (sparse23, fixed10).

One row per task (sorted by median): thin line = full min-max range, thick line = IQR,
dot = median. Replots the stored part-14b npz (no recompute).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import RESULTS_ROOT

DEFAULT_DIR = RESULTS_ROOT / "sandbox" / "perprompt_fv_norms_vanilla_sparse_opt23"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_npz", type=Path, default=DEFAULT_DIR / "fvnorm_perprompt_fixed10.npz")
    p.add_argument("--out_png", type=Path, default=DEFAULT_DIR / "fvnorm_taskranges_fixed10.png")
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.input_npz, allow_pickle=True)
    tasks, norms = list(d["tasks"]), d["norms"]  # (27,), (27, 170)

    med = np.median(norms, axis=1)
    order = np.argsort(med)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    for row, i in enumerate(order):
        lo, p25, p75, hi = (np.percentile(norms[i], q) for q in (0, 25, 75, 100))
        ax.plot([lo, hi], [row, row], color="tab:blue", lw=1, alpha=0.45, zorder=1)
        ax.plot([p25, p75], [row, row], color="tab:blue", lw=4, alpha=0.85,
                zorder=2, solid_capstyle="butt")
        ax.plot(med[i], row, "o", color="tab:red", markersize=5, zorder=3)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels([tasks[i] for i in order], fontsize=9)
    ax.set_xlabel(r"per-prompt FV norm  $\|v^j_A\|$  (sparse23 heads, fixed 10-shot)")
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("Per-task per-prompt FV norm distributions — 170 prompts/task, 27 tasks\n"
                 "thin line = min-max, thick = IQR, dot = median; SANDBOX sparse23 head set",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_png)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
