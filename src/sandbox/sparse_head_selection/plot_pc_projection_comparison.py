#!/usr/bin/env python
"""SANDBOX: per-task bar chart for the top-29-PC projection steering eval (CPU).

Reads top29pc_projection_vs_fullfv.csv and draws grouped bars per train task:
full sparse23 FV vs top-29-PC projection vs 83-PC (c=1) ceiling, with the
no-intervention baseline as markers. Tasks sorted by full-FV accuracy.
"""
import argparse
import csv
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

DEFAULT_DIR = RESULTS_ROOT / "sandbox" / "sparse_pc_selection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_csv", type=Path, default=DEFAULT_DIR / "top29pc_projection_vs_fullfv.csv")
    p.add_argument("--out_png", type=Path, default=DEFAULT_DIR / "top29pc_projection_pertask.png")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input_csv) as f:
        rows = [r for r in csv.DictReader(f)]
    mean = next(r for r in rows if r["task"] == "MEAN")
    rows = [r for r in rows if r["task"] != "MEAN"]
    rows.sort(key=lambda r: float(r["full_sparse23_fv_L9"]))

    tasks = [r["task"] for r in rows]
    x = np.arange(len(tasks))
    full = [float(r["full_sparse23_fv_L9"]) for r in rows]
    proj29 = [float(r["top29pc_proj_L9"]) for r in rows]
    proj83 = [float(r["proj83_c1_L9"]) for r in rows]
    noint = [float(r["no_intervention"]) for r in rows]

    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.bar(x - 0.27, full, width=0.27, color="tab:blue",
           label=f"full sparse23 FV (mean {float(mean['full_sparse23_fv_L9']):.3f})")
    ax.bar(x, proj29, width=0.27, color="tab:orange",
           label=f"top-29-PC projection (mean {float(mean['top29pc_proj_L9']):.3f})")
    ax.bar(x + 0.27, proj83, width=0.27, color="tab:green", alpha=0.65,
           label=f"83-PC projection, c=1 (mean {float(mean['proj83_c1_L9']):.3f})")
    ax.plot(x, noint, "k_", markersize=11, markeredgewidth=1.6,
            label=f"no intervention (mean {float(mean['no_intervention']):.3f})", linestyle="none")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("zero-shot full-label accuracy @L9")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    ax.set_title("SANDBOX: sparse23 fixed10-mean FV vs its top-29-PC / 83-PC projections — "
                 "single cue-token injection @L9, 20 train tasks, same 1720 datapoints "
                 "(loto_vs_canonical protocol)", fontsize=10)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
