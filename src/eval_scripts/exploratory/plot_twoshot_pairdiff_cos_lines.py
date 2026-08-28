#!/usr/bin/env python
"""Plot the Stream S mean-cosine figures from cos_grid.json (CPU-only, no torch).

Metric: SIGNED mean_i cos(d_i, x) per (view, layer, filter) — see
analyze_twoshot_pairdiff_fv_preimage.py. Linear y-axis with a zero line; random baselines
(isotropic + activation-covariance) drawn as mean lines with +/-2 sd bands; mean_dir
(= ||mean_i unit(d_i)||) is the analytic maximum over unit directions.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import FV_FORMATION_DIR

COLORS = {"inv_fv_diff": "tab:red", "inv_fv_diff_pcak16": "tab:purple",
          "inv_fv_diff_tsvdk16": "tab:blue", "fv_diff": "tab:green",
          "mean_dir": "tab:gray", "random": "k", "random_actcov": "tab:orange"}
LABELS = {"inv_fv_diff": "inv(fv_diff)", "inv_fv_diff_pcak16": "inv(fv_diff) PCA-k16",
          "inv_fv_diff_tsvdk16": "inv(fv_diff) TSVD-k16",
          "fv_diff": "fv_diff", "mean_dir": "mean_dir (max)",
          "random": "random (isotropic)", "random_actcov": "random (act-cov)"}
BANDED = ["random", "random_actcov"]   # drawn as mean line +/-2 sd band


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", type=Path,
                   default=FV_FORMATION_DIR / "preimage_analysis/twoshot_pairdiff_fv_preimage/train_varicl_max4_top40")
    p.add_argument("--pairs", nargs="+",
                   default=["antonym_synonym", "next_number_digits_prev_number_digits"])
    p.add_argument("--suffix", type=str, default="",
                   help="Filename suffix before .png (empty = write cos_lines_*.png).")
    return p.parse_args()


def as_arr(ys):
    return np.array([np.nan if y is None else y for y in ys], dtype=float)


def main():
    args = parse_args()
    for pair_name in args.pairs:
        out_dir = args.results_root / pair_name
        g = json.loads((out_dir / "cos_grid.json").read_text())
        grid, counts = g["mean_cos"], g["n_pairs"]
        views = [rv["view"] for rv in g["role_views"] if rv["view"] in grid]
        xs = np.array(g["layers"])
        for filt in ["all", "both_correct"]:
            fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
            for ax, v in zip(axes.flat, views):
                gg = grid[v][filt]
                for d in BANDED:
                    m, sd = as_arr(gg[d]), as_arr(gg[d + "_sd"])
                    ax.fill_between(xs, m - 2 * sd, m + 2 * sd, color=COLORS[d], alpha=0.15,
                                    linewidth=0)
                    ax.plot(xs, m, color=COLORS[d], lw=1,
                            ls=":" if d == "random" else "-.", label=LABELS[d])
                ax.plot(xs, as_arr(gg["mean_dir"]), color=COLORS["mean_dir"], ls="--", lw=1.5,
                        label=LABELS["mean_dir"])
                for d in ["inv_fv_diff", "inv_fv_diff_pcak16", "inv_fv_diff_tsvdk16", "fv_diff"]:
                    ys = as_arr(gg[d]) if d in gg else None
                    if ys is None or np.isnan(ys).all():
                        continue
                    ax.plot(xs, ys, color=COLORS[d], lw=1.5, label=LABELS[d])
                ax.axhline(0, color="k", lw=0.6)
                ax.set_title(v, fontsize=9)
                ax.grid(alpha=0.3)
            for ax in axes.flat[len(views):]:
                ax.axis("off")
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f"{pair_name}: signed mean cos(diff vector, x) vs layer "
                         f"({filt}, n={counts[filt]}; bands = baseline ±2 sd)", fontsize=11)
            fig.supxlabel("layer")
            fig.tight_layout()
            fig.savefig(out_dir / f"cos_lines_{filt}{args.suffix}.png", dpi=150)
            plt.close(fig)
        print(f"plotted {pair_name}")


if __name__ == "__main__":
    main()
