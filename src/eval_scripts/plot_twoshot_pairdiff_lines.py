#!/usr/bin/env python
"""Re-plot the Stream S line figures from explained_grid.json (CPU-only, no torch).

Reads the per-pair explained_grid.json written by analyze_twoshot_pairdiff_fv_preimage.py and
redraws the lines_{centering}_{filter}.png figures with a configurable set of direction lines
(default: exact pre-image + controls, no damped arm — per user request 2026-07-06).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import FV_FORMATION_DIR

COLORS = {"damped": "tab:blue", "exact": "tab:red", "fv_diff": "tab:green",
          "top_pc": "tab:gray", "random": "k", "random_actcov": "tab:orange"}
STYLES = {"top_pc": {"ls": "--"}, "random": {"ls": ":", "lw": 1},
          "random_actcov": {"ls": "-.", "lw": 1}}
# Display names (JSON keys unchanged): make it obvious the red line is the ridge INVERSE of
# the FV difference, vs the raw FV difference direction itself.
LABELS = {"exact": "inv(fv_diff)", "damped": "inv(fv_diff) damped",
          "random": "random (isotropic)", "random_actcov": "random (act-cov)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", type=Path,
                   default=FV_FORMATION_DIR / "twoshot_pairdiff_fv_preimage/train_varicl_max4_top40")
    p.add_argument("--pairs", nargs="+",
                   default=["antonym_synonym", "next_number_digits_prev_number_digits"])
    p.add_argument("--directions", nargs="+",
                   default=["exact", "fv_diff", "top_pc", "random", "random_actcov"])
    p.add_argument("--suffix", type=str, default="",
                   help="Filename suffix before .png (empty = overwrite lines_*.png in place).")
    return p.parse_args()


def main():
    args = parse_args()
    for pair_name in args.pairs:
        out_dir = args.results_root / pair_name
        g = json.loads((out_dir / "explained_grid.json").read_text())
        grid, counts = g["explained"], g["n_pairs"]
        views = [rv["view"] for rv in g["role_views"] if rv["view"] in grid]
        n_layers = len(g["layers"])
        for cent in ["centered", "uncentered"]:
            for filt in ["all", "both_correct"]:
                fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
                for ax, v in zip(axes.flat, views):
                    xs = list(range(n_layers))
                    for d in args.directions:
                        ys = [np.nan if y is None else y for y in grid[v][filt][cent][d]]
                        style = dict(color=COLORS[d], lw=1.5)
                        style.update(STYLES.get(d, {}))
                        ax.plot(xs, ys, label=LABELS.get(d, d), **style)
                    ax.set_title(v, fontsize=9)
                    ax.set_yscale("log")
                    ax.grid(alpha=0.3)
                for ax in axes.flat[len(views):]:
                    ax.axis("off")
                axes.flat[0].legend(fontsize=8)
                fig.suptitle(f"{pair_name}: explained pair-diff variance vs layer "
                             f"({cent}, {filt}, n={counts[filt]}) — log scale", fontsize=11)
                fig.supxlabel("layer")
                fig.tight_layout()
                fig.savefig(out_dir / f"lines_{cent}_{filt}{args.suffix}.png", dpi=150)
                plt.close(fig)
        print(f"re-plotted {pair_name} (directions: {args.directions})")


if __name__ == "__main__":
    main()
