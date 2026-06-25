"""
Combine the 4 label->query-final cosine-shift heatmaps (2 tasks x 2 alphas) into ONE figure
with a SHARED colour bar, for direct cross-panel comparison. Pure plotting -- reads the saved
grids (no model / GPU). See steer_label_cos_heatmap.py for how the grids are produced.

Layout: rows = task, cols = alpha. Shared symmetric diverging scale across all 4 panels.
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import LABEL_GEOMETRY_DIR

TASKS = ["antonym_synonym", "next_number_digits_prev_number_digits"]
TASK_LABEL = {"antonym_synonym": "antonym→synonym",
              "next_number_digits_prev_number_digits": "prev→next (digits)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "oneshot_label_intervention_cos_heatmap"))
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    summaries = {t: json.load(open(root / f"{t}_summary.json")) for t in TASKS}

    # load grids; shared symmetric scale across all panels
    grids = {}
    vmax = 0.0
    for t in TASKS:
        for a in args.alphas:
            g = np.load(root / f"{t}_alpha{a:g}_grid.npy")
            grids[(t, a)] = g
            vmax = max(vmax, float(np.nanmax(np.abs(g))))
    vmax = vmax or 1e-6
    n_layers = next(iter(grids.values())).shape[0]

    nrows, ncols = len(TASKS), len(args.alphas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for r, t in enumerate(TASKS):
        n = summaries[t]["n_pairs"]
        peaks = {gv["alpha"]: gv for gv in summaries[t]["grids"].values()}
        for c, a in enumerate(args.alphas):
            ax = axes[r][c]
            im = ax.imshow(grids[(t, a)], origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, aspect="equal")
            ax.plot([0, n_layers - 1], [0, n_layers - 1], color="k", lw=0.6, ls=":", alpha=0.5)
            pk = peaks.get(a)
            sub = f"  peak +{pk['peak_shift']:.3f} @ i{pk['peak_intervention_layer']}/k{pk['peak_read_layer']}" if pk else ""
            ax.set_title(f"{TASK_LABEL[t]}, α={a:g}{sub}", fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("intervention layer (label token)")
            if c == 0:
                ax.set_ylabel("read layer (query-final token)")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, label="steered − baseline cosine (shared scale)")
    fig.suptitle("Label-token → query-final cosine shift  (shared colour scale)", fontsize=12)
    out = root / "figures" / "combined_2x2_cos_shift_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (shared vmax={vmax:.4f})")


if __name__ == "__main__":
    main()
