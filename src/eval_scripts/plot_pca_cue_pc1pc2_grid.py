#!/usr/bin/env python
"""PC1-vs-PC2-only grid across ICL positions, from pca_cue_token_icl_evolution outputs.

Reads each variant's projections.npz + pca_model.json (no activation reloading) and writes
figures/grid_pc1_pc2.png: one panel per ICL position (2x5 for the standard 10), per-prompt
points colored by task + black-edged task means (train 'o', test '^'), shared axes.
Task colors match the main script (same palette, same train-sorted + test-sorted order).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FV_FORMATION_DIR
from eval_scripts.pca_cue_token_icl_evolution import build_task_colors, legend_handles


def parse_args():
    p = argparse.ArgumentParser(description="PC1-vs-PC2 grid over ICL positions per PCA variant.")
    p.add_argument("--study_dir", type=Path, default=FV_FORMATION_DIR / "pca_cue_token_icl_evolution")
    p.add_argument("--variants", nargs="+", default=["pca_all_positions", "pca_final_cue"])
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--point_size", type=float, default=6.0)
    p.add_argument("--mean_point_size", type=float, default=110.0)
    return p.parse_args()


def main():
    args = parse_args()
    run_config = json.loads((args.study_dir / "run_config.json").read_text())
    train_tasks, test_tasks = run_config["train_tasks"], run_config["test_tasks"]
    all_tasks = train_tasks + test_tasks
    task_group = {t: "train" for t in train_tasks}
    task_group.update({t: "test" for t in test_tasks})
    task_colors = build_task_colors(all_tasks)

    for variant in args.variants:
        vdir = args.study_dir / variant
        z = np.load(vdir / "projections.npz")
        evr = json.loads((vdir / "pca_model.json").read_text())["explained_variance_ratio"]
        coords, tasks, icls = z["coords"], z["task"], z["icl_index"]
        icl_indices = sorted(np.unique(icls).tolist())

        lo, hi = np.percentile(coords[:, :2], [1.0, 99.0], axis=0)
        pad = (hi - lo) * 0.06
        lims = [(lo[i] - pad[i], hi[i] + pad[i]) for i in range(2)]

        nrows = int(np.ceil(len(icl_indices) / args.ncols))
        fig, axes = plt.subplots(nrows, args.ncols, figsize=(4.2 * args.ncols, 4.0 * nrows),
                                 squeeze=False)
        for k, icl in enumerate(icl_indices):
            ax = axes[k // args.ncols][k % args.ncols]
            sel_icl = icls == icl
            for task in all_tasks:
                pts = coords[sel_icl & (tasks == task)]
                ax.scatter(pts[:, 0], pts[:, 1], s=args.point_size, alpha=args.alpha,
                           color=task_colors[task], linewidths=0, rasterized=True, zorder=1)
            for task in all_tasks:
                mean = coords[sel_icl & (tasks == task)][:, :2].mean(axis=0)
                ax.scatter([mean[0]], [mean[1]], s=args.mean_point_size,
                           marker="o" if task_group[task] == "train" else "^",
                           color=task_colors[task], edgecolors="black", linewidths=1.0, zorder=3)
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_title(f"icl{icl:02d}/pre", fontsize=11)
            if k // args.ncols == nrows - 1:
                ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
            if k % args.ncols == 0:
                ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
        for k in range(len(icl_indices), nrows * args.ncols):
            axes[k // args.ncols][k % args.ncols].axis("off")

        layer = run_config["layer_index"]
        fig.suptitle(f"{variant} | L{layer} cue tokens | PC1 vs PC2 across ICL positions "
                     f"(PCA fit on {len(train_tasks)} train tasks)", fontsize=14)
        fig.legend(handles=legend_handles(all_tasks, task_group, task_colors, args.mean_point_size),
                   loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 0.99, 0.96))
        out = vdir / "figures" / "grid_pc1_pc2.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[{variant}] wrote {out}")


if __name__ == "__main__":
    main()
