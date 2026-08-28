#!/usr/bin/env python
"""Per-ICL-example 3D FV-PCA plots: pre-label + label-token mapped predictions together.

One figure per ICL example index n: the 27 task FVs in their top-3 PCs, plus the ridge-mapped
predictions from the 7 TEST tasks' mean activations at that example's pre-label ':' token and
last label token (each cell at its best-test-MSE layer). Dashed connectors join each prediction
to the true FV. Reads the predictions saved by plot_fv_pca3d_icl_trajectories.py -- run that
first if the npz is missing.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    load_function_vector,
    load_json,
)

TRAIN_COLOR = "#4878a8"
TEST_COLOR = "#d62728"
PRE_COLOR = "#2ca02c"
LAST_COLOR = "#9467bd"


def parse_args():
    p = argparse.ArgumentParser(description="Per-ICL 3D FV-PCA plots with pre+label predictions.")
    p.add_argument("--icl_indices", nargs="+", type=int, default=[8, 9, 10])
    p.add_argument("--predictions_npz", type=Path,
                   default=ARTIFACTS_ROOT / "fulldim_ridge_weight_matrices/icl_trajectory_predictions.npz")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/weight_heatmaps")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.predictions_npz, allow_pickle=False)
    test_tasks = [str(t) for t in data["test_tasks"]]

    manifest = load_json(args.task_manifest)
    train_tasks = sorted(manifest["train_tasks"])
    assert test_tasks == sorted(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    tasks = train_tasks + test_tasks
    fv = torch.stack([load_function_vector(args.fv_root, t) for t in tasks]).numpy()

    center = fv.mean(axis=0)
    _, s, vt = np.linalg.svd(fv - center, full_matrices=False)
    var_frac = (s ** 2) / np.sum(s ** 2)
    comps = vt[:3]
    fv3 = (fv - center) @ comps.T
    n_train = len(train_tasks)
    tr, te = fv3[:n_train], fv3[n_train:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for icl in args.icl_indices:
        pre3 = (data[f"pre_icl{icl:02d}"] - center) @ comps.T
        last3 = (data[f"last_icl{icl:02d}"] - center) @ comps.T

        fig = plt.figure(figsize=(17, 8))
        for pi, (elev, azim) in enumerate([(18, -60), (25, 120)]):
            ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
            ax.scatter(*tr.T, c=TRAIN_COLOR, s=35, alpha=0.8, label="train FV (20)")
            ax.scatter(*te.T, c=TEST_COLOR, s=70, marker="*", edgecolors="black",
                       linewidths=0.5, label="test FV (7)")
            ax.scatter(*pre3.T, c=PRE_COLOR, s=60, marker="X",
                       label=f"mapped mean act, pre-label ':' (icl{icl:02d})")
            ax.scatter(*last3.T, c=LAST_COLOR, s=60, marker="^",
                       label=f"mapped mean act, last label tok (icl{icl:02d})")
            for t3, p3, l3 in zip(te, pre3, last3):
                ax.plot(*np.stack([p3, t3]).T, color=PRE_COLOR, linewidth=0.9, linestyle="--", alpha=0.7)
                ax.plot(*np.stack([l3, t3]).T, color=LAST_COLOR, linewidth=0.9, linestyle="--", alpha=0.7)
            for p, name in zip(tr, train_tasks):
                ax.text(*p, name, fontsize=5, color=TRAIN_COLOR, alpha=0.8)
            for p, name in zip(te, test_tasks):
                ax.text(*p, name, fontsize=7, color=TEST_COLOR, fontweight="bold")
            ax.set_xlabel(f"FV PC1 ({var_frac[0]:.0%})", fontsize=8)
            ax.set_ylabel(f"FV PC2 ({var_frac[1]:.0%})", fontsize=8)
            ax.set_zlabel(f"FV PC3 ({var_frac[2]:.0%})", fontsize=8)
            ax.view_init(elev=elev, azim=azim)
            if pi == 0:
                ax.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"27 task FVs in their top-3 PCs ({var_frac[:3].sum():.0%} var) + ridge-mapped "
                     f"test predictions at ICL example {icl} (pre-label and label token, "
                     "best layer each)", fontsize=12)
        fig.tight_layout()
        out = args.output_dir / f"fv_pca3d_pair_icl{icl:02d}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
