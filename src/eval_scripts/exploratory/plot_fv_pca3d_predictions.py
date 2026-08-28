#!/usr/bin/env python
"""3D PCA view of the 27 task FVs plus the ridge maps' predictions for the 7 test tasks.

PCA (top 3 components) is fit on the 27 true function vectors. For each saved ridge weight bank
(plot_fulldim_ridge_weight_heatmaps.py), the 7 TEST tasks' mean activations are pushed through
the map and the predicted FVs are projected into the same PCA coordinates; a dashed line connects
each prediction to its true FV. Train predictions are omitted -- they land on top of their own
FVs (diag cos ~0.995) and only add clutter.
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
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    load_task_role_pooled,
    role_load_icl_index,
)
from eval_scripts.exploratory.plot_fulldim_ridge_weight_heatmaps import DEFAULT_CELLS, ROLE_SHORT, cell_key

TRAIN_COLOR = "#4878a8"
TEST_COLOR = "#d62728"
PRED_COLOR = "#2ca02c"


def parse_args():
    p = argparse.ArgumentParser(description="3D FV PCA plot with test-task ridge predictions.")
    p.add_argument("--cells", nargs="+", default=None,
                   help="Cells as icl:role:layer. Default: the 6 study cells.")
    p.add_argument("--weights_dir", type=Path, default=ARTIFACTS_ROOT / "fulldim_ridge_weight_matrices")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/weight_heatmaps")
    return p.parse_args()


def main():
    args = parse_args()
    cells = DEFAULT_CELLS if args.cells is None else [
        (int(i), r, int(l)) for i, r, l in (c.split(":") for c in args.cells)]

    manifest = load_json(args.task_manifest)
    train_tasks = sorted(manifest["train_tasks"])
    test_tasks = sorted(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    tasks = train_tasks + test_tasks
    fv = torch.stack([load_function_vector(args.fv_root, t) for t in tasks]).numpy()  # [27, 4096]

    # PCA fit on the 27 true FVs.
    center = fv.mean(axis=0)
    u, s, vt = np.linalg.svd(fv - center, full_matrices=False)
    var_frac = (s ** 2) / np.sum(s ** 2)
    comps = vt[:3]                                   # [3, 4096]
    fv3 = (fv - center) @ comps.T                    # [27, 3]
    print(f"top-3 FV PCs explain {var_frac[:3].sum():.1%} of FV variance "
          f"({', '.join(f'{v:.1%}' for v in var_frac[:3])})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for icl, role, layer in cells:
        key = cell_key(icl, role, layer)
        bank = torch.load(args.weights_dir / f"{key}.pt", map_location="cpu", weights_only=False)
        w, ybar = bank["weight"], bank["ybar"]
        fmean, fstd, xbar = bank["feature_mean"], bank["feature_std"], bank["xbar"]
        root = args.query_activations_root if icl == QUERY_ICL_INDEX else Path(
            args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)

        preds = []
        for task in test_tasks:
            acts = load_task_role_pooled(root, task, args.splits, role, load_icl)[:, layer, :].to(torch.float32)
            x_std = (acts.mean(dim=0) - fmean) / fstd
            preds.append((ybar + (x_std - xbar) @ w).numpy())
        pred3 = (np.stack(preds) - center) @ comps.T   # [7, 3]

        fig = plt.figure(figsize=(17, 8))
        for pi, (elev, azim) in enumerate([(18, -60), (25, 120)]):
            ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
            tr, te = fv3[:len(train_tasks)], fv3[len(train_tasks):]
            ax.scatter(*tr.T, c=TRAIN_COLOR, s=35, alpha=0.85, label="train FV (20)")
            ax.scatter(*te.T, c=TEST_COLOR, s=55, alpha=0.95, label="test FV (7)")
            ax.scatter(*pred3.T, c=PRED_COLOR, s=70, marker="X", alpha=0.95,
                       label="mapped mean activation (test)")
            for a, b in zip(te, pred3):
                ax.plot(*np.stack([a, b]).T, color=PRED_COLOR, linewidth=1, linestyle="--", alpha=0.7)
            for p, name in zip(tr, train_tasks):
                ax.text(*p, name, fontsize=5, color=TRAIN_COLOR, alpha=0.8)
            for p, name in zip(te, test_tasks):
                ax.text(*p, name, fontsize=6.5, color=TEST_COLOR, fontweight="bold")
            for p, name in zip(pred3, test_tasks):
                ax.text(*p, f"pred:{name}", fontsize=6, color=PRED_COLOR, fontstyle="italic")
            ax.set_xlabel(f"FV PC1 ({var_frac[0]:.0%})", fontsize=8)
            ax.set_ylabel(f"FV PC2 ({var_frac[1]:.0%})", fontsize=8)
            ax.set_zlabel(f"FV PC3 ({var_frac[2]:.0%})", fontsize=8)
            ax.view_init(elev=elev, azim=azim)
            if pi == 0:
                ax.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"27 task FVs in their top-3 PCs (fit on all 27; {var_frac[:3].sum():.0%} var) "
                     f"+ ridge-mapped test predictions -- {key.replace('_', ' ')}", fontsize=12)
        fig.tight_layout()
        out = args.output_dir / f"fv_pca3d_{key}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
