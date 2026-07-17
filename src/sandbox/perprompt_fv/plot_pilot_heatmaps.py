#!/usr/bin/env python
"""SANDBOX: (token position x layer) heatmaps for the per-prompt-target ridge grid.

Renders, in the same style/axes as the canonical study's combined heatmaps
(`merge_fulldim_ridge_results.render_heatmap`):
  * test_mse_fv (log10) and test_r2_fv        -- comparable to the canonical study
  * test_mse_pp (log10) and test_r2_pp        -- vs the per-prompt targets
  * delta_test_r2_fv = new test_r2_fv - old test_r2 (diverging, centered at 0)
Outputs land next to summary_vs_canonical_all.csv under the pilot root.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_scripts.merge_fulldim_ridge_results import (  # noqa: E402
    position_key,
    position_label,
    render_heatmap,
)
from utils.paths import RESULTS_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/summary_vs_canonical_all.csv")
    return p.parse_args()


def render_heatmap_diverging(pos_labels, layers, grid, title, out_path, vmax, cbar_label):
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.32), max(5, len(pos_labels) * 0.3)))
    im = ax.imshow(np.array(grid, dtype=float), aspect="auto", cmap="coolwarm",
                   interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=6)
    ax.set_yticks(range(len(pos_labels)))
    ax.set_yticklabels(pos_labels, fontsize=6)
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_ylabel("token position (icl/role)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label(cbar_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_side_by_side(pos_labels, layers, grids, panel_titles, out_path, cbar_label, suptitle):
    """N heatmap panels on ONE shared color scale + one colorbar, for direct comparison."""
    vmin = float(np.nanmin([np.nanmin(g) for g in grids]))
    vmax = float(np.nanmax([np.nanmax(g) for g in grids]))
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(max(8, len(layers) * 0.32) * n * 0.85,
                                            max(5, len(pos_labels) * 0.3)), sharey=True)
    for k, (ax, grid, title) in enumerate(zip(axes, grids, panel_titles)):
        im = ax.imshow(np.array(grid, dtype=float), aspect="auto", cmap="viridis",
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=6)
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_title(title, fontsize=10)
        if k == 0:
            ax.set_yticks(range(len(pos_labels)))
            ax.set_yticklabels(pos_labels, fontsize=6)
            ax.set_ylabel("token position (icl/role)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    fig.suptitle(suptitle, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.summary_csv.parent
    with open(args.summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    pos_set = sorted({(int(r["icl_index"]), r["token_role"]) for r in rows}, key=lambda ir: position_key(*ir))
    layer_set = sorted({int(r["layer"]) for r in rows})
    pos_index = {pos: i for i, pos in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}
    pos_labels = [position_label(icl, role) for icl, role in pos_set]

    grids = {k: np.full((len(pos_set), len(layer_set)), np.nan) for k in
             ("mse_fv", "r2_fv", "mse_pp", "r2_pp", "dr2_fv", "old_r2_fv")}
    for r in rows:
        i = pos_index[(int(r["icl_index"]), r["token_role"])]
        j = layer_index[int(r["layer"])]
        grids["mse_fv"][i, j] = float(r["new_test_mse_fv"])
        grids["r2_fv"][i, j] = float(r["new_test_r2_fv"])
        grids["mse_pp"][i, j] = float(r["new_test_mse_pp"])
        grids["r2_pp"][i, j] = float(r["new_test_r2_pp"])
        if r["old_test_r2_fv"] not in ("", "None"):
            grids["old_r2_fv"][i, j] = float(r["old_test_r2_fv"])
            grids["dr2_fv"][i, j] = float(r["new_test_r2_fv"]) - float(r["old_test_r2_fv"])

    sup = "SANDBOX GPT-J full-dim ridge: activation → per-prompt top-40 head-sum target (4096 → 4096)"
    render_heatmap(pos_labels, layer_set, grids["mse_fv"], "test_mse vs stored FV (comparable to canonical)",
                   out_dir / "heatmap_test_mse_fv.png", log_scale=True, cmap="viridis", suptitle=sup)
    render_heatmap(pos_labels, layer_set, grids["r2_fv"], "test_r2 vs stored FV (train-mean baseline)",
                   out_dir / "heatmap_test_r2_fv.png", log_scale=False, cmap="viridis", suptitle=sup)
    render_heatmap(pos_labels, layer_set, grids["mse_pp"], "test_mse vs per-prompt targets",
                   out_dir / "heatmap_test_mse_pp.png", log_scale=True, cmap="viridis", suptitle=sup)
    render_heatmap(pos_labels, layer_set, grids["r2_pp"], "test_r2 vs per-prompt targets (train-mean baseline)",
                   out_dir / "heatmap_test_r2_pp.png", log_scale=False, cmap="viridis", suptitle=sup)
    render_side_by_side(
        pos_labels, layer_set, [grids["old_r2_fv"], grids["r2_fv"]],
        ["canonical: FV-broadcast targets", "SANDBOX: per-prompt head-sum targets"],
        out_dir / "heatmap_test_r2_fv_side_by_side.png",
        cbar_label="test_r2 vs stored varicl_top40 test FVs (train-mean baseline)",
        suptitle="GPT-J full-dim ridge (4096 → 4096): test R² vs stored varicl_top40 test FVs, shared color scale",
    )
    vmax = float(np.nanmax(np.abs(grids["dr2_fv"])))
    render_heatmap_diverging(pos_labels, layer_set, grids["dr2_fv"],
                             "Δ test_r2 vs stored FV (per-prompt targets − canonical broadcast)",
                             out_dir / "heatmap_delta_test_r2_fv.png", vmax=vmax,
                             cbar_label="Δ test_r2 (red = per-prompt better)")
    for name in ("heatmap_test_mse_fv", "heatmap_test_r2_fv", "heatmap_test_mse_pp",
                 "heatmap_test_r2_pp", "heatmap_test_r2_fv_side_by_side", "heatmap_delta_test_r2_fv"):
        print(f"wrote {out_dir / (name + '.png')}")


if __name__ == "__main__":
    main()
