#!/usr/bin/env python
"""Per-TEST-TASK R^2 heatmaps for the full-dim ridge, from the stored per_test_task_mse.

compute_fulldim_ridge_r2.py rescales the POOLED test_mse into one R^2 grid; this script does the
same per test task. Each shard's metrics.json stores per_test_task_mse for every (token position,
layer) cell, and each task's target is its single FV broadcast to that task's rows, so the
per-task R^2 denominator is the constant

    V_task = ||fv_task - ybar_train||^2 / hidden      (train-mean baseline, as in the pooled R^2)

and R^2_task(cell) = 1 - per_task_mse(cell) / V_task. No regression is re-fit.

Outputs (under <input_dir>/per_task_r2/): one heatmap PNG per task (shared color scale across the
requested tasks so panels are comparable), a combined panel figure, and per_task_r2.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from eval_scripts.regress_activation_to_fv_fulldim_ridge import load_function_vector, load_json
from eval_scripts.merge_fulldim_ridge_results import (
    position_key,
    position_label,
    render_heatmap,
    run_title,
)


def parse_args():
    p = argparse.ArgumentParser(description="Per-test-task R^2 heatmaps from stored per_test_task_mse.")
    p.add_argument("--input_dir", type=Path,
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/varicl_top40")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--task_manifest", type=Path,
                   default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--train_tasks", nargs="+", default=None, help="Override train tasks (baseline mean).")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Test tasks to plot (default: every key of per_test_task_mse).")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Default: <input_dir>/per_task_r2")
    return p.parse_args()


def load_shard_rows(input_dir):
    rows = []
    shard_jsons = sorted(input_dir.glob("shard_icl*/metrics.json"))
    if not shard_jsons:
        raise FileNotFoundError(f"No shard_icl*/metrics.json under {input_dir}")
    for path in shard_jsons:
        rows.extend(load_json(path))
    missing = [r for r in rows if "per_test_task_mse" not in r]
    if missing:
        raise ValueError(f"{len(missing)} cells lack per_test_task_mse (old run?) in {input_dir}")
    return rows


def main():
    args = parse_args()
    out_dir = args.output_dir if args.output_dir is not None else args.input_dir / "per_task_r2"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.task_manifest)
    train_tasks = list(args.train_tasks) if args.train_tasks is not None else list(manifest["train_tasks"])

    rows = load_shard_rows(args.input_dir)
    available = sorted(rows[0]["per_test_task_mse"].keys())
    tasks = list(args.tasks) if args.tasks is not None else available
    absent = sorted(set(tasks) - set(available))
    if absent:
        raise ValueError(f"Tasks {absent} not in per_test_task_mse (available: {available})")

    train_fv = torch.stack([load_function_vector(args.fv_root, t) for t in train_tasks], dim=0)
    ybar_train = train_fv.mean(dim=0)
    hidden = train_fv.shape[1]
    v_task = {}
    for task in tasks:
        fv = load_function_vector(args.fv_root, task)
        v_task[task] = float(torch.sum((fv - ybar_train) ** 2)) / hidden
        print(f"V({task} | train-mean baseline) = {v_task[task]:.4f}")

    # Shared (position x layer) axes across all shards.
    pos_set = sorted({(int(r["icl_example_index"]), r["token_role"]) for r in rows},
                     key=lambda ir: position_key(*ir))
    layer_set = sorted({int(r["layer"]) for r in rows})
    pos_index = {pos: i for i, pos in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}
    pos_labels = [position_label(*p) for p in pos_set]

    grids = {task: np.full((len(pos_set), len(layer_set)), np.nan) for task in tasks}
    for r in rows:
        i = pos_index[(int(r["icl_example_index"]), r["token_role"])]
        j = layer_index[int(r["layer"])]
        for task in tasks:
            grids[task][i, j] = 1.0 - float(r["per_test_task_mse"][task]) / v_task[task]

    # Shared color scale so per-task panels are directly comparable.
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
    vmin, vmax = float(finite.min()), float(finite.max())

    csv_path = out_dir / "per_task_r2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "icl_example_index", "token_role", "layer", "per_task_test_mse", "test_r2"])
        for r in rows:
            for task in tasks:
                mse = float(r["per_test_task_mse"][task])
                w.writerow([task, r["icl_example_index"], r["token_role"], r["layer"],
                            mse, 1.0 - mse / v_task[task]])
    print(f"Wrote {csv_path}")

    suptitle = run_title(args.input_dir.name)
    summary = {"input_dir": str(args.input_dir), "fv_root": str(args.fv_root),
               "baseline": "train-mean FV", "shared_color_scale": [vmin, vmax], "tasks": {}}
    for task in tasks:
        g = grids[task]
        # Same renderer as the pooled heatmap, but pin the shared scale via clipping-free norm:
        # render_heatmap has no vmin/vmax args, so draw directly here with the same layout.
        fig, ax = plt.subplots(figsize=(max(8, len(layer_set) * 0.32), max(5, len(pos_set) * 0.3)))
        im = ax.imshow(g, aspect="auto", cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(layer_set)))
        ax.set_xticklabels(layer_set, fontsize=6)
        ax.set_yticks(range(len(pos_set)))
        ax.set_yticklabels(pos_labels, fontsize=6)
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_ylabel("token position (icl/role)")
        ax.set_title(f"{task}: test_r2 (train-mean baseline)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025)
        cbar.set_label("test_r2", fontsize=8)
        fig.suptitle(suptitle, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        out_png = out_dir / f"test_r2_heatmap_{task}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        finite_mask = np.isfinite(g)
        best_flat = np.nanargmax(np.where(finite_mask, g, -np.inf))
        bi, bj = np.unravel_index(best_flat, g.shape)
        summary["tasks"][task] = {
            "V_train_mean_baseline": v_task[task],
            "best_r2": float(g[bi, bj]),
            "best_cell": {"position": pos_labels[bi], "layer": int(layer_set[bj])},
            "median_r2": float(np.nanmedian(g)),
            "min_r2": float(np.nanmin(g[finite_mask])),
            "cells_beating_baseline": int(np.sum(g[finite_mask] > 0)),
            "n_cells": int(finite_mask.sum()),
        }
        s = summary["tasks"][task]
        print(f"{task}: best R^2 {s['best_r2']:.3f} at {s['best_cell']['position']} "
              f"L{s['best_cell']['layer']} | median {s['median_r2']:.3f} | "
              f">0 in {s['cells_beating_baseline']}/{s['n_cells']} cells -> {out_png.name}")

    # Combined panel, one row per task, shared scale.
    n = len(tasks)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * max(7, len(layer_set) * 0.3),
                                      nrows * max(4.5, len(pos_set) * 0.16)))
    for k, task in enumerate(tasks):
        ax = axes[k // ncols][k % ncols]
        im = ax.imshow(grids[task], aspect="auto", cmap="viridis", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(0, len(layer_set), 2))
        ax.set_xticklabels(layer_set[::2], fontsize=6)
        ax.set_yticks(range(len(pos_set)))
        ax.set_yticklabels(pos_labels, fontsize=5)
        ax.set_title(task, fontsize=10)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{suptitle}\nper-task test_r2 (train-mean baseline, shared scale)", fontsize=10)
    cbar = fig.colorbar(im, ax=[a for row in axes for a in row], fraction=0.02)
    cbar.set_label("test_r2", fontsize=8)
    panel_png = out_dir / "test_r2_heatmap_panel.png"
    fig.savefig(panel_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {panel_png}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
