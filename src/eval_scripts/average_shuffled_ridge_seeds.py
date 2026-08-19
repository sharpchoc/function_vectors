#!/usr/bin/env python
"""Average the shuffled-label full-dim ridge control across permutation seeds.

Each seed dir is a complete run of regress_activation_to_fv_fulldim_ridge.py with
--shuffle_train_labels --shuffle_seed {s}, already merged (merge_fulldim_ridge_results.py) and
R^2-annotated (compute_fulldim_ridge_r2.py), i.e. containing combined_metrics_with_r2.csv.

This script averages test_mse / test_r2 (and train metrics, for reference) per (token position,
layer) cell across the seeds and renders the seed-mean MSE and R^2 heatmaps on the same grid and
color conventions as the real (unshuffled) run, so the two are directly comparable side by side.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FV_FORMATION_DIR
from eval_scripts.merge_fulldim_ridge_results import (
    position_key,
    position_label,
    render_heatmap,
    run_title,
)


MEAN_FIELDS = ["test_mse", "test_r2", "test_r2_testmean_baseline", "train_mse", "train_r2"]


def parse_args():
    p = argparse.ArgumentParser(description="Average shuffled-label ridge control across seeds.")
    p.add_argument("--seed_dirs", nargs="+", type=Path, default=None,
                   help="Per-seed run dirs (default: fulldim_ridge_activation_to_fv_shuffled_seed{0,1,2}).")
    p.add_argument("--output_dir", type=Path,
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/controls/shuffled")
    return p.parse_args()


def load_seed(seed_dir):
    csv_path = seed_dir / "combined_metrics_with_r2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run merge + compute_fulldim_ridge_r2 first.")
    cells = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            key = (int(r["icl_example_index"]), r["token_role"], int(r["layer"]))
            cells[key] = r
    return cells


def main():
    args = parse_args()
    seed_dirs = args.seed_dirs or [
        FV_FORMATION_DIR / f"activation_to_fv_decoding/fulldim_ridge/controls/shuffled_seed{s}" for s in range(3)
    ]
    per_seed = [load_seed(d) for d in seed_dirs]
    keys = set(per_seed[0])
    for d, cells in zip(seed_dirs[1:], per_seed[1:]):
        if set(cells) != keys:
            raise ValueError(f"Cell set mismatch between {seed_dirs[0]} and {d}")
    keys = sorted(keys, key=lambda k: (position_key(k[0], k[1]), k[2]))
    print(f"Averaging {len(keys)} cells over {len(seed_dirs)} seeds.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for key in keys:
        icl, role, layer = key
        row = {"icl_example_index": icl, "token_role": role, "layer": layer}
        for field in MEAN_FIELDS:
            vals = [float(cells[key][field]) for cells in per_seed]
            row[field] = float(np.mean(vals))
            for s, v in enumerate(vals):
                row[f"{field}_seed{s}"] = v
        rows_out.append(row)

    fields = list(rows_out[0].keys())
    out_csv = args.output_dir / "combined_metrics_seed_mean.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {out_csv} ({len(rows_out)} rows).")

    # Heatmaps on the same grid/conventions as the real run.
    pos_set = sorted({(r["icl_example_index"], r["token_role"]) for r in rows_out},
                     key=lambda ir: position_key(*ir))
    layer_set = sorted({r["layer"] for r in rows_out})
    pos_index = {pos: i for i, pos in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}
    mse_grid = np.full((len(pos_set), len(layer_set)), np.nan)
    r2_grid = np.full((len(pos_set), len(layer_set)), np.nan)
    for r in rows_out:
        i = pos_index[(r["icl_example_index"], r["token_role"])]
        j = layer_index[r["layer"]]
        mse_grid[i, j] = r["test_mse"]
        r2_grid[i, j] = r["test_r2"]
    pos_labels = [position_label(*p) for p in pos_set]
    suptitle = run_title(args.output_dir.name)
    render_heatmap(pos_labels, layer_set, mse_grid, "test_mse (seed mean)",
                   args.output_dir / "combined_test_mse_heatmap.png", log_scale=True, cmap="viridis_r",
                   suptitle=suptitle)
    render_heatmap(pos_labels, layer_set, r2_grid, "test_r2 (seed mean, train-mean baseline)",
                   args.output_dir / "combined_test_r2_heatmap.png", log_scale=False, cmap="viridis",
                   suptitle=suptitle)
    print("Wrote heatmaps: combined_test_mse_heatmap.png, combined_test_r2_heatmap.png")

    def stats(rows, field):
        vals = np.array([r[field] for r in rows])
        best_i = int(np.argmax(vals)) if field.endswith("r2") else int(np.argmin(vals))
        b = rows[best_i]
        return {
            "median": float(np.median(vals)),
            "best": float(vals[best_i]),
            "best_cell": position_label(b["icl_example_index"], b["token_role"]) + f"/L{b['layer']}",
            "frac_positive": float(np.mean(vals > 0)) if field.endswith("r2") else None,
        }

    summary = {
        "seed_dirs": [str(d) for d in seed_dirs],
        "n_seeds": len(seed_dirs),
        "n_cells": len(rows_out),
        "seed_mean": {f: stats(rows_out, f) for f in ["test_mse", "test_r2"]},
        "per_seed": [
            {
                "dir": str(d),
                "test_mse_median": float(np.median([float(c["test_mse"]) for c in cells.values()])),
                "test_r2_median": float(np.median([float(c["test_r2"]) for c in cells.values()])),
                "test_r2_max": float(np.max([float(c["test_r2"]) for c in cells.values()])),
            }
            for d, cells in zip(seed_dirs, per_seed)
        ],
    }
    with open(args.output_dir / "combined_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote combined_summary.json. Seed-mean test_r2: median={summary['seed_mean']['test_r2']['median']:.4f} "
          f"best={summary['seed_mean']['test_r2']['best']:.4f} at {summary['seed_mean']['test_r2']['best_cell']}")


if __name__ == "__main__":
    main()
