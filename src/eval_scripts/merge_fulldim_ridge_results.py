#!/usr/bin/env python
"""Merge the per-ICL shard outputs of regress_activation_to_fv_fulldim_ridge.py.

Concatenates every shard_icl*/metrics.csv into combined_metrics.csv, renders heatmaps of
test_mse and best_alpha over (token position x layer), and writes a combined_summary.json with
the best cell overall and per token position.

A "token position" here is the pair (icl_example_index, token_role); the final prompt token is
(icl 10, last_prompt_token).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROLE_ORDER = ["pre_label_token", "first_label_token", "last_label_token", "last_prompt_token"]
ROLE_SHORT = {
    "pre_label_token": "pre",
    "first_label_token": "first",
    "last_label_token": "last",
    "last_prompt_token": "finaltok",
}


def parse_args():
    p = argparse.ArgumentParser(description="Merge full-dim ridge shard results + heatmaps.")
    p.add_argument("--input_dir", type=Path, default=Path("results/fulldim_ridge_activation_to_fv"))
    p.add_argument("--expected_cells", type=int, default=899,
                   help="Expected total cell count (31 token positions x 29 layers). 0 to skip the check.")
    return p.parse_args()


def position_key(icl, role):
    return (int(icl), ROLE_ORDER.index(role) if role in ROLE_ORDER else 99)


def position_label(icl, role):
    return f"icl{int(icl):02d}/{ROLE_SHORT.get(role, role)}"


def load_rows(input_dir):
    rows = []
    shard_csvs = sorted(input_dir.glob("shard_icl*/metrics.csv"))
    if not shard_csvs:
        raise FileNotFoundError(f"No shard_icl*/metrics.csv under {input_dir}")
    for csv_path in shard_csvs:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows, shard_csvs


def render_heatmap(positions, layers, grid, title, out_path, log_scale=False, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.32), max(5, len(positions) * 0.3)))
    data = np.array(grid, dtype=float)
    if log_scale:
        with np.errstate(divide="ignore"):
            data = np.log10(data)
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=6)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions, fontsize=6)
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_ylabel("token position (icl/role)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("log10 " + title if log_scale else title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    rows, shard_csvs = load_rows(args.input_dir)
    print(f"Loaded {len(rows)} cells from {len(shard_csvs)} shards.")

    # Sorted axes.
    pos_set = sorted({(int(r["icl_example_index"]), r["token_role"]) for r in rows},
                     key=lambda ir: position_key(*ir))
    layer_set = sorted({int(r["layer"]) for r in rows})
    pos_index = {pos: i for i, pos in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}

    mse_grid = np.full((len(pos_set), len(layer_set)), np.nan)
    alpha_grid = np.full((len(pos_set), len(layer_set)), np.nan)
    for r in rows:
        i = pos_index[(int(r["icl_example_index"]), r["token_role"])]
        j = layer_index[int(r["layer"])]
        mse_grid[i, j] = float(r["test_mse"])
        alpha_grid[i, j] = float(r["best_alpha"])

    # combined_metrics.csv
    fields = ["icl_example_index", "token_role", "layer", "feature_dim", "target_dim",
              "best_alpha", "cv_mse", "alpha_pinned", "train_sample_count", "test_sample_count",
              "train_mse", "test_mse", "train_mean_squared_l2", "test_mean_squared_l2"]
    combined_csv = args.input_dir / "combined_metrics.csv"
    rows_sorted = sorted(rows, key=lambda r: (position_key(int(r["icl_example_index"]), r["token_role"]),
                                              int(r["layer"])))
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {combined_csv} ({len(rows_sorted)} rows).")

    # Heatmaps.
    pos_labels = [position_label(*p) for p in pos_set]
    render_heatmap(pos_labels, layer_set, mse_grid, "test_mse",
                   args.input_dir / "combined_test_mse_heatmap.png", log_scale=True, cmap="viridis_r")
    render_heatmap(pos_labels, layer_set, alpha_grid, "best_alpha",
                   args.input_dir / "combined_best_alpha_heatmap.png", log_scale=True, cmap="magma")
    print("Wrote heatmaps: combined_test_mse_heatmap.png, combined_best_alpha_heatmap.png")

    # Summary.
    finite = [r for r in rows if np.isfinite(float(r["test_mse"]))]
    best_overall = min(finite, key=lambda r: float(r["test_mse"]))
    per_position_best = {}
    pinned = [position_label(int(r["icl_example_index"]), r["token_role"]) + f"/L{r['layer']}"
              for r in rows if str(r.get("alpha_pinned", "")).lower() == "true"]
    for pos in pos_set:
        cells = [r for r in finite
                 if (int(r["icl_example_index"]), r["token_role"]) == pos]
        if cells:
            b = min(cells, key=lambda r: float(r["test_mse"]))
            per_position_best[position_label(*pos)] = {
                "layer": int(b["layer"]), "test_mse": float(b["test_mse"]),
                "best_alpha": float(b["best_alpha"]),
            }
    summary = {
        "n_cells": len(rows),
        "n_token_positions": len(pos_set),
        "n_layers": len(layer_set),
        "best_overall": {
            "token_position": position_label(int(best_overall["icl_example_index"]), best_overall["token_role"]),
            "icl_example_index": int(best_overall["icl_example_index"]),
            "token_role": best_overall["token_role"],
            "layer": int(best_overall["layer"]),
            "test_mse": float(best_overall["test_mse"]),
            "best_alpha": float(best_overall["best_alpha"]),
        },
        "best_per_token_position": per_position_best,
        "alpha_pinned_cells": pinned,
    }
    with open(args.input_dir / "combined_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote combined_summary.json. Best overall: {summary['best_overall']}")
    if pinned:
        print(f"WARNING: {len(pinned)} cells have alpha pinned to a grid endpoint; consider widening --alphas.")

    if args.expected_cells and len(rows) != args.expected_cells:
        print(f"WARNING: expected {args.expected_cells} cells but found {len(rows)}.")


if __name__ == "__main__":
    main()
