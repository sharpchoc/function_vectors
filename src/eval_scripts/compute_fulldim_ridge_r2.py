#!/usr/bin/env python
"""Post-hoc R^2 for the full-dim (4096->4096) ridge regression, from stored test_mse.

The full-dim ridge fits activation -> function-vector per (token position, layer) cell, but the
TARGET set is identical in every cell: the 7 test-task function vectors, each broadcast to that
task's 170 activation rows. Only the input features X change from cell to cell. So the R^2
denominator (total sum of squares of the targets about a constant baseline) is a single constant
shared by all cells, and

    R^2(cell) = 1 - SS_res(cell) / SS_tot ,  SS_res(cell) = test_mse(cell) * test_n * hidden
              = 1 - test_mse(cell) / V ,      V = SS_tot / (test_n * hidden)

where V is the per-element target variance about the baseline mean. No regression is re-fit here;
we only load the FV targets to get V, then rescale the stored test_mse (and train_mse) grids.

Baseline for R^2: the TRAIN-target mean (ybar) -- exactly the constant that ridge_eig_prep centers
on, i.e. "did the learned map beat predicting the mean training FV?". The test-target-mean variant
(sklearn's default) is also reported for reference; switching the headline metric is a one-liner.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    load_function_vector,
    load_json,
)
from eval_scripts.merge_fulldim_ridge_results import (
    position_key,
    position_label,
    render_heatmap,
    run_title,
)


def parse_args():
    p = argparse.ArgumentParser(description="Post-hoc R^2 for full-dim ridge from stored MSE.")
    p.add_argument("--input_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/main")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--task_manifest", type=Path,
                   default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--train_tasks", nargs="+", default=None, help="Override train tasks.")
    p.add_argument("--test_tasks", nargs="+", default=None, help="Override test tasks (default: 7).")
    return p.parse_args()


def per_element_variance(fvs_stack, baseline):
    """Mean over rows of ||fv - baseline||^2 / hidden (the R^2 denominator per element)."""
    diff = fvs_stack - baseline.unsqueeze(0)
    return float(torch.mean(torch.sum(diff ** 2, dim=1))) / fvs_stack.shape[1]


def main():
    args = parse_args()

    manifest = load_json(args.task_manifest)
    train_tasks = list(args.train_tasks) if args.train_tasks is not None else list(manifest["train_tasks"])
    test_tasks = list(args.test_tasks) if args.test_tasks is not None else list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)

    train_fv = torch.stack([load_function_vector(args.fv_root, t) for t in train_tasks], dim=0)
    test_fv = torch.stack([load_function_vector(args.fv_root, t) for t in test_tasks], dim=0)
    hidden = train_fv.shape[1]

    ybar_train = train_fv.mean(dim=0)          # constant ridge centers on (train-mean baseline)
    ybar_test = test_fv.mean(dim=0)            # sklearn-style baseline (mean of the eval targets)

    # Denominators (per-element target variance about each baseline).
    v_test_trainbase = per_element_variance(test_fv, ybar_train)   # headline test R^2 denominator
    v_test_testbase = per_element_variance(test_fv, ybar_test)     # reference (sklearn convention)
    v_train_trainbase = per_element_variance(train_fv, ybar_train)  # train R^2 denominator

    print(f"tasks: {len(train_tasks)} train, {len(test_tasks)} test | hidden={hidden}")
    print(f"V(test | train-mean baseline)  = {v_test_trainbase:.4f}")
    print(f"V(test | test-mean baseline)   = {v_test_testbase:.4f}")
    print(f"V(train | train-mean baseline) = {v_train_trainbase:.4f}")

    combined_csv = args.input_dir / "combined_metrics.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(f"{combined_csv} not found; run merge_fulldim_ridge_results.py first.")
    with open(combined_csv) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        test_mse = float(r["test_mse"])
        train_mse = float(r["train_mse"])
        r["test_r2"] = 1.0 - test_mse / v_test_trainbase
        r["test_r2_testmean_baseline"] = 1.0 - test_mse / v_test_testbase
        r["train_r2"] = 1.0 - train_mse / v_train_trainbase

    # Write augmented CSV.
    out_csv = args.input_dir / "combined_metrics_with_r2.csv"
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows).")

    # R^2 heatmap over the same (token position x layer) grid as the MSE heatmap.
    pos_set = sorted({(int(r["icl_example_index"]), r["token_role"]) for r in rows},
                     key=lambda ir: position_key(*ir))
    layer_set = sorted({int(r["layer"]) for r in rows})
    pos_index = {pos: i for i, pos in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}

    r2_grid = np.full((len(pos_set), len(layer_set)), np.nan)
    for r in rows:
        i = pos_index[(int(r["icl_example_index"]), r["token_role"])]
        j = layer_index[int(r["layer"])]
        r2_grid[i, j] = r["test_r2"]

    pos_labels = [position_label(*p) for p in pos_set]
    render_heatmap(pos_labels, layer_set, r2_grid, "test_r2 (train-mean baseline)",
                   args.input_dir / "combined_test_r2_heatmap.png", log_scale=False, cmap="viridis",
                   suptitle=run_title(args.input_dir.name))
    print("Wrote combined_test_r2_heatmap.png")

    finite = [r for r in rows if np.isfinite(r["test_r2"])]
    best = max(finite, key=lambda r: r["test_r2"])
    n_beat = sum(1 for r in finite if r["test_r2"] > 0)
    print(f"\nBest test R^2 = {best['test_r2']:.4f} at "
          f"{position_label(int(best['icl_example_index']), best['token_role'])} "
          f"L{best['layer']} (test_mse={float(best['test_mse']):.4f}, train R^2={best['train_r2']:.4f})")
    print(f"{n_beat}/{len(finite)} cells beat the train-mean baseline (test_r2 > 0).")


if __name__ == "__main__":
    main()
