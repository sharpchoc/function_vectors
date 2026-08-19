#!/usr/bin/env python
"""Consistency gate: newly fit ridge preimage cells must reproduce the stored R2-study MSEs.

The full-dim ridge R2 study (results/direction3_fv_formation/
fulldim_ridge_activation_to_fv_varicl_top40/combined_metrics.csv) already recorded, for every
(icl_example_index, token_role, layer), the best alpha, CV MSE and pooled test-task MSE of the
SAME fit that fit_ridge_preimages_multicell.py performs (same loader, standardizer, alpha grid,
LOO-CV selection, test-MSE formula, task manifest, FV targets). This script joins each cell's
diagnostics.json against the stored CSV rows on capture_layer == layer (layer 0 = embedding is
excluded; the preimage fit starts at capture layer 1) and FAILS (exit 1) unless, over all
compared rows, best_alpha matches exactly and the relative test_mse difference is < --tol.

Any failure means: STOP the pipeline and report to the user. Do not proceed or rationalize the
discrepancy (per user instruction 2026-07-16).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff/train_varicl_top40")
    p.add_argument("--r2_csv", type=Path,
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/varicl_top40"
                   / "combined_metrics.csv")
    p.add_argument("--cells", nargs="+", required=True,
                   help="Cell dir names, e.g. pre_label_token_icl3.")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Max allowed relative |test_mse diff| per layer.")
    return p.parse_args()


def parse_cell_dir(name):
    role, icl = name.rsplit("_icl", 1)
    return role, int(icl)


def main():
    args = parse_args()
    stored = {}
    with open(args.r2_csv) as fh:
        for row in csv.DictReader(fh):
            key = (row["token_role"], int(row["icl_example_index"]), int(row["layer"]))
            stored[key] = row

    failures = []
    n_rows = 0
    for cell in args.cells:
        role, icl = parse_cell_dir(cell)
        diag_path = args.preimage_root / cell / "diagnostics.json"
        if not diag_path.exists():
            failures.append(f"{cell}: diagnostics.json missing at {diag_path}")
            continue
        diag = json.loads(diag_path.read_text())
        worst_rel, worst_layer = 0.0, None
        for entry in diag:
            layer = entry["capture_layer"]
            key = (role, icl, layer)
            if key not in stored:
                failures.append(f"{cell} L{layer}: no stored row in {args.r2_csv}")
                continue
            ref = stored[key]
            n_rows += 1
            rel = abs(entry["test_mse"] - float(ref["test_mse"])) / float(ref["test_mse"])
            if rel > worst_rel:
                worst_rel, worst_layer = rel, layer
            if entry["best_alpha"] != float(ref["best_alpha"]):
                failures.append(
                    f"{cell} L{layer}: best_alpha {entry['best_alpha']:g} != stored "
                    f"{float(ref['best_alpha']):g} (test_mse {entry['test_mse']:.6f} vs "
                    f"{float(ref['test_mse']):.6f})")
            if rel >= args.tol:
                failures.append(
                    f"{cell} L{layer}: test_mse rel diff {rel:.2e} >= {args.tol:g} "
                    f"({entry['test_mse']:.6f} vs stored {float(ref['test_mse']):.6f})")
        print(f"{cell}: max test_mse rel diff {worst_rel:.2e} (L{worst_layer})")

    print(f"\ncompared {n_rows} (cell, layer) rows across {len(args.cells)} cells")
    if failures:
        print(f"\nMSE CONSISTENCY CHECK FAILED ({len(failures)} problems):")
        for f in failures:
            print(f"  {f}")
        print("\nSTOP: report to the user before proceeding (do not self-adjudicate).")
        sys.exit(1)
    print("MSE consistency check PASSED.")


if __name__ == "__main__":
    main()
