#!/usr/bin/env python
"""Rank-k truncated-SVD (TSVD) pair-diff pre-images of the full-dim ridge maps.

Companion to fit_ridge_preimages_multicell.py (which must have run first: this script reads its
saved maps/layer_XX.pt). Motivation (see diagnose_pairdiff_preimage_spectrum.py): the ridge was
trained on 20 train-task FVs, so W has true rank <= 20; the exact inverse is dominated by the
numerical null space (cond ~ 1e9). The honest inverse keeps only the top-k singular directions:

    W_std^T = U S V^T ;  dz_k = sum_{i<k} (u_i^T fv_diff / s_i) v_i ;  dx = std * dz_k

Uses torch.svd_lowrank (randomized) for the top-k triplets - the spectrum collapses by ~9
orders of magnitude after rank ~20, so the randomized factorization is effectively exact there
(validated against the full fp64 SVDs of the diagnostics at pre_label_token_icl3 L4/8/12/20).

Outputs per cell under artifacts/preimage_pairdiff_tsvdk{k}/<fv_root name>/<cell>/:
  pairdiff_preimages/{A}__{B}_pairdiff_preimage_bank.pt
      {"preimages_by_edit_layer": {edit_layer: {"tsvd": dx, "k", "s_top", "s_k_over_s1"}}}
plus a top-level run_config.json.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.fit_ridge_preimages_multicell import DEFAULT_CELLS, DEFAULT_PAIRS, parse_cell
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    load_function_vector,
    load_json,
    torch_load_trusted,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    p.add_argument("--pair_specs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Capture layers (default 1..28, from the saved maps).")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--oversample", type=int, default=32,
                   help="svd_lowrank q = k + oversample.")
    p.add_argument("--niter", type=int, default=8)
    p.add_argument("--preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff/train_varicl_max4_top40",
                   help="Stage-1 output of fit_ridge_preimages_multicell (source of the maps).")
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: artifacts/preimage_pairdiff_tsvdk{k}/<fv_root basename>.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    stage1_cfg = load_json(args.preimage_root / "run_config.json")
    fv_root = Path(stage1_cfg["fv_root"])
    output_root = args.output_root or (ARTIFACTS_ROOT / f"preimage_pairdiff_tsvdk{args.k}"
                                       / fv_root.name)
    cells = [parse_cell(s) for s in args.cells]
    pairs = [tuple(s.split(":", 1)) for s in args.pair_specs]

    # Same fv_root as the stage-1 maps by construction (consistency rule).
    fv_diffs = {(f1, f2): (load_function_vector(fv_root, f1)
                           - load_function_vector(fv_root, f2)).float()
                for f1, f2 in pairs}

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", {
        "method": f"rank-{args.k} TSVD pre-image of the stage-1 full-dim ridge maps",
        "preimage_root": str(args.preimage_root), "fv_root": str(fv_root),
        "cells": args.cells, "pair_specs": args.pair_specs,
        "k": args.k, "oversample": args.oversample, "niter": args.niter,
        "edit_layer_mapping": "edit_layer = capture_layer - 1 (same as preimage_pairdiff)",
    })

    for role, icl in cells:
        cell_name = f"{role}_icl{icl}"
        maps_dir = args.preimage_root / cell_name / "maps"
        out_cell = output_root / cell_name / "pairdiff_preimages"
        done_marker = output_root / cell_name / "diagnostics.json"
        if done_marker.exists() and not args.overwrite:
            print(f"[{cell_name}] exists; skipping (pass --overwrite).")
            continue
        out_cell.mkdir(parents=True, exist_ok=True)

        layer_files = sorted(maps_dir.glob("layer_*.pt"))
        if args.layers is not None:
            keep = {int(l) for l in args.layers}
            layer_files = [f for f in layer_files if int(f.stem.split("_")[1]) in keep]
        pair_banks = {pair: {} for pair in pairs}
        diagnostics = []
        t0 = time.time()
        for f in layer_files:
            layer = int(f.stem.split("_")[1])
            m = torch_load_trusted(f, map_location="cpu")
            w_t = m["w_std"].float().T                       # maps dz -> fv
            std = m["std"].float()
            u, s, v = torch.svd_lowrank(w_t, q=args.k + args.oversample, niter=args.niter)
            u, s, v = u[:, :args.k], s[:args.k], v[:, :args.k]
            edit_layer = layer - 1
            for (f1, f2) in pairs:
                fv = fv_diffs[(f1, f2)]
                dz_k = v @ ((u.T @ fv) / s)
                dx = (dz_k * std).detach().cpu()
                pair_banks[(f1, f2)][edit_layer] = {
                    "tsvd": dx, "k": args.k,
                    "s_top": float(s[0]), "s_k_over_s1": float(s[-1] / s[0]),
                }
            diagnostics.append({"layer": layer, "edit_layer": edit_layer,
                                "s_top": float(s[0]), "s_k_over_s1": float(s[-1] / s[0])})
        for (f1, f2), bank in pair_banks.items():
            torch.save({"preimages_by_edit_layer": bank,
                        "definition": (f"rank-{args.k} TSVD pre-image: dz = V_k diag(1/s_k) "
                                       f"U_k^T (fv_A - fv_B); dx = std * dz; maps from "
                                       + str(args.preimage_root))},
                       out_cell / f"{f1}__{f2}_pairdiff_preimage_bank.pt")
        write_json(done_marker, diagnostics)
        print(f"[{cell_name}] {len(layer_files)} layers in {time.time()-t0:.1f}s")

    print("DONE")


if __name__ == "__main__":
    main()
