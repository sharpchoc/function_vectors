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
  preimages/{task}_tsvd_preimage_bank.pt      (--tasks; same schema, target = the task's own FV)
plus a top-level run_config.json.

Incremental: a cell is only recomputed for the requested pair/task banks that are missing
(--overwrite recomputes all requested), so adding task banks never touches existing pairdiff banks.
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
    p.add_argument("--pair_specs", nargs="*", default=DEFAULT_PAIRS,
                   help="A:B pair diffs to invert; pass with no values to disable.")
    p.add_argument("--tasks", nargs="*", default=[],
                   help="Single tasks whose own FV is inverted per layer -> "
                        "<cell>/preimages/{task}_tsvd_preimage_bank.pt")
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
    fv_tasks = {t: load_function_vector(fv_root, t).float() for t in args.tasks}

    output_root.mkdir(parents=True, exist_ok=True)
    run_config_path = output_root / "run_config.json"
    previous = load_json(run_config_path) if run_config_path.exists() else None
    if previous is not None:
        previous.pop("previous_run_configs", None)
    write_json(run_config_path, {
        "method": f"rank-{args.k} TSVD pre-image of the stage-1 full-dim ridge maps",
        "preimage_root": str(args.preimage_root), "fv_root": str(fv_root),
        "cells": args.cells, "pair_specs": args.pair_specs, "tasks": args.tasks,
        "k": args.k, "oversample": args.oversample, "niter": args.niter,
        "edit_layer_mapping": "edit_layer = capture_layer - 1 (same as preimage_pairdiff)",
        "previous_run_configs": [previous] if previous else [],
    })

    for role, icl in cells:
        cell_name = f"{role}_icl{icl}"
        maps_dir = args.preimage_root / cell_name / "maps"
        out_pair_dir = output_root / cell_name / "pairdiff_preimages"
        out_task_dir = output_root / cell_name / "preimages"
        pair_path = lambda f1, f2: out_pair_dir / f"{f1}__{f2}_pairdiff_preimage_bank.pt"
        task_path = lambda t: out_task_dir / f"{t}_tsvd_preimage_bank.pt"
        # Incremental skip: only (re)compute the requested banks that are missing.
        todo_pairs = [pr for pr in pairs if args.overwrite or not pair_path(*pr).exists()]
        todo_tasks = [t for t in args.tasks if args.overwrite or not task_path(t).exists()]
        if not todo_pairs and not todo_tasks:
            print(f"[{cell_name}] all requested banks exist; skipping (pass --overwrite).")
            continue
        if todo_pairs:
            out_pair_dir.mkdir(parents=True, exist_ok=True)
        if todo_tasks:
            out_task_dir.mkdir(parents=True, exist_ok=True)

        layer_files = sorted(maps_dir.glob("layer_*.pt"))
        if args.layers is not None:
            keep = {int(l) for l in args.layers}
            layer_files = [f for f in layer_files if int(f.stem.split("_")[1]) in keep]
        pair_banks = {pair: {} for pair in todo_pairs}
        task_banks = {t: {} for t in todo_tasks}
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

            def tsvd_entry(fv):
                dz_k = v @ ((u.T @ fv) / s)
                dx = (dz_k * std).detach().cpu()
                return {"tsvd": dx, "k": args.k,
                        "s_top": float(s[0]), "s_k_over_s1": float(s[-1] / s[0])}

            for (f1, f2) in todo_pairs:
                pair_banks[(f1, f2)][edit_layer] = tsvd_entry(fv_diffs[(f1, f2)])
            for t in todo_tasks:
                task_banks[t][edit_layer] = tsvd_entry(fv_tasks[t])
            diagnostics.append({"layer": layer, "edit_layer": edit_layer,
                                "s_top": float(s[0]), "s_k_over_s1": float(s[-1] / s[0])})
        for (f1, f2), bank in pair_banks.items():
            torch.save({"preimages_by_edit_layer": bank,
                        "definition": (f"rank-{args.k} TSVD pre-image: dz = V_k diag(1/s_k) "
                                       f"U_k^T (fv_A - fv_B); dx = std * dz; maps from "
                                       + str(args.preimage_root))},
                       pair_path(f1, f2))
        for t, bank in task_banks.items():
            torch.save({"preimages_by_edit_layer": bank,
                        "definition": (f"rank-{args.k} TSVD pre-image: dz = V_k diag(1/s_k) "
                                       f"U_k^T fv_task; dx = std * dz; maps from "
                                       + str(args.preimage_root))},
                       task_path(t))
        write_json(output_root / cell_name / "diagnostics.json", diagnostics)
        print(f"[{cell_name}] {len(layer_files)} layers | pairs={len(todo_pairs)} "
              f"tasks={len(todo_tasks)} in {time.time()-t0:.1f}s")

    print("DONE")


if __name__ == "__main__":
    main()
