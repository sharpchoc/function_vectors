#!/usr/bin/env python
"""Pair-diff pre-images through the k=16 PCA ridge (companion to fit_ridge_preimages_multicell).

Same 6 (token_role, icl) cells as the full-dim Stream S fit, but the regression is the
direction3 PCA variant (regress_activation_to_fv_pca_ridge.py): per cell x layer, activation
PCA (k_act=16) fit on the pooled 20-train rows, FV PCA (k_fv=16) fit once on the 20 train FVs,
standardized 16->16 ridge with LOO-task CV. The pre-image of a target FV difference is then

    t  = (fv_A - fv_B) @ fv_comp^T                    (16-dim FV-PC target; fv_mean cancels)
    dz = solve(A^T, t),  A = evecs (1/(evals+alpha)) c  (the fitted 16x16 std-feature -> FV-PC map)
    dx = (dz * feature_std) @ act_comp                (minimal-norm raw-space direction)

A is 16x16 and well-conditioned, so no damping/truncation gymnastics are needed - this IS the
"invert only the part of the map that exists" estimator, with the rank chosen at fit time.

CONSISTENCY (user rule): the FV root for the regression targets AND the inverted FV diff is the
single --fv_root (default train_varicl_max4_top40) - NOT the train_selected root used by the
committed pca_ridge_activation_to_fv study; the fit is redone here for that reason.

Caveat recorded per pair: the 16 FV-PCs are fit on TRAIN FVs; the held-out fv_diff target is
only partially inside that span (target_pc_coverage ~ 0.5-0.7 expected) - the pre-image can
only ever address that captured part.

Outputs per cell under artifacts/preimage_pairdiff_pcak16/<fv_root name>/<role>_icl{k}/:
  pairdiff_preimages/{A}__{B}_pairdiff_preimage_bank.pt
      {"preimages_by_edit_layer": {edit_layer: {"pca_k16": dx, "cond_A", "rel_residual",
                                                 "target_pc_coverage", "best_alpha", "cv_mse"}}}
  diagnostics.json, plus a top-level run_config.json.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.fit_prelabel_ridge_preimages import load_task_role_pooled
from src.eval_scripts.fit_ridge_preimages_multicell import DEFAULT_CELLS, DEFAULT_PAIRS, parse_cell
from src.eval_scripts.regress_activation_to_fv_pca_ridge import (
    project,
    ridge_eig_prep,
    torch_pca,
)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    role_load_icl_index,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    p.add_argument("--pair_specs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Capture layers (default 1..28; embedding layer 0 skipped).")
    p.add_argument("--k_act", type=int, default=16)
    p.add_argument("--k_fv", type=int, default=16)
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/debug/train_varicl_max4_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: artifacts/preimage_pairdiff_pcak16/<fv_root basename>.")
    p.add_argument("--alphas", nargs="+", type=float, default=None,
                   help="Default: np.logspace(-2, 6, 17) (the PCA ridge study grid).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = args.device
    alphas = list(args.alphas) if args.alphas is not None else list(np.logspace(-2, 6, 17))
    output_root = args.output_root or (ARTIFACTS_ROOT / "preimage_pairdiff_pcak16" / args.fv_root.name)

    cells = [parse_cell(s) for s in args.cells]
    pairs = [tuple(s.split(":", 1)) for s in args.pair_specs]
    pair_tasks = sorted({t for pair in pairs for t in pair})

    manifest = load_json(args.task_manifest)
    train_tasks = list(manifest["train_tasks"])

    missing = [t for t in train_tasks + pair_tasks
               if not (args.fv_root / t / f"{t}_function_vector.pt").exists()]
    if missing:
        raise FileNotFoundError(
            f"FVs missing under {args.fv_root} for: {missing}. Regression-target and pair-task "
            f"FVs must come from the same root (consistency rule).")

    fvs = {t: load_function_vector(args.fv_root, t).to(device=device, dtype=torch.float32)
           for t in train_tasks + pair_tasks}

    # FV PCA: once, on the 20 train FVs (cell/layer independent) - as in the study.
    fv_train_stack = torch.stack([fvs[t] for t in train_tasks], dim=0)
    fv_mean, fv_comp = torch_pca(fv_train_stack, args.k_fv)   # fv_comp: [k_fv, 4096]
    pair_targets = {}
    for f1, f2 in pairs:
        diff = fvs[f1] - fvs[f2]
        t16 = diff @ fv_comp.T                                # fv_mean cancels in the difference
        coverage = float((t16 @ fv_comp).norm() ** 2 / diff.norm() ** 2)
        pair_targets[(f1, f2)] = (t16, coverage)
        print(f"pair {f1}-{f2}: target_pc_coverage = {coverage:.3f}")

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", {
        "method": "PCA ridge (k_act->k_fv) pair-diff pre-image; see script docstring",
        "cells": args.cells, "pair_specs": args.pair_specs, "fv_root": str(args.fv_root),
        "k_act": args.k_act, "k_fv": args.k_fv, "alphas": [float(a) for a in alphas],
        "splits": args.splits, "std_eps": args.std_eps, "train_tasks": train_tasks,
        "target_pc_coverage": {f"{a}__{b}": c for (a, b), (_, c) in pair_targets.items()},
        "edit_layer_mapping": "edit_layer = capture_layer - 1 (same as preimage_pairdiff)",
    })

    for role, icl in cells:
        cell_name = f"{role}_icl{icl}"
        cell_dir = output_root / cell_name
        pairdiff_dir = cell_dir / "pairdiff_preimages"
        diag_path = cell_dir / "diagnostics.json"
        if diag_path.exists() and not args.overwrite:
            print(f"[{cell_name}] diagnostics exist; skipping (pass --overwrite).")
            continue
        pairdiff_dir.mkdir(parents=True, exist_ok=True)

        if icl == QUERY_ICL_INDEX:
            activations_root = args.query_activations_root
        else:
            activations_root = Path(args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)

        print(f"[{cell_name}] loading activations from {activations_root}")
        t0 = time.time()
        acts = {t: load_task_role_pooled(activations_root, t, args.splits, role, load_icl)
                for t in train_tasks}
        n_layers = next(iter(acts.values())).shape[1]
        layers = list(args.layers) if args.layers is not None else list(range(1, n_layers))
        print(f"[{cell_name}] loaded in {time.time()-t0:.1f}s | layers {layers[0]}..{layers[-1]}")

        pair_banks = {pair: {} for pair in pairs}
        diagnostics = []
        for layer in layers:
            t_cell = time.time()
            x_raw = {t: acts[t][:, layer, :].to(device=device, dtype=torch.float32)
                     for t in train_tasks}
            x_train_pool = torch.cat([x_raw[t] for t in train_tasks], dim=0)
            act_mean, act_comp = torch_pca(x_train_pool, args.k_act)   # [k_act, 4096]
            x_proj = {t: project(x_raw[t], act_mean, act_comp) for t in train_tasks}
            proj_pool = torch.cat([x_proj[t] for t in train_tasks], dim=0)
            mean = proj_pool.mean(dim=0)
            std = proj_pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
            xs = {t: (x_proj[t] - mean) / std for t in train_tasks}
            ys = {t: (fvs[t] - fv_mean) @ fv_comp.T for t in train_tasks}

            # LOO-task CV over alphas (targets in FV-PC space, as in the study)
            cv_sqerr = torch.zeros(len(alphas), dtype=torch.float32)
            cv_n = 0
            for held in train_tasks:
                fit_tasks = [t for t in train_tasks if t != held]
                x_fit = torch.cat([xs[t] for t in fit_tasks], dim=0)
                y_fit = torch.cat([ys[t].unsqueeze(0).expand(xs[t].shape[0], -1)
                                   for t in fit_tasks], dim=0)
                xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
                a_val = (xs[held] - xbar) @ evecs
                y_val = ys[held].unsqueeze(0).expand(xs[held].shape[0], -1)
                for ai, alpha in enumerate(alphas):
                    pred = (a_val / (evals + alpha)) @ c + ybar
                    cv_sqerr[ai] += torch.sum((pred - y_val) ** 2)
                cv_n += xs[held].shape[0] * args.k_fv
            cv_mse = (cv_sqerr / cv_n).numpy()
            best_idx = int(np.argmin(cv_mse))
            best_alpha = float(alphas[best_idx])

            # Refit on all 20 train tasks; explicit 16x16 map A: std-feature diff -> FV-PC diff
            x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
            y_fit = torch.cat([ys[t].unsqueeze(0).expand(xs[t].shape[0], -1)
                               for t in train_tasks], dim=0)
            xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
            A = evecs @ ((1.0 / (evals + best_alpha)).unsqueeze(1) * c)   # [k_act, k_fv]
            cond_A = float(torch.linalg.cond(A))

            edit_layer = layer - 1
            diag_row = {"layer": layer, "edit_layer": edit_layer, "best_alpha": best_alpha,
                        "cv_mse": float(cv_mse[best_idx]), "cond_A": cond_A, "pairs": {}}
            for (f1, f2) in pairs:
                t16, coverage = pair_targets[(f1, f2)]
                dz = torch.linalg.solve(A.T, t16.to(A.dtype))
                rel_res = float((dz @ A - t16).norm() / t16.norm())
                dx = ((dz * std) @ act_comp).detach().cpu()
                pair_banks[(f1, f2)][edit_layer] = {
                    "pca_k16": dx, "cond_A": cond_A, "rel_residual": rel_res,
                    "target_pc_coverage": coverage, "best_alpha": best_alpha,
                    "cv_mse": float(cv_mse[best_idx]),
                }
                diag_row["pairs"][f"{f1}__{f2}"] = {
                    "rel_residual": rel_res, "preimage_norm": float(dx.norm())}
            diagnostics.append(diag_row)
            print(f"[{cell_name}] L{layer:02d}: alpha={best_alpha:.3g} cond_A={cond_A:.2e} "
                  f"({time.time()-t_cell:.1f}s)")

        for (f1, f2), bank in pair_banks.items():
            torch.save({"preimages_by_edit_layer": bank,
                        "definition": ("PCA-k16 ridge pre-image: dz = solve(A^T, fv_diff@fv_comp^T); "
                                       "dx = (dz*std)@act_comp; fv_root = " + str(args.fv_root))},
                       pairdiff_dir / f"{f1}__{f2}_pairdiff_preimage_bank.pt")
        write_json(diag_path, diagnostics)
        print(f"[{cell_name}] wrote {pairdiff_dir}")

    print("DONE")


if __name__ == "__main__":
    main()
