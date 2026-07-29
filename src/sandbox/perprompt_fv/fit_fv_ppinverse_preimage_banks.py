#!/usr/bin/env python
"""SANDBOX: Stream W-schema TSVD banks — task FV inverted via the PER-PROMPT regression.

Sanity check (user spec 2026-07-29): recreate the Stream W 1-shot ablation figure with one
ingredient changed — the direction is the truncated pseudo-inverse of the CANONICAL task FV
(train_varicl_top40, NOT the per-prompt FVs) through the SANDBOX per-prompt ridge maps:

  refit  W per (cell, capture layer) at the stored CV alpha (identical to
         invert_perprompt_fvs_truncsvd.py, incl. its REPRO GATE on stored test MSEs)
  k      = rank90 by sigma^2 energy of W's spectrum (part-1 convention);
           GATE: must EXACTLY match the stored preimages_truncsvd/cells_summary.csv rank90
  dz     = fv @ Vh[:k]^T diag(1/S[:k]) U[:,:k]^T     -- RAW fv, NOT centered by ybar
           (user decision 2026-07-29: byte-faithful to Stream W's own uncentered convention)
  dx     = dz * std                                   -- direction un-standardization ONLY:
           no mu, no xbar (directions carry no offsets; see DECISIONS 2026-07-28 entry)

Banks are written in the EXACT Stream W schema so the untouched
`ablate_oneshot_preimage_logprob.py` (--tsvd_root) and `plot_oneshot_preimage_ablation.py`
consume them directly:

  <output_root>/<cell>/preimages/{task}_tsvd_preimage_bank.pt
      {"preimages_by_edit_layer": {edit_layer 0..27: {"tsvd": dx fp32, "k", "s_top", ...}}}

GPU recommended (168 refits + full 4096^2 SVDs; driver="gesvd" pinned on CUDA).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sandbox.perprompt_fv.invert_perprompt_fvs_truncsvd import rank_energy  # noqa: E402
from sandbox.perprompt_fv.regress_activation_to_perprompt_headsum_ridge import (  # noqa: E402
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    FINAL_PROMPT_ROLE,
    QUERY_ICL_INDEX,
    align_targets,
    load_function_vector,
    load_json,
    load_perprompt_targets,
    load_task_role_pooled,
    ridge_eig_prep,
    ridge_predict,
    role_load_icl_index,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402

# Stream W arm cells -> (icl_index, token_role)
ABLATION_CELLS = {
    "pre_label_token_icl1": (1, "pre_label_token"),
    "last_label_token_icl1": (1, "last_label_token"),
    "pre_label_token_icl2": (2, "pre_label_token"),
    "pre_label_token_icl10": (10, "pre_label_token"),
    "last_label_token_icl10": (10, "last_label_token"),
    "last_prompt_token_icl10": (10, "last_prompt_token"),
}
N_EDIT_LAYERS = 28


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC))
    p.add_argument("--cells", nargs="+", default=list(ABLATION_CELLS), choices=list(ABLATION_CELLS))
    p.add_argument("--task_manifest", type=Path,
                   default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--targets_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--metrics_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40")
    p.add_argument("--cells_summary_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/preimages_truncsvd/cells_summary.csv")
    p.add_argument("--output_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_fv_preimages/fv_ppinverse_tsvd_banks")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Capture layers subset (SMOKE ONLY — banks written only when all 28 "
                        "edit layers are covered, i.e. leave unset for real runs).")
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--energy_threshold", type=float, default=0.90)
    p.add_argument("--gate_rel_tol", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = args.device
    dtype = torch.float32

    manifest = load_json(args.task_manifest)
    train_tasks = list(manifest["train_tasks"])
    test_tasks = list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)   # pilot's eval set (repro gate must match)
    all_tasks = train_tasks + test_tasks
    direction_tasks = list(args.tasks)
    missing_dir = [t for t in direction_tasks if t not in all_tasks]
    if missing_dir:
        raise ValueError(f"direction tasks not in manifest: {missing_dir}")

    import csv
    stored_rank90 = {}
    with open(args.cells_summary_csv, newline="") as f:
        for r in csv.DictReader(f):
            stored_rank90[(int(r["icl_example_index"]), r["token_role"], int(r["layer"]))] = int(r["rank90"])

    fvs = {t: load_function_vector(args.fv_root, t).to(device=device, dtype=dtype)
           for t in direction_tasks}
    hidden = next(iter(fvs.values())).shape[0]

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "run_config.json", {
        "sandbox": True,
        "method": "rank90-energy TSVD pseudo-inverse of the PER-PROMPT ridge maps applied to "
                  "the RAW canonical task FV (uncentered, user 2026-07-29); direction "
                  "un-standardization = * std only (no mu, no xbar)",
        "fv_root": str(args.fv_root), "cells": args.cells, "tasks": direction_tasks,
        "energy_threshold": args.energy_threshold,
        "edit_layer_mapping": "edit_layer = capture_layer - 1 (Stream W convention)",
    })

    # Group requested cells by icl_index so activations/targets load once per index.
    by_icl = {}
    for cell in args.cells:
        icl, role = ABLATION_CELLS[cell]
        by_icl.setdefault(icl, []).append((cell, role))

    t0 = time.time()
    for icl, cell_roles in sorted(by_icl.items()):
        roles = sorted({r for _, r in cell_roles})
        activations_root = (args.query_activations_root if icl == QUERY_ICL_INDEX
                            else Path(args.icl_activations_root_template.format(icl=icl)))
        acts, row_keys = {}, {}
        for role in roles:
            load_icl = role_load_icl_index(role, icl)
            for task in all_tasks:
                a, keys = load_task_role_pooled(activations_root, task, args.splits, role, load_icl)
                acts[(task, role)] = a
                row_keys[(task, role)] = keys
        y_pp = {}
        for task in all_tasks:
            target_map = load_perprompt_targets(args.targets_root, task, args.splits)
            y_pp[task] = align_targets(target_map, row_keys[(task, roles[0])], task).to(
                device=device, dtype=dtype)
        metrics = {(r["token_role"], int(r["layer"])): r
                   for r in load_json(args.metrics_root / f"perprompt_shard_icl{icl}" / "metrics.json")}
        print(f"[fvpp icl{icl}] loaded X/targets for roles={roles} ({time.time()-t0:.0f}s)", flush=True)

        for cell, role in cell_roles:
            out_dir = args.output_root / cell / "preimages"
            out_paths = {t: out_dir / f"{t}_tsvd_preimage_bank.pt" for t in direction_tasks}
            if not args.overwrite and all(p.exists() for p in out_paths.values()):
                print(f"[fvpp] {cell}: all banks exist; skipping.", flush=True)
                continue
            banks = {t: {} for t in direction_tasks}
            layer_list = args.layers or list(range(1, N_EDIT_LAYERS + 1))
            for layer in layer_list:                           # capture layers; edit = layer-1
                key = (role, layer)
                best_alpha = float(metrics[key]["best_alpha"])
                x_by_task = {task: acts[(task, role)][:, layer, :].to(device=device, dtype=dtype)
                             for task in all_tasks}
                x_train_pool = torch.cat([x_by_task[t] for t in train_tasks], dim=0)
                mean = x_train_pool.mean(dim=0)
                std = x_train_pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
                xs = {task: (x_by_task[task] - mean) / std for task in all_tasks}
                x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
                y_fit = torch.cat([y_pp[t] for t in train_tasks], dim=0)
                xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
                w = evecs @ (c / (evals + best_alpha).unsqueeze(1))

                # REPRO GATE (part-1 convention, manifest test tasks)
                sqerr_fv, sqerr_pp, test_n = 0.0, 0.0, 0
                for task in test_tasks:
                    pred = ridge_predict(xs[task], xbar, ybar, evals, evecs, c, best_alpha)
                    fv_t = load_function_vector(args.fv_root, task).to(device=device, dtype=dtype)
                    sqerr_fv += float(torch.sum((pred - fv_t.unsqueeze(0)) ** 2))
                    sqerr_pp += float(torch.sum((pred - y_pp[task]) ** 2))
                    test_n += xs[task].shape[0]
                mse_fv = sqerr_fv / (test_n * hidden)
                mse_pp = sqerr_pp / (test_n * hidden)
                rel_fv = abs(mse_fv - metrics[key]["test_mse_fv"]) / metrics[key]["test_mse_fv"]
                rel_pp = abs(mse_pp - metrics[key]["test_mse_pp"]) / metrics[key]["test_mse_pp"]
                if rel_fv > args.gate_rel_tol or rel_pp > args.gate_rel_tol:
                    raise RuntimeError(f"REPRO GATE FAILED {cell} L{layer}: rel_fv={rel_fv:.2e} "
                                       f"rel_pp={rel_pp:.2e} — STOP, user adjudicates.")

                svd_kwargs = {"driver": "gesvd"} if w.is_cuda else {}
                u, s, vh = torch.linalg.svd(w, full_matrices=False, **svd_kwargs)
                sv = s.detach().cpu().numpy().astype(np.float64)
                k = rank_energy(sv, args.energy_threshold)
                if k != stored_rank90[(icl, role, layer)]:
                    raise RuntimeError(f"RANK90 GATE FAILED {cell} L{layer}: {k} != stored "
                                       f"{stored_rank90[(icl, role, layer)]} — STOP, user adjudicates.")

                for task in direction_tasks:
                    dz = ((fvs[task] @ vh[:k].T) / s[:k]) @ u[:, :k].T   # RAW fv, uncentered
                    dx = (dz * std).detach().cpu().float()
                    banks[task][layer - 1] = {"tsvd": dx, "k": int(k),
                                              "s_top": s[:4].detach().cpu().tolist(),
                                              "s_k_over_s1": float(sv[k - 1] / sv[0])}
                del x_by_task, xs, x_fit, y_fit, w, u, s, vh
            if args.layers is not None:
                print(f"[fvpp] {cell}: SMOKE ({len(layer_list)} layers) — gates passed, "
                      "banks NOT written.", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for task in direction_tasks:
                assert set(banks[task]) == set(range(N_EDIT_LAYERS))
                torch.save({"sandbox": True, "preimages_by_edit_layer": banks[task],
                            "cell": cell, "task": task,
                            "config": {"target": "RAW canonical FV (uncentered)",
                                       "map": "per-prompt ridge, rank90-energy TSVD",
                                       "unstandardize": "* std only (no mu/xbar)"}},
                           out_paths[task])
            print(f"[fvpp] {cell}: wrote {len(direction_tasks)} banks ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"[fvpp] DONE in {time.time()-t0:.0f}s -> {args.output_root}")


if __name__ == "__main__":
    main()
