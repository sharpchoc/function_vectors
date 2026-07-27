#!/usr/bin/env python
"""SANDBOX (not repo standard): truncated-SVD pre-images of per-prompt FVs.

For every cell (icl_index, token_role, layer) of the sandbox per-prompt ridge study, this
refits the SAME forward map (single 20-train-task standardizer, centered eigendecomposition
ridge, per-prompt head-sum targets) at the cell's stored CV-chosen alpha (read from the pilot
shard's metrics.json -- no CV rerun), then inverts every per-prompt FV back into activation
space with a truncated-SVD pseudo-inverse:

  forward:   y ~= (x_std - xbar) @ W + ybar,  W = U S Vh (torch row-vector convention)
  k        = rank90 by sigma^2 energy: smallest k with cum(S^2)/sum(S^2) >= 0.90
             (identical definition to diagnose_weight_spectrum.spectrum_stats)
  pre-image: x_c = (y - ybar) @ Vh[:k].T @ diag(1/S[:k]) @ U[:, :k].T
             (min-norm solution; forward(x_c) == (y - ybar) projected on span(Vh[:k]))
  stored:    x_raw = (x_c + xbar) * std + mean   -- raw activation space, fp16

Scope (user-approved 2026-07-27, ~34 GB fp16): 27 tasks (20 train + 7 test, cc/pc excluded)
x 170 prompts x 899 cells. One output file per cell.

Gates (hard stop, user adjudicates):
  * REPRO GATE: refit W must reproduce the cell's stored test_mse_fv AND test_mse_pp
    (rel tol --gate_rel_tol; the weight-spectrum diagnostic achieved rel <= 7e-6).
  * SPECTRUM GATE (gate cell icl10/pre_label_token L13 only): rank90 must equal the stored
    rank_energy_90 (441) and the singular values must match weight_spectrum/spectra.npz
    (sv_perprompt_alpha_own) to --spectrum_gate_rel_tol relative to sigma_1.
  * SELF-CONSISTENCY: forward(pre-image) must equal the rank-k projection of (y - ybar)
    to rel <= --selfcons_rel_tol. (Truncation residual itself is a diagnostic, not a gate.)
"""
import argparse
import csv
import json
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

from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402
from sandbox.perprompt_fv.regress_activation_to_perprompt_headsum_ridge import (  # noqa: E402
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    FINAL_PROMPT_ROLE,
    LABEL_ROLES,
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

GATE_CELL = (10, "pre_label_token", 13)


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: truncated-SVD pre-images of per-prompt FVs.")
    p.add_argument("--icl_index", type=int, required=True,
                   help="ICL example index (1..10). 1-9 use icl{n}_3tokens dirs; 10 uses the 4tokens (query) dir.")
    p.add_argument("--token_roles", nargs="+", default=None)
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--energy_threshold", type=float, default=0.90,
                   help="rank90 threshold on cumulative sigma^2 energy.")
    p.add_argument("--task_manifest", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--targets_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--metrics_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40",
                   help="Pilot root holding perprompt_shard_icl{n}/metrics.json (stored best_alpha + gate MSEs).")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_fv_preimages/gptj_train_varicl_top40")
    p.add_argument("--summary_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/preimages_truncsvd")
    p.add_argument("--train_tasks", nargs="+", default=None)
    p.add_argument("--test_tasks", nargs="+", default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--gate_rel_tol", type=float, default=1e-4)
    p.add_argument("--selfcons_rel_tol", type=float, default=1e-3)
    p.add_argument("--spectrum_gate_npz", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/weight_spectrum/spectra.npz")
    p.add_argument("--spectrum_gate_key", type=str, default="sv_perprompt_alpha_own")
    p.add_argument("--spectrum_gate_rank90", type=int, default=441)
    p.add_argument("--spectrum_gate_rel_tol", type=float, default=1e-3)
    p.add_argument("--no_spectrum_gate", action="store_true",
                   help="Skip the spectrum gate even on the gate cell (smoke runs on other layers only).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def cell_path(output_root, icl_index, role, layer):
    return output_root / f"icl{icl_index}" / role / f"L{layer:02d}.pt"


def rank_energy(sv, threshold):
    """Smallest k with cum(sv^2)/sum(sv^2) >= threshold -- diagnose_weight_spectrum convention."""
    energy = np.asarray(sv, dtype=np.float64) ** 2
    cum = np.cumsum(energy) / energy.sum()
    return int(np.searchsorted(cum, threshold) + 1)


SUMMARY_FIELDS = [
    "icl_example_index", "token_role", "layer", "best_alpha", "rank90", "energy_at_k",
    "sv1", "sv_k", "gate_rel_fv", "gate_rel_pp", "selfcons_max_rel",
    "trunc_resid_mean", "trunc_resid_max", "preimage_norm_mean", "n_rows",
]


def summary_row_from_cell(cell):
    return {k: cell[k] for k in SUMMARY_FIELDS}


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    dtype = torch.float32

    manifest = load_json(args.task_manifest)
    train_tasks = list(args.train_tasks) if args.train_tasks is not None else list(manifest["train_tasks"])
    test_tasks = list(args.test_tasks) if args.test_tasks is not None else list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    overlap = sorted(set(train_tasks).intersection(test_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and test: {overlap}")
    all_tasks = train_tasks + test_tasks

    if not 1 <= args.icl_index <= QUERY_ICL_INDEX:
        raise ValueError(f"--icl_index must be in 1..{QUERY_ICL_INDEX}, got {args.icl_index}")
    if args.icl_index == QUERY_ICL_INDEX:
        activations_root = args.query_activations_root
        default_roles = LABEL_ROLES + [FINAL_PROMPT_ROLE]
    else:
        activations_root = Path(args.icl_activations_root_template.format(icl=args.icl_index))
        default_roles = list(LABEL_ROLES)
    token_roles = list(args.token_roles) if args.token_roles is not None else default_roles

    # Stored per-cell alphas + gate MSEs from the pilot run.
    metrics_path = args.metrics_root / f"perprompt_shard_icl{args.icl_index}" / "metrics.json"
    stored = {(r["token_role"], int(r["layer"])): r for r in load_json(metrics_path)}
    print(f"[preimage icl{args.icl_index}] stored metrics: {len(stored)} cells from {metrics_path}")

    fvs = {task: load_function_vector(args.fv_root, task).to(device=device, dtype=dtype) for task in all_tasks}
    hidden = fvs[test_tasks[0]].shape[0]

    # Load X once per (task, role), with row keys for target alignment (regression protocol).
    t0 = time.time()
    acts, row_keys = {}, {}
    n_layers = None
    for role in token_roles:
        load_icl = role_load_icl_index(role, args.icl_index)
        for task in all_tasks:
            a, keys = load_task_role_pooled(activations_root, task, args.splits, role, load_icl)
            acts[(task, role)] = a
            row_keys[(task, role)] = keys
            if n_layers is None:
                n_layers = a.shape[1]
            elif a.shape[1] != n_layers:
                raise ValueError(f"Layer-count mismatch for {task}/{role}: {a.shape[1]} vs {n_layers}")
    layers = list(args.layers) if args.layers is not None else list(range(n_layers))
    print(f"[preimage icl{args.icl_index}] loaded X in {time.time()-t0:.1f}s | "
          f"n_layers={n_layers} | roles={token_roles} | layers={len(layers)}")

    # Per-prompt targets, aligned to row order (role-independent; asserted like the regression).
    y_pp = {}
    for task in all_tasks:
        key_sets = {tuple(row_keys[(task, role)]) for role in token_roles}
        if len(key_sets) != 1:
            raise RuntimeError(f"Row-key mismatch across roles for {task}; cannot share targets across roles.")
        target_map = load_perprompt_targets(args.targets_root, task, args.splits)
        y_pp[task] = align_targets(target_map, row_keys[(task, token_roles[0])], task).to(device=device, dtype=dtype)
    y_all = torch.cat([y_pp[t] for t in all_tasks], dim=0)
    metadata = [
        {"task": task, "split": split, "prompt_index": pi, "query_source_index": qi}
        for task in all_tasks
        for (split, pi, qi) in row_keys[(task, token_roles[0])]
    ]
    n_rows = y_all.shape[0]
    print(f"[preimage icl{args.icl_index}] aligned targets: {n_rows} rows over {len(all_tasks)} tasks")

    spectrum_ref = None
    if not args.no_spectrum_gate and args.icl_index == GATE_CELL[0]:
        spectrum_ref = np.load(args.spectrum_gate_npz)[args.spectrum_gate_key]

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for role in token_roles:
        for layer in layers:
            t_cell = time.time()
            out_path = cell_path(args.output_root, args.icl_index, role, layer)
            if out_path.exists() and not args.overwrite:
                cell = torch.load(out_path, map_location="cpu", weights_only=False)["cell"]
                summary_rows.append(summary_row_from_cell(cell))
                print(f"[preimage icl{args.icl_index}] {role} L{layer:02d}: SKIP (exists, summary reloaded)")
                continue
            key = (role, layer)
            if key not in stored:
                raise KeyError(f"No stored pilot metrics for cell icl{args.icl_index}/{role}/L{layer}; "
                               f"cannot look up best_alpha. STOP -- user adjudicates.")
            best_alpha = float(stored[key]["best_alpha"])

            # ---- Refit the forward map exactly as the pilot did ----
            x_by_task = {task: acts[(task, role)][:, layer, :].to(device=device, dtype=dtype)
                         for task in all_tasks}
            x_train_pool = torch.cat([x_by_task[t] for t in train_tasks], dim=0)
            mean = x_train_pool.mean(dim=0)
            std = x_train_pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
            xs = {task: (x_by_task[task] - mean) / std for task in all_tasks}

            x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
            y_fit = torch.cat([y_pp[t] for t in train_tasks], dim=0)
            xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
            w = evecs @ (c / (evals + best_alpha).unsqueeze(1))  # y ~= (x - xbar) @ W + ybar

            # ---- REPRO GATE: reproduce the stored test MSEs (predictions via the same path) ----
            sqerr_fv, sqerr_pp, test_n = 0.0, 0.0, 0
            for task in test_tasks:
                pred = ridge_predict(xs[task], xbar, ybar, evals, evecs, c, best_alpha)
                sqerr_fv += float(torch.sum((pred - fvs[task].unsqueeze(0)) ** 2))
                sqerr_pp += float(torch.sum((pred - y_pp[task]) ** 2))
                test_n += xs[task].shape[0]
            mse_fv = sqerr_fv / (test_n * hidden)
            mse_pp = sqerr_pp / (test_n * hidden)
            rel_fv = abs(mse_fv - stored[key]["test_mse_fv"]) / stored[key]["test_mse_fv"]
            rel_pp = abs(mse_pp - stored[key]["test_mse_pp"]) / stored[key]["test_mse_pp"]
            if rel_fv > args.gate_rel_tol or rel_pp > args.gate_rel_tol:
                raise RuntimeError(
                    f"REPRO GATE FAILED for icl{args.icl_index}/{role}/L{layer}: "
                    f"test_mse_fv={mse_fv:.8f} (stored {stored[key]['test_mse_fv']:.8f}, rel {rel_fv:.2e}), "
                    f"test_mse_pp={mse_pp:.8f} (stored {stored[key]['test_mse_pp']:.8f}, rel {rel_pp:.2e}), "
                    f"tol {args.gate_rel_tol:.1e}. STOP -- user adjudicates.")

            # ---- SVD + rank90 truncation ----
            # CUDA's default driver (gesvdj) has fp32 backend error ~1e-3 here, which fails the
            # self-consistency gate; gesvd matches CPU LAPACK accuracy (~1e-5) and is faster.
            svd_kwargs = {"driver": "gesvd"} if w.is_cuda else {}
            u, s, vh = torch.linalg.svd(w, full_matrices=False, **svd_kwargs)
            sv = s.detach().cpu().numpy().astype(np.float64)
            k = rank_energy(sv, args.energy_threshold)
            energy_at_k = float((sv[:k] ** 2).sum() / (sv ** 2).sum())

            if spectrum_ref is not None and (args.icl_index, role, layer) == GATE_CELL:
                ref = np.asarray(spectrum_ref, dtype=np.float64)
                max_rel = float(np.max(np.abs(sv[:args.spectrum_gate_rank90] - ref[:args.spectrum_gate_rank90]))
                                / ref[0])
                if k != args.spectrum_gate_rank90 or max_rel > args.spectrum_gate_rel_tol:
                    raise RuntimeError(
                        f"SPECTRUM GATE FAILED for icl{args.icl_index}/{role}/L{layer}: rank90={k} "
                        f"(expected {args.spectrum_gate_rank90}), max|dsv|/sv1={max_rel:.2e} "
                        f"(tol {args.spectrum_gate_rel_tol:.1e}) vs {args.spectrum_gate_npz}. "
                        f"STOP -- user adjudicates.")
                print(f"[preimage icl{args.icl_index}] SPECTRUM GATE PASSED: rank90={k}, "
                      f"max|dsv|/sv1={max_rel:.2e}")

            # ---- Truncated pseudo-inverse pre-images (min-norm, centered std space) ----
            y_c = y_all - ybar
            proj = y_c @ vh[:k].T                       # [n_rows, k] coords in output basis
            x_c = (proj / s[:k]) @ u[:, :k].T           # [n_rows, hidden] centered std-space pre-images

            # Self-consistency: forward(x_c) must equal the rank-k projection of y_c.
            fwd = x_c @ w
            y_proj = proj @ vh[:k]
            selfcons = (torch.linalg.norm(fwd - y_proj, dim=1)
                        / torch.linalg.norm(y_proj, dim=1).clamp_min(1e-12))
            selfcons_max = float(selfcons.max())
            if selfcons_max > args.selfcons_rel_tol:
                raise RuntimeError(
                    f"SELF-CONSISTENCY FAILED for icl{args.icl_index}/{role}/L{layer}: "
                    f"max rel {selfcons_max:.2e} > {args.selfcons_rel_tol:.1e}. STOP -- user adjudicates.")

            # Truncation residual (diagnostic): part of y_c outside the kept output subspace.
            trunc_resid = (torch.linalg.norm(y_c - y_proj, dim=1)
                           / torch.linalg.norm(y_c, dim=1).clamp_min(1e-12))

            x_raw = (x_c + xbar) * std + mean           # raw activation space
            preimage_norms = torch.linalg.norm(x_raw, dim=1)

            cell = {
                "icl_example_index": args.icl_index,
                "token_role": role,
                "layer": layer,
                "best_alpha": best_alpha,
                "rank90": k,
                "energy_threshold": args.energy_threshold,
                "energy_at_k": energy_at_k,
                "sv1": float(sv[0]),
                "sv_k": float(sv[k - 1]),
                "gate_rel_fv": rel_fv,
                "gate_rel_pp": rel_pp,
                "selfcons_max_rel": selfcons_max,
                "trunc_resid_mean": float(trunc_resid.mean()),
                "trunc_resid_max": float(trunc_resid.max()),
                "preimage_norm_mean": float(preimage_norms.mean()),
                "n_rows": int(n_rows),
                "sv": torch.from_numpy(sv.astype(np.float32)),
                "xbar": xbar.detach().cpu().float(),
                "ybar": ybar.detach().cpu().float(),
                "standardizer_mean": mean.detach().cpu().float(),
                "standardizer_std": std.detach().cpu().float(),
                "per_task_trunc_resid_mean": {
                    task: float(trunc_resid[torch.tensor([i for i, m in enumerate(metadata)
                                                          if m["task"] == task])].mean())
                    for task in all_tasks
                },
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "sandbox": True,
                "preimages": x_raw.detach().cpu().to(torch.float16),
                "trunc_resid_rel": trunc_resid.detach().cpu().float(),
                "metadata": metadata,
                "cell": cell,
                "config": {
                    "space": "raw activation space (un-standardized)",
                    "method": "truncated-SVD pseudo-inverse of the per-prompt full-dim ridge; "
                              "k = rank90 by sigma^2 energy; min-norm solution",
                    "activations_root": str(activations_root),
                    "targets_root": str(args.targets_root),
                    "metrics_source": str(metrics_path),
                    "train_tasks": train_tasks,
                    "test_tasks": test_tasks,
                    "splits": args.splits,
                    "std_eps": args.std_eps,
                },
            }, out_path)
            summary_rows.append(summary_row_from_cell(cell))
            del x_by_task, xs, x_fit, y_fit, w, u, s, vh, x_c, x_raw, fwd, y_proj, proj
            print(f"[preimage icl{args.icl_index}] {role} L{layer:02d}: rank90={k} "
                  f"(E@k={energy_at_k:.3f}, sv_k/sv1={cell['sv_k']/cell['sv1']:.3f}) "
                  f"gate rel_fv={rel_fv:.1e} rel_pp={rel_pp:.1e} selfcons={selfcons_max:.1e} "
                  f"trunc_resid={cell['trunc_resid_mean']:.3f} alpha={best_alpha:.3g} "
                  f"({time.time()-t_cell:.1f}s)")

    csv_path = args.summary_dir / f"cells_icl{args.icl_index}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    write_json(args.output_root / f"run_config_icl{args.icl_index}.json", {
        "sandbox": True,
        "icl_index": args.icl_index,
        "token_roles": token_roles,
        "layers": layers,
        "energy_threshold": args.energy_threshold,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "splits": args.splits,
        "n_rows": int(n_rows),
        "n_cells": len(summary_rows),
        "storage_dtype": "torch.float16",
        "approved_storage": "user approved ~34 GB fp16 on 2026-07-27",
        "metrics_source": str(metrics_path),
        "gate_rel_tol": args.gate_rel_tol,
        "selfcons_rel_tol": args.selfcons_rel_tol,
        "method": "SANDBOX truncated-SVD pseudo-inverse (rank90 by sigma^2 energy) of the "
                  "per-prompt full-dim ridge at stored best_alpha; raw-space fp16 pre-images",
    })
    print(f"[preimage icl{args.icl_index}] wrote {len(summary_rows)} cells -> {csv_path}")


if __name__ == "__main__":
    main()
