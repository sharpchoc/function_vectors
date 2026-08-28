#!/usr/bin/env python
"""Stage-1 of the pre-image steering experiment (Stream R).

Refits the direct full-dim (4096 -> 4096) activation->FV ridge at ONE token role / ICL index
(default: pre_label_token, icl10) for every layer, with the FV target switched to the
train_selected_top40 set, and -- unlike the original study script -- MATERIALIZES the weight
matrix W per layer. It then computes, for each requested task, the linear pre-image of that
task's FV under each layer's map:

    prediction(z) = (z - zbar) @ W_std + ybar        (z = standardized activation)
    pre-image:  solve  dz @ W_std = fv  (lstsq)  ->  dx = sigma * dz   (raw-activation space)

dx is the activation-space displacement that the fitted regression maps onto the FV -- the
candidate steering vector. Banks are keyed by EDIT layer (capture layer L -> edit_layer L-1,
i.e. the output of transformer block L-1 == the residual stream the capture read at L).
Capture layer 0 (embedding) is skipped.

Fit spec is identical to regress_activation_to_fv_fulldim_ridge.py (single 20-train-task
standardizer, centered eig solve, alpha by leave-one-train-task-out CV over logspace(-1,8,19));
helpers are imported from that script.

Outputs (under --output_root):
  maps/layer_{L:02d}.pt        W_std fp16 + standardizer + centering + alpha + fit metrics
  preimages/{task}_preimage_bank.pt   {edit_layer: dx fp32} + per-layer diagnostics
  diagnostics.json             per-layer cond(W), residuals, norms, test MSE
  run_config.json

Validation: --validate_against_study refits layer cells with the ORIGINAL top-10 target and
compares test_mse against the saved study metrics (results/.../fulldim_ridge_activation_to_fv/
shard_icl10/metrics.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    ridge_eig_prep,
    ridge_predict,
    role_load_icl_index,
    torch_load_trusted,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def load_task_role_pooled(activations_root, task, splits, token_role, expected_icl_index):
    """Like the study loader, but re-roots STALE ABSOLUTE shard paths onto the split dir.

    The capture-era index.json files store absolute paths under the old results/ layout
    (pre 2026-06-19 migration to artifacts/); the shards live next to index.json now.
    """
    parts = []
    for split in splits:
        split_dir = activations_root / task / split
        index = load_json(split_dir / "index.json")
        chunks = []
        for shard in index["shards"]:
            shard_path = Path(shard)
            if not shard_path.is_absolute() or not shard_path.exists():
                shard_path = split_dir / shard_path.name
            data = torch_load_trusted(shard_path, map_location="cpu")
            activations, shard_metadata = data["activations"], data["metadata"]
            if len(shard_metadata) != activations.shape[0]:
                raise ValueError(f"Metadata/activation mismatch in {shard_path}")
            selected = [i for i, meta in enumerate(shard_metadata)
                        if meta.get("token_role") == token_role
                        and meta.get("icl_example_index") == expected_icl_index]
            if selected:
                chunks.append(activations[selected])
        if not chunks:
            raise ValueError(f"No {token_role} activations for {task}/{split}/ICL {expected_icl_index}")
        parts.append(torch.cat(chunks, dim=0))
    return torch.cat(parts, dim=0).to(torch.float16)

DEFAULT_TARGET_TASKS = ["next_number", "prev_number", "synonym", "antonym"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--icl_index", type=int, default=QUERY_ICL_INDEX)
    p.add_argument("--token_role", type=str, default="pre_label_token")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Capture layers to fit (default 1..28; layer 0 = embedding is skipped).")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected_top40")
    p.add_argument("--target_tasks", nargs="+", default=DEFAULT_TARGET_TASKS,
                   help="Tasks whose FV pre-images to compute (FVs read from --fv_root).")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_steering/train_selected_top40_icl10_pre")
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--damped_norm_cap_mult", type=float, default=2.0,
                   help="Damped pre-image: among gamma>0 candidates whose standardized-space norm "
                        "is <= this multiple of sqrt(D) (the typical z-scored activation norm), "
                        "pick the one with the smallest relative residual ||W dz - fv||/||fv||.")
    p.add_argument("--validate_against_study", action="store_true",
                   help="Also refit with the study's top-10 FV target and compare test_mse to the "
                        "saved shard_icl10 metrics (tolerance check).")
    p.add_argument("--study_metrics_json", type=Path,
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/main/shard_icl10/metrics.json")
    p.add_argument("--study_fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fit_cell(xs, fvs, train_tasks, alphas, device, dtype=torch.float32):
    """LOO-task-CV alpha selection + final refit; identical math to the study script."""
    cv_sqerr = torch.zeros(len(alphas), device=device, dtype=dtype)
    cv_n = 0
    for held in train_tasks:
        fit_tasks = [t for t in train_tasks if t != held]
        x_fit = torch.cat([xs[t] for t in fit_tasks], dim=0)
        y_fit = torch.cat([fvs[t].unsqueeze(0).expand(xs[t].shape[0], -1) for t in fit_tasks], dim=0)
        xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
        x_val = xs[held]
        y_val = fvs[held].unsqueeze(0).expand(x_val.shape[0], -1)
        a_val = (x_val - xbar) @ evecs
        for ai, alpha in enumerate(alphas):
            pred = (a_val / (evals + alpha)) @ c + ybar
            cv_sqerr[ai] += torch.sum((pred - y_val) ** 2)
        cv_n += x_val.shape[0] * fvs[held].shape[0]
    cv_mse = (cv_sqerr / cv_n).detach().cpu().numpy()
    best_idx = int(np.argmin(cv_mse))
    best_alpha = float(alphas[best_idx])

    x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
    y_fit = torch.cat([fvs[t].unsqueeze(0).expand(xs[t].shape[0], -1) for t in train_tasks], dim=0)
    xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
    return {
        "best_alpha": best_alpha,
        "cv_mse": float(cv_mse[best_idx]),
        "alpha_pinned": bool(best_idx in (0, len(alphas) - 1)),
        "xbar": xbar, "ybar": ybar, "evals": evals, "evecs": evecs, "c": c,
        "n_fit_rows": int(x_fit.shape[0]),
    }


def test_mse_for_fit(fit, xs, fvs, test_tasks):
    sqerr, n = 0.0, 0
    for task in test_tasks:
        x_eval = xs[task]
        y_eval = fvs[task].unsqueeze(0).expand(x_eval.shape[0], -1)
        pred = ridge_predict(x_eval, fit["xbar"], fit["ybar"], fit["evals"], fit["evecs"], fit["c"], fit["best_alpha"])
        sqerr += float(torch.sum((pred - y_eval) ** 2))
        n += x_eval.shape[0]
    return sqerr / (n * fvs[test_tasks[0]].shape[0])


def materialize_w(fit):
    """W_std [D, D] such that prediction = (z - zbar) @ W_std + ybar."""
    return fit["evecs"] @ (fit["c"] / (fit["evals"] + fit["best_alpha"]).unsqueeze(1))


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = args.device
    alphas = list(args.alphas) if args.alphas is not None else list(np.logspace(-1, 8, 19))

    maps_dir = args.output_root / "maps"
    pre_dir = args.output_root / "preimages"
    diag_path = args.output_root / "diagnostics.json"
    if diag_path.exists() and not args.overwrite:
        raise FileExistsError(f"{diag_path} exists; pass --overwrite.")
    maps_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.task_manifest)
    train_tasks = list(manifest["train_tasks"])
    test_tasks = list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)

    if args.icl_index == QUERY_ICL_INDEX:
        activations_root = args.query_activations_root
    else:
        activations_root = Path(args.icl_activations_root_template.format(icl=args.icl_index))
    load_icl = role_load_icl_index(args.token_role, args.icl_index)

    # Targets for the pre-images (top-40 basis) ...
    target_fvs = {t: load_function_vector(args.fv_root, t).to(device=device, dtype=torch.float32)
                  for t in args.target_tasks}
    # ... and regression-target FVs for the FIT (same root: the 20 train + 7 test tasks).
    fvs = {t: load_function_vector(args.fv_root, t).to(device=device, dtype=torch.float32)
           for t in train_tasks + test_tasks}
    study_fvs = None
    if args.validate_against_study:
        study_fvs = {t: load_function_vector(args.study_fv_root, t).to(device=device, dtype=torch.float32)
                     for t in train_tasks + test_tasks}

    print(f"Loading {args.token_role} activations (icl_index={args.icl_index}) for "
          f"{len(train_tasks)} train + {len(test_tasks)} test tasks from {activations_root}")
    t0 = time.time()
    acts = {t: load_task_role_pooled(activations_root, t, args.splits, args.token_role, load_icl)
            for t in train_tasks + test_tasks}
    n_layers = next(iter(acts.values())).shape[1]
    layers = list(args.layers) if args.layers is not None else list(range(1, n_layers))
    if 0 in layers:
        raise ValueError("Capture layer 0 (embedding) has no matching edit hook; drop it.")
    print(f"Loaded in {time.time()-t0:.1f}s | n_layers={n_layers} | fitting layers {layers[0]}..{layers[-1]}")

    study_rows = {}
    if args.validate_against_study:
        for row in load_json(args.study_metrics_json):
            if row["token_role"] == args.token_role:
                study_rows[int(row["layer"])] = row

    banks = {t: {} for t in args.target_tasks}
    diagnostics = []
    for layer in layers:
        t_cell = time.time()
        x_by_task = {t: acts[t][:, layer, :].to(device=device, dtype=torch.float32)
                     for t in train_tasks + test_tasks}
        x_train_pool = torch.cat([x_by_task[t] for t in train_tasks], dim=0)
        mean = x_train_pool.mean(dim=0)
        std = x_train_pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
        xs = {t: (x_by_task[t] - mean) / std for t in train_tasks + test_tasks}

        fit = fit_cell(xs, fvs, train_tasks, alphas, device)
        test_mse = test_mse_for_fit(fit, xs, fvs, test_tasks)
        w_std = materialize_w(fit)  # [D, D]

        # Pre-images: dz @ W_std = fv  <=>  W_std^T dz = fv. The ridge W is badly
        # ill-conditioned (cond ~1e9 in smoke runs), so the EXACT inverse has astronomical
        # norm (noise in near-null directions). Alongside it we compute Tikhonov-DAMPED
        # pre-images dz(g) = V diag(s/(s^2+g)) U^T fv (SVD of W^T), choosing per (task,
        # layer) the smallest-norm g with rel_residual <= damped_max_residual.
        w_t64 = w_std.double().T
        u, svals, vh = torch.linalg.svd(w_t64)
        cond = float(svals[0] / svals[-1].clamp_min(1e-300))
        gammas = [0.0] + [float(c * svals[0] ** 2) for c in
                          (1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
        norm_cap = args.damped_norm_cap_mult * float(np.sqrt(w_std.shape[0]))

        layer_diag = {
            "capture_layer": layer, "edit_layer": layer - 1,
            "best_alpha": fit["best_alpha"], "cv_mse": fit["cv_mse"],
            "alpha_pinned": fit["alpha_pinned"], "test_mse": test_mse,
            "w_cond": cond, "w_smax": float(svals[0]), "w_smin": float(svals[-1]),
            "tasks": {},
        }
        for task, fv in target_fvs.items():
            fv64 = fv.double()
            utf = u.T @ fv64
            fv_norm = float(fv64.norm())
            curve = []
            for g in gammas:
                coef = svals / (svals ** 2 + g)
                sol = vh.T @ (coef * utf)
                rel = float(torch.linalg.norm(w_t64 @ sol - fv64) / fv_norm)
                curve.append({"gamma": g, "sol": sol, "rel_residual": rel,
                              "norm_std_space": float(sol.norm())})
            exact = curve[0]
            ok = [c for c in curve[1:] if c["norm_std_space"] <= norm_cap]
            damped = min(ok, key=lambda c: c["rel_residual"]) if ok else min(
                curve[1:], key=lambda c: c["norm_std_space"])
            dx_exact = (exact["sol"].float() * std).detach().cpu()
            dx_damped = (damped["sol"].float() * std).detach().cpu()
            banks[task][layer - 1] = {
                "exact": dx_exact,
                "damped": dx_damped,
                "damped_gamma": damped["gamma"],
                "damped_rel_residual": damped["rel_residual"],
                "exact_rel_residual": exact["rel_residual"],
            }
            layer_diag["tasks"][task] = {
                "fv_norm": fv_norm,
                "exact": {"preimage_norm": float(dx_exact.norm()),
                          "norm_std_space": exact["norm_std_space"],
                          "rel_residual": exact["rel_residual"]},
                "damped": {"preimage_norm": float(dx_damped.norm()),
                           "norm_std_space": damped["norm_std_space"],
                           "rel_residual": damped["rel_residual"],
                           "gamma": damped["gamma"]},
                "gamma_curve": [{k: v for k, v in c.items() if k != "sol"} for c in curve],
            }

        if args.validate_against_study:
            sfit = fit_cell(xs, study_fvs, train_tasks, alphas, device)
            s_test_mse = test_mse_for_fit(sfit, xs, study_fvs, test_tasks)
            ref = study_rows.get(layer)
            layer_diag["study_validation"] = {
                "refit_test_mse": s_test_mse,
                "study_test_mse": None if ref is None else ref["test_mse"],
                "refit_best_alpha": sfit["best_alpha"],
                "study_best_alpha": None if ref is None else ref["best_alpha"],
                "abs_diff": None if ref is None else abs(s_test_mse - ref["test_mse"]),
            }

        torch.save({
            "w_std": w_std.detach().half().cpu(),
            "mean": mean.detach().cpu(), "std": std.detach().cpu(),
            "zbar": fit["xbar"].detach().cpu(), "ybar": fit["ybar"].detach().cpu(),
            "best_alpha": fit["best_alpha"], "cv_mse": fit["cv_mse"], "test_mse": test_mse,
            "capture_layer": layer, "edit_layer": layer - 1,
            "token_role": args.token_role, "icl_index": args.icl_index,
            "fv_root": str(args.fv_root),
        }, maps_dir / f"layer_{layer:02d}.pt")

        diagnostics.append(layer_diag)
        val = ""
        if args.validate_against_study and layer_diag["study_validation"]["abs_diff"] is not None:
            val = f" | study diff={layer_diag['study_validation']['abs_diff']:.2e}"
        norms = {t: (round(d["exact"]["preimage_norm"], 0), round(d["damped"]["preimage_norm"], 1))
                 for t, d in layer_diag["tasks"].items()}
        print(f"L{layer:02d}: test_mse={test_mse:.4f} alpha={fit['best_alpha']:.3g} cond={cond:.2e} "
              f"(exact,damped) norms={norms}{val} ({time.time()-t_cell:.1f}s)", flush=True)

    for task, bank in banks.items():
        torch.save({
            "task": task,
            "preimages_by_edit_layer": bank,
            "fv_path": str(args.fv_root / task / f"{task}_function_vector.pt"),
            "token_role": args.token_role, "icl_index": args.icl_index,
            "definition": ("linear pre-image dz @ W_std = fv via SVD (fp64); dx = std * dz. "
                           "'exact' = undamped; 'damped' = min-norm Tikhonov gamma with "
                           "rel_residual <= bound."),
        }, pre_dir / f"{task}_preimage_bank.pt")

    write_json(diag_path, diagnostics)
    write_json(args.output_root / "run_config.json", {
        "icl_index": args.icl_index, "token_role": args.token_role, "layers": layers,
        "fv_root": str(args.fv_root), "target_tasks": args.target_tasks,
        "train_tasks": train_tasks, "test_tasks": test_tasks,
        "alphas": [float(a) for a in alphas], "splits": args.splits,
        "activations_root": str(activations_root), "std_eps": args.std_eps,
        "edit_layer_mapping": "edit_layer = capture_layer - 1 (block output hooks)",
    })
    print(f"Wrote {len(layers)} maps + {len(banks)} pre-image banks -> {args.output_root}")


if __name__ == "__main__":
    main()
