#!/usr/bin/env python
"""Stage-1 of the two-shot pair-diff pre-image study (Stream S).

Generalizes fit_prelabel_ridge_preimages.py (Stream R) from a single (token_role, icl_index)
cell to a LIST of cells, and adds PAIR-DIFFERENCE pre-images: for each task pair (A, B) the
target of the inversion is fv_A - fv_B (the damped inverse is linear in the target at fixed
gamma, so inverting the difference directly avoids mixing per-task gamma choices).

Per cell (role, icl) and capture layer L in 1..28:
    fit the full-dim 4096->4096 ridge (targets = the 20 train tasks' FVs from --fv_root,
    identical fit spec to the direction3 study), materialize W_std, and compute exact +
    Tikhonov-damped pre-images of (a) each pair task's FV and (b) each pair's FV difference.

CONSISTENCY RULE (the experiment's core requirement): the regression training targets and the
FVs being inverted are read from the SAME --fv_root, so the FV definition (head basis + shot
regime + train tasks) matches the way the regression is trained. The script asserts every
needed FV exists under that root before fitting anything.

Outputs (under --output_root/<role>_icl{k}/):
  maps/layer_{L:02d}.pt                     W_std fp16 + standardizer + alpha + fit metrics
  preimages/{task}_preimage_bank.pt         {edit_layer: exact/damped dx fp32 + diagnostics}
  pairdiff_preimages/{A}__{B}_pairdiff_preimage_bank.pt   same, target fv_A - fv_B
  diagnostics.json / run_config.json
Banks are keyed by EDIT layer = capture layer - 1 (same convention as Stream R; this also
equals the two-shot capture's layer index, which stores block outputs 0..27 with no embedding
slice).

Validation: --validate_against_study refits the cells named in --validate_cells with the
ORIGINAL study target (top-10 train_selected) and compares test_mse against the saved study
metrics shard for that icl index.
"""
import argparse
import json
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

from src.eval_scripts.fit_prelabel_ridge_preimages import (
    fit_cell,
    load_task_role_pooled,
    materialize_w,
    test_mse_for_fit,
)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    role_load_icl_index,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR

DEFAULT_CELLS = [
    "pre_label_token:1",   # two-shot demo1_prelabel
    "last_label_token:1",  # two-shot demo1_label (labels single-token: first==last)
    "pre_label_token:2",   # two-shot demo2_prelabel
    "last_label_token:2",  # two-shot demo2_label
    "pre_label_token:3",   # two-shot query_final (context-matched: token after 2 full demos)
    "last_prompt_token:10",  # two-shot query_final secondary view (role-matched, 10-demo ctx)
]
DEFAULT_PAIRS = ["antonym:synonym", "next_number_digits:prev_number_digits"]
GAMMA_COEFS = (1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=DEFAULT_CELLS,
                   help="Cells as token_role:icl_index (icl 10 = query capture dir).")
    p.add_argument("--pair_specs", nargs="+", default=DEFAULT_PAIRS,
                   help="Task pairs as taskA:taskB; pre-images computed for fv_A, fv_B and fv_A - fv_B.")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Capture layers to fit (default 1..28; layer 0 = embedding is skipped).")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/debug/train_varicl_max4_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: artifacts/preimage_pairdiff/<fv_root basename>.")
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--damped_norm_cap_mult", type=float, default=2.0)
    p.add_argument("--validate_against_study", action="store_true")
    p.add_argument("--validate_cells", nargs="+", default=["pre_label_token:2"],
                   help="Cells (role:icl) to validate against the study metrics shards.")
    p.add_argument("--study_metrics_template", type=str,
                   default=str(FV_FORMATION_DIR / "fulldim_ridge_activation_to_fv/shard_icl{icl}/metrics.json"))
    p.add_argument("--study_fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_cell(spec):
    role, icl = spec.rsplit(":", 1)
    return role, int(icl)


def preimage_from_svd(u, svals, vh, target64, gammas, norm_cap):
    """Exact + damped solutions of W_std^T dz = target (SVD of W_std^T given)."""
    utf = u.T @ target64
    t_norm = float(target64.norm())
    curve = []
    for g in gammas:
        coef = svals / (svals ** 2 + g)
        sol = vh.T @ (coef * utf)
        # rel_residual uses W^T sol = U diag(s) V^T? No: W_t64 @ sol; recompute via SVD parts.
        resid = u @ (svals * (vh @ sol)) - target64
        curve.append({"gamma": g, "sol": sol,
                      "rel_residual": float(torch.linalg.norm(resid) / t_norm),
                      "norm_std_space": float(sol.norm())})
    exact = curve[0]
    ok = [c for c in curve[1:] if c["norm_std_space"] <= norm_cap]
    damped = min(ok, key=lambda c: c["rel_residual"]) if ok else min(
        curve[1:], key=lambda c: c["norm_std_space"])
    return exact, damped, curve


def bank_entry(exact, damped, std):
    dx_exact = (exact["sol"].float() * std).detach().cpu()
    dx_damped = (damped["sol"].float() * std).detach().cpu()
    return {
        "exact": dx_exact,
        "damped": dx_damped,
        "damped_gamma": damped["gamma"],
        "damped_rel_residual": damped["rel_residual"],
        "exact_rel_residual": exact["rel_residual"],
    }, dx_exact, dx_damped


def diag_entry(exact, damped, dx_exact, dx_damped, target_norm, curve):
    return {
        "target_norm": target_norm,
        "exact": {"preimage_norm": float(dx_exact.norm()),
                  "norm_std_space": exact["norm_std_space"],
                  "rel_residual": exact["rel_residual"]},
        "damped": {"preimage_norm": float(dx_damped.norm()),
                   "norm_std_space": damped["norm_std_space"],
                   "rel_residual": damped["rel_residual"],
                   "gamma": damped["gamma"]},
        "gamma_curve": [{k: v for k, v in c.items() if k != "sol"} for c in curve],
    }


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = args.device
    alphas = list(args.alphas) if args.alphas is not None else list(np.logspace(-1, 8, 19))
    output_root = args.output_root or (ARTIFACTS_ROOT / "preimage_pairdiff" / args.fv_root.name)

    cells = [parse_cell(s) for s in args.cells]
    pairs = [tuple(s.split(":", 1)) for s in args.pair_specs]
    pair_tasks = sorted({t for pair in pairs for t in pair})

    manifest = load_json(args.task_manifest)
    train_tasks = list(manifest["train_tasks"])
    test_tasks = list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)

    # --- Consistency guard: EVERY FV (regression targets AND inversion targets) must come
    # from the single --fv_root, so the FV definition matches the regression's training.
    missing = [t for t in train_tasks + test_tasks + pair_tasks
               if not (args.fv_root / t / f"{t}_function_vector.pt").exists()]
    if missing:
        raise FileNotFoundError(
            f"FVs missing under {args.fv_root} for: {missing}. All regression-target and "
            f"pair-task FVs must come from the same root (consistency rule).")
    heads_info = {}
    for mname in ("fv_manifest.json", "fv_manifest_paired.json"):
        mpath = args.fv_root / mname
        if mpath.exists():
            m = load_json(mpath)
            heads_info[mname] = {k: m.get(k) for k in ("heads_path", "n_top_heads")}

    fvs = {t: load_function_vector(args.fv_root, t).to(device=device, dtype=torch.float32)
           for t in train_tasks + test_tasks}
    target_fvs = {t: load_function_vector(args.fv_root, t).to(device=device, dtype=torch.float32)
                  for t in pair_tasks}
    study_fvs = None
    if args.validate_against_study:
        study_fvs = {t: load_function_vector(args.study_fv_root, t).to(device=device, dtype=torch.float32)
                     for t in train_tasks + test_tasks}
    validate_cells = {parse_cell(s) for s in args.validate_cells} if args.validate_against_study else set()

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", {
        "cells": args.cells, "pair_specs": args.pair_specs,
        "fv_root": str(args.fv_root), "heads_info": heads_info,
        "train_tasks": train_tasks, "test_tasks": test_tasks, "pair_tasks": pair_tasks,
        "alphas": [float(a) for a in alphas], "splits": args.splits,
        "std_eps": args.std_eps, "damped_norm_cap_mult": args.damped_norm_cap_mult,
        "edit_layer_mapping": "edit_layer = capture_layer - 1 (block output hooks; equals the "
                              "two-shot capture layer index, which has no embedding slice)",
    })

    for role, icl in cells:
        cell_name = f"{role}_icl{icl}"
        cell_dir = output_root / cell_name
        maps_dir = cell_dir / "maps"
        pre_dir = cell_dir / "preimages"
        pairdiff_dir = cell_dir / "pairdiff_preimages"
        diag_path = cell_dir / "diagnostics.json"
        if diag_path.exists() and not args.overwrite:
            print(f"[{cell_name}] diagnostics exist; skipping (pass --overwrite to refit).")
            continue
        for d in (maps_dir, pre_dir, pairdiff_dir):
            d.mkdir(parents=True, exist_ok=True)

        if icl == QUERY_ICL_INDEX:
            activations_root = args.query_activations_root
        else:
            activations_root = Path(args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)

        print(f"[{cell_name}] loading activations from {activations_root}")
        t0 = time.time()
        acts = {t: load_task_role_pooled(activations_root, t, args.splits, role, load_icl)
                for t in train_tasks + test_tasks}
        n_layers = next(iter(acts.values())).shape[1]
        layers = list(args.layers) if args.layers is not None else list(range(1, n_layers))
        if 0 in layers:
            raise ValueError("Capture layer 0 (embedding) has no matching edit hook; drop it.")
        print(f"[{cell_name}] loaded in {time.time()-t0:.1f}s | fitting layers {layers[0]}..{layers[-1]}")

        study_rows = {}
        if (role, icl) in validate_cells:
            study_metrics = load_json(Path(args.study_metrics_template.format(icl=icl)))
            study_rows = {int(r["layer"]): r for r in study_metrics if r["token_role"] == role}

        task_banks = {t: {} for t in pair_tasks}
        pair_banks = {pair: {} for pair in pairs}
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
            w_std = materialize_w(fit)

            w_t64 = w_std.double().T
            u, svals, vh = torch.linalg.svd(w_t64)
            cond = float(svals[0] / svals[-1].clamp_min(1e-300))
            gammas = [0.0] + [float(c * svals[0] ** 2) for c in GAMMA_COEFS]
            norm_cap = args.damped_norm_cap_mult * float(np.sqrt(w_std.shape[0]))

            layer_diag = {
                "cell": cell_name, "capture_layer": layer, "edit_layer": layer - 1,
                "best_alpha": fit["best_alpha"], "cv_mse": fit["cv_mse"],
                "alpha_pinned": fit["alpha_pinned"], "test_mse": test_mse,
                "w_cond": cond, "w_smax": float(svals[0]), "w_smin": float(svals[-1]),
                "tasks": {}, "pairs": {},
            }
            for task, fv in target_fvs.items():
                exact, damped, curve = preimage_from_svd(u, svals, vh, fv.double(), gammas, norm_cap)
                entry, dx_e, dx_d = bank_entry(exact, damped, std)
                task_banks[task][layer - 1] = entry
                layer_diag["tasks"][task] = diag_entry(exact, damped, dx_e, dx_d, float(fv.norm()), curve)
            for pair in pairs:
                diff64 = (target_fvs[pair[0]] - target_fvs[pair[1]]).double()
                exact, damped, curve = preimage_from_svd(u, svals, vh, diff64, gammas, norm_cap)
                entry, dx_e, dx_d = bank_entry(exact, damped, std)
                pair_banks[pair][layer - 1] = entry
                layer_diag["pairs"]["__".join(pair)] = diag_entry(
                    exact, damped, dx_e, dx_d, float(diff64.norm()), curve)

            if study_rows:
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
                "token_role": role, "icl_index": icl, "fv_root": str(args.fv_root),
            }, maps_dir / f"layer_{layer:02d}.pt")

            diagnostics.append(layer_diag)
            val = ""
            sv = layer_diag.get("study_validation")
            if sv and sv["abs_diff"] is not None:
                val = f" | study diff={sv['abs_diff']:.2e}"
            pnorms = {"__".join(p): round(d["damped"]["preimage_norm"], 1)
                      for p, d in zip(pairs, (layer_diag["pairs"]["__".join(p)] for p in pairs))}
            print(f"[{cell_name}] L{layer:02d}: test_mse={test_mse:.4f} alpha={fit['best_alpha']:.3g} "
                  f"cond={cond:.2e} pairdiff damped norms={pnorms}{val} ({time.time()-t_cell:.1f}s)",
                  flush=True)

        for task, bank in task_banks.items():
            torch.save({
                "task": task, "preimages_by_edit_layer": bank,
                "fv_path": str(args.fv_root / task / f"{task}_function_vector.pt"),
                "token_role": role, "icl_index": icl,
                "definition": ("linear pre-image dz @ W_std = fv via SVD (fp64); dx = std * dz. "
                               "'exact' = undamped; 'damped' = Tikhonov gamma selected by "
                               "norm-cap + min rel_residual rule."),
            }, pre_dir / f"{task}_preimage_bank.pt")
        for pair, bank in pair_banks.items():
            torch.save({
                "pair": list(pair), "preimages_by_edit_layer": bank,
                "fv_root": str(args.fv_root),
                "token_role": role, "icl_index": icl,
                "definition": ("pre-image of the FV DIFFERENCE fv_A - fv_B (single inversion; "
                               "damped inverse is linear in the target at fixed gamma). "
                               "Same exact/damped machinery as the per-task banks."),
            }, pairdiff_dir / f"{pair[0]}__{pair[1]}_pairdiff_preimage_bank.pt")

        write_json(diag_path, diagnostics)
        write_json(cell_dir / "run_config.json", {
            "token_role": role, "icl_index": icl, "layers": layers,
            "activations_root": str(activations_root),
            "fv_root": str(args.fv_root), "pair_specs": args.pair_specs,
        })
        print(f"[{cell_name}] wrote {len(layers)} maps + {len(task_banks)} task banks + "
              f"{len(pair_banks)} pairdiff banks -> {cell_dir}")

    print(f"DONE -> {output_root}")


if __name__ == "__main__":
    main()
