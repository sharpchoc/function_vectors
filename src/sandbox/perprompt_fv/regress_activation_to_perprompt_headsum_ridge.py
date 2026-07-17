#!/usr/bin/env python
"""SANDBOX (not repo standard): full-dim ridge with PER-PROMPT head-sum targets.

Identical protocol to `regress_activation_to_fv_fulldim_ridge.py` (same X, same single
20-train-task standardizer, same leave-one-train-task-out CV over the same alpha grid,
same eigendecomposition ridge) with ONE change: instead of broadcasting a task's FV to all
of its 170 rows, each row's target is that prompt's top-40 head-sum vector from the
sandbox capture (`capture_perprompt_head_activations.py`), row-aligned by
(split, prompt_index) and asserted on query_source_index.

Per cell it reports, on the 7 held-out test tasks:
  * test_mse_fv / test_r2_fv     -- predictions vs the stored varicl_top40 test FVs
                                    (broadcast), directly comparable to the canonical study
  * test_mse_pp / test_r2_pp     -- predictions vs the per-prompt head-sum targets
plus train MSE/R^2 vs the fit targets. R^2 uses the repo convention
(`compute_fulldim_ridge_r2.py`): 1 - MSE / V(eval targets | train-mean baseline), with the
eval-mean-baseline variant also emitted.

`--target_mode fv_broadcast` reproduces the canonical setup exactly (repro gate).
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


DEFAULT_TEST_TASKS_EXCLUDE_CC_PC = [
    "landmark-country",
    "word_length",
    "capitalize_first_letter",
    "synonym",
    "lowercase_first_letter",
    "capitalize",
    "antonym",
]
LABEL_ROLES = ["pre_label_token", "first_label_token", "last_label_token"]
QUERY_ICL_INDEX = 10
FINAL_PROMPT_ROLE = "last_prompt_token"


def role_load_icl_index(role, shard_icl_index):
    return None if role == FINAL_PROMPT_ROLE else shard_icl_index


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX full-dim ridge: activation -> per-prompt head-sum target.")
    p.add_argument("--icl_index", type=int, required=True,
                   help="ICL example index (1..10). 1-9 use icl{n}_3tokens dirs; 10 uses the 4tokens (query) dir.")
    p.add_argument("--token_roles", nargs="+", default=None)
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--target_mode", choices=["perprompt", "fv_broadcast"], default="perprompt",
                   help="'perprompt': per-prompt head-sum targets. 'fv_broadcast': canonical repro mode.")
    p.add_argument("--task_manifest", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--targets_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40")
    p.add_argument("--train_tasks", nargs="+", default=None)
    p.add_argument("--test_tasks", nargs="+", default=None)
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_function_vector(fv_root, task):
    fv_path = fv_root / task / f"{task}_function_vector.pt"
    data = torch_load_trusted(fv_path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return fv.detach().float().cpu().reshape(-1)


def load_role_activations_all_layers(activations_root, task, split, token_role, expected_icl_index):
    """Return (activations [n, n_layers, hidden], row_keys [(split, prompt_index, query_source_index)])."""
    split_dir = activations_root / task / split
    index = load_json(split_dir / "index.json")
    chunks = []
    keys = []
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = split_dir / shard_path
        elif not shard_path.exists():
            shard_path = split_dir / shard_path.name
        data = torch_load_trusted(shard_path, map_location="cpu")
        activations = data["activations"]
        shard_metadata = data["metadata"]
        if len(shard_metadata) != activations.shape[0]:
            raise ValueError(f"Metadata/activation mismatch in {shard_path}")
        selected = [
            i
            for i, meta in enumerate(shard_metadata)
            if meta.get("token_role") == token_role and meta.get("icl_example_index") == expected_icl_index
        ]
        if selected:
            chunks.append(activations[selected])
            keys.extend(
                (split, int(shard_metadata[i]["prompt_index"]), int(shard_metadata[i]["query_source_index"]))
                for i in selected
            )
    if not chunks:
        raise ValueError(f"No {token_role} activations found for {task}/{split}/ICL {expected_icl_index}")
    return torch.cat(chunks, dim=0), keys


def load_task_role_pooled(activations_root, task, splits, token_role, expected_icl_index):
    parts = []
    keys = []
    for split in splits:
        a, k = load_role_activations_all_layers(activations_root, task, split, token_role, expected_icl_index)
        parts.append(a)
        keys.extend(k)
    return torch.cat(parts, dim=0).to(torch.float16), keys


def load_perprompt_targets(targets_root, task, splits):
    """(split, prompt_index) -> (target fp32 [hidden], query_source_index) from the sandbox capture."""
    out = {}
    for split in splits:
        split_dir = targets_root / task / split
        index = load_json(split_dir / "index.json")
        for shard in index["shards"]:
            data = torch_load_trusted(split_dir / Path(shard).name, map_location="cpu")
            targets = data["targets"].float()
            for i, meta in enumerate(data["metadata"]):
                out[(split, int(meta["prompt_index"]))] = (targets[i], int(meta["query_source_index"]))
    return out


def align_targets(target_map, row_keys, task):
    rows = []
    for split, prompt_index, query_source_index in row_keys:
        if (split, prompt_index) not in target_map:
            raise KeyError(f"ALIGNMENT FAILED: no per-prompt target for {task}/{split} prompt {prompt_index}")
        target, tgt_query = target_map[(split, prompt_index)]
        if tgt_query != query_source_index:
            raise RuntimeError(
                f"ALIGNMENT GATE FAILED for {task}/{split} prompt {prompt_index}: "
                f"query_source_index X={query_source_index} vs Y={tgt_query}. STOP -- user adjudicates."
            )
        rows.append(target)
    return torch.stack(rows, dim=0)


def ridge_eig_prep(x_fit, y_fit):
    xbar = x_fit.mean(dim=0)
    ybar = y_fit.mean(dim=0)
    xc = x_fit - xbar
    gram = xc.T @ xc
    eigvals, eigvecs = torch.linalg.eigh(gram)
    rhs = xc.T @ (y_fit - ybar)
    c = eigvecs.T @ rhs
    return xbar, ybar, eigvals, eigvecs, c


def ridge_predict(x_eval, xbar, ybar, eigvals, eigvecs, c, alpha):
    a = (x_eval - xbar) @ eigvecs
    return (a / (eigvals + alpha)) @ c + ybar


def per_element_variance(rows, baseline):
    """Repo R^2 denominator: mean over rows of ||row - baseline||^2 / hidden."""
    diff = rows - baseline.unsqueeze(0)
    return float(torch.mean(torch.sum(diff ** 2, dim=1)) / rows.shape[1])


def main():
    args = parse_args()
    torch.manual_seed(0)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    manifest = load_json(args.task_manifest)
    train_tasks = list(args.train_tasks) if args.train_tasks is not None else list(manifest["train_tasks"])
    test_tasks = list(args.test_tasks) if args.test_tasks is not None else list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    overlap = sorted(set(train_tasks).intersection(test_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and test: {overlap}")
    all_tasks = train_tasks + test_tasks

    alphas = list(args.alphas) if args.alphas is not None else list(np.logspace(-1, 8, 19))

    if not 1 <= args.icl_index <= QUERY_ICL_INDEX:
        raise ValueError(f"--icl_index must be in 1..{QUERY_ICL_INDEX}, got {args.icl_index}")
    if args.icl_index == QUERY_ICL_INDEX:
        activations_root = args.query_activations_root
        default_roles = LABEL_ROLES + [FINAL_PROMPT_ROLE]
    else:
        activations_root = Path(args.icl_activations_root_template.format(icl=args.icl_index))
        default_roles = list(LABEL_ROLES)
    token_roles = list(args.token_roles) if args.token_roles is not None else default_roles

    shard_dir = args.output_dir / f"{args.target_mode}_shard_icl{args.icl_index}"
    metrics_csv = shard_dir / "metrics.csv"
    if metrics_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_csv} exists; pass --overwrite to replace.")
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sandbox icl{args.icl_index}|{args.target_mode}] activations_root={activations_root}")
    print(f"[sandbox icl{args.icl_index}|{args.target_mode}] roles={token_roles} | "
          f"train={len(train_tasks)} test={len(test_tasks)} | alphas={len(alphas)}")

    # Stored FVs (Test-A targets; also fit targets in fv_broadcast mode).
    fvs = {task: load_function_vector(args.fv_root, task).to(device=device, dtype=dtype) for task in all_tasks}

    # Load X once per (task, role), with row keys for target alignment.
    t0 = time.time()
    acts = {}
    row_keys = {}
    n_layers = None
    n_rows = None
    for role in token_roles:
        load_icl = role_load_icl_index(role, args.icl_index)
        for task in all_tasks:
            a, keys = load_task_role_pooled(activations_root, task, args.splits, role, load_icl)
            acts[(task, role)] = a
            row_keys[(task, role)] = keys
            if n_layers is None:
                n_layers, n_rows = a.shape[1], a.shape[0]
            elif a.shape[1] != n_layers:
                raise ValueError(f"Layer-count mismatch for {task}/{role}: {a.shape[1]} vs {n_layers}")
    layers = list(args.layers) if args.layers is not None else list(range(n_layers))
    print(f"[sandbox icl{args.icl_index}|{args.target_mode}] loaded X in {time.time()-t0:.1f}s | "
          f"n_layers={n_layers} rows/task={n_rows} | layers={len(layers)}")

    # Per-prompt targets, aligned to each (task, role)'s row order. Row keys are role-independent
    # (same prompts); assert that so we can store one aligned matrix per task.
    y_pp = {}
    for task in all_tasks:
        key_sets = {tuple(row_keys[(task, role)]) for role in token_roles}
        if len(key_sets) != 1:
            raise RuntimeError(f"Row-key mismatch across roles for {task}; cannot share targets across roles.")
        target_map = load_perprompt_targets(args.targets_root, task, args.splits)
        y_pp[task] = align_targets(target_map, row_keys[(task, token_roles[0])], task).to(device=device, dtype=dtype)
    print(f"[sandbox icl{args.icl_index}|{args.target_mode}] aligned per-prompt targets for {len(y_pp)} tasks")

    def fit_targets(task, n):
        if args.target_mode == "perprompt":
            y = y_pp[task]
            if y.shape[0] != n:
                raise ValueError(f"Row count mismatch for {task}: targets {y.shape[0]} vs features {n}")
            return y
        return fvs[task].unsqueeze(0).expand(n, -1)

    # ---- R^2 denominators (cell-independent; repo convention: train-mean baseline). ----
    hidden = fvs[test_tasks[0]].shape[0]
    train_fv_stack = torch.stack([fvs[t] for t in train_tasks], dim=0)
    test_fv_rows = torch.cat([fvs[t].unsqueeze(0).expand(acts[(t, token_roles[0])].shape[0], -1)
                              for t in test_tasks], dim=0)
    y_fit_all = torch.cat([fit_targets(t, acts[(t, token_roles[0])].shape[0]) for t in train_tasks], dim=0)
    y_pp_test_all = torch.cat([y_pp[t] for t in test_tasks], dim=0)

    ybar_fit = y_fit_all.mean(dim=0)                       # train-mean baseline (fit-target space)
    ybar_train_fv = train_fv_stack.mean(dim=0)             # canonical study's train-mean-FV baseline
    v_train = per_element_variance(y_fit_all, ybar_fit)
    v_test_fv_trainbase = per_element_variance(test_fv_rows, ybar_train_fv)
    v_test_fv_testbase = per_element_variance(test_fv_rows, test_fv_rows.mean(dim=0))
    v_test_pp_trainbase = per_element_variance(y_pp_test_all, ybar_fit)
    v_test_pp_testbase = per_element_variance(y_pp_test_all, y_pp_test_all.mean(dim=0))
    print(f"[sandbox icl{args.icl_index}|{args.target_mode}] V(train fit)={v_train:.4f} | "
          f"V(test FV|train-mean)={v_test_fv_trainbase:.4f} V(test FV|test-mean)={v_test_fv_testbase:.4f} | "
          f"V(test pp|train-mean)={v_test_pp_trainbase:.4f} V(test pp|test-mean)={v_test_pp_testbase:.4f}")

    rows_out = []
    for role in token_roles:
        for layer in layers:
            t_cell = time.time()
            x_by_task = {
                task: acts[(task, role)][:, layer, :].to(device=device, dtype=dtype)
                for task in all_tasks
            }
            x_train_pool = torch.cat([x_by_task[t] for t in train_tasks], dim=0)
            mean = x_train_pool.mean(dim=0)
            std = x_train_pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
            xs = {task: (x_by_task[task] - mean) / std for task in all_tasks}

            # ---- Leave-one-train-task-out CV to pick alpha ----
            cv_sqerr = torch.zeros(len(alphas), device=device, dtype=dtype)
            cv_n = 0
            for held in train_tasks:
                fit_tasks = [t for t in train_tasks if t != held]
                x_fit = torch.cat([xs[t] for t in fit_tasks], dim=0)
                y_fit = torch.cat([fit_targets(t, xs[t].shape[0]) for t in fit_tasks], dim=0)
                xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
                x_val = xs[held]
                y_val = fit_targets(held, x_val.shape[0])
                a_val = (x_val - xbar) @ evecs
                for ai, alpha in enumerate(alphas):
                    pred = (a_val / (evals + alpha)) @ c + ybar
                    cv_sqerr[ai] += torch.sum((pred - y_val) ** 2)
                cv_n += x_val.shape[0] * hidden
            cv_mse = (cv_sqerr / cv_n).detach().cpu().numpy()
            best_idx = int(np.argmin(cv_mse))
            best_alpha = float(alphas[best_idx])

            # ---- Refit on all 20 train tasks at best alpha; evaluate ----
            x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
            y_fit = torch.cat([fit_targets(t, xs[t].shape[0]) for t in train_tasks], dim=0)
            xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)

            train_pred = ridge_predict(x_fit, xbar, ybar, evals, evecs, c, best_alpha)
            train_mse = float(torch.mean((train_pred - y_fit) ** 2))

            test_sqerr_fv = 0.0
            test_sqerr_pp = 0.0
            test_n = 0
            per_task_mse_fv = {}
            per_task_mse_pp = {}
            for task in test_tasks:
                x_eval = xs[task]
                pred = ridge_predict(x_eval, xbar, ybar, evals, evecs, c, best_alpha)
                diff_fv = pred - fvs[task].unsqueeze(0).expand(x_eval.shape[0], -1)
                diff_pp = pred - y_pp[task]
                per_task_mse_fv[task] = float(torch.mean(diff_fv ** 2))
                per_task_mse_pp[task] = float(torch.mean(diff_pp ** 2))
                test_sqerr_fv += float(torch.sum(diff_fv ** 2))
                test_sqerr_pp += float(torch.sum(diff_pp ** 2))
                test_n += x_eval.shape[0]
            test_mse_fv = test_sqerr_fv / (test_n * hidden)
            test_mse_pp = test_sqerr_pp / (test_n * hidden)

            rows_out.append({
                "icl_example_index": args.icl_index,
                "token_role": role,
                "layer": layer,
                "target_mode": args.target_mode,
                "best_alpha": best_alpha,
                "cv_mse": float(cv_mse[best_idx]),
                "alpha_pinned": bool(best_idx in (0, len(alphas) - 1)),
                "train_sample_count": int(x_fit.shape[0]),
                "test_sample_count": int(test_n),
                "train_mse": train_mse,
                "train_r2": 1.0 - train_mse / v_train,
                "test_mse_fv": test_mse_fv,
                "test_r2_fv": 1.0 - test_mse_fv / v_test_fv_trainbase,
                "test_r2_fv_testmean_baseline": 1.0 - test_mse_fv / v_test_fv_testbase,
                "test_mse_pp": test_mse_pp,
                "test_r2_pp": 1.0 - test_mse_pp / v_test_pp_trainbase,
                "test_r2_pp_testmean_baseline": 1.0 - test_mse_pp / v_test_pp_testbase,
                "per_test_task_mse_fv": per_task_mse_fv,
                "per_test_task_mse_pp": per_task_mse_pp,
                "cv_curve": [{"alpha": float(a), "cv_mse": float(m)} for a, m in zip(alphas, cv_mse)],
            })
            print(f"[sandbox icl{args.icl_index}|{args.target_mode}] {role} L{layer:02d}: "
                  f"test_mse_fv={test_mse_fv:.5f} (R2={rows_out[-1]['test_r2_fv']:.3f}) "
                  f"test_mse_pp={test_mse_pp:.5f} (R2={rows_out[-1]['test_r2_pp']:.3f}) "
                  f"train_mse={train_mse:.5f} alpha={best_alpha:.3g}"
                  f"{' PINNED' if rows_out[-1]['alpha_pinned'] else ''} ({time.time()-t_cell:.1f}s)")

    csv_fields = [
        "icl_example_index", "token_role", "layer", "target_mode", "best_alpha", "cv_mse",
        "alpha_pinned", "train_sample_count", "test_sample_count", "train_mse", "train_r2",
        "test_mse_fv", "test_r2_fv", "test_r2_fv_testmean_baseline",
        "test_mse_pp", "test_r2_pp", "test_r2_pp_testmean_baseline",
    ]
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r[k] for k in csv_fields})
    write_json(shard_dir / "metrics.json", rows_out)
    write_json(shard_dir / "run_config.json", {
        "sandbox": True,
        "icl_index": args.icl_index,
        "target_mode": args.target_mode,
        "activations_root": str(activations_root),
        "targets_root": str(args.targets_root),
        "fv_root": str(args.fv_root),
        "task_manifest": str(args.task_manifest),
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "token_roles": token_roles,
        "layers": layers,
        "splits": args.splits,
        "alphas": [float(a) for a in alphas],
        "device": device,
        "dtype": args.dtype,
        "std_eps": args.std_eps,
        "r2_denominators": {
            "v_train_fit": v_train,
            "v_test_fv_trainmean": v_test_fv_trainbase,
            "v_test_fv_testmean": v_test_fv_testbase,
            "v_test_pp_trainmean": v_test_pp_trainbase,
            "v_test_pp_testmean": v_test_pp_testbase,
        },
        "n_cells": len(rows_out),
        "method": "SANDBOX direct full-dim ridge (4096->4096), per-prompt top-40 head-sum targets; "
                  "single 20-train standardizer; LOO-task CV; repo R^2 convention (train-mean baseline)",
    })
    print(f"[sandbox icl{args.icl_index}|{args.target_mode}] wrote {len(rows_out)} cells -> {metrics_csv}")


if __name__ == "__main__":
    main()
