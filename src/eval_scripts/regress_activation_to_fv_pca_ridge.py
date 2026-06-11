#!/usr/bin/env python
"""Direct PCA-space ridge regression: activation PCs -> function-vector PCs (k_act -> k_fv).

Companion to regress_activation_to_fv_fulldim_ridge.py. For each (token position, layer) cell:
  * Fit an ACTIVATION PCA (k_act components) fresh on the pooled 20-train-task rows at THIS cell;
    project every task's activations -> k_act features X.
  * Fit an FV PCA (k_fv components) ONCE on the 20 train-task FVs (layer/position-independent);
    project every task's FV -> k_fv target Y (one target per task, broadcast to its rows).
  * Ridge X -> Y, lambda by leave-one-train-task-out CV (20 folds); a single standardizer fit on
    the pooled 20-train k_act features is reused everywhere. Refit on all 20 train at best lambda.
  * Predict the 7 test tasks, RECONSTRUCT the prediction back to 4096-d (fv_mean + pred @ fv_PC),
    and report MSE there -- the headline `test_mse`, in the same units as the full-dim ridge.

This is the "direct" PCA projection (the repo standard): the activation side and FV side use
separate bases and the regression maps one to the other; it does NOT concatenate them (the older
"joint" projection leaked FV-basis features into X). Default k_act = k_fv = 16 (the saturation
corner found by the k-sweeps), giving one ridge per cell -> a 31x29 = 899-cell grid, sharded by
ICL index like the full-dim run.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


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
# The final prompt token is the query's last token; it is stored with icl_example_index=None
# (not tied to any label example), unlike the label roles which carry a concrete index.
FINAL_PROMPT_ROLE = "last_prompt_token"


def role_load_icl_index(role, shard_icl_index):
    """The icl_example_index to match in metadata when loading a given role."""
    return None if role == FINAL_PROMPT_ROLE else shard_icl_index


def parse_args():
    p = argparse.ArgumentParser(description="Direct PCA-space ridge regression: activation PCs -> FV PCs.")
    p.add_argument("--icl_index", type=int, required=True,
                   help="ICL example index (1..10). 1-9 use icl{n}_3tokens dirs; 10 uses the 4tokens (query) dir.")
    p.add_argument("--token_roles", nargs="+", default=None,
                   help="Override token roles. Default: pre/first/last for 1-9; + last_prompt_token for 10.")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Layers to process (default: all 0..28).")
    p.add_argument("--k_act", type=int, default=16, help="Number of activation PCA components (features).")
    p.add_argument("--k_fv", type=int, default=16, help="Number of FV PCA components (targets).")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=Path("results/function_vectors/gpt-j/train_selected"))
    p.add_argument("--icl_activations_root_template", type=str,
                   default="results/residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens")
    p.add_argument("--query_activations_root", type=Path,
                   default=Path("results/residual_activations/gptj_56tasks_170prompts_4tokens"))
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   help="Activation splits pooled into the 170 feature rows per task.")
    p.add_argument("--output_dir", type=Path, default=Path("results/pca_ridge_activation_to_fv"))
    p.add_argument("--train_tasks", nargs="+", default=None, help="Override train tasks.")
    p.add_argument("--test_tasks", nargs="+", default=None,
                   help="Override test tasks. Default: 9 test minus country-currency/product-company (7).")
    p.add_argument("--alphas", nargs="+", type=float, default=None,
                   help="Ridge alpha grid. Default: np.logspace(-2, 6, 17).")
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
    if not fv_path.exists():
        raise FileNotFoundError(fv_path)
    data = torch_load_trusted(fv_path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return fv.detach().float().cpu().reshape(-1)


def load_role_activations_all_layers(activations_root, task, split, token_role, expected_icl_index):
    """Return (n_selected, n_layers, hidden) float activations for one task/split/role/ICL.

    Mirrors the loader in sweep_layer_activation_to_fv_ols.py.
    """
    split_dir = activations_root / task / split
    index = load_json(split_dir / "index.json")
    chunks = []
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = split_dir / shard_path
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
    if not chunks:
        raise ValueError(f"No {token_role} activations found for {task}/{split}/ICL {expected_icl_index}")
    return torch.cat(chunks, dim=0)


def load_task_role_pooled(activations_root, task, splits, token_role, expected_icl_index):
    """Pool the requested splits into a single [n_rows, n_layers, hidden] fp16 tensor."""
    parts = [
        load_role_activations_all_layers(activations_root, task, split, token_role, expected_icl_index)
        for split in splits
    ]
    return torch.cat(parts, dim=0).to(torch.float16)


def torch_pca(x, k):
    """Top-k PCA via randomized low-rank SVD of centered x. Returns (mean[D], components[k, D]).

    components rows are orthonormal principal directions (right singular vectors of centered x)."""
    n, d = x.shape
    q = min(n, d, k + 10)
    mean = x.mean(dim=0)
    # pca_lowrank centers internally; we also return the explicit mean for projection/reconstruction.
    _, _, v = torch.pca_lowrank(x, q=q, center=True)  # v: [d, q] principal directions as columns
    comp = v[:, :k].T.contiguous()                    # [k, d]
    return mean, comp


def project(x, mean, comp):
    """[n, D] -> [n, k] in the PCA basis."""
    return (x - mean) @ comp.T


def reconstruct(proj, mean, comp):
    """[n, k] -> [n, D] back in the original space."""
    return mean + proj @ comp


def fv_pc_floor(fv_raw, fv_mean, fv_comp):
    """MSE (over elements) that a perfect FV-PC prediction still incurs: variance of fv_raw outside
    the top-k FV-PC subspace. fv_raw: [m, D]."""
    proj = (fv_raw - fv_mean) @ fv_comp.T
    recon = fv_mean + proj @ fv_comp
    return float(torch.mean((fv_raw - recon) ** 2))


def ridge_eig_prep(x_fit, y_fit):
    """Centered ridge solve via eigendecomposition of the feature Gram (dimension-agnostic)."""
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

    alphas = list(args.alphas) if args.alphas is not None else list(np.logspace(-2, 6, 17))

    # Resolve activation directory + roles for this ICL shard.
    if not 1 <= args.icl_index <= QUERY_ICL_INDEX:
        raise ValueError(f"--icl_index must be in 1..{QUERY_ICL_INDEX}, got {args.icl_index}")
    if args.icl_index == QUERY_ICL_INDEX:
        activations_root = args.query_activations_root
        default_roles = LABEL_ROLES + ["last_prompt_token"]
    else:
        activations_root = Path(args.icl_activations_root_template.format(icl=args.icl_index))
        default_roles = list(LABEL_ROLES)
    token_roles = list(args.token_roles) if args.token_roles is not None else default_roles

    shard_dir = args.output_dir / f"shard_icl{args.icl_index}"
    metrics_csv = shard_dir / "metrics.csv"
    if metrics_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_csv} exists; pass --overwrite to replace.")
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[icl{args.icl_index}] activations_root={activations_root}")
    print(f"[icl{args.icl_index}] roles={token_roles} | train={len(train_tasks)} test={len(test_tasks)} "
          f"| k_act={args.k_act} k_fv={args.k_fv} | alphas={len(alphas)}")

    # ---- FV PCA: fit ONCE on the 20 train FVs; project all tasks' FVs to k_fv targets. ----
    fv_raw = {task: load_function_vector(args.fv_root, task).to(device=device, dtype=dtype) for task in all_tasks}
    fv_train_stack = torch.stack([fv_raw[t] for t in train_tasks], dim=0)  # [20, D]
    fv_mean, fv_comp = torch_pca(fv_train_stack, args.k_fv)                # fv_comp: [k_fv, D]
    fv_proj = {task: project(fv_raw[task].unsqueeze(0), fv_mean, fv_comp).squeeze(0) for task in all_tasks}
    # FV-PC reconstruction floor on the (unique) test FVs.
    fv_test_stack = torch.stack([fv_raw[t] for t in test_tasks], dim=0)
    floor = fv_pc_floor(fv_test_stack, fv_mean, fv_comp)

    # Load activations once per (task, role): [n_rows, n_layers, hidden] fp16 on CPU.
    t0 = time.time()
    acts = {}
    n_layers = None
    n_rows = None
    for role in token_roles:
        load_icl = role_load_icl_index(role, args.icl_index)
        for task in all_tasks:
            a = load_task_role_pooled(activations_root, task, args.splits, role, load_icl)
            acts[(task, role)] = a
            if n_layers is None:
                n_layers, n_rows = a.shape[1], a.shape[0]
            elif a.shape[1] != n_layers:
                raise ValueError(f"Layer-count mismatch for {task}/{role}: {a.shape[1]} vs {n_layers}")
    layers = list(args.layers) if args.layers is not None else list(range(n_layers))
    print(f"[icl{args.icl_index}] loaded activations in {time.time()-t0:.1f}s | "
          f"n_layers={n_layers} rows/task={n_rows} | layers={len(layers)} | fv_recon_floor={floor:.5f}")

    hidden = fv_train_stack.shape[1]
    rows_out = []
    for role in token_roles:
        for layer in layers:
            t_cell = time.time()
            x_raw = {task: acts[(task, role)][:, layer, :].to(device=device, dtype=dtype) for task in all_tasks}

            # ---- Activation PCA fit on pooled 20-train rows at THIS cell; project all tasks. ----
            x_train_pool = torch.cat([x_raw[t] for t in train_tasks], dim=0)  # [3400, D]
            act_mean, act_comp = torch_pca(x_train_pool, args.k_act)          # act_comp: [k_act, D]
            x_proj = {task: project(x_raw[task], act_mean, act_comp) for task in all_tasks}  # [n, k_act]

            # ---- Single standardizer on the pooled-train k_act features; applied everywhere. ----
            proj_train_pool = torch.cat([x_proj[t] for t in train_tasks], dim=0)
            mean = proj_train_pool.mean(dim=0)
            std = proj_train_pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
            xs = {task: (x_proj[task] - mean) / std for task in all_tasks}

            # ---- Leave-one-train-task-out CV to pick alpha (in k_fv target space). ----
            cv_sqerr = torch.zeros(len(alphas), device=device, dtype=dtype)
            cv_n = 0
            for held in train_tasks:
                fit_tasks = [t for t in train_tasks if t != held]
                x_fit = torch.cat([xs[t] for t in fit_tasks], dim=0)
                y_fit = torch.cat([fv_proj[t].unsqueeze(0).expand(xs[t].shape[0], -1) for t in fit_tasks], dim=0)
                xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)
                x_val = xs[held]
                y_val = fv_proj[held].unsqueeze(0).expand(x_val.shape[0], -1)
                a_val = (x_val - xbar) @ evecs
                for ai, alpha in enumerate(alphas):
                    pred = (a_val / (evals + alpha)) @ c + ybar
                    cv_sqerr[ai] += torch.sum((pred - y_val) ** 2)
                cv_n += x_val.shape[0] * args.k_fv
            cv_mse = (cv_sqerr / cv_n).detach().cpu().numpy()
            best_idx = int(np.argmin(cv_mse))
            best_alpha = float(alphas[best_idx])

            # ---- Refit on all 20 train tasks at best alpha; evaluate train + test. ----
            x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
            y_fit = torch.cat([fv_proj[t].unsqueeze(0).expand(xs[t].shape[0], -1) for t in train_tasks], dim=0)
            xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)

            def eval_split(tasks):
                """Return (pca_sqerr, fv_sqerr, fv_l2sum, n_rows, per_task_fv_mse)."""
                pca_sq = 0.0
                fv_sq = 0.0
                fv_l2 = 0.0
                n = 0
                per_task = {}
                for task in tasks:
                    pred_proj = ridge_predict(xs[task], xbar, ybar, evals, evecs, c, best_alpha)
                    y_proj = fv_proj[task].unsqueeze(0).expand(pred_proj.shape[0], -1)
                    pca_sq += float(torch.sum((pred_proj - y_proj) ** 2))
                    pred_fv = reconstruct(pred_proj, fv_mean, fv_comp)          # [n, D]
                    true_fv = fv_raw[task].unsqueeze(0).expand(pred_fv.shape[0], -1)
                    diff = pred_fv - true_fv
                    per_task[task] = float(torch.mean(diff ** 2))
                    fv_sq += float(torch.sum(diff ** 2))
                    fv_l2 += float(torch.sum(torch.sum(diff ** 2, dim=1)))
                    n += pred_proj.shape[0]
                return pca_sq, fv_sq, fv_l2, n, per_task

            tr_pca_sq, tr_fv_sq, tr_fv_l2, tr_n, _ = eval_split(train_tasks)
            te_pca_sq, te_fv_sq, te_fv_l2, te_n, per_task_mse = eval_split(test_tasks)

            train_mse = tr_fv_sq / (tr_n * hidden)       # reconstructed 4096-d
            test_mse = te_fv_sq / (te_n * hidden)        # reconstructed 4096-d  (PRIMARY)
            pca_train_mse = tr_pca_sq / (tr_n * args.k_fv)
            pca_test_mse = te_pca_sq / (te_n * args.k_fv)
            train_msl2 = tr_fv_l2 / tr_n
            test_msl2 = te_fv_l2 / te_n

            rows_out.append({
                "icl_example_index": args.icl_index,
                "token_role": role,
                "layer": layer,
                "k_act": args.k_act,
                "k_fv": args.k_fv,
                "feature_dim": args.k_act,
                "target_dim": args.k_fv,
                "best_alpha": best_alpha,
                "cv_mse": float(cv_mse[best_idx]),
                "alpha_pinned": bool(best_idx in (0, len(alphas) - 1)),
                "train_sample_count": int(tr_n),
                "test_sample_count": int(te_n),
                "train_mse": train_mse,
                "test_mse": test_mse,
                "pca_train_mse": pca_train_mse,
                "pca_test_mse": pca_test_mse,
                "fv_recon_floor_test_mse": floor,
                "train_mean_squared_l2": train_msl2,
                "test_mean_squared_l2": test_msl2,
                "per_test_task_mse": per_task_mse,
                "cv_curve": [{"alpha": float(a), "cv_mse": float(m)} for a, m in zip(alphas, cv_mse)],
            })
            print(f"[icl{args.icl_index}] {role} L{layer:02d}: "
                  f"test_mse={test_mse:.5f} (floor={floor:.5f}) train_mse={train_mse:.5f} "
                  f"alpha={best_alpha:.3g}{' PINNED' if rows_out[-1]['alpha_pinned'] else ''} ({time.time()-t_cell:.1f}s)")

    # ---- Write outputs (CSV schema is a superset of the full-dim run, so the merge script works). ----
    csv_fields = [
        "icl_example_index", "token_role", "layer", "k_act", "k_fv", "feature_dim", "target_dim",
        "best_alpha", "cv_mse", "alpha_pinned", "train_sample_count", "test_sample_count",
        "train_mse", "test_mse", "pca_train_mse", "pca_test_mse", "fv_recon_floor_test_mse",
        "train_mean_squared_l2", "test_mean_squared_l2",
    ]
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r[k] for k in csv_fields})
    write_json(shard_dir / "metrics.json", rows_out)
    write_json(shard_dir / "run_config.json", {
        "icl_index": args.icl_index,
        "activations_root": str(activations_root),
        "query_activations_root": str(args.query_activations_root),
        "fv_root": str(args.fv_root),
        "task_manifest": str(args.task_manifest),
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "token_roles": token_roles,
        "layers": layers,
        "splits": args.splits,
        "k_act": args.k_act,
        "k_fv": args.k_fv,
        "alphas": [float(a) for a in alphas],
        "fv_recon_floor_test_mse": floor,
        "device": device,
        "dtype": args.dtype,
        "std_eps": args.std_eps,
        "n_cells": len(rows_out),
        "method": ("direct PCA-space ridge (k_act->k_fv); activation PCA fit per-cell on 20 train, "
                   "FV PCA fit once on 20 train FVs; single 20-train standardizer; LOO-task CV; "
                   "test_mse reported in reconstructed 4096-d FV space"),
    })
    print(f"[icl{args.icl_index}] wrote {len(rows_out)} cells -> {metrics_csv}")


if __name__ == "__main__":
    main()
