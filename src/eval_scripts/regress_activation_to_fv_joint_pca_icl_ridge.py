#!/usr/bin/env python
"""Ridge variant of the per-ICL activation->function-vector joint-PCA regression.

Same setup as ``regress_activation_to_fv_joint_pca_icl.py`` (independent regression per
ICL example x token role x k, joint PCA space, fit on train tasks only, evaluated on
held-out test tasks), but using Ridge regression with the regularization strength
(lambda / alpha) chosen per regression via a held-out validation split.

Alpha is selected by leave-k-tasks-out cross-validation over the train tasks: the train
tasks are partitioned into CV folds (default = leave-one-task-out), and for each candidate
alpha every fold is held out in turn, fitting on the other folds and accumulating the
held-out (pooled) MSE. This rotates every task through validation, so no single unlucky
task draw can dominate the alpha decision (which it could with a single random split).
Per regression the pipeline is:
  1. For each alpha, run the task-level CV and record the pooled cross-validated MSE.
  2. Select the alpha with the lowest cross-validated MSE.
  3. Refit Ridge with that alpha on the *full* train-task set and report train MSE
     (full train) and held-out test MSE, exactly like the OLS script.

The fold partition is seeded and shared across every (ICL example, token role, k), and is
logged in run_config.json.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Demonstration examples have no final prompt token, so it is dropped relative to the
# non-ICL regression script.
DEFAULT_TOKEN_ROLES = [
    "pre_label_token",
    "first_label_token",
    "last_label_token",
]
TOKEN_TITLES = {
    "pre_label_token": "Pre-label",
    "first_label_token": "First label",
    "last_label_token": "Last label",
    "last_prompt_token": "Last prompt",
}
# With standardized features the meaningful regularization transition sits in the mid-range
# of this grid (alpha competes with eigenvalues of the standardized design ~ n_samples).
DEFAULT_ALPHAS = [float(a) for a in np.logspace(-2, 6, 17)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ridge regression of layer activations to task function vectors in the joint "
            "PCA space, independently per ICL example, with alpha (lambda) selected per "
            "regression on a validation split."
        )
    )
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument(
        "--activations_root_template",
        type=str,
        default="results/residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens",
        help="Template with {icl} placeholder for ICL-specific activation roots.",
    )
    parser.add_argument(
        "--pca_root",
        type=Path,
        default=Path("results/pca_abstractive_icl_examples_fv_activation_scatter"),
        help="Root with shared fv_pca.pt and per-ICL icl{N}/activation_pca_{role}.pt bases.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/joint_pca_activation_to_fv_regression_icl_ridge"))
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--k_values", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
                        help="Candidate Ridge alpha (lambda) values to search over.")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True,
                        help="Standardize features (zero mean, unit variance) before Ridge so alpha is "
                             "on a meaningful scale. Use --no-standardize to fit on raw joint-PCA features.")
    parser.add_argument("--cv_folds", type=int, default=0,
                        help="Number of task-level CV folds for alpha selection. 0 (default) = leave-one-task-out "
                             "(n_folds = number of train tasks). Otherwise K-fold over the train tasks.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling tasks into CV folds.")
    parser.add_argument("--train_tasks", nargs="+", default=None, help="Optional override for smoke tests.")
    parser.add_argument("--test_tasks", nargs="+", default=None, help="Optional override for smoke tests.")
    return parser.parse_args()


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


def load_task_manifest(args):
    manifest = load_json(args.task_manifest)
    train_tasks = args.train_tasks if args.train_tasks is not None else manifest["train_tasks"]
    test_tasks = args.test_tasks if args.test_tasks is not None else manifest["test_tasks"]
    overlap = sorted(set(train_tasks).intersection(test_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and test: {overlap}")
    return manifest, list(train_tasks), list(test_tasks)


def load_function_vector(fv_root, task):
    fv_path = fv_root / task / f"{task}_function_vector.pt"
    if not fv_path.exists():
        raise FileNotFoundError(fv_path)
    data = torch_load_trusted(fv_path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return fv.detach().float().cpu().reshape(-1).numpy()


def load_pca_artifact(path):
    artifact = torch_load_trusted(path, map_location="cpu")
    return {
        "components": artifact["components"].detach().float().cpu().numpy(),
        "mean": artifact["mean"].detach().float().cpu().numpy(),
        "explained_variance": artifact["explained_variance"].detach().float().cpu().numpy(),
        "explained_variance_ratio": artifact["explained_variance_ratio"].detach().float().cpu().numpy(),
        "fit_tasks": artifact.get("fit_tasks"),
        "projected_tasks": artifact.get("projected_tasks"),
    }


def load_role_activations(activations_root, task, split, layer, token_role, expected_icl_index):
    split_dir = activations_root / task / split
    index_path = split_dir / "index.json"
    index = load_json(index_path)
    chunks = []
    metadata = []
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = split_dir / shard_path
        data = torch_load_trusted(shard_path, map_location="cpu")
        activations = data["activations"]
        shard_metadata = data["metadata"]
        if len(shard_metadata) != activations.shape[0]:
            raise ValueError(f"Metadata/activation mismatch in {shard_path}")
        if layer < 0 or layer >= activations.shape[1]:
            raise IndexError(f"Layer {layer} is outside activation shape {tuple(activations.shape)} in {shard_path}")
        selected = [
            i
            for i, meta in enumerate(shard_metadata)
            if meta.get("token_role") == token_role and meta.get("icl_example_index") == expected_icl_index
        ]
        if selected:
            chunks.append(activations[selected, layer, :].float())
            metadata.extend(shard_metadata[i] for i in selected)
    if not chunks:
        raise ValueError(f"No {token_role} activations found for {task}/{split}/ICL {expected_icl_index}")
    return torch.cat(chunks, dim=0).numpy(), metadata


def project_joint(x, activation_pca, fv_pca, k):
    activation_part = (x - activation_pca["mean"]) @ activation_pca["components"][:k].T
    fv_part = (x - fv_pca["mean"]) @ fv_pca["components"][:k].T
    return np.concatenate([activation_part, fv_part], axis=1)


def project_joint_vector(x, activation_pca, fv_pca, k):
    return project_joint(x.reshape(1, -1), activation_pca, fv_pca, k).reshape(-1)


def make_task_folds(train_tasks, cv_folds, seed):
    """Partition train tasks into CV folds (each fold is the held-out task set).

    cv_folds <= 0 or >= len(train_tasks) gives leave-one-task-out (one task per fold).
    Otherwise the tasks are shuffled (seeded) and split into cv_folds groups. The partition
    is made once and shared across every regression so selections are comparable."""
    n = len(train_tasks)
    if cv_folds <= 0 or cv_folds >= n:
        return [[t] for t in train_tasks]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [train_tasks[i] for i in order]
    return [list(chunk) for chunk in np.array_split(shuffled, cv_folds)]


def cv_scheme_label(n_folds, n_train_tasks):
    if n_folds >= n_train_tasks:
        return "leave-one-task-out CV"
    return f"{n_folds}-fold task CV"


def stack_tasks(tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k):
    x_chunks = []
    y_chunks = []
    counts = {}
    for task in tasks:
        acts = activations_by_task[task]
        x_task = project_joint(acts, activation_pca, fv_pca, k)
        y_task = project_joint_vector(fvs_by_task[task], activation_pca, fv_pca, k)
        x_chunks.append(x_task)
        y_chunks.append(np.repeat(y_task.reshape(1, -1), x_task.shape[0], axis=0))
        counts[task] = int(x_task.shape[0])
    return np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0), counts


def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def mean_squared_l2(y_true, y_pred):
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def make_model(alpha, standardize):
    """Ridge estimator, optionally preceded by per-feature standardization so that alpha
    acts on a comparable scale across all eigendirections."""
    ridge = Ridge(alpha=alpha)
    if standardize:
        return Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
    return ridge


def cv_select_alpha(train_tasks, folds, projx_by_task, projy_by_task, alphas, standardize):
    """Select alpha by leave-k-tasks-out CV. For each alpha, every fold is held out in turn
    (fit on the other folds), and the squared errors on held-out tasks are pooled into one
    cross-validated MSE. Returns (best_alpha, best_cv_mse, cv_curve).

    projx_by_task[task] is the joint-PCA feature matrix for that task; projy_by_task[task]
    is the (1D) joint-PCA target. Working from pre-projected arrays keeps the CV cheap."""
    def assemble(tasklist):
        x = np.concatenate([projx_by_task[t] for t in tasklist], axis=0)
        y = np.concatenate(
            [np.repeat(projy_by_task[t].reshape(1, -1), projx_by_task[t].shape[0], axis=0) for t in tasklist],
            axis=0,
        )
        return x, y

    fold_sets = [set(f) for f in folds]
    cv_curve = []
    best_alpha = None
    best_cv = np.inf
    for alpha in alphas:
        sq_err = 0.0
        n_elem = 0
        for fold, fset in zip(folds, fold_sets):
            fit_tasks = [t for t in train_tasks if t not in fset]
            x_fit, y_fit = assemble(fit_tasks)
            x_val, y_val = assemble(fold)
            model = make_model(alpha, standardize)
            model.fit(x_fit, y_fit)
            pred = model.predict(x_val)
            sq_err += float(np.sum((pred - y_val) ** 2))
            n_elem += int(y_val.size)
        cv_mse = sq_err / n_elem
        cv_curve.append({"alpha": float(alpha), "cv_mse": cv_mse})
        if cv_mse < best_cv:
            best_cv = cv_mse
            best_alpha = float(alpha)
    return best_alpha, best_cv, cv_curve


def save_model(model, output_dir, token_role, k, metadata):
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{token_role}_k{k:02d}"
    # Unwrap the Ridge step (coef_ is in standardized-feature space when a scaler is used).
    ridge = model.named_steps["ridge"] if isinstance(model, Pipeline) else model
    payload = {
        "coef": torch.from_numpy(ridge.coef_).float(),
        "intercept": torch.from_numpy(np.atleast_1d(ridge.intercept_)).float(),
        "alpha": float(ridge.alpha),
        "metadata": metadata,
    }
    if isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        payload["coef_space"] = "standardized_features"
        payload["scaler_mean"] = torch.from_numpy(scaler.mean_).float()
        payload["scaler_scale"] = torch.from_numpy(scaler.scale_).float()
    else:
        payload["coef_space"] = "raw_features"
    torch.save(payload, model_dir / f"{stem}.pt")
    write_json(model_dir / f"{stem}.json", metadata)


METRIC_FIELDNAMES = [
    "icl_example_index",
    "token_role",
    "k",
    "feature_dim",
    "target_dim",
    "best_alpha",
    "cv_mse",
    "train_sample_count",
    "test_sample_count",
    "train_mse",
    "test_mse",
    "train_mean_squared_l2",
    "test_mean_squared_l2",
]


def write_metrics_csv(rows, output_dir, filename="regression_metrics.csv"):
    path = output_dir / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def metric_matrix(rows, row_keys, k_values, key, row_key_fn):
    matrix = np.full((len(row_keys), len(k_values)), np.nan, dtype=np.float64)
    by_key = {(row_key_fn(row), int(row["k"])): float(row[key]) for row in rows}
    for i, rk in enumerate(row_keys):
        for j, k in enumerate(k_values):
            matrix[i, j] = by_key.get((rk, k), np.nan)
    return matrix


def annotate_heatmap(ax, matrix, fmt="{:.3g}"):
    finite = matrix[np.isfinite(matrix)]
    threshold = np.nanmedian(finite) if finite.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                color = "white" if val > threshold else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7, color=color)


def plot_train_test_heatmaps(rows, row_keys, row_labels, k_values, output_dir, row_key_fn, ylabel, suptitle, prefix=""):
    train_matrix = metric_matrix(rows, row_keys, k_values, "train_mse", row_key_fn)
    test_matrix = metric_matrix(rows, row_keys, k_values, "test_mse", row_key_fn)
    vmin = float(np.nanmin([np.nanmin(train_matrix), np.nanmin(test_matrix)]))
    vmax = float(np.nanmax([np.nanmax(train_matrix), np.nanmax(test_matrix)]))

    height = max(4.8, 0.5 * len(row_keys) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, height), constrained_layout=True)
    for ax, matrix, title in zip(axes, [train_matrix, test_matrix], ["Train-task MSE", "Held-out test-task MSE"]):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(row_keys)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("k (PCA components per space)")
        annotate_heatmap(ax, matrix)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(im, ax=axes, shrink=0.88, label="MSE in joint PCA space")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    png = output_dir / f"{prefix}mse_heatmap_train_test.png"
    pdf = output_dir / f"{prefix}mse_heatmap_train_test.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_ratio_heatmap(rows, row_keys, row_labels, k_values, output_dir, row_key_fn, ylabel, suptitle, prefix=""):
    train_matrix = metric_matrix(rows, row_keys, k_values, "train_mse", row_key_fn)
    test_matrix = metric_matrix(rows, row_keys, k_values, "test_mse", row_key_fn)
    ratio = test_matrix / np.maximum(train_matrix, 1e-12)

    height = max(4.8, 0.5 * len(row_keys) + 2.0)
    fig, ax = plt.subplots(figsize=(8.4, height), constrained_layout=True)
    im = ax.imshow(ratio, aspect="auto", cmap="magma")
    ax.set_title("Overfit ratio = held-out test MSE / train MSE  (>1 = worse on held-out tasks)")
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("k (PCA components per space)")
    ax.set_ylabel(ylabel)
    annotate_heatmap(ax, ratio)
    fig.colorbar(im, ax=ax, shrink=0.88, label="MSE ratio")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    png = output_dir / f"{prefix}mse_ratio_heatmap.png"
    pdf = output_dir / f"{prefix}mse_ratio_heatmap.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_alpha_heatmap(rows, row_keys, row_labels, k_values, output_dir, row_key_fn, ylabel, suptitle, prefix=""):
    alpha_matrix = metric_matrix(rows, row_keys, k_values, "best_alpha", row_key_fn)
    finite = alpha_matrix[np.isfinite(alpha_matrix)]
    if finite.size == 0:
        return None, None
    norm = LogNorm(vmin=max(float(finite.min()), 1e-12), vmax=float(finite.max()))

    height = max(4.8, 0.5 * len(row_keys) + 2.0)
    fig, ax = plt.subplots(figsize=(8.4, height), constrained_layout=True)
    im = ax.imshow(alpha_matrix, aspect="auto", cmap="cividis", norm=norm)
    ax.set_title("Selected Ridge alpha (lambda), chosen by leave-k-tasks-out CV over train tasks")
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("k (PCA components per space)")
    ax.set_ylabel(ylabel)
    annotate_heatmap(ax, alpha_matrix)
    fig.colorbar(im, ax=ax, shrink=0.88, label="alpha (log scale)")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    png = output_dir / f"{prefix}selected_alpha_heatmap.png"
    pdf = output_dir / f"{prefix}selected_alpha_heatmap.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_icl_example(args, icl_index, train_tasks, test_tasks, all_tasks, fvs_by_task, fv_pca, max_k,
                    folds):
    activations_root = Path(args.activations_root_template.format(icl=icl_index))
    if not activations_root.exists():
        raise FileNotFoundError(activations_root)
    icl_pca_dir = args.pca_root / f"icl{icl_index}"
    icl_output_dir = args.output_dir / f"icl{icl_index}"
    icl_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_counts_by_role = {}
    pca_artifacts_by_role = {}

    for token_role in args.token_roles:
        activation_pca = load_pca_artifact(icl_pca_dir / f"activation_pca_{token_role}.pt")
        max_activation_components = activation_pca["components"].shape[0]
        if max_k > max_activation_components:
            raise ValueError(
                f"ICL {icl_index} {token_role}: requested max k={max_k}, but activation PCA only "
                f"has {max_activation_components} components"
            )
        pca_artifacts_by_role[token_role] = {
            "activation_pca_components": int(max_activation_components),
            "activation_pca_fit_tasks": activation_pca.get("fit_tasks"),
        }

        activations_by_task = {}
        for task in all_tasks:
            activations_by_task[task], _ = load_role_activations(
                activations_root, task, args.split, args.layer, token_role, expected_icl_index=icl_index
            )

        for k in args.k_values:
            # Pre-project each task once for this k, then run task-level CV cheaply.
            projx_by_task = {
                t: project_joint(activations_by_task[t], activation_pca, fv_pca, k) for t in all_tasks
            }
            projy_by_task = {
                t: project_joint_vector(fvs_by_task[t], activation_pca, fv_pca, k) for t in all_tasks
            }

            # Full train (refit target) and held-out test.
            x_train_full, y_train_full, train_counts = stack_tasks(
                train_tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k
            )
            x_test, y_test, test_counts = stack_tasks(
                test_tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k
            )
            expected_dim = 2 * k
            for name, arr in [("train", x_train_full), ("test", x_test)]:
                if arr.shape[1] != expected_dim:
                    raise ValueError(f"ICL {icl_index} {token_role} k={k}: {name} matrix is not {expected_dim}D")

            # Alpha by leave-k-tasks-out CV over the train tasks.
            best_alpha, best_cv_mse, cv_curve = cv_select_alpha(
                train_tasks, folds, projx_by_task, projy_by_task, args.alphas, args.standardize
            )

            # Refit on the full train-task set with the selected alpha.
            model = make_model(best_alpha, args.standardize)
            model.fit(x_train_full, y_train_full)
            y_train_pred = model.predict(x_train_full)
            y_test_pred = model.predict(x_test)

            row = {
                "icl_example_index": int(icl_index),
                "token_role": token_role,
                "k": int(k),
                "feature_dim": int(expected_dim),
                "target_dim": int(expected_dim),
                "best_alpha": float(best_alpha),
                "cv_mse": float(best_cv_mse),
                "train_sample_count": int(x_train_full.shape[0]),
                "test_sample_count": int(x_test.shape[0]),
                "train_mse": mse(y_train_full, y_train_pred),
                "test_mse": mse(y_test, y_test_pred),
                "train_mean_squared_l2": mean_squared_l2(y_train_full, y_train_pred),
                "test_mean_squared_l2": mean_squared_l2(y_test, y_test_pred),
            }
            rows.append(row)

            save_model(
                model,
                icl_output_dir,
                token_role,
                k,
                {
                    **row,
                    "layer": args.layer,
                    "split": args.split,
                    "alphas_searched": list(args.alphas),
                    "cv_curve": cv_curve,
                    "cv_folds": [list(f) for f in folds],
                    "n_folds": len(folds),
                    "seed": args.seed,
                    "train_tasks": train_tasks,
                    "test_tasks": test_tasks,
                    "train_counts_by_task": train_counts,
                    "test_counts_by_task": test_counts,
                    "feature_definition": "[(activation - activation_pca_mean) @ activation_pcs[:k], (activation - fv_pca_mean) @ fv_pcs[:k]]",
                    "target_definition": "[(function_vector - activation_pca_mean) @ activation_pcs[:k], (function_vector - fv_pca_mean) @ fv_pcs[:k]]",
                    "model": "sklearn.linear_model.Ridge (fit_intercept=True)",
                    "standardize_features": bool(args.standardize),
                    "alpha_selection": "min leave-k-tasks-out CV MSE over alphas (pooled over held-out train tasks); refit on full train set",
                    "mse_definition": "mean((predicted_joint_pca_fv - actual_joint_pca_fv) ** 2)",
                },
            )

        sample_counts_by_role[token_role] = {
            "train_sample_count": int(sum(activations_by_task[task].shape[0] for task in train_tasks)),
            "test_sample_count": int(sum(activations_by_task[task].shape[0] for task in test_tasks)),
        }

    # Per-ICL heatmaps: rows = token_role (mirrors the original non-ICL layout).
    metrics_csv = write_metrics_csv(rows, icl_output_dir)
    write_json(icl_output_dir / "regression_metrics.json", rows)
    role_labels = [TOKEN_TITLES.get(role, role) for role in args.token_roles]
    feat_note = "standardized feats" if args.standardize else "raw feats"
    cv_note = cv_scheme_label(len(folds), len(train_tasks))
    context = f"layer {args.layer} · {args.split} split · Ridge, {feat_note}, lambda by {cv_note}"
    role_key_fn = lambda r: r["token_role"]
    heatmap_png, heatmap_pdf = plot_train_test_heatmaps(
        rows, args.token_roles, role_labels, args.k_values, icl_output_dir,
        row_key_fn=role_key_fn, ylabel="Token role",
        suptitle=f"ICL example {icl_index}: Ridge activation → function-vector regression in joint PCA space ({context})",
    )
    ratio_png, ratio_pdf = plot_ratio_heatmap(
        rows, args.token_roles, role_labels, args.k_values, icl_output_dir,
        row_key_fn=role_key_fn, ylabel="Token role",
        suptitle=f"ICL example {icl_index}: Ridge activation → function-vector overfit ratio ({context})",
    )
    alpha_png, alpha_pdf = plot_alpha_heatmap(
        rows, args.token_roles, role_labels, args.k_values, icl_output_dir,
        row_key_fn=role_key_fn, ylabel="Token role",
        suptitle=f"ICL example {icl_index}: selected Ridge lambda (layer {args.layer} · {args.split} split)",
    )

    icl_config = {
        "icl_example_index": int(icl_index),
        "activations_root": str(activations_root),
        "pca_dir": str(icl_pca_dir),
        "output_dir": str(icl_output_dir),
        "sample_counts_by_role": sample_counts_by_role,
        "activation_pca_by_role": pca_artifacts_by_role,
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(icl_output_dir / "regression_metrics.json"),
        "mse_heatmap_png": str(heatmap_png),
        "mse_heatmap_pdf": str(heatmap_pdf),
        "mse_ratio_heatmap_png": str(ratio_png),
        "mse_ratio_heatmap_pdf": str(ratio_pdf),
        "selected_alpha_heatmap_png": str(alpha_png) if alpha_png else None,
        "selected_alpha_heatmap_pdf": str(alpha_pdf) if alpha_pdf else None,
    }
    write_json(icl_output_dir / "run_config.json", icl_config)
    print(metrics_csv)
    print(ratio_png)
    return rows, icl_config


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, train_tasks, test_tasks = load_task_manifest(args)
    all_tasks = train_tasks + test_tasks

    # Function-vector PCA basis is shared across ICL examples (FVs do not depend on ICL position).
    fv_pca = load_pca_artifact(args.pca_root / "fv_pca.pt")
    max_fv_components = fv_pca["components"].shape[0]
    max_k = max(args.k_values)
    if max_k > max_fv_components:
        raise ValueError(f"Requested max k={max_k}, but FV PCA only has {max_fv_components} components")

    fvs_by_task = {task: load_function_vector(args.fv_root, task) for task in all_tasks}

    # Build the task-level CV folds once (seeded); shared across every regression.
    folds = make_task_folds(train_tasks, args.cv_folds, args.seed)
    print(f"Alpha selection: {cv_scheme_label(len(folds), len(train_tasks))} "
          f"({len(folds)} folds over {len(train_tasks)} train tasks)")

    all_rows = []
    icl_configs = []
    for icl_index in args.icl_example_indices:
        rows, icl_config = run_icl_example(
            args, icl_index, train_tasks, test_tasks, all_tasks, fvs_by_task, fv_pca, max_k,
            folds
        )
        all_rows.extend(rows)
        icl_configs.append(icl_config)

    # Combined outputs: rows = (ICL example, token_role).
    combined_csv = write_metrics_csv(all_rows, args.output_dir)
    write_json(args.output_dir / "regression_metrics.json", all_rows)

    combined_row_keys = [(icl, role) for icl in args.icl_example_indices for role in args.token_roles]
    combined_row_labels = [
        f"ICL{icl} / {TOKEN_TITLES.get(role, role)}"
        for icl in args.icl_example_indices
        for role in args.token_roles
    ]
    combined_key_fn = lambda r: (int(r["icl_example_index"]), r["token_role"])
    icl_span = (
        f"ICL examples {args.icl_example_indices[0]}–{args.icl_example_indices[-1]}"
        if len(args.icl_example_indices) > 1
        else f"ICL example {args.icl_example_indices[0]}"
    )
    feat_note = "standardized feats" if args.standardize else "raw feats"
    cv_note = cv_scheme_label(len(folds), len(train_tasks))
    combined_context = f"layer {args.layer} · {args.split} split · Ridge, {feat_note}, lambda by {cv_note}"
    combined_heatmap_png, combined_heatmap_pdf = plot_train_test_heatmaps(
        all_rows, combined_row_keys, combined_row_labels, args.k_values, args.output_dir,
        row_key_fn=combined_key_fn, ylabel="ICL example / token role", prefix="combined_",
        suptitle=f"Ridge activation → function-vector regression in joint PCA space, {icl_span} ({combined_context})",
    )
    combined_ratio_png, combined_ratio_pdf = plot_ratio_heatmap(
        all_rows, combined_row_keys, combined_row_labels, args.k_values, args.output_dir,
        row_key_fn=combined_key_fn, ylabel="ICL example / token role", prefix="combined_",
        suptitle=f"Ridge activation → function-vector overfit ratio, {icl_span} ({combined_context})",
    )
    combined_alpha_png, combined_alpha_pdf = plot_alpha_heatmap(
        all_rows, combined_row_keys, combined_row_labels, args.k_values, args.output_dir,
        row_key_fn=combined_key_fn, ylabel="ICL example / token role", prefix="combined_",
        suptitle=f"Selected Ridge lambda, {icl_span} (layer {args.layer} · {args.split} split)",
    )

    run_config = {
        "task_manifest": str(args.task_manifest),
        "manifest": manifest,
        "fv_root": str(args.fv_root),
        "activations_root_template": args.activations_root_template,
        "pca_root": str(args.pca_root),
        "output_dir": str(args.output_dir),
        "icl_example_indices": args.icl_example_indices,
        "layer": args.layer,
        "split": args.split,
        "token_roles": args.token_roles,
        "k_values": args.k_values,
        "model": "ridge",
        "standardize_features": bool(args.standardize),
        "alphas": list(args.alphas),
        "cv_scheme": cv_scheme_label(len(folds), len(train_tasks)),
        "cv_folds_requested": args.cv_folds,
        "n_folds": len(folds),
        "cv_folds": [list(f) for f in folds],
        "seed": args.seed,
        "alpha_selection": "per (icl, token_role, k): min leave-k-tasks-out CV MSE (pooled over held-out train tasks), then refit on full train set",
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "fv_pca_fit_tasks": fv_pca.get("fit_tasks"),
        "fv_pca_components": int(max_fv_components),
        "regression_fit_on": "train_tasks_only",
        "metrics_row_count": len(all_rows),
        "combined_metrics_csv": str(combined_csv),
        "combined_metrics_json": str(args.output_dir / "regression_metrics.json"),
        "combined_mse_heatmap_png": str(combined_heatmap_png),
        "combined_mse_heatmap_pdf": str(combined_heatmap_pdf),
        "combined_mse_ratio_heatmap_png": str(combined_ratio_png),
        "combined_mse_ratio_heatmap_pdf": str(combined_ratio_pdf),
        "combined_selected_alpha_heatmap_png": str(combined_alpha_png) if combined_alpha_png else None,
        "combined_selected_alpha_heatmap_pdf": str(combined_alpha_pdf) if combined_alpha_pdf else None,
        "per_icl_runs": icl_configs,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(combined_csv)
    print(combined_heatmap_png)
    print(combined_ratio_png)


if __name__ == "__main__":
    main()
