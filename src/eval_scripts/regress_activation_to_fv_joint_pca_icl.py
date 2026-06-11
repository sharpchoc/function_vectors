#!/usr/bin/env python
"""Regress layer activations to task function vectors in a joint PCA space, per ICL example.

This is the ICL-example counterpart to ``regress_activation_to_fv_joint_pca.py``. The
original script regresses activations of the final query region (including the
``last_prompt_token``) to the task function vector. Here we instead run the *same*
joint-PCA regression independently for each in-context demonstration example
(ICL examples 1..N). Demonstration examples have no final prompt token, so the
``last_prompt_token`` role is intentionally dropped; only the three label-region roles
are used.

For each ICL example index we use that example's own activation PCA basis (fit on the
train tasks by ``pca_abstractive_icl_examples_fv_activation_scatter.py``) and the shared
function-vector PCA basis. Regressions are fit on train-task samples only and evaluated
on held-out test tasks.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regress layer activations to task function vectors in the joint PCA space "
            "(token-role activation PCs + function-vector PCs), independently per ICL example."
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
    parser.add_argument("--output_dir", type=Path, default=Path("results/joint_pca_activation_to_fv_regression_icl"))
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--k_values", nargs="+", type=int, default=list(range(1, 11)))
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


def build_split_matrices(tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k):
    x_chunks = []
    y_chunks = []
    counts = {}
    for task in tasks:
        x_task = project_joint(activations_by_task[task], activation_pca, fv_pca, k)
        y_task = project_joint_vector(fvs_by_task[task], activation_pca, fv_pca, k)
        x_chunks.append(x_task)
        y_chunks.append(np.repeat(y_task.reshape(1, -1), x_task.shape[0], axis=0))
        counts[task] = int(x_task.shape[0])
    return np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0), counts


def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def mean_squared_l2(y_true, y_pred):
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def save_model(model, output_dir, token_role, k, metadata):
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{token_role}_k{k:02d}"
    torch.save(
        {
            "coef": torch.from_numpy(model.coef_).float(),
            "intercept": torch.from_numpy(model.intercept_).float(),
            "metadata": metadata,
        },
        model_dir / f"{stem}.pt",
    )
    write_json(model_dir / f"{stem}.json", metadata)


METRIC_FIELDNAMES = [
    "icl_example_index",
    "token_role",
    "k",
    "feature_dim",
    "target_dim",
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
    """Generic metric matrix. ``row_keys`` is the ordered list of row identities and
    ``row_key_fn(row)`` maps a metrics row to its row identity."""
    matrix = np.full((len(row_keys), len(k_values)), np.nan, dtype=np.float64)
    by_key = {(row_key_fn(row), int(row["k"])): float(row[key]) for row in rows}
    for i, rk in enumerate(row_keys):
        for j, k in enumerate(k_values):
            matrix[i, j] = by_key.get((rk, k), np.nan)
    return matrix


def annotate_heatmap(ax, matrix):
    finite = matrix[np.isfinite(matrix)]
    threshold = np.nanmedian(finite) if finite.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                color = "white" if val > threshold else "black"
                ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=7, color=color)


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


def run_icl_example(args, icl_index, train_tasks, test_tasks, all_tasks, fvs_by_task, fv_pca, max_k):
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
            # Fit on train-task samples only; evaluate on held-out test tasks.
            x_train, y_train, train_counts = build_split_matrices(
                train_tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k
            )
            x_test, y_test, test_counts = build_split_matrices(
                test_tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k
            )
            expected_dim = 2 * k
            if x_train.shape[1] != expected_dim or y_train.shape[1] != expected_dim:
                raise ValueError(f"ICL {icl_index} {token_role} k={k}: train matrices are not {expected_dim}D")
            if x_test.shape[1] != expected_dim or y_test.shape[1] != expected_dim:
                raise ValueError(f"ICL {icl_index} {token_role} k={k}: test matrices are not {expected_dim}D")

            model = LinearRegression()
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            row = {
                "icl_example_index": int(icl_index),
                "token_role": token_role,
                "k": int(k),
                "feature_dim": int(expected_dim),
                "target_dim": int(expected_dim),
                "train_sample_count": int(x_train.shape[0]),
                "test_sample_count": int(x_test.shape[0]),
                "train_mse": mse(y_train, y_train_pred),
                "test_mse": mse(y_test, y_test_pred),
                "train_mean_squared_l2": mean_squared_l2(y_train, y_train_pred),
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
                    "train_tasks": train_tasks,
                    "test_tasks": test_tasks,
                    "train_counts_by_task": train_counts,
                    "test_counts_by_task": test_counts,
                    "feature_definition": "[(activation - activation_pca_mean) @ activation_pcs[:k], (activation - fv_pca_mean) @ fv_pcs[:k]]",
                    "target_definition": "[(function_vector - activation_pca_mean) @ activation_pcs[:k], (function_vector - fv_pca_mean) @ fv_pcs[:k]]",
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
    context = f"layer {args.layer} · {args.split} split"
    heatmap_png, heatmap_pdf = plot_train_test_heatmaps(
        rows, args.token_roles, role_labels, args.k_values, icl_output_dir,
        row_key_fn=lambda r: r["token_role"], ylabel="Token role",
        suptitle=f"ICL example {icl_index}: activation → function-vector regression in joint PCA space ({context})",
    )
    ratio_png, ratio_pdf = plot_ratio_heatmap(
        rows, args.token_roles, role_labels, args.k_values, icl_output_dir,
        row_key_fn=lambda r: r["token_role"], ylabel="Token role",
        suptitle=f"ICL example {icl_index}: activation → function-vector overfit ratio ({context})",
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

    all_rows = []
    icl_configs = []
    for icl_index in args.icl_example_indices:
        rows, icl_config = run_icl_example(
            args, icl_index, train_tasks, test_tasks, all_tasks, fvs_by_task, fv_pca, max_k
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
    combined_context = f"layer {args.layer} · {args.split} split"
    combined_heatmap_png, combined_heatmap_pdf = plot_train_test_heatmaps(
        all_rows, combined_row_keys, combined_row_labels, args.k_values, args.output_dir,
        row_key_fn=combined_key_fn, ylabel="ICL example / token role", prefix="combined_",
        suptitle=f"Activation → function-vector regression in joint PCA space, {icl_span} ({combined_context})",
    )
    combined_ratio_png, combined_ratio_pdf = plot_ratio_heatmap(
        all_rows, combined_row_keys, combined_row_labels, args.k_values, args.output_dir,
        row_key_fn=combined_key_fn, ylabel="ICL example / token role", prefix="combined_",
        suptitle=f"Activation → function-vector overfit ratio, {icl_span} ({combined_context})",
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
        "per_icl_runs": icl_configs,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(combined_csv)
    print(combined_heatmap_png)
    print(combined_ratio_png)


if __name__ == "__main__":
    main()
