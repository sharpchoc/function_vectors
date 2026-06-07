#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression


DEFAULT_TOKEN_ROLES = [
    "pre_label_token",
    "first_label_token",
    "last_label_token",
    "last_prompt_token",
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
            "spanned by token-role activation PCs and function-vector PCs."
        )
    )
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument(
        "--activations_root",
        type=Path,
        default=Path("results/residual_activations/gptj_56tasks_170prompts_4tokens"),
    )
    parser.add_argument("--pca_root", type=Path, default=Path("results/pca_abstractive_fv_activation_scatter"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/joint_pca_activation_to_fv_regression"))
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


def load_role_activations(activations_root, task, split, layer, token_role):
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
        selected = [i for i, meta in enumerate(shard_metadata) if meta.get("token_role") == token_role]
        if selected:
            chunks.append(activations[selected, layer, :].float())
            metadata.extend(shard_metadata[i] for i in selected)
    if not chunks:
        raise ValueError(f"No {token_role} activations found for {task}/{split}")
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


def write_metrics_csv(rows, output_dir):
    path = output_dir / "regression_metrics.csv"
    fieldnames = [
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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def metric_matrix(rows, token_roles, k_values, key):
    matrix = np.full((len(token_roles), len(k_values)), np.nan, dtype=np.float64)
    by_key = {(row["token_role"], int(row["k"])): float(row[key]) for row in rows}
    for i, token_role in enumerate(token_roles):
        for j, k in enumerate(k_values):
            matrix[i, j] = by_key[(token_role, k)]
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


def plot_train_test_heatmaps(rows, token_roles, k_values, output_dir):
    train_matrix = metric_matrix(rows, token_roles, k_values, "train_mse")
    test_matrix = metric_matrix(rows, token_roles, k_values, "test_mse")
    vmin = float(np.nanmin([train_matrix.min(), test_matrix.min()]))
    vmax = float(np.nanmax([train_matrix.max(), test_matrix.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, matrix, title in zip(axes, [train_matrix, test_matrix], ["Train-task MSE", "Held-out test-task MSE"]):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(token_roles)))
        ax.set_yticklabels([TOKEN_TITLES.get(role, role) for role in token_roles])
        ax.set_xlabel("k")
        annotate_heatmap(ax, matrix)
    axes[0].set_ylabel("Token role")
    fig.colorbar(im, ax=axes, shrink=0.88, label="MSE in joint PCA space")
    png = output_dir / "mse_heatmap_train_test.png"
    pdf = output_dir / "mse_heatmap_train_test.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_ratio_heatmap(rows, token_roles, k_values, output_dir):
    train_matrix = metric_matrix(rows, token_roles, k_values, "train_mse")
    test_matrix = metric_matrix(rows, token_roles, k_values, "test_mse")
    ratio = test_matrix / np.maximum(train_matrix, 1e-12)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    im = ax.imshow(ratio, aspect="auto", cmap="magma")
    ax.set_title("Held-out test MSE / train MSE")
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(range(len(token_roles)))
    ax.set_yticklabels([TOKEN_TITLES.get(role, role) for role in token_roles])
    ax.set_xlabel("k")
    ax.set_ylabel("Token role")
    annotate_heatmap(ax, ratio)
    fig.colorbar(im, ax=ax, shrink=0.88, label="MSE ratio")
    png = output_dir / "mse_ratio_heatmap.png"
    pdf = output_dir / "mse_ratio_heatmap.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, train_tasks, test_tasks = load_task_manifest(args)
    all_tasks = train_tasks + test_tasks
    fv_pca = load_pca_artifact(args.pca_root / "fv_pca.pt")
    max_fv_components = fv_pca["components"].shape[0]
    max_k = max(args.k_values)
    if max_k > max_fv_components:
        raise ValueError(f"Requested max k={max_k}, but FV PCA only has {max_fv_components} components")

    fvs_by_task = {task: load_function_vector(args.fv_root, task) for task in all_tasks}
    rows = []
    sample_counts_by_role = {}
    pca_artifacts_by_role = {}

    for token_role in args.token_roles:
        activation_pca = load_pca_artifact(args.pca_root / f"activation_pca_{token_role}.pt")
        max_activation_components = activation_pca["components"].shape[0]
        if max_k > max_activation_components:
            raise ValueError(
                f"Requested max k={max_k}, but {token_role} activation PCA only has {max_activation_components} components"
            )
        pca_artifacts_by_role[token_role] = {
            "activation_pca_components": int(max_activation_components),
            "activation_pca_fit_tasks": activation_pca.get("fit_tasks"),
        }

        activations_by_task = {}
        for task in all_tasks:
            activations_by_task[task], _ = load_role_activations(
                args.activations_root, task, args.split, args.layer, token_role
            )

        for k in args.k_values:
            x_train, y_train, train_counts = build_split_matrices(
                train_tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k
            )
            x_test, y_test, test_counts = build_split_matrices(
                test_tasks, activations_by_task, fvs_by_task, activation_pca, fv_pca, k
            )
            expected_dim = 2 * k
            if x_train.shape[1] != expected_dim or y_train.shape[1] != expected_dim:
                raise ValueError(f"{token_role} k={k}: train matrices are not {expected_dim}D")
            if x_test.shape[1] != expected_dim or y_test.shape[1] != expected_dim:
                raise ValueError(f"{token_role} k={k}: test matrices are not {expected_dim}D")

            model = LinearRegression()
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            row = {
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
                args.output_dir,
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

    metrics_csv = write_metrics_csv(rows, args.output_dir)
    write_json(args.output_dir / "regression_metrics.json", rows)
    heatmap_png, heatmap_pdf = plot_train_test_heatmaps(rows, args.token_roles, args.k_values, args.output_dir)
    ratio_png, ratio_pdf = plot_ratio_heatmap(rows, args.token_roles, args.k_values, args.output_dir)

    run_config = {
        "task_manifest": str(args.task_manifest),
        "manifest": manifest,
        "fv_root": str(args.fv_root),
        "activations_root": str(args.activations_root),
        "pca_root": str(args.pca_root),
        "output_dir": str(args.output_dir),
        "layer": args.layer,
        "split": args.split,
        "token_roles": args.token_roles,
        "k_values": args.k_values,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "fv_pca_fit_tasks": fv_pca.get("fit_tasks"),
        "fv_pca_components": int(max_fv_components),
        "activation_pca_by_role": pca_artifacts_by_role,
        "sample_counts_by_role": sample_counts_by_role,
        "metrics_row_count": len(rows),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(args.output_dir / "regression_metrics.json"),
        "mse_heatmap_png": str(heatmap_png),
        "mse_heatmap_pdf": str(heatmap_pdf),
        "mse_ratio_heatmap_png": str(ratio_png),
        "mse_ratio_heatmap_pdf": str(ratio_pdf),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(metrics_csv)
    print(heatmap_png)
    print(ratio_png)


if __name__ == "__main__":
    main()
