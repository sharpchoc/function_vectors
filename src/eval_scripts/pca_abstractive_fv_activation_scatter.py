#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


DEFAULT_TOKEN_ROLES = [
    "pre_label_token",
    "first_label_token",
    "last_label_token",
    "last_prompt_token",
]
TOKEN_TITLES = {
    "pre_label_token": "Pre-label token",
    "first_label_token": "First label token",
    "last_label_token": "Last label token",
    "last_prompt_token": "Last prompt token",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit PCA bases on train abstractive task function vectors and token-position "
            "activations, then project all abstractive tasks into FV-PC1 by activation-PC1 grids."
        )
    )
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument(
        "--activations_root",
        type=Path,
        default=Path("results/residual_activations/gptj_56tasks_170prompts_4tokens"),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=Path("results/pca_abstractive_fv_activation_scatter"))
    parser.add_argument("--train_tasks", nargs="+", default=None, help="Optional override for smoke tests.")
    parser.add_argument("--test_tasks", nargs="+", default=None, help="Optional override for smoke tests.")
    parser.add_argument("--alpha", type=float, default=0.36)
    parser.add_argument("--activation_point_size", type=float, default=9.0)
    parser.add_argument("--fv_point_size", type=float, default=170.0)
    parser.add_argument("--seed", type=int, default=42)
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
    return fv.detach().float().cpu().reshape(-1)


def fit_pca(x, n_components, label):
    if x.ndim != 2:
        raise ValueError(f"{label} PCA input must be 2D, got shape {x.shape}")
    max_components = min(x.shape[0], x.shape[1])
    if n_components > max_components:
        raise ValueError(f"{label} PCA requested {n_components} components, but max is {max_components}")
    pca = PCA(n_components=n_components)
    pca.fit(x)
    return pca


def pca_to_artifact(pca, fit_tasks, projected_tasks, extra):
    artifact = {
        "components": torch.from_numpy(pca.components_).float(),
        "mean": torch.from_numpy(pca.mean_).float(),
        "explained_variance": torch.from_numpy(pca.explained_variance_).float(),
        "explained_variance_ratio": torch.from_numpy(pca.explained_variance_ratio_).float(),
        "fit_tasks": fit_tasks,
        "projected_tasks": projected_tasks,
        **extra,
    }
    return artifact


def pca_json_summary(pca, fit_tasks, projected_tasks, extra):
    return {
        "n_components": int(pca.n_components_),
        "input_feature_count": int(pca.n_features_in_),
        "fit_tasks": fit_tasks,
        "projected_tasks": projected_tasks,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        **extra,
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


def build_task_colors(tasks):
    palettes = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for palette in palettes:
        colors.extend(palette(i) for i in range(palette.N))
    return {task: colors[i % len(colors)] for i, task in enumerate(tasks)}


def write_coordinate_csv(rows, output_dir):
    csv_path = output_dir / "pca_projection_points.csv"
    fieldnames = [
        "token_role",
        "task",
        "task_group",
        "point_type",
        "x_fv_pc1_raw_dot",
        "y_activation_pc1_raw_dot",
        "prompt_index",
        "query_source_index",
        "query_input",
        "query_output",
        "demo_source_index",
        "demo_input",
        "demo_output",
        "token_position",
        "token_text",
        "token_label",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_grid(plot_rows, token_roles, tasks, task_groups, task_colors, args, output_dir):
    n_cols = 2
    n_rows = int(np.ceil(len(token_roles) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.8, 8.4), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax_idx, token_role in enumerate(token_roles):
        ax = axes_flat[ax_idx]
        role_rows = [row for row in plot_rows if row["token_role"] == token_role]
        for task in tasks:
            task_rows = [row for row in role_rows if row["task"] == task and row["point_type"] == "activation"]
            if task_rows:
                marker = "o" if task_groups[task] == "train" else "^"
                ax.scatter(
                    [row["x_fv_pc1_raw_dot"] for row in task_rows],
                    [row["y_activation_pc1_raw_dot"] for row in task_rows],
                    s=args.activation_point_size,
                    alpha=args.alpha,
                    color=task_colors[task],
                    marker=marker,
                    linewidths=0,
                )
            fv_rows = [row for row in role_rows if row["task"] == task and row["point_type"] == "function_vector"]
            for row in fv_rows:
                marker = "X" if task_groups[task] == "train" else "*"
                ax.scatter(
                    row["x_fv_pc1_raw_dot"],
                    row["y_activation_pc1_raw_dot"],
                    s=args.fv_point_size,
                    color=task_colors[task],
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=20,
                )
        ax.axhline(0, color="0.75", linewidth=0.9)
        ax.axvline(0, color="0.75", linewidth=0.9)
        ax.grid(alpha=0.18)
        ax.set_title(TOKEN_TITLES.get(token_role, token_role))
        ax.set_xlabel("Raw dot with FV PC1")
        ax.set_ylabel(f"Raw dot with {TOKEN_TITLES.get(token_role, token_role)} activation PC1")

    for empty_ax in axes_flat[len(token_roles):]:
        empty_ax.axis("off")

    marker_handles = [
        mlines.Line2D([], [], color="0.2", marker="o", linestyle="None", markersize=6, label="train activation"),
        mlines.Line2D([], [], color="0.2", marker="^", linestyle="None", markersize=6, label="test activation"),
        mlines.Line2D([], [], color="0.2", marker="X", linestyle="None", markersize=9, label="train FV"),
        mlines.Line2D([], [], color="0.2", marker="*", linestyle="None", markersize=11, label="test FV"),
    ]
    task_handles = [
        mlines.Line2D(
            [],
            [],
            color=task_colors[task],
            marker="s",
            linestyle="None",
            markersize=6,
            label=f"{task} ({task_groups[task]})",
        )
        for task in tasks
    ]
    fig.legend(
        handles=marker_handles + task_handles,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        frameon=False,
        fontsize=7.2,
        ncol=1,
    )
    fig.suptitle(f"Abstractive tasks at layer {args.layer}: FV PC1 vs token-role activation PC1", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.78, 0.97))

    png_path = output_dir / f"layer_{args.layer:02d}_{args.split}_abstractive_fv_activation_pca_grid.png"
    pdf_path = output_dir / f"layer_{args.layer:02d}_{args.split}_abstractive_fv_activation_pca_grid.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, train_tasks, test_tasks = load_task_manifest(args)
    all_tasks = train_tasks + test_tasks
    task_groups = {task: "train" for task in train_tasks}
    task_groups.update({task: "test" for task in test_tasks})
    task_colors = build_task_colors(all_tasks)

    fvs = {task: load_function_vector(args.fv_root, task).numpy() for task in all_tasks}
    fv_fit = np.stack([fvs[task] for task in train_tasks], axis=0)
    fv_pca = fit_pca(fv_fit, args.n_components, "FV")
    torch.save(
        pca_to_artifact(
            fv_pca,
            fit_tasks=train_tasks,
            projected_tasks=all_tasks,
            extra={"kind": "function_vector", "raw_dot_projection": True},
        ),
        args.output_dir / "fv_pca.pt",
    )
    write_json(
        args.output_dir / "fv_pca.json",
        pca_json_summary(
            fv_pca,
            fit_tasks=train_tasks,
            projected_tasks=all_tasks,
            extra={"kind": "function_vector", "raw_dot_projection": True},
        ),
    )

    fv_pc1 = fv_pca.components_[0]
    activation_pcas = {}
    activation_metadata = {}
    activation_arrays = {}
    role_record_counts = {}

    for token_role in args.token_roles:
        fit_chunks = []
        for task in train_tasks:
            x_task, _ = load_role_activations(args.activations_root, task, args.split, args.layer, token_role)
            fit_chunks.append(x_task)
        x_fit = np.concatenate(fit_chunks, axis=0)
        role_record_counts[token_role] = int(x_fit.shape[0])
        pca = fit_pca(x_fit, args.n_components, f"{token_role} activation")
        activation_pcas[token_role] = pca
        torch.save(
            pca_to_artifact(
                pca,
                fit_tasks=train_tasks,
                projected_tasks=all_tasks,
                extra={
                    "kind": "activation",
                    "token_role": token_role,
                    "layer": args.layer,
                    "split": args.split,
                    "raw_dot_projection": True,
                    "fit_record_count": int(x_fit.shape[0]),
                },
            ),
            args.output_dir / f"activation_pca_{token_role}.pt",
        )
        write_json(
            args.output_dir / f"activation_pca_{token_role}.json",
            pca_json_summary(
                pca,
                fit_tasks=train_tasks,
                projected_tasks=all_tasks,
                extra={
                    "kind": "activation",
                    "token_role": token_role,
                    "layer": args.layer,
                    "split": args.split,
                    "raw_dot_projection": True,
                    "fit_record_count": int(x_fit.shape[0]),
                },
            ),
        )

        activation_arrays[token_role] = {}
        activation_metadata[token_role] = {}
        for task in all_tasks:
            x_task, meta_task = load_role_activations(args.activations_root, task, args.split, args.layer, token_role)
            activation_arrays[token_role][task] = x_task
            activation_metadata[token_role][task] = meta_task

    rows = []
    for token_role in args.token_roles:
        activation_pc1 = activation_pcas[token_role].components_[0]
        for task in all_tasks:
            x_task = activation_arrays[token_role][task]
            x_coords = x_task @ fv_pc1
            y_coords = x_task @ activation_pc1
            for x, y, meta in zip(x_coords, y_coords, activation_metadata[token_role][task]):
                rows.append(
                    {
                        "token_role": token_role,
                        "task": task,
                        "task_group": task_groups[task],
                        "point_type": "activation",
                        "x_fv_pc1_raw_dot": float(x),
                        "y_activation_pc1_raw_dot": float(y),
                        "prompt_index": meta.get("prompt_index"),
                        "query_source_index": meta.get("query_source_index"),
                        "query_input": meta.get("query_input"),
                        "query_output": meta.get("query_output"),
                        "demo_source_index": meta.get("demo_source_index"),
                        "demo_input": meta.get("demo_input"),
                        "demo_output": meta.get("demo_output"),
                        "token_position": meta.get("token_position"),
                        "token_text": meta.get("token_text"),
                        "token_label": meta.get("token_label"),
                    }
                )

            fv = fvs[task]
            rows.append(
                {
                    "token_role": token_role,
                    "task": task,
                    "task_group": task_groups[task],
                    "point_type": "function_vector",
                    "x_fv_pc1_raw_dot": float(fv @ fv_pc1),
                    "y_activation_pc1_raw_dot": float(fv @ activation_pc1),
                    "prompt_index": None,
                    "query_source_index": None,
                    "query_input": None,
                    "query_output": None,
                    "demo_source_index": None,
                    "demo_input": None,
                    "demo_output": None,
                    "token_position": None,
                    "token_text": None,
                    "token_label": None,
                }
            )

    csv_path = write_coordinate_csv(rows, args.output_dir)
    png_path, pdf_path = plot_grid(rows, args.token_roles, all_tasks, task_groups, task_colors, args, args.output_dir)

    config = {
        "manifest": manifest,
        "task_manifest": str(args.task_manifest),
        "fv_root": str(args.fv_root),
        "activations_root": str(args.activations_root),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "layer": args.layer,
        "token_roles": args.token_roles,
        "n_components": args.n_components,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "all_tasks": all_tasks,
        "fv_pca_fit_shape": list(fv_fit.shape),
        "activation_pca_fit_record_counts": role_record_counts,
        "projected_fv_count": len(all_tasks),
        "projected_activation_count_per_role": {
            role: int(sum(activation_arrays[role][task].shape[0] for task in all_tasks))
            for role in args.token_roles
        },
        "coordinate_csv": str(csv_path),
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
    }
    write_json(args.output_dir / "run_config.json", config)
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
