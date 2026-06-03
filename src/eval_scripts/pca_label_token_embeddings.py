#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA on saved layer-0 embedding activations for every ICL label token "
            "from two tasks, then plot the first two PCs. Label tokens that appear in "
            "both tasks are colored with a shared-token color."
        )
    )
    parser.add_argument("--activations_root", type=Path, required=True, help="Root containing <task>/<split>/index.json.")
    parser.add_argument("--tasks", nargs=2, default=["antonym", "synonym"], help="Exactly two task names to compare.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--layer_index", type=int, default=0, help="Activation slice to analyze. Default 0 is the embedding slice.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_records_per_task",
        type=int,
        default=None,
        help="Optional deterministic subsample per task after loading all label-token records.",
    )
    parser.add_argument(
        "--allow_non_embedding_layer0",
        action="store_true",
        help="Allow running on activation sets that were not extracted with --include_embeddings.",
    )
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_label_token(meta):
    label = meta.get("demo_output")
    if label is None:
        label = meta.get("token_text", "")
    return str(label).strip()


def load_task_label_activations(activations_root, task, split, layer_index, require_embeddings=True):
    task_split_dir = activations_root / task / split
    index_path = task_split_dir / "index.json"
    index = load_json(index_path)
    include_embeddings = bool(index.get("config", {}).get("include_embeddings", False))
    if require_embeddings and not include_embeddings:
        raise ValueError(
            f"{index_path} was not extracted with include_embeddings=true. "
            "Use an activations_root such as results/residual_activations/gptj_with_embeddings, "
            "or pass --allow_non_embedding_layer0 if you intentionally want transformer block 0."
        )

    activations = []
    metadata = []
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = task_split_dir / shard_path
        data = torch_load_trusted(shard_path, map_location="cpu")
        shard_acts = data["activations"]
        shard_metadata = data["metadata"]
        if len(shard_metadata) != shard_acts.shape[0]:
            raise ValueError(f"Metadata/activation count mismatch in {shard_path}")
        if layer_index < 0 or layer_index >= shard_acts.shape[1]:
            raise IndexError(f"--layer_index {layer_index} is outside activation shape {tuple(shard_acts.shape)}")

        selected = [i for i, meta in enumerate(shard_metadata) if meta.get("token_role") == "label_token"]
        if selected:
            activations.append(shard_acts[selected, layer_index, :].float())
            metadata.extend(shard_metadata[i] for i in selected)

    if not activations:
        raise ValueError(f"No label-token activations found for {task}/{split}")
    return torch.cat(activations, dim=0).numpy(), metadata, include_embeddings


def subsample_records(x, metadata, max_records, seed):
    if max_records is None:
        return x, metadata
    if max_records <= 0:
        raise ValueError("--max_records_per_task must be positive")
    n_select = min(max_records, x.shape[0])
    rng = np.random.default_rng(seed)
    selected = np.array(sorted(rng.choice(np.arange(x.shape[0]), size=n_select, replace=False)), dtype=np.int64)
    return x[selected], [metadata[i] for i in selected]


def load_all_label_activations(args):
    xs = []
    labels = []
    metadata = []
    task_token_sets = {}
    include_embeddings_by_task = {}

    for task_idx, task in enumerate(args.tasks):
        x_task, task_metadata, include_embeddings = load_task_label_activations(
            args.activations_root,
            task,
            args.split,
            args.layer_index,
            require_embeddings=not args.allow_non_embedding_layer0,
        )
        x_task, task_metadata = subsample_records(
            x_task,
            task_metadata,
            args.max_records_per_task,
            seed=args.seed + task_idx,
        )
        xs.append(x_task)
        labels.extend([task_idx] * x_task.shape[0])
        metadata.extend(task_metadata)
        task_token_sets[task] = {normalize_label_token(meta) for meta in task_metadata}
        include_embeddings_by_task[task] = include_embeddings

    shared_tokens = sorted(set.intersection(*(task_token_sets[task] for task in args.tasks)))
    return np.concatenate(xs, axis=0), np.array(labels), metadata, shared_tokens, include_embeddings_by_task


def build_point_rows(projections, labels, metadata, tasks, shared_tokens):
    shared_token_set = set(shared_tokens)
    rows = []
    for i, (pc1, pc2) in enumerate(projections):
        meta = metadata[i]
        label_token = normalize_label_token(meta)
        task = tasks[int(labels[i])]
        color_group = "shared_token" if label_token in shared_token_set else task
        rows.append(
            {
                "pc1": float(pc1),
                "pc2": float(pc2),
                "task": task,
                "color_group": color_group,
                "label_token": label_token,
                "icl_example_index": meta.get("icl_example_index"),
                "prompt_index": meta.get("prompt_index"),
                "query_input": meta.get("query_input"),
                "query_output": meta.get("query_output"),
                "demo_input": meta.get("demo_input"),
                "demo_output": meta.get("demo_output"),
                "token_text": meta.get("token_text"),
                "token_position": meta.get("token_position"),
            }
        )
    return rows


def write_point_csv(rows, output_dir):
    csv_path = output_dir / "label_token_embedding_pca_points.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_pca(rows, tasks, output_dir, explained_variance_ratio, title):
    colors = {
        tasks[0]: "#4c78a8",
        tasks[1]: "#f58518",
        "shared_token": "#54a24b",
    }
    labels = {
        tasks[0]: f"{tasks[0]} only",
        tasks[1]: f"{tasks[1]} only",
        "shared_token": "shared token",
    }

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    for group in [tasks[0], tasks[1], "shared_token"]:
        group_rows = [row for row in rows if row["color_group"] == group]
        if not group_rows:
            continue
        ax.scatter(
            [row["pc1"] for row in group_rows],
            [row["pc2"] for row in group_rows],
            s=14,
            alpha=0.55,
            color=colors[group],
            label=f"{labels[group]} (n={len(group_rows)})",
            linewidths=0,
        )

    ax.axhline(0, color="0.75", linewidth=1)
    ax.axvline(0, color="0.75", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% var.)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    plot_path = output_dir / "label_token_embedding_pca_scatter.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def default_output_dir(args):
    layer_tag = "embedding_layer0" if args.layer_index == 0 else f"layer{args.layer_index}"
    return args.activations_root / "pca" / f"{args.tasks[0]}_vs_{args.tasks[1]}_label_tokens_{args.split}_{layer_tag}"


def main():
    args = parse_args()
    if len(args.tasks) != 2:
        raise ValueError("This script expects exactly two tasks")

    output_dir = args.output_dir or default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations, labels, metadata, shared_tokens, include_embeddings_by_task = load_all_label_activations(args)
    print(f"Loaded {activations.shape[0]} label-token activations with resid_dim={activations.shape[1]}")
    print(f"Shared label tokens: {len(shared_tokens)}")

    pca = PCA(n_components=2, random_state=args.seed)
    projections = pca.fit_transform(activations)
    rows = build_point_rows(projections, labels, metadata, args.tasks, shared_tokens)

    csv_path = write_point_csv(rows, output_dir)
    plot_path = plot_pca(
        rows,
        args.tasks,
        output_dir,
        pca.explained_variance_ratio_,
        title=f"Label-token embedding PCA ({args.tasks[0]} vs {args.tasks[1]}, {args.split})",
    )

    config = {
        "activations_root": str(args.activations_root),
        "tasks": args.tasks,
        "split": args.split,
        "layer_index": args.layer_index,
        "max_records_per_task": args.max_records_per_task,
        "n_records": int(activations.shape[0]),
        "resid_dim": int(activations.shape[1]),
        "include_embeddings_by_task": include_embeddings_by_task,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "shared_tokens": shared_tokens,
    }
    config_path = output_dir / "label_token_embedding_pca_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(plot_path)
    print(csv_path)
    print(config_path)


if __name__ == "__main__":
    main()
