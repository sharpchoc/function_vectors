#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train layer-wise linear probes for task separability from saved residual activations."
    )
    parser.add_argument("--activations_root", type=Path, required=True, help="Root containing <task>/<split>/index.json.")
    parser.add_argument("--tasks", nargs=2, default=["antonym", "synonym"], help="Exactly two task names to classify.")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--token_role", choices=["final_token", "label_token"], default="final_token")
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=None, help="ICL label-token positions to probe, e.g. 1 2 ... 10. Only used with --token_role label_token.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Defaults to <activations_root>/probes.")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength for logistic regression.")
    parser.add_argument(
        "--max_train_records_per_task",
        type=int,
        default=None,
        help="If set, deterministically subsample this many training records per task after loading activations. Test records are never subsampled.",
    )
    return parser.parse_args()


def load_index(task_dir):
    index_path = task_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    with open(index_path, "r") as f:
        return json.load(f)


def metadata_matches(meta, token_role, icl_example_index):
    if meta.get("token_role") != token_role:
        return False
    if token_role == "label_token" and icl_example_index is not None:
        return int(meta.get("icl_example_index")) == int(icl_example_index)
    return True


def load_split(activations_root, task, split, token_role, icl_example_index=None):
    index = load_index(activations_root / task / split)
    acts = []
    metadata = []

    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = activations_root / task / split / shard_path
        data = torch.load(shard_path, map_location="cpu")
        shard_acts = data["activations"]
        shard_metadata = data["metadata"]
        if len(shard_metadata) != shard_acts.shape[0]:
            raise ValueError(f"Metadata/activation count mismatch in {shard_path}")

        selected_indices = [
            i for i, meta in enumerate(shard_metadata) if metadata_matches(meta, token_role, icl_example_index)
        ]
        if selected_indices:
            acts.append(shard_acts[selected_indices])
            metadata.extend(shard_metadata[i] for i in selected_indices)

    descriptor = token_role if icl_example_index is None else f"{token_role}[{icl_example_index}]"
    if not acts:
        raise ValueError(f"No records matching {descriptor} for {task}/{split}")

    return torch.cat(acts, dim=0).float().numpy(), metadata


def load_dataset(activations_root, tasks, split, token_role, icl_example_index=None):
    xs = []
    ys = []
    all_metadata = []

    for label, task in enumerate(tasks):
        activations, metadata = load_split(activations_root, task, split, token_role, icl_example_index)
        xs.append(activations)
        ys.append(np.full(activations.shape[0], label, dtype=np.int64))
        all_metadata.extend(metadata)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if x.ndim != 3:
        raise ValueError(f"Expected activations shaped [records, layers, resid_dim], got {x.shape}")
    return x, y, all_metadata


def subsample_train_records(x, y, metadata, max_records_per_task, seed):
    if max_records_per_task is None:
        return x, y, metadata, {int(label): int((y == label).sum()) for label in np.unique(y)}
    if max_records_per_task <= 0:
        raise ValueError("--max_train_records_per_task must be positive")

    rng = np.random.default_rng(seed)
    selected_indices = []
    selected_counts = {}
    for label in sorted(np.unique(y)):
        label_indices = np.where(y == label)[0]
        n_select = min(max_records_per_task, len(label_indices))
        selected_counts[int(label)] = int(n_select)
        selected_indices.extend(rng.choice(label_indices, size=n_select, replace=False).tolist())

    selected_indices = np.array(sorted(selected_indices), dtype=np.int64)
    return x[selected_indices], y[selected_indices], [metadata[i] for i in selected_indices], selected_counts


def train_layer_probe(x_train, y_train, x_test, y_test, layer, args):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=args.seed,
            solver="lbfgs",
        ),
    )
    clf.fit(x_train[:, layer, :], y_train)

    train_probs = clf.predict_proba(x_train[:, layer, :])
    test_probs = clf.predict_proba(x_test[:, layer, :])
    train_pred = np.argmax(train_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)

    return {
        "layer": layer,
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_log_loss": float(log_loss(y_train, train_probs)),
        "test_log_loss": float(log_loss(y_test, test_probs)),
    }


def write_metrics(metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "layer_probe_metrics.json"
    csv_path = output_dir / "layer_probe_metrics.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    return json_path, csv_path


def plot_accuracies(metrics, output_dir, title):
    layers = np.array([row["layer"] for row in metrics])
    test_acc = np.array([row["test_accuracy"] for row in metrics])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, test_acc, marker="o", linewidth=2, label="Test")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test accuracy")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "layer_probe_test_accuracy.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_combined_accuracies(series, output_dir, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, metrics in series:
        layers = np.array([row["layer"] for row in metrics])
        test_acc = np.array([row["test_accuracy"] for row in metrics])
        ax.plot(layers, test_acc, marker="o", linewidth=1.8, markersize=4, label=label)

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    if series:
        ax.set_xticks(np.array([row["layer"] for row in series[0][1]]))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_layer_probe_test_accuracy.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_probe(args, token_role, icl_example_index, output_dir):
    descriptor = token_role if icl_example_index is None else f"{token_role}_icl{icl_example_index}"

    print(f"Loading train activations for {descriptor}")
    x_train, y_train, train_metadata = load_dataset(
        args.activations_root, args.tasks, args.train_split, token_role, icl_example_index
    )
    raw_train_records = len(y_train)
    raw_train_counts = dict(zip(args.tasks, np.bincount(y_train, minlength=2).tolist()))
    x_train, y_train, train_metadata, sampled_train_counts = subsample_train_records(
        x_train, y_train, train_metadata, args.max_train_records_per_task, args.seed
    )

    print(f"Loading test activations for {descriptor}")
    x_test, y_test, test_metadata = load_dataset(
        args.activations_root, args.tasks, args.test_split, token_role, icl_example_index
    )

    n_layers = x_train.shape[1]
    if x_test.shape[1] != n_layers or x_test.shape[2] != x_train.shape[2]:
        raise ValueError(f"Train/test activation shape mismatch: train={x_train.shape}, test={x_test.shape}")

    print(f"Train records used: {len(y_train)} / {raw_train_records}; test records: {len(y_test)}; layers: {n_layers}; resid_dim: {x_train.shape[2]}")
    print(f"Raw train task counts: {raw_train_counts}")
    print(f"Train task counts used: {dict(zip(args.tasks, np.bincount(y_train, minlength=2).tolist()))}")
    print(f"Test task counts: {dict(zip(args.tasks, np.bincount(y_test, minlength=2).tolist()))}")

    metrics = []
    for layer in range(n_layers):
        row = train_layer_probe(x_train, y_train, x_test, y_test, layer, args)
        metrics.append(row)
        print(
            f"{descriptor} layer {layer:2d}: train_acc={row['train_accuracy']:.4f}, "
            f"test_acc={row['test_accuracy']:.4f}, test_log_loss={row['test_log_loss']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "probe_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "tasks": args.tasks,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "token_role": token_role,
                "icl_example_index": icl_example_index,
                "activations_root": str(args.activations_root),
                "raw_train_records": int(raw_train_records),
                "train_records_used": int(len(train_metadata)),
                "test_records": int(len(test_metadata)),
                "raw_train_counts": raw_train_counts,
                "sampled_train_counts_by_label": sampled_train_counts,
                "max_train_records_per_task": args.max_train_records_per_task,
                "n_layers": int(n_layers),
                "resid_dim": int(x_train.shape[2]),
                "C": args.C,
                "max_iter": args.max_iter,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    json_path, csv_path = write_metrics(metrics, output_dir)
    title_suffix = token_role if icl_example_index is None else f"label token, ICL example {icl_example_index}"
    plot_path = plot_accuracies(
        metrics,
        output_dir,
        title=f"Task separability probe ({args.tasks[0]} vs {args.tasks[1]}, {title_suffix})",
    )

    print(json_path)
    print(csv_path)
    print(plot_path)
    return descriptor, metrics


def default_output_dir(args, token_role, icl_example_index):
    descriptor = token_role if icl_example_index is None else f"{token_role}_icl{icl_example_index}"
    return args.activations_root / "probes" / f"{args.tasks[0]}_vs_{args.tasks[1]}_{descriptor}"


def main():
    args = parse_args()
    if len(args.tasks) != 2:
        raise ValueError("This script expects exactly two tasks for binary classification")

    if args.token_role == "label_token":
        indices = args.icl_example_indices or list(range(1, 11))
    else:
        indices = [None]

    base_output_dir = args.output_dir
    combined_series = []
    combined_output_dir = base_output_dir or args.activations_root / "probes" / f"{args.tasks[0]}_vs_{args.tasks[1]}_{args.token_role}_combined"
    for icl_example_index in indices:
        output_dir = base_output_dir or default_output_dir(args, args.token_role, icl_example_index)
        if base_output_dir and len(indices) > 1:
            output_dir = base_output_dir / f"icl{icl_example_index}"
        descriptor, metrics = run_probe(args, args.token_role, icl_example_index, output_dir)
        combined_series.append((descriptor, metrics))

    if len(combined_series) > 1:
        combined_path = plot_combined_accuracies(
            combined_series,
            combined_output_dir,
            title=f"Task separability probes ({args.tasks[0]} vs {args.tasks[1]}, {args.token_role})",
        )
        print(combined_path)


if __name__ == "__main__":
    main()
