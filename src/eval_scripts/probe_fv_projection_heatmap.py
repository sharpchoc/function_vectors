#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_fv_projection_scatter import compute_task_fv
from src.utils.model_utils import load_gpt_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train binary probes on residual activations projected to the 2D space "
            "spanned by two task function vectors. Repeats over ICL label-token "
            "positions and layers, then writes an accuracy heatmap."
        )
    )
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument("--activations_root", type=Path, required=True)
    parser.add_argument("--tasks", nargs=2, default=["antonym", "synonym"])
    parser.add_argument("--x_task", type=str, default="synonym")
    parser.add_argument("--y_task", type=str, default="antonym")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--recompute_function_vectors", action="store_true")
    parser.add_argument("--revision", type=str, default=None)
    return parser.parse_args()


def load_saved_function_vector(task, fv_root, n_top_heads):
    fv_path = fv_root / task / f"{task}_function_vector.pt"
    if not fv_path.exists():
        return None

    data = torch_load_trusted(fv_path, map_location="cpu")
    if isinstance(data, dict):
        saved_n_top_heads = data.get("n_top_heads")
        if saved_n_top_heads is not None and int(saved_n_top_heads) != int(n_top_heads):
            return None
        fv = data["function_vector"]
        top_heads = data.get("top_heads", [])
    else:
        fv = data
        top_heads = []
    return fv.detach().float().cpu().reshape(-1), top_heads


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def metadata_matches(meta, token_role, icl_example_index):
    if meta.get("token_role") != token_role:
        return False
    if icl_example_index is not None:
        return int(meta.get("icl_example_index")) == int(icl_example_index)
    return True


def load_task_activations(activations_root, task, split, token_role, icl_example_index):
    task_split_dir = activations_root / task / split
    index = load_json(task_split_dir / "index.json")
    activations = []
    metadata = []

    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = task_split_dir / shard_path
        data = torch_load_trusted(shard_path, map_location="cpu")
        shard_acts = data["activations"]
        shard_metadata = data["metadata"]
        selected = [
            i for i, meta in enumerate(shard_metadata) if metadata_matches(meta, token_role, icl_example_index)
        ]
        if selected:
            activations.append(shard_acts[selected].float())
            metadata.extend(shard_metadata[i] for i in selected)

    if not activations:
        descriptor = token_role if icl_example_index is None else f"{token_role}[{icl_example_index}]"
        raise ValueError(f"No {descriptor} activations found for {task}/{split}")
    return torch.cat(activations, dim=0), metadata


def load_all_activations(activations_root, tasks, split, token_role, icl_example_index):
    all_acts = []
    labels = []
    metadata = []
    for label, task in enumerate(tasks):
        acts, task_metadata = load_task_activations(activations_root, task, split, token_role, icl_example_index)
        all_acts.append(acts)
        labels.extend([label] * acts.shape[0])
        metadata.extend(task_metadata)
    return torch.cat(all_acts, dim=0), np.array(labels), metadata


def load_function_vectors(args):
    fvs = {}
    top_heads = {}
    missing = []

    if not args.recompute_function_vectors:
        for task in args.tasks:
            saved = load_saved_function_vector(task, args.fv_root, args.n_top_heads)
            if saved is None:
                missing.append(task)
            else:
                fvs[task], top_heads[task] = saved

    if args.recompute_function_vectors or missing:
        print("Loading model for function-vector reconstruction")
        torch.set_grad_enabled(False)
        model, _, model_config = load_gpt_model_and_tokenizer(
            args.model_name, device=args.device, revision=args.revision
        )
        model.eval()

        tasks_to_compute = args.tasks if args.recompute_function_vectors else missing
        for task in tasks_to_compute:
            fvs[task], top_heads[task] = compute_task_fv(
                task,
                args.fv_root,
                model,
                model_config,
                args.n_top_heads,
                save=True,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fvs, top_heads


def project_activations(activations, x_fv, y_fv):
    basis = torch.stack([x_fv, y_fv], dim=1)
    return torch.matmul(activations.float(), basis).numpy()


def train_probe(x_train, y_train, x_test, y_test, args):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=args.seed,
            solver="lbfgs",
        ),
    )
    clf.fit(x_train, y_train)

    train_probs = clf.predict_proba(x_train)
    test_probs = clf.predict_proba(x_test)
    train_pred = np.argmax(train_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)
    return {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_log_loss": float(log_loss(y_train, train_probs)),
        "test_log_loss": float(log_loss(y_test, test_probs)),
    }


def write_metrics(metrics, output_dir):
    json_path = output_dir / "fv_projection_probe_metrics.json"
    csv_path = output_dir / "fv_projection_probe_metrics.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    return json_path, csv_path


def plot_heatmap(accuracy_grid, icl_example_indices, output_dir, title):
    fig_width = max(9.5, accuracy_grid.shape[1] * 0.34)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    im = ax.imshow(accuracy_grid, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ICL label-token example")
    ax.set_xticks(np.arange(accuracy_grid.shape[1]))
    ax.set_yticks(np.arange(len(icl_example_indices)))
    ax.set_yticklabels([str(i) for i in icl_example_indices])
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test accuracy")
    fig.tight_layout()

    output_path = output_dir / "fv_projection_probe_accuracy_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def default_output_dir(args):
    return (
        args.activations_root
        / "probes"
        / f"{args.tasks[0]}_vs_{args.tasks[1]}_fv_projection_label_token_heatmap"
    )


def main():
    args = parse_args()
    if len(args.tasks) != 2:
        raise ValueError("This script expects exactly two tasks for binary classification")
    if args.x_task not in args.tasks or args.y_task not in args.tasks:
        raise ValueError("--x_task and --y_task must both be included in --tasks")

    output_dir = args.output_dir or default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    fvs, top_heads = load_function_vectors(args)
    x_fv = fvs[args.x_task]
    y_fv = fvs[args.y_task]

    metrics = []
    accuracy_rows = []
    n_layers = None
    for icl_example_index in args.icl_example_indices:
        print(f"Loading label-token activations for ICL example {icl_example_index}")
        train_acts, y_train, _ = load_all_activations(
            args.activations_root, args.tasks, args.train_split, "label_token", icl_example_index
        )
        test_acts, y_test, _ = load_all_activations(
            args.activations_root, args.tasks, args.test_split, "label_token", icl_example_index
        )

        if train_acts.ndim != 3 or test_acts.ndim != 3:
            raise ValueError(f"Expected activations [records, layers, resid_dim], got {train_acts.shape}, {test_acts.shape}")
        if train_acts.shape[1:] != test_acts.shape[1:]:
            raise ValueError(f"Train/test activation shape mismatch: train={train_acts.shape}, test={test_acts.shape}")
        if n_layers is None:
            n_layers = train_acts.shape[1]
        elif n_layers != train_acts.shape[1]:
            raise ValueError("Layer count changed across ICL examples")

        train_proj = project_activations(train_acts, x_fv, y_fv)
        test_proj = project_activations(test_acts, x_fv, y_fv)

        row_acc = []
        for layer in range(n_layers):
            row = train_probe(train_proj[:, layer, :], y_train, test_proj[:, layer, :], y_test, args)
            row.update(
                {
                    "icl_example_index": int(icl_example_index),
                    "layer": int(layer),
                    "train_records": int(len(y_train)),
                    "test_records": int(len(y_test)),
                }
            )
            metrics.append(row)
            row_acc.append(row["test_accuracy"])
            print(
                f"icl={icl_example_index:2d} layer={layer:2d}: "
                f"train_acc={row['train_accuracy']:.4f} test_acc={row['test_accuracy']:.4f}"
            )
        accuracy_rows.append(row_acc)

    accuracy_grid = np.array(accuracy_rows)
    json_path, csv_path = write_metrics(metrics, output_dir)
    heatmap_path = plot_heatmap(
        accuracy_grid,
        args.icl_example_indices,
        output_dir,
        title=f"2D FV-projection probe accuracy ({args.tasks[0]} vs {args.tasks[1]})",
    )

    config = {
        "tasks": args.tasks,
        "x_task": args.x_task,
        "y_task": args.y_task,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "icl_example_indices": args.icl_example_indices,
        "activations_root": str(args.activations_root),
        "fv_root": str(args.fv_root),
        "n_top_heads": args.n_top_heads,
        "n_layers": int(n_layers),
        "C": args.C,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "top_heads": {task: [(int(l), int(h), float(s)) for l, h, s in heads] for task, heads in top_heads.items()},
    }
    with open(output_dir / "fv_projection_probe_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(json_path)
    print(csv_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
