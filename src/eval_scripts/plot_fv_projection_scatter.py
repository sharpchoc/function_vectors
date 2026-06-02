#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.extract_utils import compute_function_vector
from src.utils.model_utils import load_gpt_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project saved residual activations onto task function vectors and plot layer-wise scatter plots."
    )
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"), help="Root containing <task> FV result directories.")
    parser.add_argument("--activations_root", type=Path, required=True, help="Root containing <task>/<split>/index.json activation shards.")
    parser.add_argument("--tasks", nargs=2, default=["antonym", "synonym"], help="Two tasks: y-axis task first, x-axis task second by default.")
    parser.add_argument("--x_task", type=str, default="synonym", help="Task whose FV is used for the x-axis projection.")
    parser.add_argument("--y_task", type=str, default="antonym", help="Task whose FV is used for the y-axis projection.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--token_role", type=str, default="final_token")
    parser.add_argument("--icl_example_index", type=int, default=None, help="Optional label-token ICL index to filter to.")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--save_function_vectors", action="store_true", help="Save reconstructed function vectors under each FV task directory.")
    parser.add_argument("--revision", type=str, default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_task_fv(task, fv_root, model, model_config, n_top_heads, save=False):
    task_dir = fv_root / task
    mean_path = task_dir / f"{task}_mean_head_activations.pt"
    ie_path = task_dir / f"{task}_indirect_effect.pt"
    if not mean_path.exists():
        raise FileNotFoundError(mean_path)
    if not ie_path.exists():
        raise FileNotFoundError(ie_path)

    # compute_function_vector passes top-k indices through NumPy, so keep these tensors on CPU.
    mean_activations = torch.load(mean_path, map_location="cpu")
    indirect_effect = torch.load(ie_path, map_location="cpu")
    fv, top_heads = compute_function_vector(
        mean_activations,
        indirect_effect,
        model,
        model_config=model_config,
        n_top_heads=n_top_heads,
    )
    fv = fv.detach().float().cpu().reshape(-1)

    if save:
        save_path = task_dir / f"{task}_function_vector.pt"
        torch.save({"function_vector": fv, "top_heads": top_heads, "n_top_heads": n_top_heads}, save_path)
        print(save_path)

    return fv, top_heads


def metadata_matches(meta, token_role, icl_example_index):
    if meta.get("token_role") != token_role:
        return False
    if icl_example_index is not None:
        return int(meta.get("icl_example_index")) == int(icl_example_index)
    return True


def load_task_activations(activations_root, task, split, token_role, icl_example_index=None):
    task_split_dir = activations_root / task / split
    index = load_json(task_split_dir / "index.json")
    activations = []
    metadata = []

    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = task_split_dir / shard_path
        data = torch.load(shard_path, map_location="cpu")
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


def load_all_activations(activations_root, tasks, split, token_role, icl_example_index=None):
    all_acts = []
    labels = []
    metadata = []
    for label, task in enumerate(tasks):
        acts, task_metadata = load_task_activations(activations_root, task, split, token_role, icl_example_index)
        all_acts.append(acts)
        labels.extend([label] * acts.shape[0])
        metadata.extend(task_metadata)
    return torch.cat(all_acts, dim=0), np.array(labels), metadata


def draw_layer_scatter(ax, layer, x_vals, y_vals, labels, tasks, x_task, y_task, show_legend=False):
    colors = ["#4c78a8", "#f58518"]
    for label, task in enumerate(tasks):
        mask = labels == label
        ax.scatter(x_vals[mask], y_vals[mask], s=18, alpha=0.68, color=colors[label % len(colors)], label=task)

    ax.axhline(0, color="0.75", linewidth=1)
    ax.axvline(0, color="0.75", linewidth=1)
    ax.set_title(f"Layer {layer}")
    ax.grid(alpha=0.2)
    if show_legend:
        ax.legend(frameon=False)


def plot_layer_scatter(layer, x_vals, y_vals, labels, tasks, x_task, y_task, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    draw_layer_scatter(ax, layer, x_vals, y_vals, labels, tasks, x_task, y_task, show_legend=True)
    ax.set_xlabel(f"Dot with {x_task} FV")
    ax.set_ylabel(f"Dot with {y_task} FV")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layer_{layer:02d}_fv_projection_scatter.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_all_layers_scatter(projections, labels, tasks, x_task, y_task, output_dir):
    n_layers = len(projections)
    n_cols = 4
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    all_x = np.concatenate([projection[0] for projection in projections])
    all_y = np.concatenate([projection[1] for projection in projections])
    x_pad = max((all_x.max() - all_x.min()) * 0.05, 1e-6)
    y_pad = max((all_y.max() - all_y.min()) * 0.05, 1e-6)
    x_limits = (all_x.min() - x_pad, all_x.max() + x_pad)
    y_limits = (all_y.min() - y_pad, all_y.max() + y_pad)

    for layer, (x_vals, y_vals) in enumerate(projections):
        ax = axes[layer]
        draw_layer_scatter(ax, layer, x_vals, y_vals, labels, tasks, x_task, y_task, show_legend=(layer == 0))
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    for ax in axes[n_layers:]:
        ax.axis("off")

    fig.supxlabel(f"Dot with {x_task} FV")
    fig.supylabel(f"Dot with {y_task} FV")
    fig.suptitle("Residual activations projected onto task function vectors", y=0.995)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_layers_fv_projection_scatter.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if args.x_task not in args.tasks or args.y_task not in args.tasks:
        raise ValueError("--x_task and --y_task must both be included in --tasks")

    output_dir = args.output_dir
    if output_dir is None:
        token_tag = args.token_role if args.icl_example_index is None else f"{args.token_role}_icl{args.icl_example_index}"
        output_dir = args.activations_root / "fv_projection_scatter" / f"{args.split}_{token_tag}"

    print("Loading model for function-vector reconstruction")
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device, revision=args.revision)
    model.eval()

    fvs = {}
    top_heads = {}
    for task in args.tasks:
        fvs[task], top_heads[task] = compute_task_fv(
            task,
            args.fv_root,
            model,
            model_config,
            args.n_top_heads,
            save=args.save_function_vectors,
        )
        print(f"{task} top_heads: {top_heads[task]}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading residual activations")
    activations, labels, metadata = load_all_activations(
        args.activations_root,
        args.tasks,
        args.split,
        args.token_role,
        args.icl_example_index,
    )
    if activations.ndim != 3:
        raise ValueError(f"Expected activations [records, layers, resid_dim], got {tuple(activations.shape)}")

    x_fv = fvs[args.x_task]
    y_fv = fvs[args.y_task]
    n_layers = activations.shape[1]
    print(f"records={activations.shape[0]}, layers={n_layers}, resid_dim={activations.shape[2]}")

    summary = {
        "tasks": args.tasks,
        "x_task": args.x_task,
        "y_task": args.y_task,
        "split": args.split,
        "token_role": args.token_role,
        "icl_example_index": args.icl_example_index,
        "n_records": int(activations.shape[0]),
        "n_layers": int(n_layers),
        "resid_dim": int(activations.shape[2]),
        "top_heads": {task: [(int(l), int(h), float(s)) for l, h, s in heads] for task, heads in top_heads.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "projection_config.json", "w") as f:
        json.dump(summary, f, indent=2)

    output_paths = []
    projections = []
    for layer in range(n_layers):
        layer_acts = activations[:, layer, :]
        x_vals = torch.matmul(layer_acts, x_fv).numpy()
        y_vals = torch.matmul(layer_acts, y_fv).numpy()
        projections.append((x_vals, y_vals))
        output_path = plot_layer_scatter(layer, x_vals, y_vals, labels, args.tasks, args.x_task, args.y_task, output_dir)
        output_paths.append(str(output_path))
        print(output_path)

    all_layers_path = plot_all_layers_scatter(projections, labels, args.tasks, args.x_task, args.y_task, output_dir)
    output_paths.append(str(all_layers_path))
    print(all_layers_path)

    with open(output_dir / "plots.json", "w") as f:
        json.dump(output_paths, f, indent=2)


if __name__ == "__main__":
    main()
