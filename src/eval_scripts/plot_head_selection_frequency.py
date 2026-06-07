#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot how often each layer/head is selected in saved task function vectors."
    )
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_name", type=str, default="head_selection_frequency_heatmap.png")
    parser.add_argument("--n_layers", type=int, default=None, help="Defaults to the max selected layer + 1.")
    parser.add_argument("--n_heads", type=int, default=None, help="Defaults to the max selected head + 1.")
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task subset. Defaults to every task under --fv_root.")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_top_heads(task_dir):
    task = task_dir.name
    metadata_path = task_dir / f"{task}_function_vector_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return data["top_heads"]

    fv_path = task_dir / f"{task}_function_vector.pt"
    if fv_path.exists():
        data = torch_load_trusted(fv_path, map_location="cpu")
        if isinstance(data, dict) and "top_heads" in data:
            return data["top_heads"]

    raise FileNotFoundError(f"No top_heads found for {task_dir}")


def iter_task_dirs(fv_root, tasks):
    if tasks is not None:
        for task in tasks:
            task_dir = fv_root / task
            if not task_dir.exists():
                raise FileNotFoundError(task_dir)
            yield task_dir
        return

    for task_dir in sorted(path for path in fv_root.iterdir() if path.is_dir()):
        if (task_dir / f"{task_dir.name}_function_vector_metadata.json").exists() or (
            task_dir / f"{task_dir.name}_function_vector.pt"
        ).exists():
            yield task_dir


def collect_head_counts(fv_root, tasks):
    counts = Counter()
    per_task = {}
    for task_dir in iter_task_dirs(fv_root, tasks):
        top_heads = load_top_heads(task_dir)
        per_task[task_dir.name] = [
            {"layer": int(layer), "head": int(head), "score": float(score)} for layer, head, score in top_heads
        ]
        for layer, head, _ in top_heads:
            counts[(int(layer), int(head))] += 1

    if not per_task:
        raise ValueError(f"No function-vector head selections found under {fv_root}")
    return counts, per_task


def make_count_grid(counts, n_layers, n_heads):
    if n_layers is None:
        n_layers = max(layer for layer, _ in counts) + 1
    if n_heads is None:
        n_heads = max(head for _, head in counts) + 1

    grid = np.zeros((n_layers, n_heads), dtype=int)
    for (layer, head), count in counts.items():
        if layer >= n_layers or head >= n_heads:
            raise ValueError(f"Head L{layer}H{head} is outside grid shape {(n_layers, n_heads)}")
        grid[layer, head] = count
    return grid


def plot_grid(grid, n_tasks, output_path, title):
    fig_width = max(9.5, grid.shape[1] * 0.55)
    fig_height = max(9.0, grid.shape[0] * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(grid, aspect="auto", origin="upper", cmap="viridis", vmin=0, vmax=max(1, grid.max()))
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))

    threshold = grid.max() * 0.55
    for layer in range(grid.shape[0]):
        for head in range(grid.shape[1]):
            count = int(grid[layer, head])
            text_color = "white" if count > threshold else "black"
            ax.text(head, layer, str(count), ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Number of task function vectors selected out of {n_tasks}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary(counts, per_task, grid, output_path):
    summary = {
        "n_tasks": len(per_task),
        "n_layers": int(grid.shape[0]),
        "n_heads": int(grid.shape[1]),
        "head_counts": [
            {"layer": layer, "head": head, "count": count}
            for (layer, head), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        ],
        "per_task_top_heads": per_task,
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def main():
    args = parse_args()
    counts, per_task = collect_head_counts(args.fv_root, args.tasks)
    grid = make_count_grid(counts, args.n_layers, args.n_heads)

    output_dir = args.output_dir or args.fv_root / "plots"
    output_path = output_dir / args.output_name
    title = args.title or f"Function-vector head selection frequency ({len(per_task)} tasks)"

    plot_grid(grid, len(per_task), output_path, title)
    summary_path = write_summary(counts, per_task, grid, output_path)

    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
