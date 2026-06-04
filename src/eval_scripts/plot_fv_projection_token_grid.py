#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


TOKEN_ROLES = ["pre_label_token", "first_label_token", "last_label_token"]
TOKEN_TITLES = {
    "pre_label_token": "Pre-label token",
    "first_label_token": "First label token",
    "last_label_token": "Last label token",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot activations for several ICL examples and token positions projected "
            "onto the 2D space defined by two task function vectors."
        )
    )
    parser.add_argument("--activations_root", type=Path, required=True)
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument("--tasks", nargs=2, default=["antonym", "synonym"])
    parser.add_argument("--x_task", type=str, default="synonym")
    parser.add_argument("--y_task", type=str, default="antonym")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--point_size", type=float, default=12.0)
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_function_vector(fv_root, task):
    fv_path = fv_root / task / f"{task}_function_vector.pt"
    if not fv_path.exists():
        raise FileNotFoundError(fv_path)
    data = torch_load_trusted(fv_path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return fv.detach().float().cpu().reshape(-1)


def wanted_key(meta, wanted_keys):
    role = meta.get("token_role")
    icl_example_index = meta.get("icl_example_index")
    if icl_example_index is None:
        return None
    key = (int(icl_example_index), role)
    return key if key in wanted_keys else None


def load_task_grid_activations(args, task, wanted_keys):
    split_dir = args.activations_root / task / args.split
    index = load_json(split_dir / "index.json")
    by_key = {key: [] for key in wanted_keys}

    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = split_dir / shard_path
        data = torch_load_trusted(shard_path, map_location="cpu")
        shard_acts = data["activations"]
        shard_metadata = data["metadata"]
        selected_by_key = {key: [] for key in wanted_keys}
        for i, meta in enumerate(shard_metadata):
            key = wanted_key(meta, wanted_keys)
            if key is not None:
                selected_by_key[key].append(i)
        for key, selected in selected_by_key.items():
            if selected:
                by_key[key].append(shard_acts[selected, args.layer, :].float())

    projected_ready = {}
    for key, chunks in by_key.items():
        if not chunks:
            raise ValueError(f"No records for {task}/{args.split} icl={key[0]} role={key[1]}")
        projected_ready[key] = torch.cat(chunks, dim=0)
    return projected_ready


def load_grid_data(args, basis):
    wanted_keys = {(icl, role) for icl in args.icl_example_indices for role in TOKEN_ROLES}
    grid_data = {key: [] for key in wanted_keys}

    for task in args.tasks:
        task_acts = load_task_grid_activations(args, task, wanted_keys)
        for key, activations in task_acts.items():
            if basis.shape[0] != activations.shape[1]:
                raise ValueError(
                    f"Activation/FV dimension mismatch for {task}: "
                    f"activations={activations.shape[1]}, fv={basis.shape[0]}"
                )
            grid_data[key].append(torch.matmul(activations, basis).numpy())

    return grid_data


def plot_grid(args, grid_data, fv_markers, output_dir):
    n_rows = len(args.icl_example_indices)
    n_cols = len(TOKEN_ROLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.55 * n_rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    all_points = np.concatenate([points for cell in grid_data.values() for points in cell], axis=0)
    marker_points = np.array([[marker[0], marker[1]] for marker in fv_markers.values()])
    limit_points = np.concatenate([all_points, marker_points], axis=0)
    x_pad = max((limit_points[:, 0].max() - limit_points[:, 0].min()) * 0.05, 1e-6)
    y_pad = max((limit_points[:, 1].max() - limit_points[:, 1].min()) * 0.05, 1e-6)
    x_limits = (limit_points[:, 0].min() - x_pad, limit_points[:, 0].max() + x_pad)
    y_limits = (limit_points[:, 1].min() - y_pad, limit_points[:, 1].max() + y_pad)
    colors = ["#4c78a8", "#f58518"]

    for row, icl_example_index in enumerate(args.icl_example_indices):
        for col, token_role in enumerate(TOKEN_ROLES):
            ax = axes[row, col]
            for label, task in enumerate(args.tasks):
                points = grid_data[(icl_example_index, token_role)][label]
                ax.scatter(points[:, 0], points[:, 1], s=args.point_size, alpha=args.alpha, color=colors[label], label=task)
            for task, marker in fv_markers.items():
                ax.scatter(
                    marker[0],
                    marker[1],
                    marker="X",
                    s=360,
                    linewidths=2.2,
                    edgecolors="white",
                    color=fv_markers[task][2],
                    label=f"{task} FV",
                    zorder=20,
                    clip_on=False,
                )
            ax.axhline(0, color="0.75", linewidth=0.8)
            ax.axvline(0, color="0.75", linewidth=0.8)
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(TOKEN_TITLES[token_role])
            if col == 0:
                ax.set_ylabel(f"ICL {icl_example_index}\nDot with {args.y_task} FV")
            if row == n_rows - 1:
                ax.set_xlabel(f"Dot with {args.x_task} FV")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    fig.legend(unique_handles, unique_labels, loc="upper center", ncol=len(unique_labels), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"Layer {args.layer}: token positions projected onto task function vectors", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    output_dir.mkdir(parents=True, exist_ok=True)
    icl_tag = "-".join(str(i) for i in args.icl_example_indices)
    output_path = output_dir / f"layer_{args.layer:02d}_icl{icl_tag}_pre_first_last_label_fv_projection_grid.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if args.x_task not in args.tasks or args.y_task not in args.tasks:
        raise ValueError("--x_task and --y_task must both be included in --tasks")

    fvs = {task: load_function_vector(args.fv_root, task) for task in args.tasks}
    basis = torch.stack([fvs[args.x_task], fvs[args.y_task]], dim=1)
    fv_marker_colors = {args.x_task: "#2ca02c", args.y_task: "#d62728"}
    fv_markers = {
        task: (
            float(torch.matmul(fv, basis)[0]),
            float(torch.matmul(fv, basis)[1]),
            fv_marker_colors.get(task, "#b279a2"),
        )
        for task, fv in fvs.items()
    }
    grid_data = load_grid_data(args, basis)

    output_dir = args.output_dir or args.activations_root / "fv_projection_token_position_grid"
    output_path = plot_grid(args, grid_data, fv_markers, output_dir)

    config = {
        "tasks": args.tasks,
        "x_task": args.x_task,
        "y_task": args.y_task,
        "split": args.split,
        "layer": args.layer,
        "icl_example_indices": args.icl_example_indices,
        "token_roles": TOKEN_ROLES,
        "activations_root": str(args.activations_root),
        "fv_root": str(args.fv_root),
        "fv_markers": {task: [x, y] for task, (x, y, _) in fv_markers.items()},
        "output_path": str(output_path),
    }
    with open(output_dir / f"layer_{args.layer:02d}_fv_projection_token_grid_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(output_path)


if __name__ == "__main__":
    main()
