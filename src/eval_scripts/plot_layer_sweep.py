#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = {
    "fv": {
        "label": "Function vector",
        "result_file": "zs_results_layer_sweep.json",
    },
    "avg_hs": {
        "label": "Average activation",
        "result_glob": "mean_layer_intervention_zs_results_sweep_*.json",
    },
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def topk_lookup(topk_items, k):
    return {int(rank): float(acc) for rank, acc in topk_items}[k]


def resolve_result_file(task_dir, method_config):
    if "result_file" in method_config:
        result_file = task_dir / method_config["result_file"]
        if not result_file.exists():
            raise FileNotFoundError(result_file)
        return result_file

    matches = sorted(task_dir.glob(method_config["result_glob"]), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {method_config['result_glob']} under {task_dir}")
    return matches[-1]


def load_baseline(baseline_task_dir, k):
    baseline = load_json(baseline_task_dir / "model_baseline.json")
    zero_shot = topk_lookup(baseline["0"]["clean_topk"], k)
    max_shot = max(int(shot) for shot in baseline.keys())
    max_shot_acc = topk_lookup(baseline[str(max_shot)]["clean_topk"], k)
    return zero_shot, max_shot, max_shot_acc


def load_layer_sweep(task_dir, method, k):
    method_config = METHODS[method]
    result_file = resolve_result_file(task_dir, method_config)
    sweep = load_json(result_file)
    layers = sorted(int(layer) for layer in sweep.keys())
    accuracies = [topk_lookup(sweep[str(layer)]["intervention_topk"], k) for layer in layers]
    return layers, accuracies, result_file


def plot_task(task, methods, fv_root, avg_hs_root, baseline_root, output_dir, k):
    roots = {"fv": fv_root, "avg_hs": avg_hs_root}
    zero_shot, max_shot, max_shot_acc = load_baseline(baseline_root / task, k)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    all_values = [zero_shot, max_shot_acc]
    used_files = []
    layers_for_ticks = None

    for method in methods:
        layers, accuracies, result_file = load_layer_sweep(roots[method] / task, method, k)
        layers_for_ticks = layers
        ax.plot(layers, accuracies, marker="o", linewidth=2, label=METHODS[method]["label"])
        all_values.extend(accuracies)
        used_files.append(result_file)

    ax.axhline(zero_shot, linestyle=":", color="0.35", linewidth=2, label="0-shot ICL baseline")
    ax.axhline(max_shot_acc, linestyle=":", color="0.05", linewidth=2, label=f"{max_shot}-shot ICL baseline")

    ax.set_title(f"{task.capitalize()} steering, top-{k}")
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel(f"Top-{k} accuracy")
    ax.set_xticks(layers_for_ticks)
    ax.set_ylim(0, max(1.0, max(all_values) * 1.08))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    method_tag = "_vs_".join(methods)
    output_path = output_dir / f"{task}_{method_tag}_top{k}_layer_sweep.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, used_files


def main():
    parser = argparse.ArgumentParser(description="Plot steering accuracy by intervention layer.")
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"), help="Root containing FV task result directories.")
    parser.add_argument("--avg_hs_root", type=Path, default=Path("results/gptj_avg_hs"), help="Root containing average hidden-state task result directories.")
    parser.add_argument("--baseline_root", type=Path, default=None, help="Root containing model_baseline.json task dirs. Defaults to --fv_root.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for plot PNGs. Defaults to <fv_root>/plots.")
    parser.add_argument("--tasks", nargs="+", default=["antonym", "synonym"], help="Task names to plot. Pass one task, e.g. --tasks antonym, to plot only that task.")
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 2], help="Top-k values to plot.")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["fv", "avg_hs"], help="Methods to overlay.")
    args = parser.parse_args()

    baseline_root = args.baseline_root or args.fv_root
    output_dir = args.output_dir or args.fv_root / "plots"

    for task in args.tasks:
        for k in args.topks:
            output_path, used_files = plot_task(
                task, args.methods, args.fv_root, args.avg_hs_root, baseline_root, output_dir, k
            )
            sources = ", ".join(str(path) for path in used_files)
            print(f"{output_path} <- {sources}")


if __name__ == "__main__":
    main()
