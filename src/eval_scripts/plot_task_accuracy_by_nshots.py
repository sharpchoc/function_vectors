"""
Plot baseline GPT-J accuracy vs number of ICL examples (reads the per-cell JSONs; CPU-only).

Companion to compute_task_accuracy_by_nshots.py: one line per task, x = n_shots (0..10),
y = top-k accuracy on the test split. Hue slots follow the same fixed categorical order used
in plot_tenshot_strip_lines.py so the two figure families sit next to each other cleanly.

Output: gptj_accuracy_by_nshots_top{k}.png, by default next to the strip-study figures in
results/direction2_label_geometry/tenshot_strip_intervention_cos_heatmap/figures/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import GENERAL_DIR, LABEL_GEOMETRY_DIR

TASKS = ["antonym", "synonym", "next_number_digits", "prev_number_digits"]
TASK_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]  # fixed categorical slots 1-4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", type=Path, default=GENERAL_DIR / "task_accuracies" / "by_nshots")
    p.add_argument("--n_shots", type=int, nargs="+", default=list(range(0, 11)))
    p.add_argument("--top_k", type=int, nargs="+", default=[1])
    p.add_argument("--output_dir", type=Path,
                   default=LABEL_GEOMETRY_DIR / "tenshot_strip_intervention_cos_heatmap" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    for k in args.top_k:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for task, color in zip(TASKS, TASK_COLORS):
            y = []
            for n in args.n_shots:
                f = args.input_root / f"{task}_n{n}.json"
                y.append(json.loads(f.read_text())["topk"][str(k)] if f.exists() else np.nan)
            ax.plot(args.n_shots, y, "-", color=color, linewidth=2, marker="o",
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.7, label=task)
        ax.set_xticks(args.n_shots)
        ax.set_xlabel("number of ICL examples (n_shots)")
        ax.set_ylabel(f"top-{k} accuracy (test split)")
        ax.set_ylim(-0.02, 1.0)
        ax.set_title(f"GPT-J baseline top-{k} accuracy vs ICL examples")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)
        ax.legend(title="task", fontsize=8, title_fontsize=8, framealpha=0.9, loc="lower right")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / f"gptj_accuracy_by_nshots_top{k}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
