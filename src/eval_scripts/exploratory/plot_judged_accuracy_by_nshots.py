"""
Plot GPT-4-judged vs gold-first-token top-1 accuracy by n_shots (antonym/synonym; CPU-only).

Companion to compute_judged_accuracy_by_nshots.py: solid line = judge-scored top-1 (valid
answer, same-word echoes false), dashed line = gold-first-token top-1 re-derived from the SAME
records. Hue slots follow the fixed categorical order of plot_task_accuracy_by_nshots.py.

Output: gptj_judged_accuracy_by_nshots_top1.png next to the strip-study figures.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.paths import GENERAL_DIR, LABEL_GEOMETRY_DIR

TASK_COLORS = {"antonym": "#2a78d6", "synonym": "#1baf7a",   # same slots as the gold figure
               "next_number_digits": "#eda100", "prev_number_digits": "#008300"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", type=Path,
                   default=GENERAL_DIR / "task_accuracies" / "by_nshots_judged")
    p.add_argument("--tasks", nargs="+", default=["antonym", "synonym"])
    p.add_argument("--gold_only_tasks", nargs="+", default=[],
                   help="Tasks drawn from the gold-scored by_nshots run only (no judge records); "
                        "closed-answer tasks where gold top-1 is already exact.")
    p.add_argument("--gold_root", type=Path,
                   default=GENERAL_DIR / "task_accuracies" / "by_nshots")
    p.add_argument("--n_shots", type=int, nargs="+", default=list(range(0, 11)))
    p.add_argument("--output_name", type=str, default="gptj_judged_accuracy_by_nshots_top1.png")
    p.add_argument("--output_dir", type=Path,
                   default=LABEL_GEOMETRY_DIR / "tenshot_strip_intervention_cos_heatmap" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    label_ys = []

    def stagger(y):  # nudge end-of-line labels apart when curves end at the same height
        while any(abs(y - u) < 0.035 for u in label_ys):
            y -= 0.035
        label_ys.append(y)
        return y

    for task in args.tasks:
        color = TASK_COLORS[task]
        judged, gold = [], []
        for n in args.n_shots:
            recs = json.loads((args.input_root / f"{task}_n{n}.json").read_text())["records"]
            judged.append(sum(r["judge_correct"] for r in recs) / len(recs))
            gold.append(sum(r["gold_rank"] == 0 for r in recs) / len(recs))
        ax.plot(args.n_shots, judged, "-", color=color, linewidth=2, marker="o",
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
                label=f"{task} (judged)")
        ax.plot(args.n_shots, gold, "--", color=color, linewidth=1.6, marker="o",
                markersize=3.5, alpha=0.65, label=f"{task} (gold top-1)")
        ax.annotate(task, (args.n_shots[-1] + 0.15, stagger(judged[-1])), color=color,
                    fontsize=8, va="center", annotation_clip=False)
    for task in args.gold_only_tasks:
        color = TASK_COLORS[task]
        y = [json.loads((args.gold_root / f"{task}_n{n}.json").read_text())["topk"]["1"]
             for n in args.n_shots]
        ax.plot(args.n_shots, y, "-", color=color, linewidth=2, marker="o",
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
                label=f"{task} (gold = exact)")
        ax.annotate(task, (args.n_shots[-1] + 0.15, stagger(y[-1])), color=color,
                    fontsize=8, va="center", annotation_clip=False)
    ax.set_xticks(args.n_shots)
    ax.set_xlim(args.n_shots[0] - 0.3, args.n_shots[-1] + 1.1)
    ax.set_xlabel("number of ICL examples (n_shots)")
    ax.set_ylabel("top-1 accuracy (test split)")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("GPT-J top-1 accuracy vs ICL examples — GPT-4.1-judged vs gold-first-token")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / args.output_name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
