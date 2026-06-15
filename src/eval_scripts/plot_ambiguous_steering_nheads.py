"""Per-task steering overlay: task-specific FV vs train-pooled-head FV at top-10/20/40.
2 panels (zero-shot+FV, 10-shot-shuffled+FV), 4 series each. No GPU."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["magnitude", "identity", "count_vowels", "count_consonants"]
NS = [10, 20, 40]
OUT = Path("results/heldout_ambiguous_eval_nheads_plots")
OUT.mkdir(parents=True, exist_ok=True)


def by_layer(block, key):
    d = {int(k): v for k, v in block[key].items()}
    xs = sorted(d)
    return xs, [d[x] for x in xs]


for task in TASKS:
    summaries = {n: json.loads((Path(f"results/heldout_ambiguous_eval_top{n}") / task /
                                "comparison_summary.json").read_text()) for n in NS}
    nfilt = summaries[10]["n_filtered_test_examples"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(axes[0], "zs_intervention_top1_by_layer", "Zero-shot + FV"),
                           (axes[1], "fs_shuffled_intervention_top1_by_layer", "10-shot shuffled + FV")]:
        # task-specific (same across n) — plot once
        xs, ys = by_layer(summaries[10]["task_specific_heads"], key)
        ax.plot(xs, ys, "k--", marker="s", ms=4, label="Task-specific")
        for n in NS:
            xs, ys = by_layer(summaries[n]["multitask_heads"], key)
            ax.plot(xs, ys, marker="o", ms=4, label=f"Train top-{n}")
        ax.set_title(title); ax.set_xlabel("Edit layer"); ax.set_ylim(0, 1.02)
        ax.set_ylabel("Intervention top-1 accuracy"); ax.grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{task}  (n_test={nfilt})", fontsize=14)
    fig.tight_layout()
    p = OUT / f"{task}_steering_nheads.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
