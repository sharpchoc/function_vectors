"""Re-plot the qfin attention-knockout bars from summary.json (pure plotting, no GPU).
See ablate_qfinal_attention.py for how the metrics are produced."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import LABEL_GEOMETRY_DIR

COND_COLOR = {"clean": "#888888", "ko_demo2_prelabel": "#c44e52",
              "ko_both_labels": "#4c72b0", "ko_demo2_qcolon": "#55a868"}
COND_LABEL = {"clean": "clean", "ko_demo2_prelabel": "cut qfin→demo2 pre-label (test)",
              "ko_both_labels": "cut qfin→both labels (+ctrl)", "ko_demo2_qcolon": "cut qfin→demo2 'Q:' (−ctrl)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(LABEL_GEOMETRY_DIR / "twoshot" / "qfinal_attn_knockout"))
    args = p.parse_args()
    root = Path(args.root)
    summary = json.load(open(root / "summary.json"))
    tasks, conds, results = summary["tasks"], summary["conditions"], summary["results"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    width = 0.8 / len(conds)
    for ax, metric, ylabel in [(axes[0], "top1", "first-token top-1 accuracy"),
                               (axes[1], "mean_gold_logit", "mean gold-token logit @ qfin")]:
        for ci, cond in enumerate(conds):
            vals = [results[t]["conditions"][cond][metric] for t in tasks]
            ax.bar(np.arange(len(tasks)) + ci * width, vals, width,
                   color=COND_COLOR[cond], label=COND_LABEL[cond])
        ax.set_xticks(np.arange(len(tasks)) + (len(conds) - 1) / 2 * width)
        ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Does qfin read task info directly from demo-2 pre-label? "
                 "(attention knockout at all layers; test vs ±controls)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = root / "figures" / "qfinal_attn_knockout_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
