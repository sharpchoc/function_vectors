#!/usr/bin/env python
"""Train vs held-out FV-steering generalisation figure from an aggregate_eval_headset.py summary CSV
(columns task,group,n_heads,{zs,shuf,mix}_{base,best,bestL}). Mirrors results/qwen25_fv/round2_96_prunedfail/
train_heldout_generalisation.png (whose ad-hoc plotter was not committed). Written 2026-09-23.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <csv dir>/train_heldout_generalisation.png")
    ap.add_argument("--model_label", default="")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    out = a.out or a.csv.parent / "train_heldout_generalisation.png"
    settings = [("zs", "zero-shot"), ("shuf", "shuffled-label 10-shot"), ("mix", "mixed-task 10-shot")]
    groups = ["train", "heldout"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150, sharey=True)
    for ax, (k, title) in zip(axes, settings):
        x = np.arange(len(groups)); w = 0.36
        base = [np.mean([float(r[f"{k}_base"]) for r in rows if r["group"] == g]) for g in groups]
        best = [np.mean([float(r[f"{k}_best"]) for r in rows if r["group"] == g]) for g in groups]
        n = [sum(r["group"] == g for r in rows) for g in groups]
        b1 = ax.bar(x - w / 2, base, w, color="0.7", label="unsteered")
        b2 = ax.bar(x + w / 2, best, w, color="tab:blue", label="FV steered (best layer)")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, [f"{g} (n={c})" for g, c in zip(groups, n)])
        ax.set_title(title); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("full-label teacher-forced accuracy")
    axes[0].legend(fontsize=8)
    nh = rows[0]["n_heads"]
    fig.suptitle(f"Pooled sparse FV ({nh} heads) steering: train vs held-out tasks {('— ' + a.model_label) if a.model_label else ''}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
