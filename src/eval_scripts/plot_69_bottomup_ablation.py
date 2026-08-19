#!/usr/bin/env python
"""Figures + summary CSV for the bottom-up read-feature ablation baselines.

Reads the per-task eval JSONs written by ablate_readdir_labeltokens.py
(artifacts/69_task_run/bottom_up_ablation/n{1,6}shot/<task>.json) plus the reused
unablated baselines (zero_shot / real_1shot / real_6shot) from
bottom_up_read_features/steering_results/sixshot_dummy/per_task_acc.csv.

Writes into results/69_task_run/bottom_up_read_features/ablation/:
  per_task_acc.csv          task, group, cf_task, zero_shot, then per n in {1,6}:
                            baseline, attnmask, mean_ablation, zero_ablation,
                            cf_mean_ablation, cf_zero_ablation
  aggregate_bars.png        two panels (1-shot | 6-shot), mean accuracy over 69 tasks
  per_task_bars_1shot.png   69 tasks x 7 bars
  per_task_bars_6shot.png
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation"
BASE_CSV = (TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
            / "sixshot_dummy" / "per_task_acc.csv")
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"

CONDS = ("baseline", "attnmask", "mean_ablation", "zero_ablation",
         "cf_mean_ablation", "cf_zero_ablation")
COLOR = {"baseline": "#8a8a8a", "attnmask": "#8e6bb5", "mean_ablation": "#2f7fe0",
         "zero_ablation": "#d94f3d", "cf_mean_ablation": "#a9c9ef",
         "cf_zero_ablation": "#efb2a9", "zero_shot": "#d9d9d9"}
LABEL = {"baseline": "unablated", "attnmask": "cue attn-masked from labels",
         "mean_ablation": "mean-ablate read dir", "zero_ablation": "zero-ablate read dir",
         "cf_mean_ablation": "mean-ablate counterfactual dir",
         "cf_zero_ablation": "zero-ablate counterfactual dir", "zero_shot": "0-shot"}


def main():
    split = json.load(open(SPLIT))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)

    base = {r["task"]: r for r in csv.DictReader(open(BASE_CSV))}
    assert set(tasks) <= set(base), f"missing baseline rows: {set(tasks) - set(base)}"

    rows = {}
    for t in tasks:
        rows[t] = {"task": t, "group": group[t],
                   "zero_shot": float(base[t]["zero_shot"]),
                   "n1_baseline": float(base[t]["real_1shot"]),
                   "n6_baseline": float(base[t]["real_6shot"])}
        for n in (1, 6):
            r = json.load(open(AR / f"n{n}shot" / f"{t}.json"))
            assert r["n_prompts"] == 150
            rows[t].setdefault("cf_task", r["cf_task"])
            assert rows[t]["cf_task"] == r["cf_task"]
            for c in CONDS[1:]:
                rows[t][f"n{n}_{c}"] = r["conditions"][c]["acc"]

    OUT.mkdir(parents=True, exist_ok=True)
    cols = (["task", "group", "cf_task", "zero_shot"]
            + [f"n{n}_{c}" for n in (1, 6) for c in CONDS])
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    # ---------------- aggregate figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=200, sharey=True)
    for ax, n in zip(axes, (1, 6)):
        names = list(CONDS) + ["zero_shot"]
        vals = [np.mean([rows[t][f"n{n}_{c}"] if c != "zero_shot" else rows[t]["zero_shot"]
                         for t in tasks]) for c in names]
        x = np.arange(len(names))
        ax.bar(x, vals, color=[COLOR[c] for c in names], width=0.72)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[c] for c in names], rotation=28, ha="right", fontsize=9.5)
        ax.set_title(f"{n}-shot", fontsize=14)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    fig.suptitle("Ablating the bottom-up read direction (L6 label-mean, all layers, "
                 "all demo-label tokens)", fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "aggregate_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- per-task figures ----------------
    for n in (1, 6):
        names = list(CONDS) + ["zero_shot"]
        fig, ax = plt.subplots(figsize=(26, 7.5), dpi=170)
        x = np.arange(len(tasks))
        w = 0.115
        for ci, c in enumerate(names):
            vals = [rows[t][f"n{n}_{c}"] if c != "zero_shot" else rows[t]["zero_shot"]
                    for t in tasks]
            ax.bar(x + (ci - (len(names) - 1) / 2) * w, vals, w,
                   color=COLOR[c], label=LABEL[c])
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=90, fontsize=7.5)
        ax.set_ylabel("accuracy")
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9, ncol=4, loc="upper right")
        ax.set_title(f"Bottom-up read-direction ablation, per task ({n}-shot)",
                     fontsize=14, fontweight="bold", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"per_task_bars_{n}shot.png", bbox_inches="tight")
        plt.close(fig)

    for n in (1, 6):
        print(f"n{n}: " + "  ".join(
            f"{c}={np.mean([rows[t][f'n{n}_{c}'] for t in tasks]):.3f}" for c in CONDS))
    print(f"wrote {OUT}/per_task_acc.csv, aggregate_bars.png, per_task_bars_{{1,6}}shot.png")


if __name__ == "__main__":
    main()
