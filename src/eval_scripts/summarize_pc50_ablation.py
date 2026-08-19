#!/usr/bin/env python
"""Aggregate the PC50 label-token ablation eval (ablate_pc50_labeltokens.py) into a summary
table + figure. Reads artifacts/69_task_run/pc50_ablation/eval/<task>.json (55 train tasks,
9 conditions each: baseline + 4 brackets x {zero, mean}), writes to
results/69_task_run/pc50_ablation/:
  ablation_drops.png   per-condition mean accuracy + mean drop vs baseline (bars) with
                       per-task drop distributions (strip)
  per_task_acc.csv     task x condition accuracy matrix
  summary.csv          per-condition mean/median acc, mean/median drop vs baseline
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
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

EVAL_DIR = ARTIFACTS_ROOT / "69_task_run" / "pc50_ablation" / "eval"
OUT_DIR = TASK69_RUN_DIR / "pc50_ablation"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")
CONDS = ["baseline"] + [f"{b}__{m}" for b in BRACKETS for m in ("zero", "mean")]


def main():
    files = sorted(EVAL_DIR.glob("*.json"))
    assert len(files) == 55, len(files)
    tasks, acc = [], {c: [] for c in CONDS}
    for f in files:
        d = json.load(open(f))
        tasks.append(d["task"])
        for c in CONDS:
            acc[c].append(d["conditions"][c]["acc"])
    A = {c: np.array(v) for c, v in acc.items()}
    base = A["baseline"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task"] + CONDS)
        for i, t in enumerate(tasks):
            w.writerow([t] + [A[c][i] for c in CONDS])

    rows = []
    for c in CONDS:
        drop = base - A[c]
        rows.append([c, round(float(A[c].mean()), 4), round(float(np.median(A[c])), 4),
                     round(float(drop.mean()), 4), round(float(np.median(drop)), 4)])
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_acc", "median_acc", "mean_drop_vs_baseline",
                    "median_drop_vs_baseline"])
        w.writerows(rows)
        print(*["  ".join(str(x) for x in r) for r in rows], sep="\n")

    ab_conds = CONDS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=150)
    ax = axes[0]
    xs = np.arange(len(CONDS))
    ax.bar(xs, [A[c].mean() for c in CONDS],
           color=["0.4"] + ["tab:blue", "tab:cyan"] * 4)
    ax.set_xticks(xs, CONDS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean accuracy (55 train tasks)")
    ax.set_title("temperature-1 sampled accuracy, clean 10-shot prompts")
    ax.grid(alpha=0.25, axis="y")
    ax = axes[1]
    rng = np.random.RandomState(0)
    for i, c in enumerate(ab_conds):
        drop = base - A[c]
        ax.scatter(np.full(len(drop), i) + rng.uniform(-0.14, 0.14, len(drop)), drop,
                   s=10, alpha=0.45, color="tab:blue" if c.endswith("zero") else "tab:cyan")
        ax.plot([i - 0.25, i + 0.25], [drop.mean()] * 2, color="black", lw=2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(ab_conds)), ab_conds, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy drop vs baseline (per task)")
    ax.set_title("drop from ablating the top-50 uncentered read-dir PCs\n"
                 "at demo-label tokens, all 28 layers (black = mean)")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("PC50 read-direction ablation eval — 55 train tasks, 150 prompts each, "
                 "T=1 sampling, exact match", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_drops.png", bbox_inches="tight")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
