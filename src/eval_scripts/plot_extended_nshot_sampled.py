#!/usr/bin/env python
"""Merge + plot the extended-tasks n-shot sampled sweep (CPU).

Reads artifacts/extended_tasks_nshot/results/<task>.json (from
compute_extended_nshot_sampled.py --mode run) and writes to
results/general/extended_tasks_nshot_sweep/:
  - nshot_accuracy.csv       task x n accuracy + Wilson 95% CI + origin/lane
  - nshot_grid.png           142 small-multiple curves (originals shaded differently)
  - nshot_aggregate.png      mean curves: all / original / new + per-lane means
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import ARTIFACTS_ROOT, GENERAL_DIR  # noqa: E402

IN_ROOT = ARTIFACTS_ROOT / "extended_tasks_nshot" / "results"
OUT = GENERAL_DIR / "extended_tasks_nshot_sweep"
TASK_ROOT = REPO_ROOT / "dataset_files" / "extended_tasks"
N_SHOTS = list(range(0, 7))


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(TASK_ROOT / "manifest.json"))["tasks"]
    files = sorted(IN_ROOT.glob("*.json"))
    print(f"{len(files)} task result files")

    rows, curves = [], {}
    for f in files:
        recs = json.load(open(f))
        task = f.stem
        assert len(recs) == len(N_SHOTS) * 50, (task, len(recs))
        info = manifest.get(task, {})
        accs = {}
        for n in N_SHOTS:
            sub = [r for r in recs if r["n"] == n]
            acc = sum(r["match"] for r in sub) / len(sub)
            lo, hi = wilson(acc, len(sub))
            accs[n] = acc
            rows.append({"task": task, "origin": info.get("origin", "?"),
                         "lane": info.get("lane", ""), "n_shots": n, "n_prompts": len(sub),
                         "accuracy": round(acc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        curves[task] = (info.get("origin", "?"), accs)

    with open(OUT / "nshot_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # grid of small multiples
    tasks = sorted(curves)
    ncols, nrows = 12, math.ceil(len(tasks) / 12)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 1.7 * nrows), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, tasks):
        origin, accs = curves[task]
        color = "tab:blue" if origin == "new" else "tab:red"
        ax.plot(N_SHOTS, [accs[n] for n in N_SHOTS], "o-", ms=2.5, lw=1.2, color=color)
        ax.set_title(task, fontsize=6.5)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    fig.suptitle("GPT-J n-shot accuracy (T=1.0 sampled, full-label match; 50 prompts/point) — "
                 "blue=new task, red=original", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT / "nshot_grid.png", dpi=140)
    plt.close(fig)

    # aggregate
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    groups = {"all (142)": tasks,
              "original (42)": [t for t in tasks if curves[t][0] != "new"],
              "new (100)": [t for t in tasks if curves[t][0] == "new"]}
    for label, ts in groups.items():
        means = [sum(curves[t][1][n] for t in ts) / len(ts) for n in N_SHOTS]
        axes[0].plot(N_SHOTS, means, "o-", label=label)
    axes[0].set_xlabel("n shots"); axes[0].set_ylabel("mean accuracy"); axes[0].set_ylim(0, 1)
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title("Mean accuracy vs n")

    lanes = defaultdict(list)
    for t in tasks:
        lane = manifest.get(t, {}).get("lane") or "original"
        lanes[lane].append(t)
    for lane, ts in sorted(lanes.items(), key=lambda kv: -len(kv[1]))[:10]:
        means = [sum(curves[t][1][n] for t in ts) / len(ts) for n in N_SHOTS]
        axes[1].plot(N_SHOTS, means, "o-", ms=3, label=f"{lane} ({len(ts)})")
    axes[1].set_xlabel("n shots"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3); axes[1].set_title("By lane (top 10 by size)")
    fig.suptitle("extended_tasks n-shot sweep (GPT-J, sampled T=1.0)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "nshot_aggregate.png", dpi=160)
    plt.close(fig)

    hard = [t for t in tasks if curves[t][1][6] < 0.1]
    print(f"wrote {OUT}; tasks with n=6 accuracy < 0.1: {len(hard)}: {hard}")


if __name__ == "__main__":
    main()
