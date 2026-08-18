#!/usr/bin/env python
"""Aggregate the seed-split ridge study (ridge_labeltoken_seedsplits.py, seeds 1001-1010).

Writes to results/69_task_run/labeltoken_fv_ridge/seedsplits/:
  seed_summary.csv       per seed: train/test pooled R^2, lambda, test task list
  per_task_heldout.csv   per task: how often held out, mean heldout R^2 (own-mean and
                         pool-ref), canonical-split values for comparison
  seed_r2.png            (A) test R^2 per seed with the canonical split marked;
                         (B) per-task mean heldout pool-ref R^2, tasks sorted, canonical
                         heldout tasks highlighted
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge" / "seedsplits"
OUT = TASK69_RUN_DIR / "labeltoken_fv_ridge" / "seedsplits"
CANONICAL_TEST_R2 = 0.4641
SEEDS = list(range(1001, 1011))


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    canonical_heldout = set(split["heldout_tasks"])
    data = {s: json.load(open(AR / f"seed{s}.json")) for s in SEEDS}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "seed_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_alpha", "r2_train_uniform", "r2_test_uniform",
                    "r2_test_weighted", "test_tasks"])
        for s in SEEDS:
            d = data[s]
            w.writerow([s, d["best_alpha"], d["r2_train_uniform"], d["r2_test_uniform"],
                        d["r2_test_weighted"], " ".join(d["test_tasks"])])
    te = np.array([data[s]["r2_test_uniform"] for s in SEEDS])
    tr = np.array([data[s]["r2_train_uniform"] for s in SEEDS])
    print(f"test R2 over {len(SEEDS)} seeds: mean {te.mean():.4f} sd {te.std():.4f} "
          f"range {te.min():.3f}-{te.max():.3f} | canonical {CANONICAL_TEST_R2}")
    print(f"train R2: mean {tr.mean():.4f} sd {tr.std():.4f}")

    per_task = defaultdict(list)
    for s in SEEDS:
        for t, r in data[s]["per_test_task"].items():
            per_task[t].append((r["r2_ownmean"], r["r2_poolref"]))
    rows = []
    for t, vals in sorted(per_task.items()):
        own = np.mean([v[0] for v in vals])
        pool = np.mean([v[1] for v in vals])
        rows.append([t, len(vals), round(float(own), 4), round(float(pool), 4),
                     "canonical_heldout" if t in canonical_heldout else ""])
    with open(OUT / "per_task_heldout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "times_heldout", "mean_r2_ownmean", "mean_r2_poolref", "note"])
        w.writerows(rows)
    never = [t for t in sorted(set(split["train_tasks"]) | canonical_heldout)
             if t not in per_task]
    print(f"tasks never held out across the 10 seeds: {len(never)}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150,
                             gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    ax.bar(range(len(SEEDS)), te, color="tab:blue")
    ax.axhline(CANONICAL_TEST_R2, color="tab:red", ls="--",
               label=f"canonical split = {CANONICAL_TEST_R2}")
    ax.axhline(float(te.mean()), color="0.3", ls=":",
               label=f"seed mean = {te.mean():.3f} (sd {te.std():.3f})")
    ax.set_xticks(range(len(SEEDS)), [str(s) for s in SEEDS], rotation=45, fontsize=7.5)
    ax.set_ylabel("held-out pooled R^2 (uniform)")
    ax.set_title("(A) test R^2 by split seed", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    rows_s = sorted(rows, key=lambda r: r[3])
    cols = ["tab:red" if r[4] else "tab:blue" for r in rows_s]
    ax.bar(range(len(rows_s)), [r[3] for r in rows_s], color=cols)
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xticks(range(len(rows_s)),
                  [r[0] + (" *" if r[4] else "") for r in rows_s],
                  rotation=90, fontsize=5.8)
    ax.set_ylabel("mean held-out R^2 (pool-ref)")
    ax.set_title("(B) per-task held-out R^2 averaged over the seeds where the task was "
                 "held out (red * = canonical held-out task)", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Seed-split robustness of the avg-label-token -> FV ridge", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "seed_r2.png", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
