#!/usr/bin/env python
"""Summarise the 6-shot RANDOM-label steering experiment (sixshot_randomlabel_steer.py).

Reads artifacts/69_task_run/raw_mean_steering/sixshot_randomlabel/<task>.json and, for
context, the references already computed in this study: sixshot_dummy/<task>.json (real
6-shot baseline + dummy-slot steering) and read_dir_steering_1shot/<task>__*.json (0-shot,
real 1-shot).

Writes to results/69_task_run/bottom_up_read_features/steering_results/sixshot_randomlabel/:
  by_task.png     per-task bars: random-6 unsteered | random-6 steered (best alpha) |
                  dummy-6 steered (best alpha) | real 6-shot   (* = held-out task)
  alpha_curve.png mean accuracy vs alpha for random-label vs dummy-label steering, with
                  unsteered / real-6 / real-1-shot reference lines
  summary.csv, per_task_acc.csv
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
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

RL = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sixshot_randomlabel"
SS = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sixshot_dummy"
REF = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / "sixshot_randomlabel"
ALPHAS = (0.5, 1.0, 2.0, 4.0)


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(t for t in group if (RL / f"{t}.json").exists())
    missing = [t for t in sorted(group) if t not in tasks]
    if missing:
        print(f"WARNING: missing {len(missing)}: {missing[:5]}")
    d = {t: json.load(open(RL / f"{t}.json")) for t in tasks}
    dd = {t: json.load(open(SS / f"{t}.json")) for t in tasks}
    grp = np.array([group[t] for t in tasks])

    base = np.array([d[t]["conditions"]["random6_baseline"]["acc"] for t in tasks])
    steer = {a: np.array([d[t]["conditions"][f"random6_steer_a{a}"]["acc"] for t in tasks])
             for a in ALPHAS}
    best = np.max(np.stack([steer[a] for a in ALPHAS]), axis=0)
    # references from the dummy-slot study (same protocol / prompt bank)
    real6 = np.array([dd[t]["conditions"]["real6_baseline"]["acc"] for t in tasks])
    dsteer = {a: np.array([dd[t]["conditions"][f"dummy6_steer_a{a}"]["acc"] for t in tasks])
              for a in ALPHAS}
    dbest = np.max(np.stack([dsteer[a] for a in ALPHAS]), axis=0)
    r1 = np.array([json.load(open(REF / f"{t}__real_1shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    zs = np.array([json.load(open(REF / f"{t}__zero_shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    gold_coll = np.array([np.mean(d[t]["n_label_eq_gold"]) for t in tasks])
    print(f"mean sampled-label==gold collisions per prompt: {gold_coll.mean():.3f} "
          f"(max task {tasks[int(np.argmax(gold_coll))]} {gold_coll.max():.2f})")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [["condition", "task_group", "mean_acc", "median_acc"]]
    named = [("zero_shot", zs), ("random6_unsteered", base),
             ("random6_steered_best", best), ("dummy6_steered_best", dbest),
             ("real_1shot", r1), ("real_6shot", real6)] + \
            [(f"random6_steer_a{a}", steer[a]) for a in ALPHAS]
    for name, arr in named:
        for g in ("train", "heldout", "all"):
            m = np.ones(len(tasks), bool) if g == "all" else grp == g
            rows.append([name, g, round(float(arr[m].mean()), 4),
                         round(float(np.median(arr[m])), 4)])
    with open(OUT / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for r in rows:
        if r[1] == "all":
            print("  ".join(str(x) for x in r))

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
    ax.plot(ALPHAS, [float(steer[a].mean()) for a in ALPHAS], "o-", color="tab:red", lw=2,
            label="6-shot RANDOM labels, steered at all label tokens (L6)")
    ax.plot(ALPHAS, [float(dsteer[a].mean()) for a in ALPHAS], "o--", color="tab:purple",
            lw=1.6, label="6-shot dummy '_', steered at all 6 slots (L6)")
    ax.axhline(float(base.mean()), color="0.45", ls=":", lw=1.2,
               label=f"6-shot random unsteered = {base.mean():.3f}")
    ax.axhline(float(real6.mean()), color="tab:green", lw=1.4,
               label=f"real 6-shot demos = {real6.mean():.3f}")
    ax.axhline(float(r1.mean()), color="tab:olive", ls="--", lw=1.2,
               label=f"real 1-shot demo = {r1.mean():.3f}")
    ax.set_xscale("log"); ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
    ax.set_xlabel("alpha (x the vector's own norm)")
    ax.set_ylabel(f"mean T=1 exact-match accuracy ({len(tasks)} tasks)")
    ax.set_title("6-shot random-label steering at L6 (all label tokens)", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curve.png", bbox_inches="tight")

    order = np.argsort(best)
    labels = [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in order]
    x = np.arange(len(tasks))
    w = 0.21
    fig, ax = plt.subplots(figsize=(max(15, 0.4 * len(tasks)), 7.0), dpi=150)
    ax.bar(x - 1.5 * w, base[order], w, color="0.72", label="6-shot random, unsteered")
    ax.bar(x - 0.5 * w, best[order], w, color="tab:red",
           label="6-shot random, steered (best alpha)")
    ax.bar(x + 0.5 * w, dbest[order], w, color="tab:purple",
           label="6-shot dummy '_', steered (best alpha)")
    ax.bar(x + 1.5 * w, real6[order], w, color="tab:green", label="real 6-shot demos")
    ax.set_xticks(x, labels, rotation=90, fontsize=6.4)
    ax.set_ylabel("T=1 sampled exact-match accuracy (150 prompts)")
    ax.set_title("6-shot random-label steering at L6 by task — * = held-out", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "by_task.png", bbox_inches="tight")

    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["task", "group", "zero_shot", "random6_unsteered", "real_1shot",
                     "random6_steered_best", "dummy6_steered_best", "real_6shot",
                     "mean_labels_eq_gold"] + [f"random6_steer_a{a}" for a in ALPHAS])
        for i, t in enumerate(tasks):
            w_.writerow([t, group[t], zs[i], base[i], r1[i], best[i], dbest[i], real6[i],
                         round(float(gold_coll[i]), 3)] + [steer[a][i] for a in ALPHAS])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
