#!/usr/bin/env python
"""Summarise the narrow-patching experiment (narrow_patch_L6.py) vs plain raw-mean steering.

Reads artifacts/69_task_run/raw_mean_steering/narrow_patch/<task>.json, the plain-steering
L6 numbers from artifacts/69_task_run/raw_mean_steering/<task>.json, and the existing
0-shot / real-1-shot baselines. Writes to results/69_task_run/raw_mean_steering/narrow_patch/:

  patch_vs_steer.png   per-task bars: unsteered | plain raw-mean steering @L6 (best alpha) |
                       narrow patch (pure, alpha=1) | real 1-shot   (* = held-out)
  alpha_curve.png      mean accuracy vs alpha for the patch family (alpha=0 remove-only
                       control included) with plain-steering L6 and baselines as lines
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

NP_AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "narrow_patch"
RMS_AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering"
REF = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / "narrow_patch"
PATCH_ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
STEER_ALPHAS = (0.5, 1.0, 2.0, 4.0)
LAYER = 6


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)
    grp = np.array([group[t] for t in tasks])

    npd = {t: json.load(open(NP_AR / f"{t}.json")) for t in tasks}
    rms = {t: json.load(open(RMS_AR / f"{t}.json")) for t in tasks}
    patch = {a: np.array([npd[t]["conditions"][f"patch_a{a}"]["acc"] for t in tasks])
             for a in PATCH_ALPHAS}
    steer_best = np.array([max(rms[t]["conditions"][f"L{LAYER}_a{a}"]["acc"]
                               for a in STEER_ALPHAS) for t in tasks])
    base = np.array([rms[t]["conditions"]["baseline"]["acc"] for t in tasks])
    zs = np.array([json.load(open(REF / f"{t}__zero_shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    r1 = np.array([json.load(open(REF / f"{t}__real_1shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    patch_best = np.max(np.stack([patch[a] for a in PATCH_ALPHAS if a > 0]), axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [["condition", "task_group", "mean_acc", "median_acc"]]
    named = [("unsteered", base), ("zero_shot", zs), ("real_1shot", r1),
             (f"steer_L{LAYER}_best", steer_best), ("patch_best_nonzero", patch_best)] + \
            [(f"patch_a{a}", patch[a]) for a in PATCH_ALPHAS]
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

    # alpha curve
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    ax.plot(PATCH_ALPHAS, [float(patch[a].mean()) for a in PATCH_ALPHAS], "o-",
            color="tab:purple", lw=2, label="narrow patch: (I-P)z + alpha*P m_A")
    ax.axhline(float(steer_best.mean()), color="tab:red", lw=1.4,
               label=f"plain steering @L{LAYER} (best alpha) = {steer_best.mean():.3f}")
    ax.axhline(float(r1.mean()), color="tab:green", lw=1.2,
               label=f"real 1-shot = {r1.mean():.3f}")
    ax.axhline(float(base.mean()), color="0.45", ls=":", lw=1.1,
               label=f"unsteered = {base.mean():.3f}")
    ax.set_xlabel("alpha on the replacement term (0 = remove-only, 1 = pure patch)")
    ax.set_ylabel(f"mean accuracy ({len(tasks)} tasks)")
    ax.set_title("Narrow 41-PC patching at the '_' slot, L6", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curve.png", bbox_inches="tight")

    # per-task bars
    order = np.argsort(patch[1.0])
    labels = [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in order]
    x = np.arange(len(tasks))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(15, 0.4 * len(tasks)), 7.0), dpi=150)
    ax.bar(x - 1.5 * w, base[order], w, color="0.72", label="unsteered '_' scaffold")
    ax.bar(x - 0.5 * w, steer_best[order], w, color="tab:red",
           label=f"plain raw-mean steering @L{LAYER} (best alpha)")
    ax.bar(x + 0.5 * w, patch[1.0][order], w, color="tab:purple",
           label="narrow patch (pure, alpha=1)")
    ax.bar(x + 1.5 * w, r1[order], w, color="tab:green", label="real 1-shot demo")
    ax.set_xticks(x, labels, rotation=90, fontsize=6.4)
    ax.set_ylabel("T=1 sampled exact-match accuracy (150 prompts)")
    ax.set_title("Narrow 41-PC patching vs plain steering at the label slot (L6) — "
                 "* = held-out", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "patch_vs_steer.png", bbox_inches="tight")

    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["task", "group", "unsteered", "zero_shot", "real_1shot",
                     f"steer_L{LAYER}_best"] + [f"patch_a{a}" for a in PATCH_ALPHAS])
        for i, t in enumerate(tasks):
            w_.writerow([t, group[t], base[i], zs[i], r1[i], steer_best[i]] +
                        [patch[a][i] for a in PATCH_ALPHAS])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
