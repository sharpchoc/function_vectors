#!/usr/bin/env python
"""Retention curve: how few of the 40 read-feature PCs keep the steering effect?

Reads artifacts/69_task_run/raw_mean_steering/sparse_pc40/eval/<task>.json
(eval_sparse_pc40.py) and writes to results/69_task_run/raw_mean_steering/sparse_pc40/:

  retention_curve.png  x = number of PCs kept, y = steering accuracy (best alpha), with the
                       full read feature as the 100% reference and the all-40 truncation shown.
  alpha_curves.png     accuracy vs alpha, one line per subset size.
  summary.csv          per condition x alpha: mean/median accuracy, train/heldout split.
  per_task_acc.csv     per task, every condition at its best alpha.
"""
import csv
import json
import re
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

AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sparse_pc40" / "eval"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "dimensionality_analysis" / "sparse_pc40"
ALPHAS = (0.5, 1.0, 2.0, 4.0)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAPHITE, GRAY = "#2a78d6", "#52514e", "#b8b7b2"


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(t for t in group if (AR / f"{t}.json").exists())
    missing = [t for t in sorted(group) if t not in tasks]
    if missing:
        print(f"WARNING: missing {len(missing)}: {missing[:5]}")
    data = {t: json.load(open(AR / f"{t}.json")) for t in tasks}
    grp = np.array([group[t] for t in tasks])
    sets = data[tasks[0]]["selected_sets"]

    vnames = sorted({c.rsplit("_a", 1)[0] for c in data[tasks[0]]["conditions"]
                     if c != "baseline"})
    acc = {v: {a: np.array([data[t]["conditions"][f"{v}_a{a}"]["acc"] for t in tasks])
               for a in ALPHAS} for v in vnames}
    best = {v: np.max(np.stack([acc[v][a] for a in ALPHAS]), axis=0) for v in vnames}
    base = np.array([data[t]["conditions"]["baseline"]["acc"] for t in tasks])

    def n_dims(v):
        if v == "full":
            return None
        if v == "pc40":
            return 40
        key = v[len("sel_"):]
        return len(sets.get(key, sets.get(key.replace("lam", ""), [])))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [["condition", "n_pcs", "alpha", "task_group", "mean_acc", "median_acc"]]
    for v in ["baseline"] + vnames:
        for a in (["-"] if v == "baseline" else ALPHAS):
            arr = base if v == "baseline" else acc[v][a]
            for g in ("train", "heldout", "all"):
                m = np.ones(len(tasks), bool) if g == "all" else grp == g
                rows.append([v, n_dims(v) if v != "baseline" else 0, a, g,
                             round(float(arr[m].mean()), 4),
                             round(float(np.median(arr[m])), 4)])
    with open(OUT / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    full_mean = float(best["full"].mean())
    pts = []
    for v in vnames:
        k = n_dims(v)
        if k is None:
            continue
        pts.append((k, float(best[v].mean()), v))
    pts.sort()
    print(f"full read feature (best alpha): {full_mean:.4f}   unsteered {base.mean():.4f}")
    for k, m, v in pts:
        print(f"  {k:>2d} PCs ({v}): {m:.4f}   = {m/full_mean:.1%} of the full feature")

    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group", "baseline", "full"] +
                   [f"{v}_{n_dims(v)}pc" for v in vnames if v != "full"])
        for i, t in enumerate(tasks):
            w.writerow([t, group[t], base[i], round(float(best["full"][i]), 4)] +
                       [round(float(best[v][i]), 4) for v in vnames if v != "full"])

    # retention curve
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    xs = [k for k, _, _ in pts]
    ys = [m for _, m, _ in pts]
    ax.plot(xs, ys, "o-", color=BLUE, lw=2.6, ms=9, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    for i, (k, m, _) in enumerate(pts):
        # stagger labels so near-identical points (24 vs 25 PCs) do not overlap
        dy = 13 if i % 2 == 0 else -20
        ax.annotate(f"{m/full_mean:.0%}", (k, m), textcoords="offset points",
                    xytext=(0, dy), fontsize=11.5, color=BLUE, fontweight="bold", ha="center")
    ax.axhline(full_mean, color=GRAPHITE, lw=1.6,
               label=f"full read feature = {full_mean:.3f}")
    ax.axhline(float(base.mean()), color=GRAY, ls=":", lw=1.4,
               label=f"unsteered = {base.mean():.3f}")
    ax.set_xlabel("number of PC directions kept", fontsize=13, color=INK2)
    ax.set_ylabel("steering accuracy", fontsize=13, color=INK2)
    ax.set_title("The steering effect is spread across many directions",
                 fontsize=16, fontweight="bold", color=INK, loc="left", pad=14)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "retention_curve.png", bbox_inches="tight", facecolor=SURFACE)

    # alpha curves per subset
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    shades = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
    ordered = sorted([p for p in pts], key=lambda p: p[0])
    for i, (k, _, v) in enumerate(ordered):
        ax.plot(ALPHAS, [acc[v][a].mean() for a in ALPHAS], "o-", ms=6, lw=2,
                color=shades[i % len(shades)], label=f"{k} PCs")
    ax.plot(ALPHAS, [acc["full"][a].mean() for a in ALPHAS], "o--", ms=6, lw=2,
            color=GRAPHITE, label="full read feature")
    ax.axhline(float(base.mean()), color=GRAY, ls=":", lw=1.3)
    ax.set_xscale("log"); ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
    ax.set_xlabel("steering strength α", fontsize=12.5, color=INK2)
    ax.set_ylabel("steering accuracy", fontsize=12.5, color=INK2)
    ax.set_title("Dose response by subspace size", fontsize=14, fontweight="bold",
                 color=INK, loc="left", pad=12)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=9.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curves.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
