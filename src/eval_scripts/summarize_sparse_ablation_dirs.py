#!/usr/bin/env python
"""Summarize the sparse ablation-direction optimization (train_sparse_ablation_dirs.py).

Reads artifacts/69_task_run/sparse_ablation_dirs/{train,eval}/ and writes to
results/69_task_run/Read_direction_geometry/dot_perhead_unit_sparse_optimisation/:
  sparsity_curve.png   (A) n_selected vs mean T=1 accuracy (train / heldout tasks, with
                       baseline line); (B) per-task accuracy drop distributions per lambda
  c_vectors.png        the learned c per lambda (139 bars each), selected dirs marked
  per_task_acc.csv     task x condition accuracies (+ group column)
  summary.csv          per-lambda: n_selected, mean/median acc + drop, split by group
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "sparse_ablation_dirs"
OUT_DIR = TASK69_RUN_DIR / "Read_direction_geometry" / "dot_perhead_unit_sparse_optimisation"
LAMS = (0.003, 0.01, 0.03, 0.1)


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})

    trains = {lam: torch.load(AR / "train" / f"lambda_{lam}.pt", map_location="cpu",
                              weights_only=False) for lam in LAMS}
    conds = ["baseline"] + [f"lam{lam}_n{trains[lam]['n_selected']}" for lam in LAMS]

    files = sorted((AR / "eval").glob("*.json"))
    assert len(files) == 69, len(files)
    tasks, acc = [], {c: [] for c in conds}
    for f in files:
        d = json.load(open(f))
        tasks.append(d["task"])
        for c in conds:
            acc[c].append(d["conditions"][c]["acc"])
    A = {c: np.array(v) for c, v in acc.items()}
    grp = np.array([group[t] for t in tasks])
    base = A["baseline"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group"] + conds)
        for i, t in enumerate(tasks):
            w.writerow([t, grp[i]] + [A[c][i] for c in conds])

    rows = []
    for lam, c in zip((None,) + LAMS, conds):
        n_sel = 0 if lam is None else trains[lam]["n_selected"]
        for g in ("train", "heldout", "all"):
            m = np.ones(len(tasks), bool) if g == "all" else grp == g
            drop = (base - A[c])[m]
            rows.append([c, n_sel, g, round(float(A[c][m].mean()), 4),
                         round(float(np.median(A[c][m])), 4),
                         round(float(drop.mean()), 4), round(float(np.median(drop)), 4)])
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_selected", "task_group", "mean_acc", "median_acc",
                    "mean_drop", "median_drop"])
        w.writerows(rows)
        for r in rows:
            print("  ".join(str(x) for x in r))

    # (A) sparsity curve + (B) per-task drops
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=150)
    ax = axes[0]
    for g, col, mk in (("train", "tab:blue", "o"), ("heldout", "tab:red", "s")):
        m = grp == g
        ns = [trains[lam]["n_selected"] for lam in LAMS]
        ys = [float(A[f"lam{lam}_n{trains[lam]['n_selected']}"][m].mean()) for lam in LAMS]
        ax.plot(ns, ys, mk + "-", color=col, label=f"{g} tasks (ablated)")
        ax.axhline(float(base[m].mean()), color=col, ls=":", lw=1,
                   label=f"{g} baseline {float(base[m].mean()):.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("number of ablated directions (c > 0.5)")
    ax.set_ylabel("mean T=1 sampled accuracy")
    ax.set_title("(A) sparsity vs ICL destruction (mean over tasks)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1]
    rng = np.random.RandomState(0)
    ab_conds = conds[1:]
    for i, c in enumerate(ab_conds):
        for g, col in (("train", "tab:blue"), ("heldout", "tab:red")):
            m = grp == g
            drop = (base - A[c])[m]
            ax.scatter(np.full(m.sum(), i) + rng.uniform(-0.16, 0.16, m.sum()), drop,
                       s=10, alpha=0.5, color=col)
        ax.plot([i - 0.25, i + 0.25], [(base - A[c]).mean()] * 2, color="black", lw=2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(ab_conds)), ab_conds, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("accuracy drop vs baseline (per task)")
    ax.set_title("(B) per-task drops (blue=train, red=heldout, black=mean)")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Sparse ablation-direction optimization over the 139 dot_perhead-unit PCs — "
                 "mean-ablation at demo-label tokens, all layers, 69 tasks, T=1", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sparsity_curve.png", bbox_inches="tight")

    fig, axes = plt.subplots(len(LAMS), 1, figsize=(13, 2.1 * len(LAMS)), dpi=150, sharex=True)
    for ax, lam in zip(axes, LAMS):
        cvec = trains[lam]["c"].numpy()
        sel = set(trains[lam]["selected"])
        cols = ["tab:red" if j in sel else "tab:blue" for j in range(len(cvec))]
        ax.bar(np.arange(len(cvec)), cvec, color=cols, width=1.0)
        ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
        ax.set_ylabel("c")
        ax.set_title(f"lambda={lam}: {len(sel)} selected (red)", fontsize=9)
        ax.set_ylim(0, 1.02)
    axes[-1].set_xlabel("PC index (dot_perhead unit, centered pooled PCs, sorted by variance)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c_vectors.png", bbox_inches="tight")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
