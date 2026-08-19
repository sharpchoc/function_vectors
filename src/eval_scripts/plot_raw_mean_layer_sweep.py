#!/usr/bin/env python
"""Summarise the raw mean-activation layer sweep (sweep_raw_mean_layers.py).

Reads artifacts/69_task_run/raw_mean_steering/<task>.json plus the existing 0-shot /
real-1-shot baselines, and writes to results/69_task_run/raw_mean_steering/:

  layer_curve.png     THE headline: mean accuracy over all 69 tasks vs injection layer
                      (one line per alpha + best-over-alpha), with unsteered / 0-shot /
                      real-1-shot reference lines and the task-agnostic shared-mean control
  by_task_best.png    per-task bars at the best layer: 0-shot | unsteered | raw mean |
                      real 1-shot   (* = held-out task)
  by_task_heatmap.png per-task x per-layer accuracy grid (best alpha per cell)
  layer_summary.csv   per layer x alpha: mean/median accuracy, train/heldout split
  per_task_by_layer.csv  task x layer matrix (best alpha per cell) + the three baselines
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

AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering"
REF = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "layer_selection"
ALPHAS = (0.5, 1.0, 2.0, 4.0)
LAYERS = list(range(28))


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(t for t in group if (AR / f"{t}.json").exists())
    missing = [t for t in sorted(group) if t not in tasks]
    if missing:
        print(f"WARNING: {len(missing)} tasks missing: {missing}")
    data = {t: json.load(open(AR / f"{t}.json")) for t in tasks}
    grp = np.array([group[t] for t in tasks])

    base = np.array([data[t]["conditions"]["baseline"]["acc"] for t in tasks])
    zs = np.array([json.load(open(REF / f"{t}__zero_shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    r1 = np.array([json.load(open(REF / f"{t}__real_1shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])

    # acc[layer][alpha] -> per-task array; and the shared-mean control if present
    acc = {l: {a: np.array([data[t]["conditions"][f"L{l}_a{a}"]["acc"] for t in tasks])
               for a in ALPHAS} for l in LAYERS}
    has_shared = all(f"sharedL0_a1.0" in data[t]["conditions"] for t in tasks)
    shared = ({l: {a: np.array([data[t]["conditions"][f"sharedL{l}_a{a}"]["acc"]
                                for t in tasks]) for a in ALPHAS} for l in LAYERS}
              if has_shared else None)
    best = {l: np.max(np.stack([acc[l][a] for a in ALPHAS]), axis=0) for l in LAYERS}
    mean_by_layer = {l: float(np.mean(best[l])) for l in LAYERS}
    best_layer = max(mean_by_layer, key=mean_by_layer.get)
    print(f"best layer = L{best_layer} (mean {mean_by_layer[best_layer]:.4f} over "
          f"{len(tasks)} tasks)")
    for l in LAYERS:
        per_a = "  ".join(f"a{a}={np.mean(acc[l][a]):.3f}" for a in ALPHAS)
        print(f"  L{l:<2d} best-over-alpha={mean_by_layer[l]:.4f}   {per_a}")

    OUT.mkdir(parents=True, exist_ok=True)
    # ---- layer_summary.csv ----
    with open(OUT / "layer_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "alpha", "mean_acc_all", "median_acc_all", "mean_acc_train",
                    "mean_acc_heldout", "mean_shared_control"])
        for l in LAYERS:
            for a in ALPHAS:
                v = acc[l][a]
                sh = float(np.mean(shared[l][a])) if shared else ""
                w.writerow([l, a, round(float(v.mean()), 4), round(float(np.median(v)), 4),
                            round(float(v[grp == "train"].mean()), 4),
                            round(float(v[grp == "heldout"].mean()), 4),
                            round(sh, 4) if sh != "" else ""])
            w.writerow([l, "best", round(mean_by_layer[l], 4),
                        round(float(np.median(best[l])), 4),
                        round(float(best[l][grp == "train"].mean()), 4),
                        round(float(best[l][grp == "heldout"].mean()), 4), ""])

    # ---- layer_curve.png (the headline) ----
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    for a, col in zip(ALPHAS, ("#c6dbef", "#6baed6", "#2171b5", "#08306b")):
        ax.plot(LAYERS, [float(np.mean(acc[l][a])) for l in LAYERS], "o-", ms=3,
                color=col, lw=1.2, label=f"raw mean, alpha={a}")
    ax.plot(LAYERS, [mean_by_layer[l] for l in LAYERS], "o-", ms=4.5, color="tab:red",
            lw=2.0, label="raw mean, best alpha per layer")
    if shared:
        ax.plot(LAYERS, [float(np.mean(np.max(np.stack([shared[l][a] for a in ALPHAS]),
                                              axis=0))) for l in LAYERS],
                "s--", ms=3.5, color="tab:purple", lw=1.3,
                label="shared-mean control (no task identity), best alpha")
    ax.axhline(float(r1.mean()), color="tab:green", lw=1.4,
               label=f"real 1-shot demo = {r1.mean():.3f}")
    ax.axhline(float(base.mean()), color="0.45", ls=":", lw=1.2,
               label=f"unsteered '_' scaffold = {base.mean():.3f}")
    ax.axhline(float(zs.mean()), color="tab:brown", ls="-.", lw=1.1,
               label=f"0-shot = {zs.mean():.3f}")
    ax.set_xticks(LAYERS, [str(l) for l in LAYERS], fontsize=8)
    ax.set_xlabel("injection layer (mean taken at the same layer)")
    ax.set_ylabel(f"mean T=1 exact-match accuracy ({len(tasks)} tasks)")
    ax.set_title("Raw mean-activation steering at the label slot, swept over depth\n"
                 "1-shot 'Q: {input} / A: _' scaffold, injection at the '_' token, "
                 "alpha x the vector's own norm", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "layer_curve.png", bbox_inches="tight")

    # ---- by_task_best.png ----
    bl = best[best_layer]
    order = np.argsort(bl)
    labels = [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in order]
    x = np.arange(len(tasks))
    w_ = 0.2
    fig, ax = plt.subplots(figsize=(max(15, 0.4 * len(tasks)), 7.0), dpi=150)
    ax.bar(x - 1.5 * w_, zs[order], w_, color="0.35", label="0-shot (no demo)")
    ax.bar(x - 0.5 * w_, base[order], w_, color="0.72", label="unsteered '_' scaffold")
    ax.bar(x + 0.5 * w_, bl[order], w_, color="tab:red",
           label=f"raw mean activation @L{best_layer} (best alpha)")
    ax.bar(x + 1.5 * w_, r1[order], w_, color="tab:green", label="real 1-shot demo")
    ax.set_xticks(x, labels, rotation=90, fontsize=6.4)
    ax.set_ylabel("T=1 sampled exact-match accuracy (150 prompts)")
    ax.set_title(f"Raw mean-activation steering at the best layer (L{best_layer}) by task "
                 "— * = held-out", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "by_task_best.png", bbox_inches="tight")

    # ---- by_task_heatmap.png (task x layer) ----
    M = np.stack([best[l] for l in LAYERS], axis=1)   # (tasks, layers)
    row_order = np.argsort(-M.max(axis=1))
    fig, ax = plt.subplots(figsize=(12, max(9, 0.16 * len(tasks))), dpi=150)
    im = ax.imshow(M[row_order], aspect="auto", cmap="magma", vmin=0,
                   vmax=float(np.percentile(M, 99)))
    ax.set_xticks(range(len(LAYERS)), [str(l) for l in LAYERS], fontsize=7)
    ax.set_yticks(range(len(tasks)),
                  [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in row_order],
                  fontsize=5.5)
    ax.set_xlabel("injection layer")
    fig.colorbar(im, ax=ax, label="accuracy (best alpha)")
    ax.set_title("Raw mean-activation steering: task x layer (best alpha per cell)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "by_task_heatmap.png", bbox_inches="tight")

    # ---- per_task_by_layer.csv ----
    with open(OUT / "per_task_by_layer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group", "zero_shot", "unsteered", "real_1shot"] +
                   [f"L{l}" for l in LAYERS] + ["best_layer", "best_acc"])
        for i, t in enumerate(tasks):
            row_vals = [round(float(best[l][i]), 4) for l in LAYERS]
            bi = int(np.argmax(row_vals))
            w.writerow([t, group[t], zs[i], base[i], r1[i]] + row_vals +
                       [LAYERS[bi], row_vals[bi]])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
