#!/usr/bin/env python
"""Layer curve + summary CSV for the fixed-vector (L6/7 mean + top-1 dir) 1-shot sweep.

Aggregates artifacts/69_task_run/l67top1_steering/<task>.json (sweep_l67top1_layers.py):
per layer, the mean over tasks of the per-task best-alpha accuracy ("best" convention of
the raw-mean sweep). Overlays the raw matched-site m_A(L) curve (layer_selection/
layer_summary.csv) and the real-1-shot baseline for reference.

Writes results/69_task_run/bottom_up_read_features/steering_results/l67top1/:
  sweep_layer_summary.csv   layer, alpha rows + per-task-best row (mean/median, train/heldout)
  sweep_layer_curve.png     presentation curve
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

import os
_SUF = os.environ.get("L67_SUFFIX", "")          # "" (bank b) or "_bankA"
AR = ARTIFACTS_ROOT / "69_task_run" / ("l67top1_steering" + _SUF)
REFDIR = TASK69_RUN_DIR / "bottom_up_read_features" / "layer_selection"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / ("l67top1" + _SUF)
ALPHAS = (0.5, 1.0, 2.0, 4.0)
LAYERS = list(range(28))


def main():
    files = sorted(AR.glob("*.json"))
    assert len(files) == 69, f"expected 69 task files, found {len(files)}"
    data = {f.stem: json.load(open(f)) for f in files}
    tasks = sorted(data)
    grp = {t: data[t]["group"] for t in tasks}

    acc = {l: {a: np.array([data[t]["conditions"][f"L{l}_a{a}"]["acc"] for t in tasks])
               for a in ALPHAS} for l in LAYERS}
    best = {l: np.max(np.stack([acc[l][a] for a in ALPHAS]), axis=0) for l in LAYERS}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sweep_layer_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "alpha", "mean_acc_all", "median_acc_all",
                    "mean_acc_train", "mean_acc_heldout"])
        tr = np.array([grp[t] == "train" for t in tasks])
        for l in LAYERS:
            for a in ALPHAS:
                v = acc[l][a]
                w.writerow([l, a, round(float(v.mean()), 4), round(float(np.median(v)), 4),
                            round(float(v[tr].mean()), 4), round(float(v[~tr].mean()), 4)])
            v = best[l]
            w.writerow([l, "best", round(float(v.mean()), 4), round(float(np.median(v)), 4),
                        round(float(v[tr].mean()), 4), round(float(v[~tr].mean()), 4)])

    curve = [float(best[l].mean()) for l in LAYERS]
    # reference: raw matched-site m_A(L) sweep + real 1-shot baseline
    raw = {}
    for r in csv.DictReader(open(REFDIR / "layer_summary.csv")):
        if r["alpha"] == "best":
            raw[int(r["layer"])] = float(r["mean_acc_all"])
    r1 = np.mean([float(r["real_1shot"])
                  for r in csv.DictReader(open(REFDIR / "per_task_by_layer.csv"))])

    TEAL, PURPLE, INK, MUTED, GREEN = "#0e7c6b", "#7c3aad", "#181c1e", "#5d6771", "#2e8b57"
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.plot(LAYERS, curve, color=PURPLE, lw=2.2, marker="o", ms=5, mfc=PURPLE,
            mec="white", mew=0.9, zorder=4,
            label="fixed $w_A$ = L6/7 mean + top-1 dir (best $\\alpha$)")
    ax.plot(LAYERS, [raw[l] for l in LAYERS], color=TEAL, lw=1.6, alpha=0.55, zorder=3,
            label="matched-site raw mean $m_A(\\ell)$ (best $\\alpha$)")
    ax.axhline(r1, color=GREEN, lw=1.6, ls=(0, (5, 2.5)), zorder=2,
               label=f"real 1-shot demonstration = {r1:.3f}")
    pk = int(np.argmax(curve))
    ax.set_xlim(-0.6, 27.6)
    ax.set_ylim(0, max(0.245, max(curve) + 0.04))
    ax.set_xticks(range(0, 28, 3))
    ax.set_xlabel("injection layer", color=INK, fontsize=11)
    ax.set_ylabel("steered accuracy (mean, 69 tasks)", color=INK, fontsize=11)
    ax.set_title("Dummy target token steering with the fixed L6/7+top-1 vector (1-shot)",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#c9ccc7")
    ax.tick_params(colors=MUTED, labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_layer_curve.png", facecolor="white")
    print(f"wrote {OUT}/sweep_layer_curve.png  peak L{pk} = {curve[pk]:.4f} "
          f"(raw-mean peak for reference: L{max(raw, key=raw.get)} = {max(raw.values()):.4f})")


if __name__ == "__main__":
    main()
