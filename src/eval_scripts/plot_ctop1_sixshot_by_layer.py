#!/usr/bin/env python
"""6-shot dummy-target steering accuracy of u_A = c + n_A v1 vs injection layer.

Reads the per-layer summaries written by plot_l67top1_sixshot.py
(steering_results/ctop1/sixshot_L{L}/sixshot_summary.csv, L in {0,1,5,6,7}) and plots the
69-task mean accuracy for (i) the fixed dose alpha=2 and (ii) the per-task best alpha, with
dashed reference lines for the full-mean-at-L6 steering result and real 6-shot demonstrations
(both read from the sixshot_L0 summary's reference columns).

Writes results/69_task_run/bottom_up_read_features/steering_results/ctop1/
  sixshot_by_layer.csv   layer, n_tasks, mean a0.5..a4.0, mean best, train/heldout splits
  sixshot_by_layer.png
"""
import csv
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
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

import os
CTOP1 = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / os.environ.get("BYLAYER_NAME", "ctop1")
LAYERS = (0, 1, 5, 6, 7)
ALPHAS = ("a0.5", "a1.0", "a2.0", "a4.0")


def load(layer):
    rows = list(csv.DictReader(open(CTOP1 / f"sixshot_L{layer}" / "sixshot_summary.csv")))
    assert len(rows) == 69, (layer, len(rows))
    return rows


def mean(rows, key, group=None):
    vals = [float(r[key]) for r in rows if group is None or r["group"] == group]
    return float(np.mean(vals))


def main():
    per_layer = {L: load(L) for L in LAYERS}
    ref_rows = per_layer[0]
    ref_fullmean = mean(ref_rows, "ref_fullmeanL6_best")
    ref_real6 = mean(ref_rows, "ref_real6")

    out_rows = []
    for L, rows in per_layer.items():
        out_rows.append({"layer": L, "n_tasks": len(rows),
                         **{f"mean_{a}": round(mean(rows, a), 4) for a in ALPHAS},
                         "mean_best": round(mean(rows, "best"), 4),
                         "train_a2.0": round(mean(rows, "a2.0", "train"), 4),
                         "heldout_a2.0": round(mean(rows, "a2.0", "heldout"), 4),
                         "train_best": round(mean(rows, "best", "train"), 4),
                         "heldout_best": round(mean(rows, "best", "heldout"), 4)})
    out_rows.append({"layer": "ref_fullmeanL6_best", "n_tasks": 69, "mean_best": round(ref_fullmean, 4)})
    out_rows.append({"layer": "ref_real6", "n_tasks": 69, "mean_best": round(ref_real6, 4)})
    fields = list(out_rows[0])
    with open(CTOP1 / "sixshot_by_layer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    xs = np.arange(len(LAYERS))
    a2 = [mean(per_layer[L], "a2.0") for L in LAYERS]
    best = [mean(per_layer[L], "best") for L in LAYERS]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(ref_real6, ls="--", lw=1.2, color="0.35", zorder=2)
    ax.text(xs[-1] + 0.12, ref_real6 + 0.008, f"real 6-shot demos {ref_real6:.3f}",
            ha="right", va="bottom", fontsize=9, color="0.35")
    ax.axhline(ref_fullmean, ls="--", lw=1.2, color="#0e7c6b", zorder=2)
    ax.text(xs[0] - 0.35, ref_fullmean + 0.008,
            f"full mean $m_A$(L6) @L6, best $\\alpha$  {ref_fullmean:.3f}",
            ha="left", va="bottom", fontsize=9, color="#0e7c6b")
    ax.plot(xs, best, "-o", color="#eb6834", lw=2, ms=6, label="per-task best $\\alpha$", zorder=4)
    ax.plot(xs, a2, "-o", color="#2a78d6", lw=2, ms=6, label="$\\alpha$ = 2", zorder=4)
    for x, y in zip(xs, best):
        ax.text(x, y + 0.012, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5, color="#181c1e")
    for x, y in zip(xs, a2):
        ax.text(x, y - 0.014, f"{y:.3f}", ha="center", va="top", fontsize=8.5, color="#181c1e")
    ax.set_xticks(xs, [f"L{L}" for L in LAYERS])
    ax.set_xlim(-0.4, len(LAYERS) - 0.6)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("injection layer of " + ("$s_A = c + u_A$" if os.environ.get("BYLAYER_NAME", "ctop1").startswith("meanresid") else "$u_A = c + n_A v_1$") + " (block output, all six target slots)")
    ax.set_ylabel("6-shot dummy-target accuracy (mean, 69 tasks)")
    ax.set_title("6-shot steering by injection layer", loc="left", fontsize=11.5, pad=10)
    ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(CTOP1 / "sixshot_by_layer.png", facecolor="white")
    print(f"wrote {CTOP1}/sixshot_by_layer.{{png,csv}}")
    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
