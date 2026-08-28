#!/usr/bin/env python
"""Figures + CSVs for the two-knob carrier/unique steering grid (twoknob_dummy).

Writes into results/.../steering_results/twoknob_dummy/:
  per_task_acc.csv   per-task accuracy for every (a,b) cell + baseline
  summary.csv        aggregate means (all cells, baseline, references)
  knob_heatmap.png   5x5 mean-accuracy heatmap (carrier knob a vs unique knob b),
                     references in the title band
  knob_curves.png    diagonal a=b vs the two edges (a-only, b-only) as curves
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

AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "twoknob_dummy"
SR = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
OUT = SR / "twoknob_dummy"
KNOBS = (0.0, 1.0, 2.0, 4.0, 8.0)
BLUE, ORANGE, GRAY = "#2f7fe0", "#e07b2f", "#6b7280"


def agg(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["task_group"] == "all":
            out[r["condition"]] = float(r["mean_acc"])
    return out


def main():
    rows = {}
    for f in sorted(AR.glob("*.json")):
        r = json.load(open(f))
        row = {"task": r["task"], "group": r["group"],
               "baseline": r["conditions"]["dummy6_baseline"]["acc"]}
        for a in KNOBS:
            for b in KNOBS:
                row[f"c{a:g}_u{b:g}"] = \
                    r["conditions"][f"dummy6_twoknob_c{a:g}_u{b:g}"]["acc"]
        rows[r["task"]] = row
    tasks = sorted(rows)
    assert len(tasks) == 69

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[tasks[0]].keys())
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(c):
        return float(np.mean([rows[t][c] for t in tasks]))

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_acc"])
        for c in cols[2:]:
            w.writerow([c, round(mean_of(c), 4)])

    dummy_ref = agg(SR / "sixshot_dummy" / "summary.csv")
    swap_ref = agg(SR / "taskunique_svd_dummy" / "summary.csv")
    fm_best, sw_peak = dummy_ref["dummy6_steered_best"], swap_ref["dummy6_swap1_a32.0"]
    real6 = dummy_ref["real_6shot"]

    M = np.array([[mean_of(f"c{a:g}_u{b:g}") for b in KNOBS] for a in KNOBS])

    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=200)
    im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0.0,
                   vmax=max(M.max(), fm_best))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] < 0.6 * M.max() else "black")
    ax.set_xticks(range(5), [f"{b:g}" for b in KNOBS], fontsize=11)
    ax.set_yticks(range(5), [f"{a:g}" for a in KNOBS], fontsize=11)
    ax.set_xlabel("unique knob b  ($b \\cdot n_A \\cdot v_1$, x natural)", fontsize=12)
    ax.set_ylabel("carrier knob a  ($a \\cdot c_A \\cdot \\hat{c}$, x natural)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.82, label="accuracy (mean over 69 tasks)")
    ax.set_title("Two-knob steering, 6-shot '_' scaffold (L6 two-coordinate swap)\n"
                 f"refs: full mean best {fm_best:.3f} | swap $\\alpha$=32 {sw_peak:.3f} | "
                 f"real 6-shot {real6:.3f}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "knob_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=200)
    xs = np.array(KNOBS[1:])
    ax.plot(xs, [mean_of(f"c{k:g}_u{k:g}") for k in KNOBS[1:]], "-o", color=BLUE,
            lw=2.4, ms=7, label="diagonal a = b (ratio preserved)")
    ax.plot(xs, [mean_of(f"c0_u{k:g}") for k in KNOBS[1:]], "-s", color=ORANGE,
            lw=2.2, ms=6, label="unique only (a = 0)")
    ax.plot(xs, [mean_of(f"c{k:g}_u0") for k in KNOBS[1:]], "-^", color=GRAY,
            lw=2.2, ms=6, label="carrier only (b = 0)")
    ax.plot([xs[0] * 0.55], [mean_of("c0_u0")], marker="D", ms=7, color="0.35",
            ls="none", label="removal only (0,0)")
    for v, c, lab in ((fm_best, "#7fb3ea", "full mean best "),
                      (sw_peak, "#c98a5a", "swap $\\alpha$=32 "),
                      (real6, "0.25", "real 6-shot ")):
        ax.axhline(v, color=c, lw=1.6, ls=(0, (5, 3)))
        ax.annotate(f"{lab}{v:.3f}", (xs[-1], v), ha="right", va="bottom",
                    fontsize=10, color=c if c != "0.25" else "0.25")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k:g}" for k in KNOBS[1:]], fontsize=11)
    ax.set_xlabel("knob value (x natural coordinate)", fontsize=12)
    ax.set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.set_title("Diagonal vs edges of the two-knob grid", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "knob_curves.png", bbox_inches="tight")
    plt.close(fig)

    print(f"baseline={mean_of('baseline'):.3f}  removal={mean_of('c0_u0'):.3f}")
    print("diag:", "  ".join(f"{k:g}:{mean_of(f'c{k:g}_u{k:g}'):.3f}" for k in KNOBS[1:]))
    print("u-only:", "  ".join(f"{k:g}:{mean_of(f'c0_u{k:g}'):.3f}" for k in KNOBS[1:]))
    print("c-only:", "  ".join(f"{k:g}:{mean_of(f'c{k:g}_u0'):.3f}" for k in KNOBS[1:]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
