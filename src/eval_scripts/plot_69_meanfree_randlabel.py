#!/usr/bin/env python
"""Figure + CSVs for the C-arm: mean-free (full r_A) steering on the random-label scaffold.

Writes into results/.../steering_results/meanfree_randlabel/:
  per_task_acc.csv   per-task: unsteered, meanfree alphas + best
  summary.csv        aggregate means
  alpha_curve.png    meanfree-on-randlabel curve vs references (full mean randlabel best,
                     top-1 swap randlabel peak/best, real 6-shot)
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

AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "meanfree_randlabel"
SR = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
OUT = SR / "meanfree_randlabel"
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
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
               "unsteered": r["conditions"]["random6_unsteered"]["acc"]}
        for a in ALPHAS:
            row[f"meanfree_a{a}"] = r["conditions"][f"random6_meanfree_a{a}"]["acc"]
        row["meanfree_best"] = max(row[f"meanfree_a{a}"] for a in ALPHAS)
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

    rl = agg(SR / "randlabel_swap" / "summary.csv")
    dummy_ref = agg(SR / "sixshot_dummy" / "summary.csv")

    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=200)
    xs = np.array(ALPHAS)
    ax.plot(xs, [mean_of(f"meanfree_a{a}") for a in ALPHAS], "-o", color=ORANGE,
            lw=2.4, ms=7, label="mean-free $r_A$ on RANDOM-label base")
    ax.plot([xs[0] * 0.7], [mean_of("unsteered")], marker="s", ms=7, color=GRAY,
            ls="none", label="unsteered")
    refs = ((rl["fullmean_best"], BLUE, "full mean on random-label base (best) "),
            (rl["swap1_best"], "#c98a5a", "top-1 unique swap, same base (best) "),
            (dummy_ref["real_6shot"], "0.25", "real 6-shot "))
    for v, c, lab in refs:
        ax.axhline(v, color=c, lw=1.6, ls=(0, (5, 3)))
        ax.annotate(f"{lab}{v:.3f}", (xs[-1], v), ha="right", va="bottom",
                    fontsize=10, color=c if c != "0.25" else "0.25")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=10)
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.set_title("C-arm: does full-rank task-unique content close the carrier gap on the\n"
                 "random-label base? (6-shot, 69 tasks)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curve.png", bbox_inches="tight")
    plt.close(fig)

    print("  ".join(f"a{a:g}={mean_of(f'meanfree_a{a}'):.3f}" for a in ALPHAS)
          + f"  best={mean_of('meanfree_best'):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
