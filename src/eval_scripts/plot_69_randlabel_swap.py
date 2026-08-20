#!/usr/bin/env python
"""Figure + CSVs for the hypothesis-1 test: task-unique swap on the random-label scaffold.

Combines the randlabel_swap eval JSONs with the peer full-mean run
(artifacts/.../sixshot_randomlabel) and the '_'-scaffold references
(steering_results/taskunique_svd_dummy + sixshot_dummy summaries).

Writes into results/.../steering_results/randlabel_swap/:
  per_task_acc.csv    per-task: unsteered, swap alphas + best, fullmean alphas + best
  summary.csv         aggregate means
  alpha_curve.png     swap-on-randlabel curve with reference lines (fullmean randlabel
                      best, fullmean '_' best, swap '_' peak, real 6-shot)
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

AR_SW = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "randlabel_swap"
AR_FM = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sixshot_randomlabel"
SR = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
OUT = SR / "randlabel_swap"
SW_AL = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
FM_AL = (0.5, 1.0, 2.0, 4.0)
BLUE, ORANGE, GRAY = "#2f7fe0", "#e07b2f", "#6b7280"


def agg(path, prefix):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["task_group"] == "all":
            out[r["condition"]] = float(r["mean_acc"])
    return out


def main():
    rows = {}
    for f in sorted(AR_SW.glob("*.json")):
        r = json.load(open(f))
        fm = json.load(open(AR_FM / f"{r['task']}.json"))
        row = {"task": r["task"], "group": r["group"],
               "random6_unsteered": r["conditions"]["random6_unsteered"]["acc"]}
        for a in SW_AL:
            row[f"swap1_a{a}"] = r["conditions"][f"random6_swap1_a{a}"]["acc"]
        row["swap1_best"] = max(row[f"swap1_a{a}"] for a in SW_AL if a > 0)
        for a in FM_AL:
            row[f"fullmean_a{a}"] = fm["conditions"][f"random6_steer_a{a}"]["acc"]
        row["fullmean_best"] = max(row[f"fullmean_a{a}"] for a in FM_AL)
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

    dummy_ref = agg(SR / "sixshot_dummy" / "summary.csv", "")
    swap_dummy = agg(SR / "taskunique_svd_dummy" / "summary.csv", "")

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=200)
    xs = np.array(SW_AL[1:])
    ax.plot(xs, [mean_of(f"swap1_a{a}") for a in SW_AL[1:]], "-o", color=ORANGE,
            lw=2.4, ms=7, label="task-unique swap on RANDOM-label base")
    ax.plot([xs[0] * 0.55], [mean_of("swap1_a0.0")], marker="s", ms=7, color=GRAY,
            ls="none", label="$\\alpha$=0 (removal only)")
    refs = ((mean_of("fullmean_best"), BLUE, "full mean on random-label base (best) "),
            (dummy_ref["dummy6_steered_best"], "#7fb3ea", "full mean on '_' base (best) "),
            (swap_dummy["dummy6_swap1_a32.0"], "#c98a5a", "swap on '_' base (peak $\\alpha$=32) "),
            (dummy_ref["real_6shot"], "0.25", "real 6-shot "))
    for v, c, lab in refs:
        ax.axhline(v, color=c, lw=1.6, ls=(0, (5, 3)))
        ax.annotate(f"{lab}{v:.3f}", (xs[-1], v), ha="right", va="bottom",
                    fontsize=10, color=c if c != "0.25" else "0.25")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([("%g" % a) for a in xs], fontsize=10)
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.set_title("Hypothesis-1 test: real-word (wrong-task) labels lift swap AND full-mean "
                 "steering equally —\nthe carrier gap does not close (6-shot, 69 tasks)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curve.png", bbox_inches="tight")
    plt.close(fig)

    print("  ".join(f"a{a:g}={mean_of(f'swap1_a{a}'):.3f}" for a in SW_AL)
          + f"  swap_best={mean_of('swap1_best'):.3f}  fullmean_best={mean_of('fullmean_best'):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
