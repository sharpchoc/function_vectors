#!/usr/bin/env python
"""Figures + CSVs for the low-dim task-unique projection-swap steering run.

Reads artifacts/69_task_run/raw_mean_steering/taskunique_svd_dummy/<task>.json
(steer_taskunique_svd.py: h <- h - P_V h + alpha*s1*v1 at dummy '_' slots, L6) and the
meanfree_dummy per-task CSV for reference conditions (real n-shot, full-vector best,
mean-free best, shared-mean best).

Writes into results/.../steering_results/taskunique_svd_dummy/:
  per_task_acc.csv   task, group, s1, natural_coord1, references, all swap alphas + best
  summary.csv        aggregate means per condition (train/heldout/all)
  alpha_curve.png    two panels (dummy 1-shot | dummy 6-shot): mean acc vs alpha with
                     reference lines (real n-shot, full-mean best, mean-free best)
  by_task.png        per-task swap-best bars + reference markers (6-shot)
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
_BANKA = os.environ.get("BANKA") == "1"
_MR = os.environ.get("MEANRESID") == "1"   # alpha*u_A swap (mean-residual task-unique part, natural units)
AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / ("bankA_meanresid_swap_dummy" if _MR else "bankA_taskunique_svd_dummy" if _BANKA else "taskunique_svd_dummy")
SR = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
OUT = SR / ("taskunique_meanresid_swap" if _MR else "taskunique_svd_dummy")
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0) if _MR else (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0)
BLUE, ORANGE, GRAY = "#2f7fe0", "#e07b2f", "#6b7280"


def main():
    ref = {r["task"]: r for r in csv.DictReader(open(SR / "meanfree_dummy" / "per_task_acc.csv"))}
    tasks = sorted(ref)
    assert len(tasks) == 69
    rows = {}
    for t in tasks:
        r = json.load(open(AR / f"{t}.json"))
        row = {"task": t, "group": r["group"], "s1": r["s_top3"][0],
               "natural_coord1": r["natural_L6_coords"][0],
               "real_1shot": ref[t]["real_1shot"], "real_6shot": ref[t]["real_6shot"],
               "dummy1_fullvec_best": ref[t]["dummy1_fullvec_best"],
               "dummy6_fullvec_best": ref[t]["dummy6_fullvec_best"],
               "dummy1_meanfree_best": ref[t]["dummy1_meanfree_best"],
               "dummy6_meanfree_best": ref[t]["dummy6_meanfree_best"]}
        for n in (1, 6):
            row[f"dummy{n}_baseline"] = r["conditions"][f"dummy{n}_baseline"]["acc"]
            best = 0.0
            for a in ALPHAS:
                acc = r["conditions"][f"dummy{n}_swap1_a{a}"]["acc"]
                row[f"dummy{n}_swap1_a{a}"] = acc
                if a > 0:
                    best = max(best, acc)
            row[f"dummy{n}_swap1_best"] = best
        rows[t] = row

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[tasks[0]].keys())
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(col, grp=None):
        sel = [t for t in tasks if grp is None or rows[t]["group"] == grp]
        return float(np.mean([float(rows[t][col]) for t in sel]))

    conds = ([f"dummy{n}_baseline" for n in (1, 6)]
             + [f"dummy{n}_swap1_a{a}" for n in (1, 6) for a in ALPHAS]
             + [f"dummy{n}_swap1_best" for n in (1, 6)]
             + ["real_1shot", "real_6shot", "dummy1_fullvec_best", "dummy6_fullvec_best",
                "dummy1_meanfree_best", "dummy6_meanfree_best"])
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "task_group", "mean_acc"])
        for c in conds:
            for g in ("train", "heldout", None):
                w.writerow([c, g or "all", round(mean_of(c, g), 4)])

    # ---------------- alpha curve ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=200, sharey=True)
    xs = np.array(ALPHAS[1:])
    for ax, n in zip(axes, (1, 6)):
        ys = [mean_of(f"dummy{n}_swap1_a{a}") for a in ALPHAS[1:]]
        ax.plot(xs, ys, "-o", color=ORANGE, lw=2.2, ms=6,
                label=("swap steer $\\alpha\\, u_A$" if _MR else "swap steer $\\alpha\\, s_1 v_1$ (1 direction)"))
        ax.plot([xs[0] * 0.55], [mean_of(f"dummy{n}_swap1_a0.0")], marker="s", ms=7,
                color=GRAY, ls="none", label="$\\alpha$=0 (removal only)")
        for v, c, lab in ((mean_of(f"real_{n}shot"), "0.25", f"real {n}-shot"),
                          (mean_of(f"dummy{n}_fullvec_best"), BLUE, "full mean, best $\\alpha$")):
            ax.axhline(v, color=c, lw=1.5, ls=(0, (5, 3)))
            ax.annotate(f"{lab} {v:.3f}", (xs[-1], v), ha="right", va="bottom",
                        fontsize=10, color=c if c != "0.25" else "0.25")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([("%g" % a) for a in xs], fontsize=10)
        ax.axvline(8, color="0.85", lw=1)
        ax.annotate("natural scale\n($\\alpha\\approx$8)", (8, 0.6), fontsize=9,
                    color="0.5", ha="center")
        ax.set_xlabel("alpha", fontsize=12)
        ax.set_title(f"dummy {n}-shot scaffold", fontsize=13.5)
        ax.grid(axis="y", color="0.92")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("accuracy (T=1 sampled exact match,\nmean over 69 tasks)", fontsize=11)
    axes[0].legend(fontsize=10, loc="upper left", frameon=False)
    fig.suptitle("Steering with ONE task-unique direction: projection swap " +
                 ("$h \\leftarrow h - (h\\!\\cdot\\!\\hat u_A)\\hat u_A + \\alpha u_A$ at L6 dummy target slots" if _MR else "$h \\leftarrow h - (h\\!\\cdot\\!v_1)v_1 + \\alpha s_1 v_1$ at L6 dummy target slots"),
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "alpha_curve.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- per-task (6-shot) ----------------
    fig, ax = plt.subplots(figsize=(26, 7.5), dpi=170)
    x = np.arange(len(tasks))
    ax.bar(x, [float(rows[t]["dummy6_swap1_best"]) for t in tasks], 0.62, color=ORANGE,
           label="swap steer best alpha (1 direction)")
    ax.plot(x, [float(rows[t]["dummy6_fullvec_best"]) for t in tasks], ls="none",
            marker="_", ms=12, mew=2.2, color=BLUE, label="full mean best alpha")
    ax.plot(x, [float(rows[t]["real_6shot"]) for t in tasks], ls="none", marker="_",
            ms=12, mew=2.2, color="0.25", label="real 6-shot")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=90, fontsize=7.5)
    ax.set_ylabel("accuracy")
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=10, ncol=3, loc="upper right")
    ax.set_title("Single-direction swap steering per task (dummy 6-shot, best alpha)",
                 fontsize=14, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "by_task.png", bbox_inches="tight")
    plt.close(fig)

    for n in (1, 6):
        print(f"dummy{n}: " + "  ".join(
            f"a{a:g}={mean_of(f'dummy{n}_swap1_a{a}'):.3f}" for a in ALPHAS)
            + f"  best={mean_of(f'dummy{n}_swap1_best'):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
