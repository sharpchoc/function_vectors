#!/usr/bin/env python
"""Figures + summary CSV for the L5-7 top-1 (band-restricted) task-unique read-feature ablation.

Combines the meanremoved_L5to7_top1 (USER REQUEST 2026-08-31) eval JSONs (artifacts/69_task_run/bottom_up_ablation/
meanremoved_L5to7_top1/n{1,6}shot/) with the 11-direction results
(ablation/task_unique_top3/per_task_acc.csv) so each 1-dir bar sits next to its
11-dir counterpart.

Writes into results/.../ablation/task_unique_top1_L5to7/:
  per_task_acc.csv           task, group, cf_task, zero_shot, per n: baseline + mr11
                             columns (copied) + the four mr1 columns
  aggregate_bars.png         two panels (1-shot | 6-shot); bar groups mean/zero ablation,
                             each [own 3-dir, own 11-dir, cf 3-dir, cf 11-dir]
  per_task_bars_{1,6}shot.png  69 tasks x 4 mr3 bars (+ baseline markers)
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

AR = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved_L5to7_top1"
ABL = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation"
OUT = ABL / "task_unique_top1_L5to7"

JSON_CONDS = ("mean_ablation_pc5", "zero_ablation_pc5",
              "cf_mean_ablation_pc5", "cf_zero_ablation_pc5")   # names inside the JSONs
MR1 = ("mean_ablation_mr157", "zero_ablation_mr157",
       "cf_mean_ablation_mr157", "cf_zero_ablation_mr157")
MR3REF = ("mean_ablation_mr3", "zero_ablation_mr3",
        "cf_mean_ablation_mr3", "cf_zero_ablation_mr3")
COLOR = {"mean_ablation_mr157": "#2f7fe0", "cf_mean_ablation_mr157": "#a9c9ef",
         "mean_ablation_mr3": "#7fb3ea", "cf_mean_ablation_mr3": "#c9dcf2",
         "zero_ablation_mr157": "#d94f3d", "cf_zero_ablation_mr157": "#efb2a9",
         "zero_ablation_mr3": "#e89285", "cf_zero_ablation_mr3": "#f3cdc6"}


def main():
    rows = {r["task"]: dict(r)
            for r in csv.DictReader(open(ABL / "task_unique_top3" / "per_task_acc.csv"))}
    tasks = sorted(rows)
    assert len(tasks) == 69
    for t in tasks:
        for n in (1, 6):
            r = json.load(open(AR / f"n{n}shot" / f"{t}.json"))
            assert r["n_prompts"] == 150 and r["cf_task"] == rows[t]["cf_task"]
            assert r["rank"] == 1 and "meanremoved_L5to7_top1_bases" in r["bases_path"]
            for jc, cc in zip(JSON_CONDS, MR1):
                rows[t][f"n{n}_{cc}"] = r["conditions"][jc]["acc"]

    OUT.mkdir(parents=True, exist_ok=True)
    cols = (["task", "group", "cf_task", "zero_shot"]
            + [f"n{n}_{c}" for n in (1, 6) for c in ("baseline",) + MR3REF + MR1])
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(col):
        return float(np.mean([float(rows[t][col]) for t in tasks]))

    order = ["mean_ablation_mr157", "cf_mean_ablation_mr157",
             "zero_ablation_mr157", "cf_zero_ablation_mr157"]
    short = {c: ("cf " if c.startswith("cf_") else "own ") + "L6-9" for c in order}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=200, sharey=True)
    for ax, n in zip(axes, (1, 6)):
        xs = np.array([0, 1, 2.6, 3.6])
        vals = [mean_of(f"n{n}_{c}") for c in order]
        ax.bar(xs, vals, color=[COLOR[c] for c in order], width=0.8)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x, v), ha="center", va="bottom", fontsize=9.5)
        base, zs = mean_of(f"n{n}_baseline"), mean_of("zero_shot")
        ax.axhline(base, color="0.35", lw=1.6, ls=(0, (5, 3)))
        ax.axhline(zs, color="0.6", lw=1.4, ls=(0, (2, 2)))
        ax.annotate(f"unablated {base:.3f}", (3.9, base), ha="right", va="bottom",
                    fontsize=10, color="0.25")
        ax.annotate(f"0-shot {zs:.3f}", (3.9, zs), ha="right", va="bottom",
                    fontsize=10, color="0.45")
        ax.set_xticks(xs)
        ax.set_xticklabels([short[c] for c in order], fontsize=10, rotation=20, ha="right")
        ax.text(0.5, -0.16, "mean ablation", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color="#2f7fe0")
        ax.text(3.1, -0.16, "zero ablation", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color="#d94f3d")
        ax.set_title(f"{n}-shot", fontsize=14)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    fig.suptitle("Task-unique (mean-removed, layers 5-7 only) top-1 read-feature ablation "
                 "(all layers, all demo-label tokens)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUT / "aggregate_bars.png", bbox_inches="tight")
    plt.close(fig)

    for n in (1, 6):
        fig, ax = plt.subplots(figsize=(26, 7.5), dpi=170)
        x = np.arange(len(tasks))
        w = 0.2
        for ci, c in enumerate(MR1):
            vals = [float(rows[t][f"n{n}_{c}"]) for t in tasks]
            lab = ("cf " if c.startswith("cf_") else "own ") + \
                  ("mean" if "mean" in c else "zero")
            ax.bar(x + (ci - 1.5) * w, vals, w, color=COLOR[c], label=lab)
        bl = [float(rows[t][f"n{n}_baseline"]) for t in tasks]
        ax.plot(x, bl, ls="none", marker="_", ms=13, mew=2.2, color="0.25",
                label="unablated baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=90, fontsize=7.5)
        ax.set_ylabel("accuracy")
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9, ncol=5, loc="upper right")
        ax.set_title(f"L5-7 top-1 task-unique read-feature ablation, per task ({n}-shot)",
                     fontsize=14, fontweight="bold", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"per_task_bars_{n}shot.png", bbox_inches="tight")
        plt.close(fig)

    for n in (1, 6):
        print(f"n{n}: base={mean_of(f'n{n}_baseline'):.3f}  " + "  ".join(
            f"{c}={mean_of(f'n{n}_{c}'):.3f}" for c in MR1 + MR3REF))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
