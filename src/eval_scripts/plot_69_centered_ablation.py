#!/usr/bin/env python
"""Figures + summary CSV for the CENTERED-PCA rank-5 read-feature ablation.

Combines the pc5_centered eval JSONs (artifacts/69_task_run/bottom_up_ablation/
pc5_centered/n{1,6}shot/) with the uncentered rank-5 results
(multi_direction_ablation/per_task_acc.csv) so every centered bar sits next to its
uncentered counterpart.

Writes into results/.../ablation/multi_direction_ablation/centered/:
  per_task_acc.csv           task, group, cf_task, zero_shot, per n: baseline + the four
                             uncentered pc5 columns (copied) + the four centered pc5c columns
  aggregate_bars.png         two panels (1-shot | 6-shot); bar groups mean/zero ablation,
                             each [own 5-PC unc., own 5-PC centered, cf 5-PC unc.,
                             cf 5-PC centered]; dashed lines = unablated baseline and 0-shot
  per_task_bars_{1,6}shot.png  69 tasks x 4 centered bars (+ baseline markers)
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

AR = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "pc5_centered"
MDA = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation" / "multi_direction_ablation"
OUT = MDA / "centered"

PC5 = ("mean_ablation_pc5", "zero_ablation_pc5", "cf_mean_ablation_pc5", "cf_zero_ablation_pc5")
PC5C = tuple(c + "c" for c in PC5)   # CSV column names for the centered variant
COLOR = {"mean_ablation_pc5": "#7fb3ea", "mean_ablation_pc5c": "#2f7fe0",
         "cf_mean_ablation_pc5": "#c9dcf2", "cf_mean_ablation_pc5c": "#a9c9ef",
         "zero_ablation_pc5": "#e89285", "zero_ablation_pc5c": "#d94f3d",
         "cf_zero_ablation_pc5": "#f3cdc6", "cf_zero_ablation_pc5c": "#efb2a9"}


def main():
    rows = {r["task"]: dict(r) for r in csv.DictReader(open(MDA / "per_task_acc.csv"))}
    tasks = sorted(rows)
    assert len(tasks) == 69
    for t in tasks:
        for n in (1, 6):
            r = json.load(open(AR / f"n{n}shot" / f"{t}.json"))
            assert r["n_prompts"] == 150 and r["cf_task"] == rows[t]["cf_task"]
            assert "pc5_centered_bases" in r.get("bases_path", "")
            for c in PC5:
                rows[t][f"n{n}_{c}c"] = r["conditions"][c]["acc"]

    OUT.mkdir(parents=True, exist_ok=True)
    cols = (["task", "group", "cf_task", "zero_shot"]
            + [f"n{n}_{c}" for n in (1, 6) for c in ("baseline",) + PC5 + PC5C])
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(col):
        return float(np.mean([float(rows[t][col]) for t in tasks]))

    # ---------------- aggregate figure ----------------
    order = ["mean_ablation_pc5", "mean_ablation_pc5c",
             "cf_mean_ablation_pc5", "cf_mean_ablation_pc5c",
             "zero_ablation_pc5", "zero_ablation_pc5c",
             "cf_zero_ablation_pc5", "cf_zero_ablation_pc5c"]
    short = {c: ("cf " if c.startswith("cf_") else "own ")
             + ("centered" if c.endswith("c") else "uncent.") for c in order}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=200, sharey=True)
    for ax, n in zip(axes, (1, 6)):
        xs = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6])
        vals = [mean_of(f"n{n}_{c}") for c in order]
        ax.bar(xs, vals, color=[COLOR[c] for c in order], width=0.8)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x, v), ha="center", va="bottom", fontsize=9.5)
        base, zs = mean_of(f"n{n}_baseline"), mean_of("zero_shot")
        ax.axhline(base, color="0.35", lw=1.6, ls=(0, (5, 3)))
        ax.axhline(zs, color="0.6", lw=1.4, ls=(0, (2, 2)))
        ax.annotate(f"unablated {base:.3f}", (7.9, base), ha="right", va="bottom",
                    fontsize=10, color="0.25")
        ax.annotate(f"0-shot {zs:.3f}", (7.9, zs), ha="right", va="bottom",
                    fontsize=10, color="0.45")
        ax.set_xticks(xs)
        ax.set_xticklabels([short[c] for c in order], fontsize=10, rotation=20, ha="right")
        ax.text(1.5, -0.16, "mean ablation", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color="#2f7fe0")
        ax.text(6.1, -0.16, "zero ablation", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color="#d94f3d")
        ax.set_title(f"{n}-shot", fontsize=14)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    fig.suptitle("Rank-5 read-feature ablation: centered vs uncentered PCA bases "
                 "(all layers, all demo-label tokens)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUT / "aggregate_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- per-task figures ----------------
    for n in (1, 6):
        fig, ax = plt.subplots(figsize=(26, 7.5), dpi=170)
        x = np.arange(len(tasks))
        w = 0.2
        for ci, c in enumerate(PC5C):
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
        ax.set_title(f"Centered top-5-PC read-feature ablation, per task ({n}-shot)",
                     fontsize=14, fontweight="bold", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"per_task_bars_{n}shot.png", bbox_inches="tight")
        plt.close(fig)

    for n in (1, 6):
        print(f"n{n}: base={mean_of(f'n{n}_baseline'):.3f}  " + "  ".join(
            f"{c}={mean_of(f'n{n}_{c}'):.3f}" for c in PC5 + PC5C))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
