#!/usr/bin/env python
"""Summarise the FV-direction cue-token ablation study (ablate_fv_cue6.py).

Reads artifacts/69_task_run/FV_ablation/eval/<task>.json (8 ablation conditions per task)
and merges the 6-shot / 0-shot baselines from the sixshot_dummy per-task CSV (same T=1
sampled readout, same 150 prompts).

Writes to results/69_task_run/FV_ablation/:
  headline_bars.png  mean accuracy per condition (baselines + own/cf x zero/mean),
                     one panel per layer clamp (L9-27, L0-27)
  by_task_dots.png   per-task breakdown, same conditions, one panel per layer clamp
  summary.csv, per_task_acc.csv, cf_pairs.csv
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

EVAL = ARTIFACTS_ROOT / "69_task_run" / "FV_ablation" / "eval"
BASE_CSV = (TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" /
            "sixshot_dummy" / "per_task_acc.csv")
OUT = TASK69_RUN_DIR / "FV_ablation"
CFGS = ("L9to27", "L0to27")
CFG_TITLES = {"L9to27": "clamp layers 9–27", "L0to27": "clamp layers 0–27"}
# fixed condition order + colors; cf bars additionally hatched (identity not color-alone)
CONDS = ("zero_shot", "real_6shot", "own_zero", "own_mean", "cf_zero", "cf_mean")
COLORS = {"zero_shot": "#bfbfbf", "real_6shot": "#2ca02c",
          "own_zero": "#b2182b", "own_mean": "#ef8a62",
          "cf_zero": "#2166ac", "cf_mean": "#67a9cf"}
LABELS = {"zero_shot": "0-shot (no demos)", "real_6shot": "6-shot, unablated",
          "own_zero": "6-shot, own-FV zero-ablated", "own_mean": "6-shot, own-FV mean-ablated",
          "cf_zero": "6-shot, cf-task-FV zero-ablated", "cf_mean": "6-shot, cf-task-FV mean-ablated"}


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(t for t in group if (EVAL / f"{t}.json").exists())
    missing = [t for t in sorted(group) if t not in tasks]
    if missing:
        print(f"WARNING: missing {len(missing)}: {missing[:5]}")
    d = {t: json.load(open(EVAL / f"{t}.json")) for t in tasks}
    grp = np.array([group[t] for t in tasks])

    base = {}
    with open(BASE_CSV) as f:
        for row in csv.DictReader(f):
            base[row["task"]] = row
    acc = {"zero_shot": np.array([float(base[t]["zero_shot"]) for t in tasks]),
           "real_6shot": np.array([float(base[t]["real_6shot"]) for t in tasks])}
    for cfg in CFGS:
        for who in ("own", "cf"):
            for op in ("zero", "mean"):
                acc[f"{who}_{op}_{cfg}"] = np.array(
                    [d[t]["conditions"][f"{who}_{op}_{cfg}"]["acc"] for t in tasks])

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- csv outputs -------------------------------------------------------
    abl_cols = [f"{who}_{op}_{cfg}" for cfg in CFGS for who in ("own", "cf")
                for op in ("zero", "mean")]
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group", "zero_shot", "real_6shot"] + abl_cols +
                   ["cf_task", "family", "cf_family", "cos_own_cf"])
        for i, t in enumerate(tasks):
            w.writerow([t, group[t], acc["zero_shot"][i], acc["real_6shot"][i]] +
                       [acc[c][i] for c in abl_cols] +
                       [d[t]["cf_task"], d[t]["family"], d[t]["cf_family"], d[t]["cos_own_cf"]])

    rows = [["condition", "task_group", "mean_acc", "median_acc", "mean_drop_vs_6shot"]]
    for name in ["zero_shot", "real_6shot"] + abl_cols:
        for g in ("train", "heldout", "all"):
            m = np.ones(len(tasks), bool) if g == "all" else grp == g
            rows.append([name, g, round(float(acc[name][m].mean()), 4),
                         round(float(np.median(acc[name][m])), 4),
                         round(float((acc["real_6shot"][m] - acc[name][m]).mean()), 4)])
    with open(OUT / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for r in rows:
        if r[1] == "all":
            print("  ".join(str(x) for x in r))

    with open(OUT / "cf_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "family", "cf_task", "cf_family", "cos_own_cf"])
        for t in tasks:
            w.writerow([t, d[t]["family"], d[t]["cf_task"], d[t]["cf_family"],
                        d[t]["cos_own_cf"]])

    # ---- headline bars -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=150, sharey=True)
    for ax, cfg in zip(axes, CFGS):
        vals, cols, hats = [], [], []
        for c in CONDS:
            key = c if c in acc else f"{c}_{cfg}"
            vals.append(float(acc[key].mean()))
            cols.append(COLORS[c])
            hats.append("//" if c.startswith("cf_") else None)
        x = np.arange(len(CONDS))
        bars = ax.bar(x, vals, 0.62, color=cols, edgecolor="white", linewidth=1.0)
        for b, h in zip(bars, hats):
            if h:
                b.set_hatch(h)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color="0.25")
        ax.set_xticks(x, ["0-shot", "6-shot", "own\nzero", "own\nmean", "cf\nzero", "cf\nmean"],
                      fontsize=8.5)
        ax.set_title(CFG_TITLES[cfg], fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(f"mean T=1 exact-match accuracy ({len(tasks)} tasks, 150 prompts)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c],
                             hatch="//" if c.startswith("cf_") else None,
                             ec="white") for c in CONDS]
    fig.legend(handles, [LABELS[c] for c in CONDS], fontsize=7.6, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("FV-direction ablation at the final cue token (6-shot prompts, 69 tasks)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "headline_bars.png", bbox_inches="tight")

    # ---- per-task breakdown (dot rows) ------------------------------------
    order = np.argsort(-acc["real_6shot"])
    ylabels = [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in order]
    y = np.arange(len(tasks))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(9, 0.21 * len(tasks))), dpi=150,
                             sharey=True)
    for ax, cfg in zip(axes, CFGS):
        for i, oi in enumerate(order):
            ax.plot([acc[f"own_zero_{cfg}"][oi], acc["real_6shot"][oi]], [i, i],
                    color="0.85", lw=1.0, zorder=1)
        for c in CONDS:
            key = c if c in acc else f"{c}_{cfg}"
            mk = "s" if c.startswith("cf_") else "o"
            ax.scatter(acc[key][order], y, s=16, color=COLORS[c], marker=mk,
                       zorder=2, label=LABELS[c], linewidths=0)
        ax.set_yticks(y, ylabels, fontsize=5.6)
        ax.set_ylim(-1, len(tasks))
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(CFG_TITLES[cfg], fontsize=10)
        ax.grid(alpha=0.2, axis="x")
        ax.set_xlabel("T=1 exact-match accuracy")
    axes[0].legend(fontsize=6.4, loc="lower right")
    fig.suptitle("FV cue-token ablation by task (sorted by 6-shot baseline; * = held-out)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "by_task_dots.png", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
