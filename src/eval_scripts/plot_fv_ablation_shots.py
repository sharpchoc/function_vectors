#!/usr/bin/env python
"""FV-direction cue-token ablation: 6-shot vs 1-shot prompts (L9-27 clamp), one panel each.

Reads artifacts/69_task_run/FV_ablation/eval/<task>.json (6-shot study, 2026-08-19) and
eval_1shot/<task>.json (1-shot rerun, ablate_fv_cue6.py --n_shots 1 --layer_cfgs L9to27).
The unablated n-shot baseline is the in-run seed-matched real{n}_baseline; the 0-shot floor
is merged from the sixshot_dummy per-task CSV (same T=1 sampled readout, same 150 queries).

Writes to results/69_task_run/FV_ablation/:
  headline_bars_by_shots.png   mean accuracy per condition, one panel per shot count
  summary_1shot.csv, per_task_acc_1shot.csv   (6-shot equivalents already exist)
"""
import argparse
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

ABL = ARTIFACTS_ROOT / "69_task_run" / "FV_ablation"
BASE_CSV = (TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" /
            "sixshot_dummy" / "per_task_acc.csv")
OUT = TASK69_RUN_DIR / "FV_ablation"
CFG = "L9to27"
CONDS = ("zero_shot", "real", "own_zero", "own_mean", "cf_zero", "cf_mean")
COLORS = {"zero_shot": "#bfbfbf", "real": "#2ca02c",
          "own_zero": "#b2182b", "own_mean": "#ef8a62",
          "cf_zero": "#2166ac", "cf_mean": "#67a9cf"}
LABELS = {"zero_shot": "0-shot (no demos)", "real": "n-shot, unablated",
          "own_zero": "own-FV zero-ablated", "own_mean": "own-FV mean-ablated",
          "cf_zero": "cf-task-FV zero-ablated", "cf_mean": "cf-task-FV mean-ablated"}


def load(n_shots, tasks, base):
    evald = ABL / ("eval" if n_shots == 6 else f"eval_{n_shots}shot")
    d = {t: json.load(open(evald / f"{t}.json")) for t in tasks}
    acc = {"zero_shot": np.array([float(base[t]["zero_shot"]) for t in tasks]),
           "real": np.array([d[t]["conditions"][f"real{n_shots}_baseline"]["acc"]
                             for t in tasks])}
    for c in CONDS[2:]:
        acc[c] = np.array([d[t]["conditions"][f"{c}_{CFG}"]["acc"] for t in tasks])
    return d, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", type=int, default=[6, 1])
    args = ap.parse_args()
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    base = {r["task"]: r for r in csv.DictReader(open(BASE_CSV))}
    tasks = sorted(group)
    grp = np.array([group[t] for t in tasks])
    OUT.mkdir(parents=True, exist_ok=True)

    per_shot = {}
    for n in args.shots:
        d, acc = load(n, tasks, base)
        per_shot[n] = acc
        if n == 6:
            continue   # 6-shot CSVs already written by plot_fv_ablation.py
        with open(OUT / f"per_task_acc_{n}shot.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "group", "zero_shot", f"real_{n}shot"] +
                       [f"{c}_{CFG}" for c in CONDS[2:]] + ["cf_task", "cos_own_cf"])
            for i, t in enumerate(tasks):
                w.writerow([t, group[t], acc["zero_shot"][i], acc["real"][i]] +
                           [acc[c][i] for c in CONDS[2:]] + [d[t]["cf_task"], d[t]["cos_own_cf"]])
        rows = [["condition", "task_group", "mean_acc", "median_acc", f"mean_drop_vs_{n}shot"]]
        for c in CONDS:
            name = f"real_{n}shot" if c == "real" else (c if c == "zero_shot" else f"{c}_{CFG}")
            for g in ("train", "heldout", "all"):
                m = np.ones(len(tasks), bool) if g == "all" else grp == g
                rows.append([name, g, round(float(acc[c][m].mean()), 4),
                             round(float(np.median(acc[c][m])), 4),
                             round(float((acc["real"][m] - acc[c][m]).mean()), 4)])
        with open(OUT / f"summary_{n}shot.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        for r in rows:
            if r[1] == "all":
                print(f"{n}-shot  " + "  ".join(str(x) for x in r))

    # ---- headline bars: one panel per shot count ---------------------------
    fig, axes = plt.subplots(1, len(args.shots), figsize=(5.2 * len(args.shots), 4.6),
                             dpi=150, sharey=True, squeeze=False)
    for ax, n in zip(axes[0], args.shots):
        acc = per_shot[n]
        vals = [float(acc[c].mean()) for c in CONDS]
        x = np.arange(len(CONDS))
        bars = ax.bar(x, vals, 0.62, color=[COLORS[c] for c in CONDS],
                      edgecolor="white", linewidth=1.0)
        for b, c in zip(bars, CONDS):
            if c.startswith("cf_"):
                b.set_hatch("//")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color="0.25")
        ax.set_xticks(x, ["0-shot", f"{n}-shot", "own\nzero", "own\nmean", "cf\nzero", "cf\nmean"],
                      fontsize=8.5)
        ax.set_title(f"{n}-shot prompts", fontsize=10.5)
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel(f"mean T=1 exact-match accuracy ({len(tasks)} tasks, 150 prompts)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c],
                             hatch="//" if c.startswith("cf_") else None, ec="white")
               for c in CONDS]
    fig.legend(handles, [LABELS[c] for c in CONDS], fontsize=7.6, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("FV-Direction Ablation at the Final Cue Token (layers 9–27)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "headline_bars_by_shots.png", bbox_inches="tight")
    print(f"wrote {OUT / 'headline_bars_by_shots.png'}")


if __name__ == "__main__":
    main()
