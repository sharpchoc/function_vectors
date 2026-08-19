#!/usr/bin/env python
"""Alpha-sweep figure for one task's read-direction steering variants (agent_noun_to_verb).

Overlays the scaffold / injection-site variants produced by steer_read_dir_1shot.py and the
demo baselines (0-shot, blank-'_' scaffold, real 1-shot, real 10-shot from the pc50 run).

Outputs in results/69_task_run/Read_direction_geometry/steering/:
  steering_1shot_<task>.png, steering_1shot_<task>.csv
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
OUT = TASK69_RUN_DIR / "top_down_read_features" / "steering_results" / "steering"
TASK = "agent_noun_to_verb"
ALPHAS = (0.5, 1.0, 2.0, 4.0)
TENSHOT = 0.5733  # unablated baseline for this task, results/69_task_run/pc50_ablation


def main():
    runs = [
        ("const 'Input/Output', L7", f"{TASK}.json", ":"),
        ("sampled input + '_', L7", f"{TASK}__sampled_underscore.json", "-"),
        ("sampled input + '_', L7-20", f"{TASK}__sampled_underscore__L7-20.json", "--"),
        ("real 1-shot demo, L7", f"{TASK}__real_1shot.json", "-."),
    ]
    data = [(lbl, json.load(open(AR / f)), ls) for lbl, f, ls in runs]
    zs = json.load(open(AR / f"{TASK}__zero_shot.json"))["conditions"]["baseline"]["acc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150, sharey=True)
    for ax, fam, col in ((axes[0], "dot_perhead", "tab:blue"),
                         (axes[1], "cosine_perhead", "tab:orange")):
        for lbl, d, ls in data:
            ys = [d["conditions"][f"{fam}_a{a}"]["acc"] for a in ALPHAS]
            ax.plot(ALPHAS, ys, "o" + ls, color=col, alpha=1.0 if ls == "-" else 0.6,
                    label=f"steered: {lbl}")
        for lbl, d, ls in data:
            ax.axhline(d["conditions"]["baseline"]["acc"], color="0.45", ls=ls, lw=0.9,
                       label=f"unsteered: {lbl.split(',')[0]} = "
                             f"{d['conditions']['baseline']['acc']:.3f}")
        ax.axhline(zs, color="tab:red", ls="-", lw=1.1, label=f"0-shot (no demo) = {zs:.3f}")
        ax.axhline(TENSHOT, color="tab:green", ls="-", lw=1.2,
                   label=f"real 10-shot demos = {TENSHOT:.3f}")
        ax.set_xscale("log")
        ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
        ax.set_xlabel("alpha")
        ax.set_title(fam, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.2, loc="upper right")
    axes[0].set_ylabel("T=1 sampled exact-match accuracy (150 prompts)")
    axes[0].set_ylim(0, 0.62)
    fig.suptitle(f"Read-direction steering vs demo baselines — {TASK}\n"
                 "(natural-magnitude per-task read dir injected at the demo label slot)",
                 fontsize=10.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"steering_1shot_{TASK}.png", bbox_inches="tight")

    rows = [["variant", "condition", "acc"]]
    for lbl, d, _ in data:
        for c, v in d["conditions"].items():
            rows.append([lbl, c, v["acc"]])
    rows.append(["reference", "zero_shot_unsteered", zs])
    rows.append(["reference", "real_10shot_unsteered", TENSHOT])
    with open(OUT / f"steering_1shot_{TASK}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"0-shot {zs:.3f} | " + " | ".join(
        f"{lbl.split(',')[0]} base {d['conditions']['baseline']['acc']:.3f}"
        for lbl, d, _ in data))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
