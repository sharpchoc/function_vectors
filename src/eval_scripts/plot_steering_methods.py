#!/usr/bin/env python
"""Summarise the read-direction steering sweep over definition method (S1) x layer (S2).

Reads artifacts/69_task_run/read_dir_method_steering/<task>__L<cfg>.json (+ __baseline.json)
from steer_read_dir_methods.py and writes to
results/69_task_run/Read_direction_geometry/steering_methods/:
  layer_profiles.png        mean acc over the 20 tasks vs layer, one line per bracket
                            (at each bracket's best alpha), bands as separate markers,
                            with unsteered / 0-shot / real-1-shot reference lines
  methods_alpha_curves.png  mean acc vs alpha per bracket, at that bracket's best layer
  methods_by_task.png       per-task bars at the overall best (bracket, layer, alpha)
  summary.csv               mean/median acc + uplift per bracket x alpha x layer config
  per_task_acc.csv          per-task accuracies for the best cell of each bracket
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

AR = ARTIFACTS_ROOT / "69_task_run" / "read_dir_method_steering"
REF = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
OUT = TASK69_RUN_DIR / "Read_direction_geometry" / "steering_methods"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")
ALPHAS = (0.5, 1.0, 2.0, 4.0)
SINGLE = [str(l) for l in range(3, 16)]
BANDS = ["3-15", "7-11"]
COLORS = {"cosine_M": "tab:purple", "dot_M": "tab:red",
          "cosine_perhead": "tab:orange", "dot_perhead": "tab:blue"}


def main():
    tasks = sorted({p.name.split("__")[0] for p in AR.glob("*__baseline.json")})
    print(f"{len(tasks)} tasks")
    cfgs = SINGLE + BANDS
    # acc[bracket][alpha][cfg] -> array over tasks
    acc = {b: {a: {c: np.full(len(tasks), np.nan) for c in cfgs} for a in ALPHAS}
           for b in BRACKETS}
    base = np.array([json.load(open(AR / f"{t}__baseline.json"))
                     ["conditions"]["baseline"]["acc"] for t in tasks])
    for ti, t in enumerate(tasks):
        for c in cfgs:
            f = AR / f"{t}__L{c}.json"
            if not f.exists():
                continue
            d = json.load(open(f))["conditions"]
            for b in BRACKETS:
                for a in ALPHAS:
                    acc[b][a][c][ti] = d[f"{b}__a{a}"]["acc"]
    # references (computed earlier for all 69 tasks)
    zs = np.array([json.load(open(REF / f"{t}__zero_shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    r1 = np.array([json.load(open(REF / f"{t}__real_1shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in BRACKETS:
        for a in ALPHAS:
            for c in cfgs:
                v = acc[b][a][c]
                if np.isnan(v).all():
                    continue
                rows.append([b, a, c, round(float(np.nanmean(v)), 4),
                             round(float(np.nanmedian(v)), 4),
                             round(float(np.nanmean(v - base)), 4)])
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bracket", "alpha", "layer_config", "mean_acc", "median_acc", "mean_uplift"])
        w.writerows(rows)

    def mean_at(b, a, c):
        return float(np.nanmean(acc[b][a][c]))

    # best (alpha, cfg) per bracket
    best = {}
    for b in BRACKETS:
        cand = [(mean_at(b, a, c), a, c) for a in ALPHAS for c in cfgs
                if not np.isnan(acc[b][a][c]).all()]
        best[b] = max(cand)
        print(f"{b}: best mean {best[b][0]:.3f} at alpha={best[b][1]} layers={best[b][2]}")

    # ---- figure 1: layer profiles (single layers as lines, bands as markers) ----
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=150)
    xs = [int(c) for c in SINGLE]
    for b in BRACKETS:
        a_best = best[b][1]
        ys = [mean_at(b, a_best, c) for c in SINGLE]
        ax.plot(xs, ys, "o-", color=COLORS[b], label=f"{b} (alpha={a_best})")
        for c, mk in zip(BANDS, ("s", "D")):
            ax.plot([xs[-1] + 1.5 + BANDS.index(c) * 1.5], [mean_at(b, a_best, c)], mk,
                    color=COLORS[b], ms=7)
    ax.axhline(float(base.mean()), color="0.45", ls=":", lw=1,
               label=f"unsteered scaffold = {base.mean():.3f}")
    ax.axhline(float(zs.mean()), color="tab:brown", ls="-.", lw=1,
               label=f"0-shot = {zs.mean():.3f}")
    ax.axhline(float(r1.mean()), color="tab:green", ls="-", lw=1.2,
               label=f"real 1-shot demo = {r1.mean():.3f}")
    ax.set_xticks(xs + [xs[-1] + 1.5, xs[-1] + 3.0],
                  SINGLE + ["L3-15", "L7-11"], fontsize=8)
    ax.set_xlabel("injection layer (single layers; right-hand markers = bands)")
    ax.set_ylabel(f"mean T=1 exact-match accuracy ({len(tasks)} tasks)")
    ax.set_title("Read-direction steering: depth profile by definition method\n"
                 "1-shot 'Q: {input} / A: _' scaffold, injection at the '_' slot, "
                 "alpha = multiple of each method's natural magnitude", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "layer_profiles.png", bbox_inches="tight")

    # ---- figure 2: alpha curves at each bracket's best layer ----
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    for b in BRACKETS:
        c_best = best[b][2]
        ys = [mean_at(b, a, c_best) for a in ALPHAS]
        ax.plot(ALPHAS, ys, "o-", color=COLORS[b], label=f"{b} @ L{c_best}")
    ax.axhline(float(base.mean()), color="0.45", ls=":", lw=1, label="unsteered scaffold")
    ax.axhline(float(r1.mean()), color="tab:green", ls="-", lw=1.2, label="real 1-shot demo")
    ax.set_xscale("log"); ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
    ax.set_xlabel("alpha (multiple of the method's natural magnitude)")
    ax.set_ylabel(f"mean accuracy ({len(tasks)} tasks)")
    ax.set_title("Dose response at each method's best layer", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "methods_alpha_curves.png", bbox_inches="tight")

    # ---- figure 3: per-task bars at each bracket's best cell ----
    order = np.argsort(acc[max(best, key=lambda b: best[b][0])][
        best[max(best, key=lambda b: best[b][0])][1]][
        best[max(best, key=lambda b: best[b][0])][2]])
    labels = [tasks[i] for i in order]
    x = np.arange(len(tasks))
    w = 0.15
    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(tasks)), 6.2), dpi=150)
    ax.bar(x - 2.5 * w, base[order], w, color="0.7", label="unsteered scaffold")
    for k, b in enumerate(BRACKETS):
        _, a_b, c_b = best[b]
        ax.bar(x + (k - 1.5) * w, acc[b][a_b][c_b][order], w, color=COLORS[b],
               label=f"{b} (a={a_b}, L{c_b})")
    ax.bar(x + 2.5 * w, r1[order], w, color="tab:green", label="real 1-shot demo")
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.set_ylabel("T=1 exact-match accuracy (150 prompts)")
    ax.set_title("Per-task steering by read-direction method (each at its best alpha/layer)",
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "methods_by_task.png", bbox_inches="tight")

    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        head = ["task", "unsteered", "zero_shot", "real_1shot"]
        head += [f"{b}_a{best[b][1]}_L{best[b][2]}" for b in BRACKETS]
        w.writerow(head)
        for ti, t in enumerate(tasks):
            w.writerow([t, base[ti], zs[ti], r1[ti]] +
                       [acc[b][best[b][1]][best[b][2]][ti] for b in BRACKETS])
    print(f"unsteered {base.mean():.3f} | 0-shot {zs.mean():.3f} | 1-shot {r1.mean():.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
