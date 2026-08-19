#!/usr/bin/env python
"""Figures + summary CSV for mean-free read-feature steering on dummy scaffolds.

Joins the meanfree_dummy eval JSONs with the earlier full-vector runs so the mean-free
steering sits next to what it is compared against:
  1-shot: raw_mean_steering/<task>.json           L6_a* (full vector), sharedL6_a*
          (shared mean control), baseline
  6-shot: raw_mean_steering/sixshot_dummy/<task>.json  dummy6_steer_a*, dummy6_baseline,
          real6_baseline;  real_1shot from the sixshot_dummy results CSV.
"best" columns are the per-task max over that condition's alpha grid (old convention).

Writes into results/.../bottom_up_read_features/steering_results/meanfree_dummy/:
  per_task_acc.csv, aggregate_bars.png, per_task_bars_{1,6}shot.png
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

RMS = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering"
STEER = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
OUT = STEER / "meanfree_dummy"

MF_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
OLD_ALPHAS = (0.5, 1.0, 2.0, 4.0)


def main():
    ref = {r["task"]: r for r in csv.DictReader(
        open(STEER / "sixshot_dummy" / "per_task_acc.csv"))}
    tasks = sorted(ref)
    assert len(tasks) == 69
    rows = {}
    for t in tasks:
        mf = json.load(open(RMS / "meanfree_dummy" / f"{t}.json"))
        old1 = json.load(open(RMS / f"{t}.json"))["conditions"]
        old6 = json.load(open(RMS / "sixshot_dummy" / f"{t}.json"))["conditions"]
        c = mf["conditions"]
        r = {"task": t, "group": mf["group"], "norm_m": mf["norm_m"],
             "norm_meanfree": mf["norm_meanfree"], "cos_m_shared": mf["cos_m_shared"],
             "real_1shot": float(ref[t]["real_1shot"]),
             "real_6shot": old6["real6_baseline"]["acc"]}
        # infra cross-check: same prompts/protocol as the old 6-shot run -> same baseline
        # up to T=1 sampling noise (fp nondeterminism across pods breaks bit-matching)
        assert abs(c["dummy6_baseline"]["acc"] - old6["dummy6_baseline"]["acc"]) < 0.03, t
        for n, alphas in ((1, MF_ALPHAS), (6, MF_ALPHAS)):
            r[f"dummy{n}_baseline"] = c[f"dummy{n}_baseline"]["acc"]
            vals = [c[f"dummy{n}_meanfree_a{a}"]["acc"] for a in alphas]
            for a, v in zip(alphas, vals):
                r[f"dummy{n}_meanfree_a{a}"] = v
            r[f"dummy{n}_meanfree_best"] = max(vals)
        r["dummy1_fullvec_best"] = max(old1[f"L6_a{a}"]["acc"] for a in OLD_ALPHAS)
        r["dummy1_shared_best"] = max(old1[f"sharedL6_a{a}"]["acc"] for a in OLD_ALPHAS)
        r["dummy6_fullvec_best"] = max(old6[f"dummy6_steer_a{a}"]["acc"] for a in OLD_ALPHAS)
        rows[t] = r

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[tasks[0]])
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(col):
        return float(np.mean([float(rows[t][col]) for t in tasks]))

    # ---------------- aggregate figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=200, sharey=True)
    for ax, n in zip(axes, (1, 6)):
        names = ([f"dummy{n}_baseline"]
                 + [f"dummy{n}_meanfree_a{a}" for a in MF_ALPHAS]
                 + [f"dummy{n}_meanfree_best", f"dummy{n}_fullvec_best"]
                 + (["dummy1_shared_best"] if n == 1 else []))
        labels = (["dummy\nunsteered"] + [f"mean-free\nα={a:g}" for a in MF_ALPHAS]
                  + ["mean-free\nbest α", "full vector\nbest α"]
                  + (["shared mean\nbest α"] if n == 1 else []))
        colors = (["0.6"] + ["#63a3e8"] * len(MF_ALPHAS) + ["#1f5fb0", "#e8a13f"]
                  + (["#b0b0b0"] if n == 1 else []))
        vals = [mean_of(c) for c in names]
        xs = np.arange(len(names), dtype=float)
        xs[-2:] += 0.35
        if n == 1:
            xs[-1] += 0.35
        ax.bar(xs, vals, color=colors, width=0.8)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x, v), ha="center", va="bottom", fontsize=9.5)
        real = mean_of(f"real_{n}shot")
        ax.axhline(real, color="0.35", lw=1.6, ls=(0, (5, 3)))
        ax.annotate(f"real {n}-shot {real:.3f}", (xs[-1] + 0.4, real), ha="right",
                    va="bottom", fontsize=10, color="0.25")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{n}-shot dummy scaffold", fontsize=14)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy (T=1 sampled exact match, mean over 69 tasks)", fontsize=11)
    fig.suptitle("Steering with the MEAN-FREE read feature (m_A − proj on shared mean) "
                 "at L6 dummy label slots", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUT / "aggregate_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- per-task figures ----------------
    for n in (1, 6):
        fig, ax = plt.subplots(figsize=(26, 7.5), dpi=170)
        x = np.arange(len(tasks))
        w = 0.38
        ax.bar(x - w / 2, [rows[t][f"dummy{n}_meanfree_best"] for t in tasks], w,
               color="#1f5fb0", label="mean-free best α")
        ax.bar(x + w / 2, [rows[t][f"dummy{n}_fullvec_best"] for t in tasks], w,
               color="#e8a13f", label="full vector best α")
        ax.plot(x, [float(rows[t][f"real_{n}shot"]) for t in tasks], ls="none",
                marker="_", ms=13, mew=2.2, color="0.25", label=f"real {n}-shot")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=90, fontsize=7.5)
        ax.set_ylabel("accuracy")
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9, ncol=3, loc="upper right")
        ax.set_title(f"Mean-free vs full-vector read-feature steering, per task "
                     f"({n}-shot dummy scaffold)", fontsize=14, fontweight="bold", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"per_task_bars_{n}shot.png", bbox_inches="tight")
        plt.close(fig)

    for n in (1, 6):
        print(f"n{n}: dummy_base={mean_of(f'dummy{n}_baseline'):.3f}  "
              + "  ".join(f"a{a:g}={mean_of(f'dummy{n}_meanfree_a{a}'):.3f}"
                          for a in MF_ALPHAS)
              + f"  mf_best={mean_of(f'dummy{n}_meanfree_best'):.3f}"
              + f"  fullvec_best={mean_of(f'dummy{n}_fullvec_best'):.3f}"
              + (f"  shared_best={mean_of('dummy1_shared_best'):.3f}" if n == 1 else "")
              + f"  real={mean_of(f'real_{n}shot'):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
