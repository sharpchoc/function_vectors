#!/usr/bin/env python
"""Compare PC5 SUBSPACE-PATCH steering against direct additive steering (69 tasks).

Patch condition (steer_read_dir_1shot.py --patch_pcs 5): at the blank-'_' label slot, L7,
project out the top-5 uncentered PCs of the pooled per-prompt dot_perhead read directions
and add alpha * P5 v_task. Direct condition: the same v_task added without projecting
anything out (earlier sweep). Both on the same prompts/seeds.

Outputs in results/69_task_run/Read_direction_geometry/steering/:
  patch_vs_direct.png   (A) mean accuracy vs alpha for patch / direct / controls;
                        (B) per-task scatter of best-alpha patch vs best-alpha direct
  patch_vs_direct.csv   per-task accuracies for every condition
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

AR = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
OUT = TASK69_RUN_DIR / "Read_direction_geometry" / "steering"
ALPHAS = (0.5, 1.0, 2.0, 4.0)


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)

    rows = []
    for t in tasks:
        direct = json.load(open(AR / f"{t}__sampled_underscore.json"))["conditions"]
        patch = json.load(open(AR / f"{t}__sampled_underscore__patch5pc.json"))["conditions"]
        rec = {"task": t, "group": group[t],
               "baseline": direct["baseline"]["acc"],
               "projout_only": patch["projout_only"]["acc"]}
        for a in ALPHAS:
            rec[f"direct_a{a}"] = direct[f"dot_perhead_a{a}"]["acc"]
            rec[f"patch_a{a}"] = patch[f"patch_a{a}"]["acc"]
        rec["direct_best"] = max(rec[f"direct_a{a}"] for a in ALPHAS)
        rec["patch_best"] = max(rec[f"patch_a{a}"] for a in ALPHAS)
        rows.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with open(OUT / "patch_vs_direct.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)
    ax = axes[0]
    dm = [np.mean([r[f"direct_a{a}"] for r in rows]) for a in ALPHAS]
    pm = [np.mean([r[f"patch_a{a}"] for r in rows]) for a in ALPHAS]
    ax.plot(ALPHAS, dm, "o-", color="tab:blue", label="direct add (h + a*v_task)")
    ax.plot(ALPHAS, pm, "s-", color="tab:purple", label="PC5 patch (h - P h + a*P v_task)")
    ax.axhline(np.mean([r["baseline"] for r in rows]), color="0.5", ls=":",
               label=f"unsteered blank '_' = {np.mean([r['baseline'] for r in rows]):.3f}")
    ax.axhline(np.mean([r["projout_only"] for r in rows]), color="tab:red", ls="--",
               label=f"project-out only (a=0) = {np.mean([r['projout_only'] for r in rows]):.3f}")
    ax.set_xscale("log"); ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
    ax.set_xlabel("alpha"); ax.set_ylabel("mean T=1 exact-match accuracy (69 tasks)")
    ax.set_title("(A) mean over tasks vs alpha")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axes[1]
    for g, col, mk in (("train", "tab:blue", "o"), ("heldout", "tab:red", "s")):
        sel = [r for r in rows if r["group"] == g]
        ax.scatter([r["direct_best"] for r in sel], [r["patch_best"] for r in sel],
                   s=26, alpha=0.75, color=col, marker=mk, label=f"{g} tasks")
    lim = max(max(r["direct_best"] for r in rows), max(r["patch_best"] for r in rows)) * 1.08
    ax.plot([0, lim], [0, lim], color="0.6", ls="--", lw=0.9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("direct steering, best alpha"); ax.set_ylabel("PC5 patch, best alpha")
    n_better = sum(r["patch_best"] > r["direct_best"] for r in rows)
    ax.set_title(f"(B) per-task best-alpha: patch vs direct\n"
                 f"patch better on {n_better}/{len(rows)} tasks")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("PC5 subspace patching vs direct steering — blank '_' scaffold, L7, "
                 "dot_perhead per-task read direction", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "patch_vs_direct.png", bbox_inches="tight")

    d_best = np.array([r["direct_best"] for r in rows])
    p_best = np.array([r["patch_best"] for r in rows])
    print(f"direct best mean {d_best.mean():.3f} | patch best mean {p_best.mean():.3f} | "
          f"projout-only mean {np.mean([r['projout_only'] for r in rows]):.3f} | "
          f"baseline {np.mean([r['baseline'] for r in rows]):.3f}")
    print(f"patch > direct on {n_better}/{len(rows)} tasks; "
          f"mean paired diff {np.mean(p_best - d_best):+.3f}")
    print(f"per-alpha means: direct {[round(x, 3) for x in dm]} | patch {[round(x, 3) for x in pm]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
