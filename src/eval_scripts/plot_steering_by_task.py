#!/usr/bin/env python
"""Per-task bar chart of the read-direction steering sweep (steer_read_dir_1shot.py).

Reads artifacts/69_task_run/read_dir_steering_1shot/<task>__{sampled_underscore,real_1shot}
.json for all 69 tasks and plots, per task (sorted by the blank-slot steered accuracy):
  - 0-shot (no demo at all): unsteered baseline (the format-free floor)
  - blank-'_' scaffold: unsteered baseline vs BEST steered alpha (dot_perhead family)
  - real 1-shot demo: unsteered baseline (the "one real demo" reference)
Held-out tasks marked with * in the tick label.

Outputs in results/69_task_run/Read_direction_geometry/steering/:
  steering_by_task.png, steering_by_task.csv
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
FAMILIES = ("dot_perhead", "cosine_perhead")


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)

    rows = []
    for t in tasks:
        us = json.load(open(AR / f"{t}__sampled_underscore.json"))
        r1 = json.load(open(AR / f"{t}__real_1shot.json"))
        zs = json.load(open(AR / f"{t}__zero_shot.json"))
        rec = {"task": t, "group": group[t],
               "zeroshot_baseline": zs["conditions"]["baseline"]["acc"],
               "blank_baseline": us["conditions"]["baseline"]["acc"],
               "real1shot_baseline": r1["conditions"]["baseline"]["acc"]}
        for fam in FAMILIES:
            accs = {a: us["conditions"][f"{fam}_a{a}"]["acc"] for a in ALPHAS}
            best_a = max(accs, key=accs.get)
            rec[f"blank_{fam}_best"] = accs[best_a]
            rec[f"blank_{fam}_best_alpha"] = best_a
            r1accs = {a: r1["conditions"][f"{fam}_a{a}"]["acc"] for a in ALPHAS}
            rec[f"real1shot_{fam}_best"] = max(r1accs.values())
        rows.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with open(OUT / "steering_by_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    rows.sort(key=lambda r: r["blank_dot_perhead_best"])
    labels = [(r["task"] + (" *" if r["group"] == "heldout" else "")) for r in rows]
    x = np.arange(len(rows))
    w = 0.21
    fig, ax = plt.subplots(figsize=(max(14, 0.32 * len(rows)), 7.2), dpi=150)
    ax.bar(x - 1.5 * w, [r["zeroshot_baseline"] for r in rows], w, color="0.35",
           label="0-shot (no demo), unsteered")
    ax.bar(x - 0.5 * w, [r["blank_baseline"] for r in rows], w, color="0.72",
           label="blank '_' scaffold, unsteered")
    ax.bar(x + 0.5 * w, [r["blank_dot_perhead_best"] for r in rows], w, color="tab:blue",
           label="blank '_' scaffold, steered (best alpha, dot_perhead)")
    ax.bar(x + 1.5 * w, [r["real1shot_baseline"] for r in rows], w, color="tab:green",
           label="real 1-shot demo, unsteered")
    ax.set_xticks(x, labels, rotation=90, fontsize=6.2)
    ax.set_ylabel("T=1 sampled exact-match accuracy (150 prompts)")
    ax.set_title("Read-direction steering by task — injection at the demo label slot, L7, "
                 "natural-magnitude per-task read direction\n"
                 "(sorted by steered accuracy; * = held-out task)", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "steering_by_task.png", bbox_inches="tight")

    b = np.array([r["blank_baseline"] for r in rows])
    s = np.array([r["blank_dot_perhead_best"] for r in rows])
    d = np.array([r["real1shot_baseline"] for r in rows])
    z = np.array([r["zeroshot_baseline"] for r in rows])
    print(f"0-shot mean {z.mean():.3f} | blank unsteered mean {b.mean():.3f} | "
          f"steered mean {s.mean():.3f} | real 1-shot mean {d.mean():.3f}")
    print(f"steered > 0-shot on {(s > z).sum()}/{len(rows)} tasks")
    print(f"steered > blank baseline on {(s > b).sum()}/{len(rows)} tasks; "
          f"steered >= real 1-shot on {(s >= d).sum()}/{len(rows)}")
    frac = np.divide(s, d, out=np.full_like(s, np.nan), where=d > 0)
    print(f"median steered/real-1shot ratio (where 1-shot > 0): {np.nanmedian(frac):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
