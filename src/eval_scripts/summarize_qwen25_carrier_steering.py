#!/usr/bin/env python
"""Summarize Qwen2.5-7B-Instruct s_A = c + u_A steering (reviewer fix 2026-09-23; GPT-J counterpart:
results/69_task_run/bottom_up_read_features/steering_results/meanresid/).

Inputs (artifacts/qwen25_fv/meanresid_steering/): the 1-shot dummy-label block sweep (<task>.json,
conditions L{l}_a{a}) and the six-dummy-slot runs at the chosen block(s) (sixshot_L<l>/<task>.json).
Block selection rule = GPT-J's: the block maximizing the mean over tasks of each task's best-alpha accuracy.
Real 6-shot reference = real_6shot from results/qwen25_fv/read_feature_steering_6shot/per_task_acc.csv
(same prompt bank and seeding scheme).
Writes results/qwen25_fv/read_feature_steering_6shot/carrier_plus_task_specific/
  sweep_layer_summary.csv   block x alpha means over the 96 tasks (+ per-task-best)
  sixshot_by_layer.csv      six-slot means per block, alpha, and group (train/heldout/all)
  per_task_sixshot.csv      per-task six-slot accuracies with the real-6-shot reference
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import QWEN25_FV_ARTIFACTS_DIR, QWEN25_FV_DIR  # noqa: E402

A = (0.5, 1.0, 2.0, 4.0)
ART = QWEN25_FV_ARTIFACTS_DIR / "meanresid_steering"
OUT = QWEN25_FV_DIR / "read_feature_steering_6shot" / "carrier_plus_task_specific"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sw = [json.load(open(f)) for f in sorted(glob.glob(str(ART / "*.json")))]
    assert len(sw) == 96, len(sw)
    rows = []
    for L in range(28):
        M = np.array([[r["conditions"][f"L{L}_a{a}"]["acc"] for a in A] for r in sw])
        rows.append({"layer": L, **{f"mean_a{a}": M[:, i].mean() for i, a in enumerate(A)},
                     "mean_best": M.max(1).mean()})
    sweep = pd.DataFrame(rows)
    sweep["baseline"] = np.mean([r["conditions"]["baseline"]["acc"] for r in sw])
    sweep.round(4).to_csv(OUT / "sweep_layer_summary.csv", index=False)
    chosen = int(sweep.layer[sweep.mean_best.idxmax()])
    print(f"sweep: chosen block {chosen} (mean per-task best {sweep.mean_best.max():.4f}); baseline {sweep.baseline[0]:.4f}")

    ref = pd.read_csv(QWEN25_FV_DIR / "read_feature_steering_6shot" / "per_task_acc.csv").set_index("task")
    per, agg = [], []
    for d in sorted(glob.glob(str(ART / "sixshot_L*"))):
        L = int(Path(d).name.split("_L")[1])
        R = [json.load(open(f)) for f in sorted(glob.glob(d + "/*.json"))]
        if len(R) != 96:
            print(f"block {L}: only {len(R)} tasks, skipped"); continue
        for r in R:
            row = {"task": r["task"], "group": r["group"], "layer": L, "baseline": r["conditions"]["baseline"]["acc"],
                   **{f"a{a}": r["conditions"][f"a{a}"]["acc"] for a in A}, "real_6shot": ref.loc[r["task"], "real_6shot"]}
            row["best"] = max(row[f"a{a}"] for a in A); per.append(row)
    per = pd.DataFrame(per)
    for (L, g), d in list(per.groupby(["layer", "group"])) + [((L, "all"), d) for L, d in per.groupby("layer")]:
        agg.append({"layer": L, "group": g, "n_tasks": len(d), "baseline": d.baseline.mean(),
                    **{f"a{a}": d[f"a{a}"].mean() for a in A}, "best": d.best.mean(), "real_6shot": d.real_6shot.mean()})
    agg = pd.DataFrame(agg).sort_values(["layer", "group"]).round(4)
    per.round(4).to_csv(OUT / "per_task_sixshot.csv", index=False)
    agg.to_csv(OUT / "sixshot_by_layer.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
