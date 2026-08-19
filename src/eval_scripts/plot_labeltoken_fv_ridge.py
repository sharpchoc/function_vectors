#!/usr/bin/env python
"""Summarise the label-token -> per-prompt-FV ridge study (ridge_labeltoken_to_fv.py).

Reads artifacts/69_task_run/labeltoken_fv_ridge/variant_{1..10,avg}.json and writes to
results/69_task_run/labeltoken_fv_ridge/:
  r2_by_n.png     train and held-out R^2 (uniform-average) vs demo index n, with the
                  all-10-average X variant shown as horizontal reference lines
  summary.csv     per variant: best alpha (+pinned flag), train/test R^2 both conventions
  per_task_r2.csv per task x variant held-out/train R^2 (uniform)
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

AR = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge"
OUT = TASK69_RUN_DIR / "FV_linear_decodability" / "labeltoken_fv_ridge"
VARIANTS = [str(n) for n in range(1, 11)] + ["avg"]


def main():
    data = {}
    for v in VARIANTS:
        f = AR / f"variant_{v}.json"
        assert f.exists(), f"missing {f}"
        data[v] = json.load(open(f))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "best_alpha", "alpha_pinned", "r2_train_uniform",
                    "r2_test_uniform", "r2_train_weighted", "r2_test_weighted"])
        for v in VARIANTS:
            d = data[v]
            w.writerow([v, d["best_alpha"], d["alpha_pinned"], d["r2_train_uniform"],
                        d["r2_test_uniform"], d["r2_train_weighted"], d["r2_test_weighted"]])
            print(f"n={v:>3}: alpha={d['best_alpha']:g}{' PIN' if d['alpha_pinned'] else ''}"
                  f" | R2 train {d['r2_train_uniform']:.3f} test {d['r2_test_uniform']:.3f}"
                  f" (weighted {d['r2_train_weighted']:.3f}/{d['r2_test_weighted']:.3f})")

    ns = list(range(1, 11))
    tr = [data[str(n)]["r2_train_uniform"] for n in ns]
    te = [data[str(n)]["r2_test_uniform"] for n in ns]
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    ax.plot(ns, tr, "o-", color="tab:blue", label="train tasks (in-sample, 55)")
    ax.plot(ns, te, "s-", color="tab:red", label="held-out tasks (14)")
    ax.axhline(data["avg"]["r2_train_uniform"], color="tab:blue", ls="--", lw=1.1,
               label=f"avg-of-10 X, train = {data['avg']['r2_train_uniform']:.3f}")
    ax.axhline(data["avg"]["r2_test_uniform"], color="tab:red", ls="--", lw=1.1,
               label=f"avg-of-10 X, held-out = {data['avg']['r2_test_uniform']:.3f}")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(ns)
    ax.set_xlabel("demo index n (X = L6 activation at the last token of demo n's label)")
    ax.set_ylabel("R^2 (uniform average over 4096 dims)")
    ax.set_title("Ridge: nth-demo-label L6 activation -> per-prompt FV\n"
                 "(fit on 55 train tasks, lambda by 5-fold task CV)", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "r2_by_n.png", bbox_inches="tight")

    tasks = sorted(data["1"]["per_task"])
    with open(OUT / "per_task_r2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "split"] + [f"n{v}" for v in VARIANTS])
        for t in tasks:
            row = [t, data["1"]["per_task"][t]["split"]]
            row += [data[v]["per_task"][t]["r2_uniform"] for v in VARIANTS]
            w.writerow(row)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
