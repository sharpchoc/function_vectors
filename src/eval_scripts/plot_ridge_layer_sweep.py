#!/usr/bin/env python
"""Summarise the ridge layer sweep (ridge_layer_sweep.py, L5..L15, avg-of-10 label X).

Reads artifacts/69_task_run/labeltoken_fv_ridge_layer_sweep/layer_<L>.json and writes to
results/69_task_run/labeltoken_fv_ridge/layer_sweep/:
  r2_by_layer.png  three curves vs layer: train in-sample, train unseen-prompts (honest),
                   held-out tasks; plus the unseen-prompt fair-oracle line and the L6
                   reference points from the original study
  summary.csv      all recorded quantities per layer
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

AR = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge_layer_sweep"
OUT = TASK69_RUN_DIR / "FV_linear_decodability" / "labeltoken_fv_ridge" / "layer_sweep"
LAYERS = list(range(5, 16))


def main():
    d = {}
    for L in LAYERS:
        f = AR / f"layer_{L}.json"
        assert f.exists(), f"missing {f}"
        d[L] = json.load(open(f))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["layer", "best_alpha", "alpha_pinned", "r2_train_insample_uniform",
                "r2_train_unseenprompts_uniform", "r2_train_unseenprompts_oracle",
                "r2_test_uniform", "r2_train_insample_weighted",
                "r2_train_unseenprompts_weighted", "r2_test_weighted"]
        w.writerow(keys)
        for L in LAYERS:
            w.writerow([d[L][k] if k != "layer" else L for k in keys])
            print(f"L{L:>2}: alpha={d[L]['best_alpha']:g} | insample "
                  f"{d[L]['r2_train_insample_uniform']:.3f} | unseen-prompts "
                  f"{d[L]['r2_train_unseenprompts_uniform']:.3f} (oracle "
                  f"{d[L]['r2_train_unseenprompts_oracle']:.3f}) | heldout "
                  f"{d[L]['r2_test_uniform']:.3f}")

    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=150)
    ax.plot(LAYERS, [d[L]["r2_train_insample_uniform"] for L in LAYERS], "o--",
            color="tab:blue", alpha=0.55, label="train tasks, in-sample (optimistic)")
    ax.plot(LAYERS, [d[L]["r2_train_unseenprompts_uniform"] for L in LAYERS], "o-",
            color="tab:blue", label="train tasks, UNSEEN prompts (honest)")
    ax.plot(LAYERS, [d[L]["r2_train_unseenprompts_oracle"] for L in LAYERS], "-",
            color="0.4", lw=1.1, ls=":",
            label="fair oracle on the same unseen prompts (task means)")
    ax.plot(LAYERS, [d[L]["r2_test_uniform"] for L in LAYERS], "s-", color="tab:red",
            label="HELD-OUT tasks (out-of-distribution)")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("layer of the mean label-token activation (X)")
    ax.set_ylabel("R^2 (uniform average over 4096 dims)")
    ax.set_title("Ridge layer sweep: mean label-token activation -> per-prompt FV\n"
                 "(avg-of-10 X; lambda by 5-fold task CV per layer)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "r2_by_layer.png", bbox_inches="tight")
    best = max(LAYERS, key=lambda L: d[L]["r2_test_uniform"])
    print(f"best held-out layer: L{best} ({d[best]['r2_test_uniform']:.4f})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
