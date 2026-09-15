#!/usr/bin/env python
"""Summary bar chart of the code-convention steering results: mean accuracy over families for
unsteered 0-shot, steered 0-shot (best layer / α per family) and 4 in-context examples, per target pole and pooled.
Reads results/style_translation/<model>/<tag>/steering/best_config.csv, writes summary_bars.png next to it."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--tag", default="code")
    args = ap.parse_args()
    R = model_paths(args.model)["results"] / args.tag / "steering"
    rows = [r for r in csv.DictReader(open(R / "best_config.csv")) if r["step3_k4"] not in ("", None)]
    groups = [("toward the alternative convention\n(the pole the model does not default to)", [r for r in rows if r["target"] == "alt"]),
              ("toward the natural convention", [r for r in rows if r["target"] == "nat"]),
              ("both directions pooled", rows)]
    conds = [("unsteered, 0-shot", "base_accuracy", "#9e9e9e"), ("steered at the cue token, 0-shot", "accuracy", "#d62728"),
             ("4 in-context examples, no steering", "step3_k4", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    w = 0.26
    for gi, (gname, sel) in enumerate(groups):
        for ci, (cname, key, col) in enumerate(conds):
            v = np.array([float(r[key]) for r in sel]); m = v.mean(); e = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            x = gi + (ci - 1) * w
            ax.bar(x, m, w, color=col, edgecolor="black", linewidth=0.6, yerr=e, capsize=3, label=cname if gi == 0 else None)
            ax.text(x, m + e + 0.015, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=9.5)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy (mean over families)", fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    n = len({r["family"] for r in rows})
    ax.set_title(f"Zero-shot steering of coding conventions on Qwen2.5-7B base — {n} families\n"
                 "accuracy = target convention used AND correct solution (judge)\n"
                 "steered = best layer and α per family; error bars = 95% CI of the mean across families", fontsize=10)
    fig.tight_layout()
    out = R / "summary_bars.png"; fig.savefig(out, dpi=150); plt.close(fig); print("->", out)


if __name__ == "__main__":
    main()
