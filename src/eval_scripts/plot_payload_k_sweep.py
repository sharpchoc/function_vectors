#!/usr/bin/env python
"""k-sweep summary for the payload-subspace test-task ablation (CPU).

Reads the per-k output roots (ablation/attention_head_mechanisms/test7 for k=4,
.../test7_k_sweep/k{k} otherwise) and plots, per site row and op, the
min-over-start-layers of the task-averaged delta log p as a function of k — own subspace
solid, shuffled-cf dashed. The gap between the two lines is the task-specificity headroom.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.plot_oneshot_preimage_ablation import load_arm
from utils.paths import FV_FORMATION_DIR

ARMS = ["payload_zero", "payload_mean", "payload_cf_zero", "payload_cf_mean"]
TEST7 = ["landmark-country", "word_length", "capitalize_first_letter", "synonym",
         "lowercase_first_letter", "capitalize", "antonym"]
OP_COLORS = {"zero": "#256abf", "mean": "#b0451f"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--base_root", type=Path,
                   default=FV_FORMATION_DIR / "ablation/attention_head_mechanisms/test7")
    p.add_argument("--tasks", nargs="+", default=TEST7)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def root_for_k(base_root, k):
    return base_root if k == 4 else base_root.parent / "test7_k_sweep" / f"k{k}"


def main():
    args = parse_args()
    # curves[(row, op, same/cf)] = list over ks of min-over-L task-mean delta
    curves = {}
    row_names = None
    for k in args.ks:
        root = root_for_k(args.base_root, k)
        for arm in ARMS:
            per_task = []
            for task in args.tasks:
                got = load_arm(root / task, arm)
                if got is None:
                    continue
                row_names, g = got
                per_task.append(g)
            if not per_task:
                print(f"[k={k}] missing arm {arm} under {root}")
                continue
            grid = np.nanmean(np.stack(per_task), axis=0)   # (rows, 28)
            op = "zero" if arm.endswith("_zero") else "mean"
            kind = "cf" if "_cf_" in arm else "own"
            for ri, row in enumerate(row_names):
                curves.setdefault((str(row), op, kind), {})[k] = float(np.nanmin(grid[ri]))

    rows = ["cue1", "target1", "final_cue"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, constrained_layout=True)
    for ax, row in zip(axes, rows):
        for op in ("zero", "mean"):
            for kind, ls, mk in (("own", "-", "o"), ("cf", "--", "s")):
                pts = curves.get((row, op, kind), {})
                ks = [k for k in args.ks if k in pts]
                if not ks:
                    continue
                ax.plot(ks, [pts[k] for k in ks], ls, marker=mk, ms=5, lw=1.8,
                        color=OP_COLORS[op], alpha=1.0 if kind == "own" else 0.55,
                        label=f"{op}, {kind}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(args.ks)
        ax.set_xticklabels([str(k) for k in args.ks], fontsize=9)
        ax.axhline(0, color="#d9d8d3", lw=0.8)
        ax.set_title(row, fontsize=11)
        ax.set_xlabel("subspace dimension k", fontsize=9, color="#454540")
        for spine in ax.spines.values():
            spine.set_color("#d9d8d3")
        ax.tick_params(labelsize=9, color="#d9d8d3")
    axes[0].set_ylabel("min over L of task-mean Δ log p", fontsize=9, color="#454540")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Payload-subspace ablation k-sweep — mean over {len(args.tasks)} test tasks "
                 "(own subspace solid, shuffled cf dashed)", fontsize=12)
    out = args.out or (args.base_root / "figures" / "ktrend_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print("\nmin-over-L task-mean delta by k:")
    print(f"{'row':12s} {'op':5s} {'kind':4s} " + " ".join(f"k={k:>2d}" for k in args.ks))
    for (row, op, kind), pts in sorted(curves.items()):
        vals = " ".join(f"{pts.get(k, float('nan')):5.2f}" for k in args.ks)
        print(f"{row:12s} {op:5s} {kind:4s} {vals}")


if __name__ == "__main__":
    main()
