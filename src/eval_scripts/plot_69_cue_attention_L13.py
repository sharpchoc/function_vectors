#!/usr/bin/env python
"""Figures + CSVs for the L13 cue->label attention theory test (1-shot, 69 tasks).

Reads artifacts/69_task_run/raw_mean_steering/cue_attention_L13/<task>.json
(capture_cue_attention_L13.py) plus the accuracy references
(steering_results/oneshot_dummy/summary.csv for full-mean alphas,
steering_results/taskunique_svd_dummy/summary.csv for swap alphas).

Writes into results/.../steering_results/attention_to_label_1shot/:
  per_task_attn.csv     per-task mean attention for every condition
  summary.csv           aggregate means
  headline_bars.png     the four requested conditions (steering at its accuracy-best
                        alpha: full mean a2, swap a16)
  attn_vs_alpha.png     2x2: attention-vs-alpha (top) over accuracy-vs-alpha (bottom),
                        full mean (left) vs task-unique swap (right) — no dual axes
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

AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "cue_attention_L13"
SR = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
OUT = SR / "attention_to_label_1shot"
FM_AL = (0.5, 1.0, 2.0, 4.0)
SW_AL = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0)
BLUE, ORANGE, GRAY = "#2f7fe0", "#e07b2f", "#6b7280"


def acc_ref(path, prefix):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["task_group"] == "all" and r["condition"].startswith(prefix):
            out[r["condition"]] = float(r["mean_acc"])
    return out


def main():
    tasks, rows = [], {}
    for f in sorted(AR.glob("*.json")):
        r = json.load(open(f))
        tasks.append(r["task"])
        rows[r["task"]] = {"task": r["task"], "group": r["group"],
                           **{c: v["mean_attn"] for c, v in r["conditions"].items()}}
    assert len(tasks) == 69
    conds = [c for c in rows[tasks[0]] if c not in ("task", "group")]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_task_attn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "group"] + conds)
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(c):
        return float(np.mean([rows[t][c] for t in tasks]))

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_attn_L13_cue_to_label"])
        for c in conds:
            w.writerow([c, round(mean_of(c), 5)])

    acc_fm = acc_ref(SR / "oneshot_dummy" / "summary.csv", "dummy1_steer_a")
    acc_sw = acc_ref(SR / "taskunique_svd_dummy" / "summary.csv", "dummy1_swap1_a")

    # ---------------- headline bars ----------------
    hl = [("dummy1_unsteered", "unsteered\ndummy", GRAY),
          ("dummy1_swap1_a16.0", "task-unique\nswap ($\\alpha$=16)", ORANGE),
          ("dummy1_fullmean_a2.0", "full mean\n($\\alpha$=2)", BLUE),
          ("real_1shot", "real\n1-shot", "0.25")]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    xs = np.arange(4)
    vals = [mean_of(c) for c, _, _ in hl]
    ax.bar(xs, vals, 0.62, color=[c for _, _, c in hl])
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.4f}", (x, v), ha="center", va="bottom", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l, _ in hl], fontsize=11)
    ax.set_ylabel("L13 head-mean attention: final cue $\\to$ final label token",
                  fontsize=11)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("Cue$\\to$label attention at L13 (1-shot, steering at its "
                 "accuracy-best $\\alpha$, mean over 69 tasks)", fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "headline_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- attention vs alpha over accuracy vs alpha ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=200, sharex="col")
    base_attn, real_attn = mean_of("dummy1_unsteered"), mean_of("real_1shot")
    for col, (al, pref, accs, color, name) in enumerate((
            (FM_AL, "dummy1_fullmean_a", {a: acc_fm[f"dummy1_steer_a{a}"] for a in FM_AL},
             BLUE, "full mean  $z + \\alpha m_A$"),
            (SW_AL[1:], "dummy1_swap1_a", {a: acc_sw[f"dummy1_swap1_a{a}"] for a in SW_AL[1:]},
             ORANGE, "task-unique swap  $\\alpha s_1 v_1$"))):
        ax = axes[0][col]
        ax.plot(al, [mean_of(f"{pref}{a}") for a in al], "-o", color=color, lw=2.2, ms=6)
        ax.axhline(base_attn, color=GRAY, lw=1.5, ls=(0, (5, 3)))
        ax.axhline(real_attn, color="0.25", lw=1.5, ls=(0, (5, 3)))
        ax.annotate(f"unsteered {base_attn:.4f}", (al[-1], base_attn), ha="right",
                    va="bottom", fontsize=10, color=GRAY)
        ax.annotate(f"real 1-shot {real_attn:.4f}", (al[-1], real_attn), ha="right",
                    va="bottom", fontsize=10, color="0.25")
        ax.set_ylabel("L13 cue$\\to$label attention" if col == 0 else "")
        ax.set_title(name, fontsize=13)
        ax = axes[1][col]
        ax.plot(al, [accs[a] for a in al], "-s", color=color, lw=2.2, ms=6)
        ax.set_ylabel("accuracy" if col == 0 else "")
        ax.set_xlabel("alpha")
        best = max(accs, key=accs.get)
        ax.annotate(f"acc peak $\\alpha$={best:g}", (best, accs[best]),
                    xytext=(best, accs[best] + 0.012), ha="center", fontsize=10)
        for rr in (0, 1):
            axes[rr][col].set_xscale("log", base=2)
            axes[rr][col].set_xticks(list(al))
            axes[rr][col].set_xticklabels([("%g" % a) for a in al], fontsize=9)
            axes[rr][col].grid(axis="y", color="0.92")
            for s_ in ("top", "right"):
                axes[rr][col].spines[s_].set_visible(False)
    fig.suptitle("Does steering work by increasing cue$\\to$label attention? "
                 "Attention (top) vs accuracy (bottom) across $\\alpha$ (1-shot, 69 tasks)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "attn_vs_alpha.png", bbox_inches="tight")
    plt.close(fig)

    for c in conds:
        print(f"{c}: {mean_of(c):.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
