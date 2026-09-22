#!/usr/bin/env python
"""Figures + summary CSV for the task-unique read-feature ablation (û_A, mean carrier-removed
read-band residual; ablate_readdir_pc5.py with the meanresid_top1 bases).

Generalised, model-agnostic version of the MEANRESID=1 mode of plot_69_taskunique_L5to7_top1.py
(written for the Qwen2.5 port, 2026-09-22). Differences: no mr3 side-by-side columns, and the
unablated / 0-shot baselines can come either from the sixshot_dummy per-task CSV (GPT-J) or from
the FV-ablation eval JSONs' in-run zero_shot / real{n}_baseline conditions (Qwen: same prompts,
same T=1 readout).

Writes into --out:
  per_task_acc.csv        task, group, cf_task, zero_shot, per n: baseline + the four mr157 columns
  aggregate_bars.png      headline: unablated | ablate own û_A | ablate cf û_A (mean ablation)
  aggregate_bars_full.png mean/zero x own/cf bars with baseline lines, per n
  per_task_bars_{1,6}shot.png
"""
import argparse
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
from utils.paper_style import apply_paper_style, C, label_bars  # noqa: E402
apply_paper_style()

JSON_CONDS = ("mean_ablation_pc5", "zero_ablation_pc5",
              "cf_mean_ablation_pc5", "cf_zero_ablation_pc5")   # names inside the JSONs
MR1 = ("mean_ablation_mr157", "zero_ablation_mr157",
       "cf_mean_ablation_mr157", "cf_zero_ablation_mr157")
COLOR = {"mean_ablation_mr157": "#2f7fe0", "cf_mean_ablation_mr157": "#a9c9ef",
         "zero_ablation_mr157": "#d94f3d", "cf_zero_ablation_mr157": "#efb2a9"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ar", type=Path,
                    default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA_meanresid_top1")
    ap.add_argument("--out", type=Path,
                    default=TASK69_RUN_DIR / "bottom_up_read_features" / "ablation" / "task_unique_meanresid")
    ap.add_argument("--split_path", type=Path,
                    default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    ap.add_argument("--baselines", type=str,
                    default="csv:" + str(TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
                                         / "sixshot_dummy" / "per_task_acc.csv"),
                    help="'csv:<per_task_acc.csv>' (zero_shot, real_1shot, real_6shot columns) or "
                         "'fv_ablation:<root>' (zero_shot + real6_baseline from <root>/eval/<task>.json, "
                         "real1_baseline from <root>/eval_1shot/<task>.json)")
    ap.add_argument("--band_label", default="L5–7")
    ap.add_argument("--model_label", default="GPT-J-6B")
    return ap.parse_args()


def load_baselines(spec, tasks):
    kind, path = spec.split(":", 1)
    out = {}
    if kind == "csv":
        rows = {r["task"]: r for r in csv.DictReader(open(path))}
        for t in tasks:
            out[t] = {"zero_shot": float(rows[t]["zero_shot"]),
                      "n1_baseline": float(rows[t]["real_1shot"]),
                      "n6_baseline": float(rows[t]["real_6shot"])}
    elif kind == "fv_ablation":
        root = Path(path)
        for t in tasks:
            d6 = json.load(open(root / "eval" / f"{t}.json"))["conditions"]
            d1 = json.load(open(root / "eval_1shot" / f"{t}.json"))["conditions"]
            out[t] = {"zero_shot": d6["zero_shot"]["acc"], "n6_baseline": d6["real6_baseline"]["acc"],
                      "n1_baseline": d1["real1_baseline"]["acc"]}
    else:
        raise ValueError(spec)
    return out


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)
    base = load_baselines(args.baselines, tasks)
    rows = {}
    for t in tasks:
        rows[t] = {"task": t, "group": group[t], **base[t]}
        for n in (1, 6):
            r = json.load(open(args.ar / f"n{n}shot" / f"{t}.json"))
            assert r["rank"] == 1 and "meanresid_top1_bases" in r["bases_path"], r["bases_path"]
            rows[t]["cf_task"] = r["cf_task"]
            for jc, cc in zip(JSON_CONDS, MR1):
                rows[t][f"n{n}_{cc}"] = r["conditions"][jc]["acc"]
    T = len(tasks)

    args.out.mkdir(parents=True, exist_ok=True)
    cols = (["task", "group", "cf_task", "zero_shot"]
            + [f"n{n}_{c}" for n in (1, 6) for c in ("baseline",) + MR1])
    with open(args.out / "per_task_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in tasks:
            w.writerow(rows[t])

    def mean_of(col):
        return float(np.mean([float(rows[t][col]) for t in tasks]))

    order = ["mean_ablation_mr157", "cf_mean_ablation_mr157",
             "zero_ablation_mr157", "cf_zero_ablation_mr157"]
    DIR = "$\\hat u_A$"
    short = {c: ("counterfactual task's " if c.startswith("cf_") else "own ") + DIR for c in order}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=200, sharey=True)
    for ax, n in zip(axes, (1, 6)):
        xs = np.array([0, 1, 2.6, 3.6])
        vals = [mean_of(f"n{n}_{c}") for c in order]
        ax.bar(xs, vals, color=[COLOR[c] for c in order], width=0.8)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x, v), ha="center", va="bottom", fontsize=9.5)
        bl, zs = mean_of(f"n{n}_baseline"), mean_of("zero_shot")
        ax.axhline(bl, color="0.35", lw=1.6, ls=(0, (5, 3)))
        ax.axhline(zs, color="0.6", lw=1.4, ls=(0, (2, 2)))
        ax.annotate(f"unablated {bl:.3f}", (3.9, bl), ha="right", va="bottom", fontsize=10, color="0.25")
        ax.annotate(f"0-shot {zs:.3f}", (3.9, zs), ha="right", va="bottom", fontsize=10, color="0.45")
        ax.set_xticks(xs)
        ax.set_xticklabels([short[c] for c in order], fontsize=10, rotation=20, ha="right")
        ax.text(0.5, -0.16, "mean ablation", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color="#2f7fe0")
        ax.text(3.1, -0.16, "zero ablation", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color="#d94f3d")
        ax.set_title(f"{n}-shot", fontsize=14)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"accuracy (T=1 sampled exact match, mean over {T} tasks)", fontsize=11)
    fig.suptitle(f"Task-unique direction {DIR} (mean of carrier-removed {args.band_label} read features) "
                 f"ablated at every demo target token, every block — {args.model_label}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(args.out / "aggregate_bars_full.png", bbox_inches="tight")
    plt.close(fig)

    # ---- SIMPLE headline: mean-ablation only — unablated | own direction | counterfactual direction ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=150, sharey=True)
    axes[0].set_ylim(0, max(mean_of("n6_baseline"), mean_of("n6_cf_mean_ablation_mr157")) + 0.09)
    for ax, n in zip(axes, (1, 6)):
        vals = [mean_of(f"n{n}_baseline"), mean_of(f"n{n}_mean_ablation_mr157"),
                mean_of(f"n{n}_cf_mean_ablation_mr157")]
        bars = ax.bar([0, 1, 2], vals, color=[C.grey, C.read, C.accent], width=0.62, zorder=3)
        label_bars(ax, bars, fontsize=10)
        ax.set_xticks([0, 1, 2], ["unablated", "ablate own\n" + DIR, "ablate counterfactual\ntask's " + DIR])
        ax.set_title(f"{n}-shot prompts")
    axes[0].set_ylabel(f"task accuracy (mean, {T} tasks)")
    fig.suptitle(f"Ablating one task-unique direction at the demonstration target tokens ({args.model_label})",
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "aggregate_bars.png")
    plt.close(fig)

    for n in (1, 6):
        fig, ax = plt.subplots(figsize=(26, 7.5), dpi=170)
        x = np.arange(T)
        w = 0.2
        for ci, c in enumerate(MR1):
            vals = [float(rows[t][f"n{n}_{c}"]) for t in tasks]
            lab = ("cf " if c.startswith("cf_") else "own ") + ("mean" if "mean" in c else "zero")
            ax.bar(x + (ci - 1.5) * w, vals, w, color=COLOR[c], label=lab)
        bl = [float(rows[t][f"n{n}_baseline"]) for t in tasks]
        ax.plot(x, bl, ls="none", marker="_", ms=13, mew=2.2, color="0.25", label="unablated baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=90, fontsize=7.5)
        ax.set_ylabel("accuracy")
        ax.set_xlim(-0.6, T - 0.4)
        ax.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9, ncol=5, loc="upper right")
        ax.set_title(f"Task-unique direction {DIR} ablation, per task ({n}-shot, {args.model_label})",
                     fontsize=14, fontweight="bold", loc="left")
        fig.tight_layout()
        fig.savefig(args.out / f"per_task_bars_{n}shot.png", bbox_inches="tight")
        plt.close(fig)

    for n in (1, 6):
        print(f"n{n}: base={mean_of(f'n{n}_baseline'):.3f}  zero_shot={mean_of('zero_shot'):.3f}  "
              + "  ".join(f"{c}={mean_of(f'n{n}_{c}'):.3f}" for c in MR1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
