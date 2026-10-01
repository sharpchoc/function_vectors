#!/usr/bin/env python
"""Summarise the FV-head mean-ablation study (ablate_fv_heads_cue6.py).

Reads <eval_root>/<task>.json (conditions real6_baseline, fvheads_mean, rand{k}_mean) and
the zero-shot floor per task from the FV_ablation per_task_acc.csv of the same model/pool
(same prompts and readout). Writes to <out_dir>:
  summary.csv        mean/median accuracy per condition x {train, heldout, all}
  per_task_acc.csv   per-task accuracies (+ mean over the random sets)
  headline_bars.png  pooled bars: 0-shot | 6-shot unablated | FV heads mean-ablated |
                     random same-size head sets mean-ablated (mean of K draws)
  by_task_dots.png   per-task: unablated vs FV-heads-ablated vs random-ablated, sorted
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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", type=Path,
                    default=ARTIFACTS_ROOT / "69_task_run" / "FV_head_mean_ablation" / "eval")
    ap.add_argument("--zs_csv", type=Path,
                    default=TASK69_RUN_DIR / "FV_ablation" / "per_task_acc.csv",
                    help="per-task CSV with a zero_shot column (same prompts/readout)")
    ap.add_argument("--out_dir", type=Path,
                    default=TASK69_RUN_DIR / "FV_ablation" / "head_mean_ablation")
    ap.add_argument("--split_path", type=Path,
                    default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    ap.add_argument("--model_label", default="GPT-J-6B")
    return ap.parse_args()


def main():
    a = parse_args()
    split = json.load(open(a.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(t for t in group if (a.eval_root / f"{t}.json").exists())
    missing = sorted(set(group) - set(tasks))
    if missing:
        print(f"WARNING: {len(missing)} tasks missing: {missing[:6]}")
    d = {t: json.load(open(a.eval_root / f"{t}.json")) for t in tasks}
    zs_rows = {r["task"]: float(r["zero_shot"]) for r in csv.DictReader(open(a.zs_csv))}
    n_rand = d[tasks[0]]["n_random_sets"]
    n_heads = d[tasks[0]]["n_fv_heads"]
    rand_names = [f"rand{k}_mean" for k in range(n_rand)]
    grp = np.array([group[t] for t in tasks])
    acc = {"zero_shot": np.array([zs_rows[t] for t in tasks]),
           "real_6shot": np.array([d[t]["conditions"]["real6_baseline"]["acc"] for t in tasks]),
           "fvheads_mean": np.array([d[t]["conditions"]["fvheads_mean"]["acc"] for t in tasks])}
    for r in rand_names:
        acc[r] = np.array([d[t]["conditions"][r]["acc"] for t in tasks])
    acc["random_mean"] = np.mean(np.stack([acc[r] for r in rand_names]), axis=0)
    order = ["zero_shot", "real_6shot", "fvheads_mean", "random_mean"] + rand_names

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group"] + order)
        for i, t in enumerate(tasks):
            w.writerow([t, group[t]] + [acc[c][i] for c in order])
    rows = [["condition", "task_group", "mean_acc", "median_acc", "n_tasks"]]
    for c in order:
        for g in ("train", "heldout", "all"):
            m = np.ones(len(tasks), bool) if g == "all" else grp == g
            rows.append([c, g, round(float(acc[c][m].mean()), 4),
                         round(float(np.median(acc[c][m])), 4), int(m.sum())])
    with open(a.out_dir / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for r in rows:
        if r[1] == "all":
            print("  ".join(str(x) for x in r))

    labels = ["0-shot\n(no demos)", "6-shot\nunablated", f"6-shot, {n_heads} FV heads\nmean-ablated at cue",
              f"6-shot, {n_heads} random\nnon-FV heads mean-ablated\n(mean of {n_rand} draws)"]
    keys = ["zero_shot", "real_6shot", "fvheads_mean", "random_mean"]
    colors = ["#bfbfbf", "#2ca02c", "#d62728", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    x = np.arange(len(keys))
    means = [float(acc[k].mean()) for k in keys]
    sems = [float(acc[k].std(ddof=1) / np.sqrt(len(tasks))) for k in keys]
    ax.bar(x, means, 0.62, color=colors, yerr=sems, capsize=3)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.015, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel(f"mean T=1 exact-match accuracy ({len(tasks)} tasks)")
    ax.set_ylim(0, max(1.0, max(means) + 0.08))
    ax.set_title(f"Mean-ablating the FV heads at the final cue token, {a.model_label}", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(a.out_dir / "headline_bars.png", bbox_inches="tight")

    o = np.argsort(acc["real_6shot"])
    fig, ax = plt.subplots(figsize=(max(15, 0.36 * len(tasks)), 6.4), dpi=150)
    xs = np.arange(len(tasks))
    ax.vlines(xs, acc["fvheads_mean"][o], acc["real_6shot"][o], color="0.75", lw=1)
    ax.scatter(xs, acc["real_6shot"][o], s=22, color="#2ca02c", label="6-shot, unablated", zorder=3)
    ax.scatter(xs, acc["random_mean"][o], s=22, color="#1f77b4", marker="s",
               label=f"random non-FV heads mean-ablated (mean of {n_rand})", zorder=3)
    ax.scatter(xs, acc["fvheads_mean"][o], s=22, color="#d62728", marker="v",
               label=f"{n_heads} FV heads mean-ablated", zorder=4)
    ax.scatter(xs, acc["zero_shot"][o], s=12, color="0.5", marker="_", label="0-shot", zorder=2)
    ax.set_xticks(xs, [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in o],
                  rotation=90, fontsize=6.2)
    ax.set_ylabel("T=1 sampled exact-match accuracy")
    ax.set_title(f"FV-head mean ablation at the cue by task, {a.model_label} — * = held-out", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(a.out_dir / "by_task_dots.png", bbox_inches="tight")
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
