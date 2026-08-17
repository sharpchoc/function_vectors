#!/usr/bin/env python
"""Summarise the label-slot head-selection experiment (read_vector_head_selection).

Reads artifacts/69_task_run/read_vector_head_selection/eval/<task>.json (produced by
eval_label_slot_vectors.py) plus the existing 0-shot / real-1-shot artifacts, and writes to
results/69_task_run/read_vector_head_selection/:

  by_task.png        the headline per-task breakdown (all 69 tasks, * = held-out):
                     unsteered | mean-activation-difference | sparse-selected head sum |
                     real 1-shot demo
  layer_alpha.png    mean accuracy vs alpha for every vector family at L3 and L7, with the
                     cross terms (L7-selected heads injected at L3 and vice versa)
  summary.csv        per condition: mean/median accuracy, split by train / heldout / all
  per_task_acc.csv   per-task accuracies for every headline condition
  selection_summary.csv  chosen lambda, head counts, layer histogram, overlap with the
                     canonical 37-head cue-token FV set
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

AR = ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection"
REF = ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot"
SEL7 = AR / "pooled_sparse" / "selection.json"
SEL3 = (ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection_L3"
        / "pooled_sparse" / "selection.json")
FV37 = (ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43"
        / "pooled_sparse" / "selection.json")
OUT = TASK69_RUN_DIR / "read_vector_head_selection"
ALPHAS = (0.5, 1.0, 2.0, 4.0)
LAYERS = (3, 7)


def best_over_alpha(cond, prefix):
    """max over alphas of conditions named f'{prefix}_a{alpha}'."""
    vals = [cond[f"{prefix}_a{a}"]["acc"] for a in ALPHAS if f"{prefix}_a{a}" in cond]
    return max(vals) if vals else np.nan


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)
    files = {t: AR / "eval" / f"{t}.json" for t in tasks}
    missing = [t for t in tasks if not files[t].exists()]
    assert not missing, f"missing eval files: {missing[:5]}"

    data = {t: json.load(open(files[t])) for t in tasks}
    base = np.array([data[t]["conditions"]["baseline"]["acc"] for t in tasks])
    r1 = np.array([json.load(open(REF / f"{t}__real_1shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    zs = np.array([json.load(open(REF / f"{t}__zero_shot.json"))
                   ["conditions"]["baseline"]["acc"] for t in tasks])
    grp = np.array([group[t] for t in tasks])

    # every family x injection layer, best over alpha
    fam = {}
    for sel in ("L3", "L7"):
        for inj in LAYERS:
            key = f"headsum_{sel}sel@L{inj}"
            fam[key] = np.array([best_over_alpha(data[t]["conditions"], key) for t in tasks])
    for kind in ("meandiff", "rawmean"):
        for inj in LAYERS:
            key = f"{kind}@L{inj}"
            fam[key] = np.array([best_over_alpha(data[t]["conditions"], key) for t in tasks])

    OUT.mkdir(parents=True, exist_ok=True)
    # ---- summary.csv over every condition ----
    rows = []
    for name, arr in [("unsteered", base), ("zero_shot", zs), ("real_1shot", r1)] + \
                     sorted(fam.items()):
        for g in ("train", "heldout", "all"):
            m = np.ones(len(tasks), bool) if g == "all" else grp == g
            rows.append([name, g, round(float(np.nanmean(arr[m])), 4),
                         round(float(np.nanmedian(arr[m])), 4),
                         round(float(np.nanmean(arr[m] - base[m])), 4)])
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "task_group", "mean_acc", "median_acc", "mean_uplift"])
        w.writerows(rows)
    for r in rows:
        if r[1] == "all":
            print("  ".join(str(x) for x in r))

    # headline picks: the best head-sum cell and the best mean-difference cell
    head_key = max([k for k in fam if k.startswith("headsum")],
                   key=lambda k: np.nanmean(fam[k]))
    md_key = max([k for k in fam if k.startswith("meandiff")],
                 key=lambda k: np.nanmean(fam[k]))
    print(f"headline head-sum: {head_key} ({np.nanmean(fam[head_key]):.3f}) | "
          f"mean-diff: {md_key} ({np.nanmean(fam[md_key]):.3f})")

    # ---- per-task bar chart ----
    order = np.argsort(fam[head_key])
    labels = [tasks[i] + (" *" if grp[i] == "heldout" else "") for i in order]
    x = np.arange(len(tasks))
    w = 0.21
    fig, ax = plt.subplots(figsize=(max(14, 0.34 * len(tasks)), 7.0), dpi=150)
    ax.bar(x - 1.5 * w, base[order], w, color="0.7", label="unsteered (1-shot '_' scaffold)")
    ax.bar(x - 0.5 * w, fam[md_key][order], w, color="tab:orange",
           label=f"mean-activation difference ({md_key.split('@')[1]})")
    ax.bar(x + 0.5 * w, fam[head_key][order], w, color="tab:blue",
           label=f"sparse-selected head sum ({head_key})")
    ax.bar(x + 1.5 * w, r1[order], w, color="tab:green", label="real 1-shot demo")
    ax.set_xticks(x, labels, rotation=90, fontsize=6.4)
    ax.set_ylabel("T=1 sampled exact-match accuracy (150 prompts)")
    ax.set_title("Label-slot steering by task — sparse-selected head vector vs baselines\n"
                 "(each steered bar at its best alpha; * = held-out task, others are "
                 "in-sample for the head selection)", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "by_task.png", bbox_inches="tight")

    # ---- alpha curves per family ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150, sharey=True)
    for ax, inj in zip(axes, LAYERS):
        for key, col in ((f"headsum_L3sel@L{inj}", "tab:blue"),
                         (f"headsum_L7sel@L{inj}", "tab:cyan"),
                         (f"meandiff@L{inj}", "tab:orange"),
                         (f"rawmean@L{inj}", "tab:red")):
            ys = [float(np.nanmean([data[t]["conditions"][f"{key}_a{a}"]["acc"]
                                    for t in tasks])) for a in ALPHAS]
            ax.plot(ALPHAS, ys, "o-", color=col, label=key.split("@")[0])
        ax.axhline(float(base.mean()), color="0.45", ls=":", lw=1, label="unsteered")
        ax.axhline(float(r1.mean()), color="tab:green", ls="-", lw=1.2, label="real 1-shot")
        ax.set_xscale("log"); ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
        ax.set_xlabel("alpha (x the vector's natural magnitude)")
        ax.set_title(f"injection at L{inj}", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=7.5)
    axes[0].set_ylabel(f"mean accuracy ({len(tasks)} tasks)")
    fig.suptitle("Label-slot steering: dose response by vector family and injection site",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "layer_alpha.png", bbox_inches="tight")

    # ---- per-task csv ----
    with open(OUT / "per_task_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = sorted(fam)
        w.writerow(["task", "group", "unsteered", "zero_shot", "real_1shot"] + keys)
        for i, t in enumerate(tasks):
            w.writerow([t, group[t], base[i], zs[i], r1[i]] +
                       [round(float(fam[k][i]), 4) for k in keys])

    # ---- selection summary ----
    import collections
    fv = set(json.load(open(FV37))["selected_flat"])
    with open(OUT / "selection_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["selection", "inject_layer", "chosen_lambda", "n_heads",
                    "overlap_with_37head_FV", "jaccard_with_FV", "layer_histogram"])
        for tag, pth in (("L3", SEL3), ("L7", SEL7)):
            if not pth.exists():
                continue
            d = json.load(open(pth))
            s = set(d["selected_flat"])
            hist = sorted(collections.Counter(x // 16 for x in s).items())
            w.writerow([tag, d["inject_layer"], d["chosen_lambda"], d["n_selected"],
                        len(s & fv), round(len(s & fv) / len(s | fv), 4),
                        " ".join(f"L{l}:{c}" for l, c in hist)])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
