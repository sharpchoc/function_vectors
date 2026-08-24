#!/usr/bin/env python
"""Error analysis: task-unique swap vs full-mean steering on the 6-shot '_' scaffold.

Discriminator for the cue-side "generic emit-a-label drive" hypothesis: if the carrier's
advantage comes from pushing the cue into answer-emission mode, the swap's EXTRA errors
(prompts full-mean gets right and swap gets wrong, each at its per-task best alpha) should
be dominated by FORMAT failures (underscore echo, prompt continuation, garbage) rather
than valid-label mapping errors.

Categories (first match wins):
  correct     pred == gold (the eval criterion)
  near_miss   casefold+strip match to gold (format-only miss)
  underscore  pred is '_' / starts with '_' (echoes the dummy label)
  empty       empty prediction
  own_pool    valid label from the task's own output pool, wrong mapping
  other_pool  valid label of a DIFFERENT task's pool (wrong-task-style output)
  other       everything else (garbage / continuation / off-format)

Inputs are the stored eval JSONs (preds aligned by index: both scripts build items from
train_prompts.json in file order; golds come from the sixshot_dummy JSONs).

Writes results/.../steering_results/error_analysis_swap_vs_fullmean/:
  summary.csv        category shares over all 69x150 predictions, per method
  gap_breakdown.csv  swap's category distribution on gap prompts (fullmean right, swap
                     wrong) and the reverse gap
  per_task.csv       per-task counts for both methods + gap counts
  breakdown_bars.png overall shares + gap-prompt breakdown
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402
from src.sandbox.ext_steerability.sixshot_randomlabel_steer import (  # noqa: E402
    build_output_pools)

AR = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering"
PROMPTS_ROOT = REPO_ROOT / "dataset_files" / "isolation_prompts_ext"
OUT = (TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
       / "error_analysis_swap_vs_fullmean")
SW_AL = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0)
FM_AL = (0.5, 1.0, 2.0, 4.0)
CATS = ("correct", "near_miss", "underscore", "empty", "own_pool", "other_pool", "other")
BLUE, ORANGE = "#2f7fe0", "#e07b2f"


def classify(pred, gold, own_pool, other_union):
    if pred == gold:
        return "correct"
    ps = pred.strip()
    if ps.casefold() == gold.strip().casefold():
        return "near_miss"
    if ps.startswith("_"):
        return "underscore"
    if not ps:
        return "empty"
    if ps in own_pool:
        return "own_pool"
    if ps in other_union:
        return "other_pool"
    return "other"


def best_cond(conds, fmt, alphas):
    return max((fmt.format(a) for a in alphas), key=lambda c: conds[c]["acc"])


def main():
    split = json.load(
        open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    assert len(tasks) == 69
    pools = build_output_pools(tasks, PROMPTS_ROOT)
    union_all = {t: set(pools[t]) for t in tasks}

    per_task, tot = [], {"swap": Counter(), "fullmean": Counter()}
    gap_sw = Counter()   # fullmean correct, swap wrong: swap's category
    gap_fm = Counter()   # swap correct, fullmean wrong: fullmean's category
    n_gap_sw = n_gap_fm = 0
    for t in tasks:
        sw = json.load(open(AR / "taskunique_svd_dummy" / f"{t}.json"))["conditions"]
        fmj = json.load(open(AR / "sixshot_dummy" / f"{t}.json"))
        golds = fmj["golds"]
        assert len(golds) == 150
        other_union = set().union(*(union_all[o] for o in tasks if o != t))
        own = union_all[t]
        c_sw = best_cond(sw, "dummy6_swap1_a{}", SW_AL)
        c_fm = best_cond(fmj["conditions"], "dummy6_steer_a{}", FM_AL)
        p_sw, p_fm = sw[c_sw]["preds"], fmj["conditions"][c_fm]["preds"]
        row = {"task": t, "swap_cond": c_sw, "fullmean_cond": c_fm}
        cnt = {"swap": Counter(), "fullmean": Counter()}
        for g, a, b in zip(golds, p_sw, p_fm):
            ka = classify(a, g, own, other_union)
            kb = classify(b, g, own, other_union)
            cnt["swap"][ka] += 1
            cnt["fullmean"][kb] += 1
            if kb == "correct" and ka != "correct":
                gap_sw[ka] += 1
                n_gap_sw += 1
            if ka == "correct" and kb != "correct":
                gap_fm[kb] += 1
                n_gap_fm += 1
        for m in ("swap", "fullmean"):
            tot[m].update(cnt[m])
            for c in CATS:
                row[f"{m}_{c}"] = cnt[m][c]
        row["gap_fm_right_sw_wrong"] = sum(
            1 for g, a, b in zip(golds, p_sw, p_fm) if b == g and a != g)
        per_task.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task[0].keys()))
        w.writeheader()
        w.writerows(per_task)

    n_all = 69 * 150
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n"] + [f"{c}_frac" for c in CATS])
        for m in ("swap", "fullmean"):
            w.writerow([m, n_all] + [round(tot[m][c] / n_all, 4) for c in CATS])

    with open(OUT / "gap_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap", "n"] + list(CATS[1:]))
        w.writerow(["fullmean_right_swap_wrong (swap's category)", n_gap_sw]
                   + [gap_sw[c] for c in CATS[1:]])
        w.writerow(["swap_right_fullmean_wrong (fullmean's category)", n_gap_fm]
                   + [gap_fm[c] for c in CATS[1:]])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=200)
    x = np.arange(len(CATS))
    for m, col, off in (("swap", ORANGE, -0.18), ("fullmean", BLUE, 0.18)):
        axes[0].bar(x + off, [tot[m][c] / n_all for c in CATS], width=0.34,
                    color=col, label=f"{m} (per-task best $\\alpha$)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CATS, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("fraction of all 69×150 predictions", fontsize=10)
    axes[0].set_title("Prediction categories, all prompts", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False)
    x2 = np.arange(len(CATS) - 1)
    axes[1].bar(x2, [gap_sw[c] / max(n_gap_sw, 1) for c in CATS[1:]], width=0.55,
                color=ORANGE)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(CATS[1:], rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("fraction of gap prompts", fontsize=10)
    axes[1].set_title(f"What the swap outputs where ONLY full-mean is correct "
                      f"(n={n_gap_sw})", fontsize=11)
    for ax in axes:
        ax.grid(axis="y", color="0.92")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("Swap vs full-mean steering: error anatomy (6-shot '_' scaffold)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "breakdown_bars.png", bbox_inches="tight")
    plt.close(fig)

    print("overall:", {m: {c: round(tot[m][c] / n_all, 3) for c in CATS}
                       for m in ("swap", "fullmean")})
    print(f"gap fullmean-right-swap-wrong n={n_gap_sw}:",
          {c: round(gap_sw[c] / max(n_gap_sw, 1), 3) for c in CATS[1:]})
    print(f"reverse gap n={n_gap_fm}:",
          {c: round(gap_fm[c] / max(n_gap_fm, 1), 3) for c in CATS[1:]})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
