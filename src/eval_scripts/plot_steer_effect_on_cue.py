#!/usr/bin/env python
"""Summarise how label-slot steering moves the final cue token towards the task FV.

Reads artifacts/69_task_run/mean_read_steering_effect_on_write/<task>.pt
(steer_effect_on_cue.py) and writes to results/69_task_run/mean_read_steering_effect_on_write/:

  headline_cos.png    THE plot: x = steering strength alpha, y = CHANGE in cos(cue activation
                      at L13, reference) vs the alpha=0 run. Two series: the task's own FV and
                      the all-task-averaged FV. Per-task points scattered, task-mean line over.
  headline_proj.png   same, for projection magnitude onto each reference direction.
  layer_profile.png   delta cos towards the task FV vs LAYER, one line per alpha (context for
                      why L13 is the headline).
  summary.csv         per alpha: raw + delta cos/proj for both references, train/heldout split.
  per_task.csv        per task x alpha at L13.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "mean_read_steering_effect_on_write"
OUT = TASK69_RUN_DIR / "mean_read_steering_effect_on_write"
LAYER = 13
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"   # task FV / generic FV — validated adjacent pair


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(t for t in group if (AR / f"{t}.pt").exists())
    missing = [t for t in sorted(group) if t not in tasks]
    if missing:
        print(f"WARNING: missing {len(missing)}: {missing[:5]}")
    d0 = torch.load(AR / f"{tasks[0]}.pt", map_location="cpu", weights_only=False)
    alphas = list(d0["alphas"])
    grp = np.array([group[t] for t in tasks])

    # per task: mean over prompts -> (n_alpha,) at LAYER, and (n_alpha, 28) for the profile
    keys = ("cos_task", "cos_gen", "proj_task", "proj_gen")
    at_layer = {k: np.zeros((len(tasks), len(alphas))) for k in keys}
    profile = np.zeros((len(tasks), len(alphas), 28))
    for ti, t in enumerate(tasks):
        d = torch.load(AR / f"{t}.pt", map_location="cpu", weights_only=False)
        for k in keys:
            at_layer[k][ti] = d[k][:, :, LAYER].mean(dim=1).numpy()
        profile[ti] = d["cos_task"].mean(dim=1).numpy()
    delta = {k: at_layer[k] - at_layer[k][:, [0]] for k in keys}
    dprof = profile - profile[:, [0], :]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "task_group", "cos_task_raw", "cos_gen_raw", "d_cos_task",
                    "d_cos_gen", "proj_task_raw", "proj_gen_raw", "d_proj_task", "d_proj_gen"])
        for ai, a in enumerate(alphas):
            for g in ("train", "heldout", "all"):
                m = np.ones(len(tasks), bool) if g == "all" else grp == g
                w.writerow([a, g] + [round(float(x[m, ai].mean()), 5) for x in
                                     (at_layer["cos_task"], at_layer["cos_gen"],
                                      delta["cos_task"], delta["cos_gen"],
                                      at_layer["proj_task"], at_layer["proj_gen"],
                                      delta["proj_task"], delta["proj_gen"])])
    with open(OUT / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group"] +
                   [f"{k}_a{a}" for k in ("d_cos_task", "d_cos_gen", "d_proj_task",
                                          "d_proj_gen") for a in alphas])
        for ti, t in enumerate(tasks):
            row = [t, group[t]]
            for k in ("cos_task", "cos_gen", "proj_task", "proj_gen"):
                row += [round(float(delta[k][ti, ai]), 5) for ai in range(len(alphas))]
            w.writerow(row)

    for ai, a in enumerate(alphas):
        print(f"alpha={a}: d_cos_task={delta['cos_task'][:, ai].mean():+.4f} "
              f"d_cos_gen={delta['cos_gen'][:, ai].mean():+.4f} "
              f"d_proj_task={delta['proj_task'][:, ai].mean():+.2f} "
              f"d_proj_gen={delta['proj_gen'][:, ai].mean():+.2f} "
              f"(raw cos_task {at_layer['cos_task'][:, ai].mean():.4f})")

    def scatter_fig(dt, dg, ylabel, title, fname, fmt="{:+.3f}"):
        fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
        fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
        rng = np.random.RandomState(0)
        for arr, col, lab in ((dt, BLUE, "towards the task's own FV"),
                              (dg, ORANGE, "towards the all-task averaged FV")):
            for ai, a in enumerate(alphas):
                jitter = rng.uniform(-0.06, 0.06, arr.shape[0])
                ax.scatter(np.full(arr.shape[0], ai) + jitter, arr[:, ai], s=13,
                           alpha=0.32, color=col, linewidths=0, zorder=2)
            ax.plot(range(len(alphas)), arr.mean(axis=0), "o-", color=col, lw=2.4, ms=8,
                    zorder=4, label=lab, markeredgecolor=SURFACE, markeredgewidth=1.4)
            for ai in range(len(alphas)):
                if ai == len(alphas) - 1:
                    ax.annotate(fmt.format(arr[:, ai].mean()),
                                (ai, arr[:, ai].mean()), textcoords="offset points",
                                xytext=(10, 0), fontsize=11, color=col, fontweight="bold",
                                va="center")
        ax.axhline(0, color=INK2, lw=1.0, ls=":")
        ax.set_xticks(range(len(alphas)), [str(a) for a in alphas], fontsize=11)
        ax.set_xlabel("steering strength α  (× the injected vector's own norm)",
                      fontsize=11.5, color=INK2)
        ax.set_ylabel(ylabel, fontsize=11.5, color=INK2)
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK, loc="left", pad=12)
        ax.tick_params(colors=INK2)
        ax.grid(True, color=GRID, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.legend(fontsize=10, frameon=False, loc="upper left")
        fig.text(0.005, 0.005, f"GPT-J-6B, {len(tasks)} tasks. 6-shot dummy-'_' prompt; "
                 f"α·(task mean L6 label activation) added at all six label slots; readout = "
                 f"layer {LAYER} residual at the final cue token. One point per task "
                 f"(mean over its 150 prompts); line = mean over tasks.",
                 fontsize=8.5, color=INK2, ha="left", va="bottom", wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(OUT / fname, bbox_inches="tight", facecolor=SURFACE)

    scatter_fig(delta["cos_task"], delta["cos_gen"],
                f"change in cosine similarity at L{LAYER}  (vs α = 0)",
                "Steering the label slots rotates the cue-token\nrepresentation towards the "
                "task's function vector", "headline_cos.png")
    scatter_fig(delta["proj_task"], delta["proj_gen"],
                f"change in projection magnitude at L{LAYER}  (vs α = 0)",
                "…and increases how far it extends along that direction",
                "headline_proj.png", fmt="{:+.1f}")

    # layer profile of the task-FV delta
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    shades = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]
    for ai, a in enumerate(alphas):
        if a == 0:
            continue
        ax.plot(range(28), dprof[:, ai, :].mean(axis=0), "o-", ms=3.5, lw=1.8,
                color=shades[(ai - 1) % len(shades)], label=f"α = {a}")
    ax.axvline(LAYER, color=INK2, ls=":", lw=1.1)
    ax.text(LAYER + 0.4, ax.get_ylim()[1] * 0.92, f"L{LAYER}\n(headline)", fontsize=9,
            color=INK2, va="top")
    ax.axhline(0, color=INK2, lw=1.0, ls=":")
    ax.set_xlabel("layer of the cue-token readout", fontsize=11.5, color=INK2)
    ax.set_ylabel("change in cos(cue activation, task FV)", fontsize=11.5, color=INK2)
    ax.set_title("Where the injected label-slot signal shows up at the cue token",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=12)
    ax.set_xticks(range(0, 28, 2))
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "layer_profile.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
