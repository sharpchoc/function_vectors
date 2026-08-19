#!/usr/bin/env python
"""Stacked read->write panels: first-label steering, one panel per downstream cue token.

Reads artifacts/69_task_run/mean_read_steering_effect_on_write_firstlabel/<task>.pt
(steer_firstlabel_effect_on_cues.py) and writes to
results/69_task_run/read_write_relationship/bottom_up_firstlabel/:

  headline_cos_by_cue.png   3x2 grid, one panel per cue (cue2..cue6, query cue): change in
                            cos at L13 vs alpha, towards the own FV and the generic FV —
                            the same view as bottom_up/headline_cos.png, stacked by position
  excess_decay.png          task-specific excess (d_cos_task - d_cos_gen) at L13 vs cue
                            position, one line per alpha — how the effect propagates/decays
  summary.csv               per alpha x position: raw + delta cos/proj for both references
  per_task.csv              per task: d_cos_task/d_cos_gen at L13 for every alpha x position
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

AR = ARTIFACTS_ROOT / "69_task_run" / "mean_read_steering_effect_on_write_firstlabel"
OUT = TASK69_RUN_DIR / "read_write_relationship" / "bottom_up_firstlabel"
LAYER = 13
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
POS_LABEL = {"cue2": "demo-2 cue", "cue3": "demo-3 cue", "cue4": "demo-4 cue",
             "cue5": "demo-5 cue", "cue6": "demo-6 cue", "query_cue": "query cue (final)"}


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
    positions = list(d0["positions"])

    keys = ("cos_task", "cos_gen", "proj_task", "proj_gen")
    # (task, alpha, position) at LAYER, prompt-averaged
    at = {k: np.zeros((len(tasks), len(alphas), len(positions))) for k in keys}
    for ti, t in enumerate(tasks):
        d = torch.load(AR / f"{t}.pt", map_location="cpu", weights_only=False)
        assert list(d["positions"]) == positions
        for k in keys:
            at[k][ti] = d[k][:, :, :, LAYER].mean(dim=1).numpy()
    delta = {k: at[k] - at[k][:, [0], :] for k in keys}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "position", "cos_task_raw", "cos_gen_raw", "d_cos_task",
                    "d_cos_gen", "d_cos_excess", "proj_task_raw", "d_proj_task",
                    "d_proj_gen"])
        for ai, a in enumerate(alphas):
            for pi, pos in enumerate(positions):
                w.writerow([a, pos] + [round(float(x), 5) for x in (
                    at["cos_task"][:, ai, pi].mean(), at["cos_gen"][:, ai, pi].mean(),
                    delta["cos_task"][:, ai, pi].mean(), delta["cos_gen"][:, ai, pi].mean(),
                    (delta["cos_task"] - delta["cos_gen"])[:, ai, pi].mean(),
                    at["proj_task"][:, ai, pi].mean(),
                    delta["proj_task"][:, ai, pi].mean(),
                    delta["proj_gen"][:, ai, pi].mean())])
    with open(OUT / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group"] +
                   [f"d_cos_{ref}_{pos}_a{a}" for ref in ("task", "gen")
                    for pos in positions for a in alphas])
        for ti, t in enumerate(tasks):
            row = [t, group[t]]
            for ref in ("task", "gen"):
                for pi in range(len(positions)):
                    row += [round(float(delta[f"cos_{ref}"][ti, ai, pi]), 5)
                            for ai in range(len(alphas))]
            w.writerow(row)

    # ---------------- stacked headline grid ----------------
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.2), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.RandomState(0)
    for pi, (pos, ax) in enumerate(zip(positions, axes.T.flatten())):
        ax.set_facecolor(SURFACE)
        for arr, col, lab in ((delta["cos_task"][:, :, pi], BLUE, "towards own task FV"),
                              (delta["cos_gen"][:, :, pi], ORANGE, "towards generic FV")):
            for ai in range(len(alphas)):
                ax.scatter(np.full(arr.shape[0], ai) + rng.uniform(-0.06, 0.06, arr.shape[0]),
                           arr[:, ai], s=9, alpha=0.25, color=col, linewidths=0, zorder=2)
            ax.plot(range(len(alphas)), arr.mean(axis=0), "o-", color=col, lw=2.2, ms=7,
                    zorder=4, label=lab, markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(f"{arr[:, -1].mean():+.3f}", (len(alphas) - 1, arr[:, -1].mean()),
                        textcoords="offset points", xytext=(8, 0), fontsize=9.5, color=col,
                        fontweight="bold", va="center")
        ax.axhline(0, color=INK2, lw=0.9, ls=":")
        ax.set_title(POS_LABEL.get(pos, pos), fontsize=12.5, fontweight="bold", color=INK,
                     loc="left")
        ax.set_xticks(range(len(alphas)), [str(a) for a in alphas], fontsize=10)
        ax.tick_params(colors=INK2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        if pi == 0:
            ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("steering strength α", fontsize=11, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Δ cos at L{LAYER} (vs α = 0)", fontsize=11, color=INK2)
    fig.suptitle("Steering ONLY the first dummy label: effect on each downstream cue token",
                 fontsize=15, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.005, f"GPT-J-6B, {len(tasks)} tasks. 6-shot dummy-'_' prompt; "
             f"α·(task mean L6 label activation) added at the FIRST '_' slot only; readout = "
             f"layer {LAYER} residual at each later cue token (demo-1 cue precedes the "
             f"intervention and is skipped). One point per task; line = mean over tasks.",
             fontsize=8.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(OUT / "headline_cos_by_cue.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---------------- propagation / decay summary ----------------
    excess = delta["cos_task"] - delta["cos_gen"]     # (task, alpha, pos)
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    cmap = plt.get_cmap("Blues")
    for ai, a in enumerate(alphas[1:], start=1):
        ax.plot(range(len(positions)), excess[:, ai].mean(axis=0), "o-",
                color=cmap(0.35 + 0.16 * ai), lw=2.2, ms=7, label=f"α = {a:g}",
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(0, color=INK2, lw=1.0, ls=":")
    ax.set_xticks(range(len(positions)), [POS_LABEL.get(p, p) for p in positions],
                  fontsize=10, rotation=12)
    ax.set_ylabel(f"task-specific excess Δcos at L{LAYER}\n(own FV − generic FV, vs α = 0)",
                  fontsize=11, color=INK2)
    ax.set_title("How a single steered label propagates along the prompt",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=12)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "excess_decay.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    for pi, pos in enumerate(positions):
        print(f"{pos}: " + "  ".join(
            f"a{a:g} d_task={delta['cos_task'][:, ai, pi].mean():+.4f} "
            f"d_gen={delta['cos_gen'][:, ai, pi].mean():+.4f}"
            for ai, a in enumerate(alphas) if a in (1.0, 2.0)))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
