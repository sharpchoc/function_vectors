#!/usr/bin/env python
"""Poster visuals for the 6-shot random-label steering experiment.

Writes to results/.../steering_results/sixshot_randomlabel/poster_visuals/:
  aggregate_bars.png  six bars: 0-shot | random 6-shot unsteered | real 1-shot |
                      dummy '_' steered (best alpha) | random steered (best alpha) |
                      real 6-shot. Means over 69 tasks with 95% CIs, direct value labels.
  example_prompt.png  a real agent_noun_to_verb prompt from the run: six demos with
                      random wrong-task labels (injection sites highlighted), the query,
                      and the model's steered answer; each label annotated with its
                      source task (recovered via the run's seeded RNG).

Palette: dataviz reference instance (categorical slot 1 blue #2a78d6 = this study's
intervention, slot 2 orange #eb6834 = the dummy-slot variant; neutral inks for
baselines/references). CI whiskers in ink with a surface-coloured halo.
"""
import csv
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402
from src.sandbox.ext_steerability.sixshot_randomlabel_steer import (  # noqa: E402
    build_output_pools)

OUT = (TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results"
       / "sixshot_randomlabel" / "poster_visuals")
RES = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / "sixshot_randomlabel"
RL = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sixshot_randomlabel"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"     # this study's intervention
ORANGE = "#eb6834"   # the dummy-'_' variant (comparison)
GRAY_LIGHT = "#d9d8d3"   # unsteered baselines (below 3:1 -> relief rule: value labels)
GRAPHITE = "#4a4a47"     # real-demo references
EX_TASK = "agent_noun_to_verb"
N_SHOTS = 6


def ci95(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def aggregate_bars():
    rows = list(csv.DictReader(open(RES / "per_task_acc.csv")))
    col = lambda c: np.array([float(r[c]) for r in rows])
    bars = [
        ("No demos\n(0-shot)", col("zero_shot"), GRAY_LIGHT),
        ("Random wrong\nlabels, unsteered", col("random6_unsteered"), GRAY_LIGHT),
        ("Real 1-shot\ndemo", col("real_1shot"), GRAPHITE),
        ("Dummy '_',\nsteered (best α)", col("dummy6_steered_best"), ORANGE),
        ("Random labels,\nsteered (best α)", col("random6_steered_best"), BLUE),
        ("Real 6-shot\ndemos", col("real_6shot"), GRAPHITE),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(bars))
    for i, (lab, v, c) in enumerate(bars):
        m, e = float(v.mean()), ci95(v)
        ax.bar(i, m, 0.62, color=c, zorder=3)
        if m >= 0.01:  # no whisker clutter on the ~0 floor bars
            # ink whiskers with a surface halo so they read on any bar colour
            ax.errorbar(i, m, yerr=e, ecolor=SURFACE, elinewidth=4.2, capsize=7,
                        capthick=4.2, zorder=4)
            ax.errorbar(i, m, yerr=e, ecolor=INK, elinewidth=1.6, capsize=6,
                        capthick=1.6, zorder=5)
        else:
            e = 0.0
        ax.text(i, m + e + 0.018, f"{m:.2f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=INK, zorder=6)
    ax.set_xticks(x, [b[0] for b in bars], fontsize=10.5, color=INK)
    ax.set_ylabel("task accuracy", fontsize=12, color=INK2)
    ax.set_ylim(0, 0.74)
    ax.set_title("Steering a 6-shot prompt with mixed random labels",
                 fontsize=16, fontweight="bold", color=INK, loc="left", pad=14)
    ax.grid(axis="y", color="#eceae6", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="y", colors=INK2, labelsize=10)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "aggregate_bars.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT / 'aggregate_bars.png'}")


def recover_sources(task, rec_idx, demos, pools):
    """Re-run the run script's seeded sampling to recover each label's source task."""
    other = sorted(t for t in pools if t != task)
    out = []
    for si, d in enumerate(demos):
        true_out = str(d["output"]).strip()
        rng = random.Random(zlib.crc32(f"{task}|{rec_idx}|{si}".encode()))
        while True:
            src = rng.choice(other)
            lab = rng.choice(pools[src])
            if lab != true_out:
                break
        out.append((lab, src))
    return out


def example_prompt():
    d = json.load(open(RL / f"{EX_TASK}.json"))
    recs = json.load(open(REPO_ROOT / "dataset_files" / "isolation_prompts_ext"
                          / EX_TASK / "train_prompts.json"))
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    pools = build_output_pools(sorted(split["train_tasks"] + split["heldout_tasks"]),
                               REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    # first prompt whose steered (alpha=4) answer is correct, for a clean illustration
    preds = d["conditions"]["random6_steer_a4.0"]["preds"]
    ri = next(i for i, (p, g) in enumerate(zip(preds, d["golds"])) if p == g)
    rec = recs[ri]
    demos = rec["demos"][:N_SHOTS]
    labels = recover_sources(EX_TASK, ri, demos, pools)
    assert [l for l, _ in labels] == d["rand_labels"][ri], "seeded recovery mismatch"
    gold, pred = d["golds"][ri], preds[ri]

    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.24, 1)
    mono = {"family": "monospace", "fontsize": 12.5}
    y, dy = 0.945, 0.0455
    x0 = 0.035
    for (di, (lab, src)) in zip(demos, labels):
        ax.text(x0, y, f"Q: {di['input']}", color=INK, va="center", **mono)
        y -= dy
        ax.text(x0, y, "A:", color=INK, va="center", **mono)
        # highlighted injected label
        t = ax.text(x0 + 0.042, y, f" {lab} ", color=SURFACE, va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", fc=BLUE, ec="none"), **mono)
        fig.canvas.draw()
        w = ax.transData.inverted().transform_bbox(
            t.get_window_extent()).width
        ax.text(x0 + 0.062 + w, y, f"← random output from '{src}'",
                color=INK2, va="center", fontsize=10.5, style="italic")
        y -= dy * 1.28
    ax.text(x0, y, f"Q: {rec['query']['input']}", color=INK, va="center", **mono)
    y -= dy
    ax.text(x0, y, "A:", color=INK, va="center", **mono)
    ax.text(x0 + 0.042, y, f" {pred}", color=BLUE, va="center",
            fontweight="bold", **mono)
    ax.text(x0 + 0.042 + 0.02 * (len(pred) + 2), y,
            f"   ← model's steered answer (gold: '{gold}')",
            color=INK2, va="center", fontsize=10.5, style="italic")
    # intervention note
    y -= dy * 1.7
    ax.text(x0, y, f"Intervention: z ← z + α·m_A(L6) at block-6 output, "
                   f"at every highlighted label token",
            color=BLUE, fontsize=12, fontweight="bold", va="center")
    y -= dy * 0.85
    ax.text(x0, y, f"m_A = task-mean label-token activation of '{EX_TASK}' "
                   f"(the bottom-up read feature)",
            color=INK2, fontsize=10.5, va="center")
    ax.set_title("Example prompt: mixed random labels, steered",
                 fontsize=16, fontweight="bold", color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "example_prompt.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT / 'example_prompt.png'} (prompt #{ri}, gold '{gold}')")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aggregate_bars()
    example_prompt()


if __name__ == "__main__":
    main()
