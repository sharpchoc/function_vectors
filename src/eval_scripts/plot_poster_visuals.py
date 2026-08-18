#!/usr/bin/env python
"""Poster visuals for the 6-shot dummy-label steering result.

Two figures into
results/69_task_run/raw_mean_steering/sixshot_dummy/poster_visuals/:

  headline_bars.png   three bars — no steering / steered / real 6-shot demos — mean over the
                      69 tasks with 95% CIs and direct value labels
  method_diagram.png  schematic of the intervention on a 1-shot dummy prompt: where the
                      vector comes from and where it is added

Colour: dataviz reference palette (blue #2a78d6 as the single categorical hue for the
intervention, neutral inks for the reference bars); validated with the ported six-checks
script (PASS; the light gray sits below 3:1 so every bar carries a visible value label).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

SRC = TASK69_RUN_DIR / "raw_mean_steering" / "sixshot_dummy"
OUT = SRC / "poster_visuals"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # the intervention
GRAY = "#b8b7b2"      # unsteered reference
GRAPHITE = "#52514e"  # real-demo reference
ALPHA_FIXED = "dummy6_steer_a4.0"


def load():
    rows = list(csv.DictReader(open(SRC / "per_task_acc.csv")))
    def col(k):
        return np.array([float(r[k]) for r in rows])
    return rows, col


def ci95(v):
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def headline_bars():
    rows, col = load()
    series = [("No steering\n(6 dummy '_' labels)", col("dummy6_unsteered"), GRAY),
              ("Steered\n(inject task mean at all 6 label slots)", col(ALPHA_FIXED), BLUE),
              ("Real 6-shot demos\n(upper reference)", col("real_6shot"), GRAPHITE)]
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(series))
    for i, (lab, v, c) in enumerate(series):
        m = float(v.mean())
        ax.bar(i, m, width=0.62, color=c, zorder=3,
               error_kw=dict(ecolor=INK2, lw=1.2, capsize=4, zorder=4),
               yerr=ci95(v) if m > 0 else None)
        ax.text(i, m + 0.014, f"{m:.3f}", ha="center", va="bottom", fontsize=17,
                fontweight="bold", color=INK, zorder=5)
    # the headline: how much of the real-demo effect steering recovers
    steer, real = float(series[1][1].mean()), float(series[2][1].mean())
    subtitle = (f"On a prompt with no worked example, steering recovers "
                f"{steer/real:.0%} of what six real demonstrations achieve")
    ax.set_xticks(x, [s[0] for s in series], fontsize=11.5, color=INK)
    ax.set_ylabel("task accuracy  (exact match, T=1, 150 prompts/task)",
                  fontsize=11.5, color=INK2)
    ax.set_ylim(0, 0.72)
    ax.set_yticks(np.arange(0, 0.71, 0.1))
    ax.tick_params(axis="y", colors=INK2, labelsize=10)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Injecting a task's mean label-token activation makes a\n"
                 "content-free prompt behave like a demonstrated one",
                 fontsize=16, fontweight="bold", color=INK, pad=30, loc="left")
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=12, color=INK2,
            ha="left", va="bottom")
    fig.text(0.005, 0.005,
             "GPT-J-6B, 69 tasks. Prompt: six demos with real inputs and '_' as every "
             "label, then the query. Intervention: add 4x the task's mean L6 activation at "
             "the label token to all six '_' slots. Bars are means over tasks; whiskers 95% CI.",
             fontsize=8.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "headline_bars.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"headline: unsteered {series[0][1].mean():.4f} | steered {steer:.4f} | "
          f"real6 {real:.4f} | ratio {steer/real:.3f}")


def method_diagram():
    """Schematic of the intervention on a 1-shot dummy prompt."""
    fig, ax = plt.subplots(figsize=(10.6, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    toks = ["Q:", "climber", "\\n", "A:", "_", "\\n\\n", "Q:", "compiler", "\\n", "A:"]
    slot = 4          # the '_' token
    x0, w, gap, ytok = 0.45, 0.86, 0.1, 0.55
    centers = []
    for i, t in enumerate(toks):
        x = x0 + i * (w + gap)
        centers.append(x + w / 2)
        hot = (i == slot)
        ax.add_patch(FancyBboxPatch((x, ytok), w, 0.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.6 if hot else 1.0,
                                    edgecolor=BLUE if hot else GRID,
                                    facecolor="#eaf2fd" if hot else "#ffffff", zorder=3))
        ax.text(x + w / 2, ytok + 0.25, t, ha="center", va="center",
                fontsize=11.5 if hot else 10.5, color=BLUE if hot else INK2,
                fontweight="bold" if hot else "normal", zorder=4)
    ax.text(x0, ytok - 0.26, "prompt tokens  (1-shot, dummy '_' label)",
            fontsize=10, color=INK2, ha="left", va="top")
    # make the task explicit — the prompt itself never reveals it (placed under the title)
    ax.text(0.35, 4.12, "task:  agent noun  →  verb", fontsize=13,
            color=INK, fontweight="bold", ha="left", va="center")
    ax.text(0.35, 3.80, "a real demo would read  'climber → climb'  —  here the label "
            "slot is a bare '_'", fontsize=10.5, color=INK2, ha="left", va="center")

    # layer stack over the '_' column
    lx, lw_ = centers[slot] - 0.55, 1.1
    bands = [("L0-L5", 1.35, "#f2f1ee"), ("L6", 2.05, "#eaf2fd"), ("L7-L27", 2.75, "#f2f1ee")]
    for name, y, fc in bands:
        hot = name == "L6"
        ax.add_patch(FancyBboxPatch((lx, y), lw_, 0.58,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8 if hot else 1.0,
                                    edgecolor=BLUE if hot else GRID, facecolor=fc, zorder=3))
        ax.text(lx + lw_ / 2, y + 0.29, name, ha="center", va="center",
                fontsize=11 if hot else 10, color=BLUE if hot else INK2,
                fontweight="bold" if hot else "normal", zorder=4)
    ax.add_patch(FancyArrowPatch((centers[slot], ytok + 0.52), (centers[slot], 1.33),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color=INK2, zorder=2))
    ax.add_patch(FancyArrowPatch((centers[slot], 1.95), (centers[slot], 2.03),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color=INK2, zorder=2))
    ax.add_patch(FancyArrowPatch((centers[slot], 2.65), (centers[slot], 2.73),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color=INK2, zorder=2))

    # the injected vector, entering L6 from the left
    bx, by = 0.35, 2.02
    ax.add_patch(FancyBboxPatch((bx, by), 3.05, 0.62,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                linewidth=1.6, edgecolor=BLUE, facecolor="#ffffff", zorder=3))
    ax.text(bx + 1.52, by + 0.31, r"$+\ \alpha\; \bar{h}_{\rm task}$", ha="center",
            va="center", fontsize=15, color=BLUE, fontweight="bold", zorder=4)
    ax.add_patch(FancyArrowPatch((bx + 3.08, by + 0.31), (lx - 0.04, by + 0.31),
                                 arrowstyle="-|>", mutation_scale=15, lw=2.0,
                                 color=BLUE, zorder=4))
    ax.text(bx, by - 0.1, "prompt-agnostic task mean \"read feature\"",
            fontsize=10.5, color=INK2, ha="left", va="top")

    # the edited stream propagates forward and changes what the FINAL token predicts
    px, py, pw = centers[-1] - 1.62, 1.98, 1.95
    ax.add_patch(FancyArrowPatch((lx + lw_, 3.04), (px - 0.03, py + 0.28),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 linestyle=(0, (5, 3)), color=INK2, zorder=2,
                                 connectionstyle="arc3,rad=0.22"))
    ax.text(centers[slot] + 1.5, 2.92, "attended by later tokens", fontsize=9.5,
            color=INK2, ha="left", va="center", style="italic")
    # prediction, read out at the query's final cue token
    ax.add_patch(FancyBboxPatch((px, py), pw, 0.56,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                linewidth=1.6, edgecolor=BLUE, facecolor="#eaf2fd",
                                zorder=3))
    ax.text(px + pw / 2, py + 0.28, "→  'compile'", ha="center", va="center", fontsize=13.5,
            color=BLUE, fontweight="bold", zorder=4)
    ax.text(px + pw / 2, py - 0.08, "prediction at the query's 'A:' token\n"
            "— the verb for 'compiler'", fontsize=9.5, color=INK2,
            ha="center", va="top")
    ax.add_patch(FancyArrowPatch((px + pw / 2, py - 0.62), (centers[-1], ytok + 0.54),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.0,
                                 color=GRID, zorder=1))

    ax.text(0.35, 4.6, "Read feature intervention", fontsize=18, fontweight="bold",
            color=INK, ha="left", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "method_diagram.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    headline_bars()
    method_diagram()
