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




def _token_row(ax, toks, y, slot, steered, centers_out=None):
    """One rollout: the prompt tokens plus the answer the model gives.
    steered=True colours every token from the intervened slot onward."""
    x0, w, gap, h = 0.45, 0.86, 0.1, 0.5
    centers = []
    for i, t in enumerate(toks):
        x = x0 + i * (w + gap)
        centers.append(x + w / 2)
        touched = steered and i == slot   # only the edited slot, not every token
        ec = BLUE if touched else GRID
        fc = "#eaf2fd" if touched else "#ffffff"
        tc = BLUE if touched else INK2
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.5 if touched else 1.0,
                                    edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                fontsize=11 if touched else 10.5, color=tc,
                fontweight="bold" if (touched and i == slot) else "normal", zorder=4)
    # the answer
    ax_x = x0 + len(toks) * (w + gap) + 0.12
    col = BLUE if steered else "#9c9b96"
    ax.add_patch(FancyBboxPatch((ax_x, y - 0.03), 1.9, h + 0.06,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                linewidth=1.8, edgecolor=col,
                                facecolor="#eaf2fd" if steered else "#f2f1ee", zorder=3))
    ax.text(ax_x + 0.95, y + h / 2, "'compile'" if steered else "wrong answer",
            ha="center", va="center", fontsize=13 if steered else 11.5,
            color=col, fontweight="bold", zorder=4)
    if centers_out is not None:
        centers_out.extend(centers)
    return centers


def method_diagram():
    """Two rollouts of the same prompt — with and without the read feature added at L6."""
    fig, ax = plt.subplots(figsize=(12.6, 5.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    toks = ["Q:", "climber", "\\n", "A:", "_", "\\n\\n", "Q:", "compiler", "\\n", "A:"]
    slot = 4
    y_top, y_bot = 3.62, 0.40

    ax.text(0.35, 4.85, "Read feature intervention", fontsize=18, fontweight="bold",
            color=INK, ha="left", va="center")
    ax.text(0.35, 4.52, "task:  agent noun  →  verb          "
            "a real demo would read  'climber → climb'  —  here the label slot is a bare '_'",
            fontsize=10.5, color=INK2, ha="left", va="center")

    centers = _token_row(ax, toks, y_top, slot, steered=True)
    _token_row(ax, toks, y_bot, slot, steered=False)
    ax.text(0.45, y_top + 0.62, "with the read feature added", fontsize=11.5,
            color=BLUE, fontweight="bold", ha="left", va="center")
    ax.text(0.45, y_bot - 0.26, "without it  —  the same prompt, unsteered",
            fontsize=11.5, color=INK2, ha="left", va="top")

    # the same token slot in both rollouts
    ax.plot([centers[slot], centers[slot]], [y_bot + 0.52, y_top - 0.04],
            color=GRID, lw=1.0, ls=(0, (2, 3)), zorder=1)

    # layer stack at that slot: L6 carries the added feature, everything above it inherits it
    lx, lw_ = centers[slot] - 0.62, 1.24
    bands = [("L0-L5", 1.10, "#f2f1ee", GRID, INK2, 1.0),
             ("L6", 1.86, "#cfe2fa", BLUE, BLUE, 2.2),
             ("L7-L27", 2.62, "#eaf2fd", BLUE, BLUE, 1.4)]
    for name, y, fc, ec, tc, lwid in bands:
        ax.add_patch(FancyBboxPatch((lx, y), lw_, 0.66,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=lwid, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(lx + lw_ / 2, y + 0.33, name, ha="center", va="center",
                fontsize=11.5 if name != "L0-L5" else 10.5, color=tc,
                fontweight="bold" if name == "L6" else "normal", zorder=4)
    ax.text(lx + lw_ + 0.22, 2.95, "every later layer at this slot\nnow carries it",
            fontsize=10, color=BLUE, ha="left", va="center")
    ax.text(lx + lw_ + 0.22, 1.43, "unchanged", fontsize=10, color=INK2,
            ha="left", va="center")

    # the injected vector
    bx, by = 0.35, 1.84
    ax.add_patch(FancyBboxPatch((bx, by), 3.0, 0.70,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                linewidth=1.8, edgecolor=BLUE, facecolor="#ffffff",
                                zorder=3))
    ax.text(bx + 1.5, by + 0.35, r"$+\ \alpha\; \bar{h}_{\rm task}$", ha="center",
            va="center", fontsize=16, color=BLUE, fontweight="bold", zorder=4)
    ax.add_patch(FancyArrowPatch((bx + 3.03, by + 0.35), (lx - 0.04, by + 0.35),
                                 arrowstyle="-|>", mutation_scale=16, lw=2.2,
                                 color=BLUE, zorder=4))
    ax.text(bx, by - 0.12, "prompt-agnostic task mean \"read feature\"",
            fontsize=10.5, color=INK2, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(OUT / "method_diagram.png", bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")

if __name__ == "__main__":
    headline_bars()
    method_diagram()
