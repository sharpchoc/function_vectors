#!/usr/bin/env python
"""Poster-ready visuals for the 69-task train/held-out generalisation result (CPU).

Condensed, label-light versions of the per-task bars in
results/69_task_run/FV_train_test_generalisation/: aggregate steered-vs-unsteered
accuracy for the 55 train and 14 held-out tasks under the zero-shot and
mixed-task mixed-label 10-shot settings, plus a ranked per-task lift strip that
shows held-out tasks interleaving with train tasks.

Reads train_heldout_summary.csv; writes into the poster_visuals/ subfolder.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import TASK69_RUN_DIR  # noqa: E402

SRC_DIR = TASK69_RUN_DIR / "FV_train_test_generalisation"
OUT_DIR = SRC_DIR / "poster_visuals"

# Palette: documented categorical slots 1 (blue) and 2 (orange) on a light surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY = "#a3a19b"          # "no steering" — absence, not an identity
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

SETTINGS = [("zs", "Zero-shot"), ("mix", "Mixed-task, mixed-label 10-shot")]


def load():
    rows = list(csv.DictReader(open(SRC_DIR / "train_heldout_summary.csv")))
    return ([r for r in rows if r["group"] == "train"],
            [r for r in rows if r["group"] == "heldout"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d3cd")
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color="#e6e4de", lw=1.0)
    ax.set_axisbelow(True)


def bar_panel(ax, train, held, key, title, show_ylabel):
    """Three bars: train steered, held-out steered, held-out unsteered."""
    vals = [np.mean([float(r[f"{key}_best"]) for r in train]),
            np.mean([float(r[f"{key}_best"]) for r in held]),
            np.mean([float(r[f"{key}_base"]) for r in held])]
    colors = [BLUE, ORANGE, GREY]
    x = np.array([0.0, 1.0, 2.15])
    ax.bar(x, vals, width=0.68, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom",
                fontsize=19, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Train\n55 tasks", "Held-out\n14 tasks", "Held-out\nno steering"],
                       fontsize=15, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.62, 2.77)
    if show_ylabel:
        ax.set_ylabel("Task accuracy", fontsize=15, color=INK)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=18, color=INK, pad=14, fontweight="bold")
    style(ax)
    return vals


def fig_headline():
    train, held = load()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=200, facecolor=SURFACE)
    for i, (ax, (key, label)) in enumerate(zip(axes, SETTINGS)):
        bar_panel(ax, train, held, key, label, show_ylabel=(i == 0))
    fig.suptitle("Steering transfers to held-out tasks",
                 fontsize=23, color=INK, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT_DIR / "headline_bars.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def lift_panel(ax, train, held, key, title, show_ylabel):
    rows = sorted(train + held, key=lambda r: float(r[f"{key}_best"]))
    for i, r in enumerate(rows):
        b, s = float(r[f"{key}_base"]), float(r[f"{key}_best"])
        c = ORANGE if r["group"] == "heldout" else BLUE
        ax.plot([i, i], [b, s], color=c, lw=1.6, alpha=0.45, zorder=2,
                solid_capstyle="round")
        ax.plot([i], [b], marker="_", ms=7, mew=2.0, color="#8b8983", zorder=3)
        ax.plot([i], [s], marker="o", ms=6.5, color=c, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(rows)} tasks, ranked by steered accuracy", fontsize=14, color=INK2)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-1.2, len(rows) + 0.2)
    if show_ylabel:
        ax.set_ylabel("Task accuracy", fontsize=15, color=INK)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=18, color=INK, pad=14, fontweight="bold")
    style(ax)


def fig_lift():
    train, held = load()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=200, facecolor=SURFACE)
    for i, (ax, (key, label)) in enumerate(zip(axes, SETTINGS)):
        lift_panel(ax, train, held, key, label, show_ylabel=(i == 0))
    handles = [Line2D([], [], marker="o", ls="none", ms=9, color=BLUE, label="Train task"),
               Line2D([], [], marker="o", ls="none", ms=9, color=ORANGE, label="Held-out task"),
               Line2D([], [], marker="_", ls="none", ms=11, mew=2.2, color=GREY,
                      label="Same task, no steering")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=15, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Every task improves",
                 fontsize=23, color=INK, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0.055, 1, 0.945])
    p = OUT_DIR / "per_task_lift.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_combined():
    """Single-slot poster graphic: headline bars (left) + zero-shot lift strip (right)."""
    train, held = load()
    fig = plt.figure(figsize=(16.0, 6.6), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.13,
                          left=0.055, right=0.985, top=0.815, bottom=0.215)
    for i, (key, label) in enumerate(SETTINGS):
        bar_panel(fig.add_subplot(gs[0, i]), train, held, key, label, show_ylabel=(i == 0))
    ax = fig.add_subplot(gs[0, 2])
    lift_panel(ax, train, held, "zs", "Zero-shot, task by task", show_ylabel=False)
    ax.set_ylim(0, 1.12)  # align gridlines with the bar panels
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    handles = [Patch(facecolor=BLUE, label="Train tasks (55)"),
               Patch(facecolor=ORANGE, label="Held-out tasks (14)"),
               Patch(facecolor=GREY, label="No steering")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=15.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Steering transfers to held-out tasks",
                 fontsize=23, color=INK, fontweight="bold", y=0.965)
    p = OUT_DIR / "poster_summary.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_heads():
    """Which 37 of GPT-J's 448 heads the pooled sparse optimisation selected."""
    import json
    from utils.paths import ARTIFACTS_ROOT
    sel = json.load(open(ARTIFACTS_ROOT / "sandbox" / "ext_steerability" /
                         "prunedfail_seed43" / "pooled_sparse" / "selection.json"))
    heads = [(int(l), int(h), float(c)) for l, h, c in sel["selected_heads"]]
    n_layers, n_heads = 28, 16
    grid = np.zeros((n_layers, n_heads), dtype=bool)
    for l, h, _ in heads:
        grid[l, h] = True
    per_layer = grid.sum(axis=1)

    fig = plt.figure(figsize=(9.8, 8.1), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 16], hspace=0.07,
                          left=0.075, right=0.99, top=0.885, bottom=0.075)
    axb = fig.add_subplot(gs[0, 0])
    axb.set_facecolor(SURFACE)
    axb.bar(np.arange(n_layers) + 0.5, per_layer, width=0.72, color=BLUE, zorder=3)
    for l, n in enumerate(per_layer):
        if n:
            axb.text(l + 0.5, n + 0.22, str(n), ha="center", va="bottom",
                     fontsize=10.5, color=INK2)
    axb.set_xlim(0, n_layers)
    axb.set_ylim(0, max(per_layer) + 1.4)
    axb.set_yticks([])
    axb.tick_params(labelbottom=False, length=0)
    axb.set_ylabel("Heads\nper layer", fontsize=12, color=INK2, rotation=0,
                   ha="right", va="center", labelpad=14)
    for s in axb.spines.values():
        s.set_visible(False)

    ax = fig.add_subplot(gs[1, 0], sharex=axb)
    ax.set_facecolor(SURFACE)
    for l in range(n_layers):
        for h in range(n_heads):
            ax.add_patch(plt.Rectangle((l + 0.06, h + 0.06), 0.88, 0.88,
                                       facecolor=BLUE if grid[l, h] else "#eae8e2",
                                       edgecolor="none"))
    ax.set_ylim(0, n_heads)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks(np.arange(0, n_layers, 2) + 0.5)
    ax.set_xticklabels(range(0, n_layers, 2), fontsize=11, color=INK2)
    ax.set_yticks(np.arange(0, n_heads, 2) + 0.5)
    ax.set_yticklabels(range(0, n_heads, 2), fontsize=11, color=INK2)
    ax.set_xlabel("Layer", fontsize=14.5, color=INK, labelpad=8)
    ax.set_ylabel("Head", fontsize=14.5, color=INK, labelpad=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.suptitle("The selected heads",
                 fontsize=23, color=INK, fontweight="bold", y=0.965)
    p = OUT_DIR / "selected_heads.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)

    with open(OUT_DIR / "selected_heads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "c"])
        for l, h, c in sorted(heads):
            w.writerow([l, h, f"{c:.4f}"])
    print("wrote", OUT_DIR / "selected_heads.csv")


def write_table():
    train, held = load()
    p = OUT_DIR / "poster_numbers.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "group", "n_tasks", "acc_no_steering", "acc_steered",
                    "min_steered", "n_tasks_below_0.4"])
        for key, label in SETTINGS:
            for name, rs in (("train", train), ("heldout", held)):
                s = [float(r[f"{key}_best"]) for r in rs]
                b = [float(r[f"{key}_base"]) for r in rs]
                w.writerow([label, name, len(rs), f"{np.mean(b):.3f}", f"{np.mean(s):.3f}",
                            f"{min(s):.2f}", sum(1 for v in s if v < 0.4)])
    print("wrote", p)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_headline()
    fig_lift()
    fig_combined()
    fig_heads()
    write_table()
