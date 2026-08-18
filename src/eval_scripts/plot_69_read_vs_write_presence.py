#!/usr/bin/env python
"""Regenerate the FV_location read-vs-write presence poster line figures from their CSVs.

The original generating script lived in a since-deleted worktree; the plotted values
survive in results/69_task_run/FV_location/poster_visuals/*.csv (mean cos over the 69
tasks between the residual stream and the task feature, by layer x token role):
    write_* = the 37-head task FV; read_* = the read feature
    (read_vs_write_presence*   -> cosine_perhead unit read direction,
     *_label_mean*             -> unit task-mean L6 label-token residual).

Three layouts per read variant, same filenames as the originals:
    <stem>.png          one shared axis
    <stem>_dual.png     read on the left axis, write on the right (scales chosen so the
                        two bold peaks sit at the same height)
    <stem>_stacked.png  READ panel over WRITE panel
Bold lines = the peaking (token, feature) pair: write cue (solid) and read label
(dashed). Peak dot/arrow callouts from the originals are dropped (user request);
--with_peaks restores them.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

OUT = TASK69_RUN_DIR / "FV_location" / "poster_visuals"

COLOR = {"cue": "#2f7fe0", "label": "#e8623d", "input": "#2fae82"}
TOKEN_LABEL = {"cue": 'cue “A:”', "label": "label / target", "input": "input"}
BOLD = {"write": "cue", "read": "label"}  # the emphasised token per feature
LS = {"write": "-", "read": (0, (4, 1.6))}


def load(stem):
    with open(OUT / f"{stem}.csv") as f:
        rows = list(csv.DictReader(f))
    layers = [int(r["layer"]) for r in rows]
    data = {feat: {tok: np.array([float(r[f"{feat}_{tok}"]) for r in rows])
                   for tok in ("cue", "label", "input")}
            for feat in ("write", "read")}
    return layers, data


def draw_feature(ax, layers, data, feat):
    for tok in ("input", "cue", "label") if feat == "read" else ("input", "label", "cue"):
        bold = tok == BOLD[feat]
        ax.plot(layers, data[feat][tok], color=COLOR[tok], ls=LS[feat],
                lw=5.0 if bold else 2.2, alpha=1.0 if bold else 0.45,
                solid_capstyle="round", zorder=5 if bold else 3)


def mark_peak(ax, layers, series, color, text, dx, dy):
    pk = int(np.argmax(series))
    ax.scatter([layers[pk]], [series[pk]], s=140, color=color, zorder=7)
    ax.annotate(text, xy=(layers[pk], series[pk]),
                xytext=(layers[pk] + dx, series[pk] + dy),
                fontsize=18, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="-", lw=1.4, color=color, shrinkB=8,
                                connectionstyle="arc3,rad=0.25"))


def peak_texts(layers, data):
    rl = int(np.argmax(data["read"]["label"]))
    wl = int(np.argmax(data["write"]["cue"]))
    return (f"READ peaks · L{layers[rl]}  (label tokens)",
            f"WRITE peaks · L{layers[wl]}  (cue tokens)")


def style(ax):
    ax.grid(axis="y", color="0.90", lw=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=14)


def legends(ax):
    h1 = [Line2D([], [], color=COLOR[t], lw=4.0, label=TOKEN_LABEL[t])
          for t in ("cue", "label", "input")]
    leg1 = ax.legend(handles=h1, title="token type", loc="upper left",
                     bbox_to_anchor=(0.015, 0.99), fontsize=15, title_fontsize=15,
                     frameon=False, alignment="left")
    ax.add_artist(leg1)
    h2 = [Line2D([], [], color="0.45", lw=3.4, ls=LS["write"], label="write"),
          Line2D([], [], color="0.45", lw=3.4, ls=LS["read"], label="read")]
    ax.legend(handles=h2, title="feature", loc="upper left",
              bbox_to_anchor=(0.24, 0.99), fontsize=15, title_fontsize=15,
              frameon=False, alignment="left")


def title(fig):
    fig.suptitle("Read and Write Feature Locations", x=0.065, y=0.965,
                 ha="left", fontsize=26, fontweight="bold")


def plot_single(stem, with_peaks):
    layers, data = load(stem)
    fig, ax = plt.subplots(figsize=(13.8, 8.16), dpi=250)
    draw_feature(ax, layers, data, "write")
    draw_feature(ax, layers, data, "read")
    top = max(data[f][t].max() for f in data for t in data[f])
    ax.set_ylim(0, top * 1.55)
    ax.set_xticks(range(0, layers[-1] + 1, 2))
    ax.set_xlabel("layer", fontsize=18, labelpad=8)
    ax.set_ylabel("mean cos with the task feature", fontsize=18, labelpad=10)
    style(ax); legends(ax)
    if with_peaks:
        rt, wt = peak_texts(layers, data)
        mark_peak(ax, layers, data["read"]["label"], COLOR["label"], rt, -4.0, top * 0.28)
        mark_peak(ax, layers, data["write"]["cue"], COLOR["cue"], wt, 2.0, top * 0.20)
    title(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_dual(stem, with_peaks):
    layers, data = load(stem)
    fig, axL = plt.subplots(figsize=(13.8, 8.16), dpi=250)
    axR = axL.twinx()
    draw_feature(axL, layers, data, "read")
    draw_feature(axR, layers, data, "write")
    # scales chosen so the two bold peaks sit at the same visual height
    axL.set_ylim(0, data["read"]["label"].max() / 0.68)
    axR.set_ylim(0, data["write"]["cue"].max() / 0.68)
    axL.set_xticks(range(0, layers[-1] + 1, 2))
    axL.set_xlabel("layer", fontsize=18, labelpad=8)
    axL.set_ylabel("READ feature   mean cos   (dashed lines)", fontsize=17, labelpad=10)
    axR.set_ylabel("WRITE feature   mean cos   (solid lines)", fontsize=17, labelpad=14)
    style(axL)
    axR.grid(False)
    for s in ("top",):
        axR.spines[s].set_visible(False)
    axR.tick_params(labelsize=14)
    legends(axL)
    if with_peaks:
        rt, wt = peak_texts(layers, data)
        mark_peak(axL, layers, data["read"]["label"], COLOR["label"], rt, -4.5,
                  axL.get_ylim()[1] * 0.10)
        mark_peak(axR, layers, data["write"]["cue"], COLOR["cue"], wt, 2.0,
                  axR.get_ylim()[1] * 0.10)
    title(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"{stem}_dual.png", bbox_inches="tight")
    plt.close(fig)


def plot_stacked(stem, with_peaks):
    layers, data = load(stem)
    fig, (axR, axW) = plt.subplots(2, 1, figsize=(13.8, 10.08), dpi=250, sharex=True)
    draw_feature(axR, layers, data, "read")
    draw_feature(axW, layers, data, "write")
    for ax, feat in ((axR, "read"), (axW, "write")):
        ax.set_ylim(0, max(data[feat][t].max() for t in data[feat]) * 1.28)
        ax.set_ylabel(f"{feat.upper()}\nmean cos", fontsize=17, labelpad=10)
        style(ax)
    axW.set_xticks(range(0, layers[-1] + 1, 2))
    axW.set_xlabel("layer", fontsize=18, labelpad=8)
    legends(axR)
    if with_peaks:
        rt, wt = peak_texts(layers, data)
        mark_peak(axR, layers, data["read"]["label"], COLOR["label"], rt, 5.0,
                  axR.get_ylim()[1] * 0.05)
        mark_peak(axW, layers, data["write"]["cue"], COLOR["cue"], wt, 2.0,
                  axW.get_ylim()[1] * 0.06)
    title(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"{stem}_stacked.png", bbox_inches="tight")
    plt.close(fig)


def main(with_peaks=False):
    for stem in ("read_vs_write_presence", "read_vs_write_presence_label_mean"):
        plot_single(stem, with_peaks)
        plot_dual(stem, with_peaks)
        plot_stacked(stem, with_peaks)
        print(f"wrote {stem}{{,.png,_dual.png,_stacked.png}} (peaks={'on' if with_peaks else 'off'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_peaks", action="store_true",
                    help="restore the peak dot/arrow callouts of the original figures")
    a = ap.parse_args()
    main(with_peaks=a.with_peaks)
