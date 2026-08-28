"""
Line-graph summary of the 10-shot intervene-token STRIP study (reads saved grids; CPU-only).

Collapses each (combo, intervene-token, α) 29×29 layer grid to its peak Δcos (same nanmax
reduction as scalar_overview.png in plot_tenshot_strip_heatmap.py) and plots it as ONE line
chart: x = the 30 intervene tokens in sequence order (d1_in, d1_pre, d1_lab, …, d10_lab),
y = peak Δcos, one line per (task-direction, α) — 12 lines. Hue encodes the task-direction
combo, line style encodes α, so the four combos stay comparable across α.

--labels_only restricts x to the 10 demo label tokens (labels dominate; NOTE the old Δcos-era
"in/pre ≈ 0" claim does not hold under dircos — in/pre peaks reach ~0.2-0.4, see DECISIONS
2026-07-14 metric change) → figures/scalar_lines_labels_only.png.

Output: figures/scalar_lines.png / figures/scalar_lines_labels_only.png.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.paths import LABEL_GEOMETRY_DIR

N_SHOTS = 10
IKEYS = [f"d{i}_{s}" for i in range(1, N_SHOTS + 1) for s in ("in", "pre", "lab")]  # 30, sequence order

COMBOS = [
    ("antonym_synonym", "antonym_to_synonym", "ant→syn"),
    ("antonym_synonym", "synonym_to_antonym", "syn→ant"),
    ("next_number_digits_prev_number_digits", "prev_number_digits_to_next_number_digits", "prev→next"),
    ("next_number_digits_prev_number_digits", "next_number_digits_to_prev_number_digits", "next→prev"),
]

# categorical slots 1-4 (validated fixed order); α carried by line style, not hue
COMBO_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
ALPHA_STYLES = {2.0: "-", 4.0: "--", 8.0: ":"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "tenshot_strip_intervention_cos_heatmap"))
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--labels_only", action="store_true",
                   help="plot only the 10 demo label tokens (in/pre are ~0 everywhere)")
    p.add_argument("--top_k", type=int, default=1,
                   help="reduce each grid to the mean of its top-k cells (1 = plain max)")
    return p.parse_args()


def load_grid(root, task_pair, dir_name, tkey, a):
    p = Path(root) / task_pair / f"{dir_name}__{tkey}_alpha{a:g}_grid.npy"
    return np.load(p) if p.exists() else None


def reduce_grid(g, top_k):
    vals = g[np.isfinite(g)]
    if vals.size == 0:
        return np.nan
    return float(np.sort(vals.ravel())[-top_k:].mean())


def main():
    args = parse_args()
    ikeys = [f"d{i}_lab" for i in range(1, N_SHOTS + 1)] if args.labels_only else IKEYS
    x = np.arange(len(ikeys))

    fig, ax = plt.subplots(figsize=(7.5, 5.2) if args.labels_only else (13.5, 5.2))
    all_y = []
    for (task_pair, dir_name, clab), color in zip(COMBOS, COMBO_COLORS):
        for a in args.alphas:
            y = [g if g is None else reduce_grid(g, args.top_k)
                 for g in (load_grid(args.root, task_pair, dir_name, tk, a) for tk in ikeys)]
            y = np.array([np.nan if v is None else v for v in y], dtype=float)
            all_y.append(y)
            ax.plot(x, y, ALPHA_STYLES.get(a, "-"), color=color, linewidth=2,
                    marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)

    # data-driven y-range (don't force 0 into view: dircos label values sit ~0.4-0.8 and a
    # 0-anchored axis wastes the bottom half); ~15% headroom at the top for the legends.
    yv = np.concatenate(all_y)
    lo, hi = np.nanmin(yv), np.nanmax(yv)
    pad = 0.05 * max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad, hi + 3.5 * pad)
    if lo - pad <= 0.0 <= hi + 3.5 * pad:
        ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    stat = "peak" if args.top_k == 1 else f"top-{args.top_k} mean"
    ax.set_xticks(x)
    if args.labels_only:
        ax.set_xticklabels([str(d) for d in range(1, N_SHOTS + 1)])
        ax.set_xlabel("demo index (label token of demo d)")
        ax.set_title(f"10-shot strip study: {stat} dircos per demo label token")
    else:
        # demo-group separators: a light line after each demo's 3 tokens
        for d in range(1, N_SHOTS):
            ax.axvline(3 * d - 0.5, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_xticklabels([k.split("_")[1] for k in ikeys], fontsize=7, color="#555555")
        for d in range(N_SHOTS):  # demo number, centered under its 3 tokens
            ax.text(3 * d + 1, -0.085, f"demo {d + 1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#333333")
        ax.set_xlabel("intervene token (position in the 10-shot prompt)", labelpad=22)
        ax.set_title(f"10-shot strip study: {stat} dircos per intervene token")
    ax.set_xlim(-0.5, len(ikeys) - 0.5)
    red_lab = ("peak dircos at qfinal (max over intervene × read layers)" if args.top_k == 1
               else f"dircos at qfinal (mean of top-{args.top_k} intervene × read cells)")
    ax.set_ylabel(red_lab)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)

    combo_handles = [Line2D([], [], color=c, linewidth=2, label=lab)
                     for (_, _, lab), c in zip(COMBOS, COMBO_COLORS)]
    alpha_handles = [Line2D([], [], color="#666666", linewidth=2, linestyle=ALPHA_STYLES[a],
                            label=f"α={a:g}") for a in args.alphas]
    leg1 = ax.legend(handles=combo_handles, title="direction", loc="upper left", fontsize=8,
                     title_fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=alpha_handles, title="α", loc="upper left",
              bbox_to_anchor=(0.27, 1.0) if args.labels_only else (0.13, 1.0),
              fontsize=8, title_fontsize=8, framealpha=0.9)

    name = "scalar_lines_labels_only" if args.labels_only else "scalar_lines"
    if args.top_k > 1:
        name += f"_top{args.top_k}"
    out = Path(args.root) / "figures" / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
