#!/usr/bin/env python
"""Render the read-direction write-up figures as standalone PNGs (Slack-ready).

Reads the committed summary CSVs under results/69_task_run/Read_direction_geometry/ and
writes four self-contained PNGs to <that dir>/writeup_assets/:
  1_definitions.png     the four read-direction brackets and their lever settings
  2_dimensionality.png  centered-PCA dimensionality per bracket x normalization
  3_overlap.png         pairwise pooled-90% subspace overlap matrix (sequential ramp)
  4_containment.png     unit vs natural-magnitude subspace containment

Style: white surface, cool-green neutrals, one teal accent, warm accent reserved for the
cosine_M outlier; tabular figures; no chartjunk.
"""
import csv
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

RD = TASK69_RUN_DIR / "Read_direction_geometry"
OUT = RD / "writeup_assets"

INK = "#111917"
MUTED = "#5A6A66"
RULE = "#DCE2E0"
ACCENT = "#0F6E63"
SIGNAL = "#A65423"
BAND = "#EDF3F1"
SURFACE = "#FFFFFF"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")
TEAL = LinearSegmentedColormap.from_list("teal", ["#F2F7F6", "#BFDDD8", "#6FB4AA", "#2C8479", "#0C4F47"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "savefig.facecolor": SURFACE,
})


def table_fig(col_labels, rows, col_widths, title, subtitle, note=None,
              row_accents=None, align=None, width=12.0):
    """Render a text table as a figure: banded header, hairline row rules, no boxes.

    Geometry is in inches, not axes fractions, so nothing silently overflows the canvas:
    the title/subtitle/note are wrapped to the table's own width, and column widths are
    checked against the widest string each column has to hold.
    """
    n = len(rows)
    align = align or ["left"] * len(col_labels)
    # widen any column whose header or cells would not fit (≈ chars -> inches at 10pt)
    col_widths = list(col_widths)
    for j in range(len(col_labels)):
        longest = max([len(str(col_labels[j])) * 0.075]
                      + [len(str(r[j])) * 0.072 for r in rows]) + 0.18
        span = width * col_widths[j] / sum(col_widths)
        if span < longest:
            col_widths[j] *= longest / span
    scale = width / sum(col_widths)
    col_in = [w * scale for w in col_widths]

    sub_lines = textwrap.wrap(subtitle, int(width * 12.6)) if subtitle else []
    note_lines = textwrap.wrap(note, int(width * 13.8)) if note else []
    row_h, head_h = 0.40, 0.46
    top_block = 0.34 + 0.235 * len(sub_lines) + 0.22
    bot_block = (0.16 + 0.20 * len(note_lines)) if note_lines else 0.10
    fig_h = top_block + head_h + row_h * n + bot_block
    fig, ax = plt.subplots(figsize=(width, fig_h), dpi=220)
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, fig_h)
    xs = np.concatenate([[0.0], np.cumsum(col_in)])

    y = fig_h - 0.30
    ax.text(0, y, title, fontsize=15, fontweight="bold", va="top", ha="left", color=INK)
    y -= 0.40
    for line in sub_lines:
        ax.text(0, y, line, fontsize=10, va="top", ha="left", color=MUTED)
        y -= 0.235
    y -= 0.16

    ax.add_patch(plt.Rectangle((0, y - head_h), width, head_h, facecolor=BAND, edgecolor="none"))
    for j, lab in enumerate(col_labels):
        x = xs[j] + 0.09 if align[j] == "left" else xs[j + 1] - 0.09
        ax.text(x, y - head_h / 2, lab, fontsize=9.5, fontweight="bold",
                va="center", ha=align[j], color=ACCENT)
    y -= head_h
    for i, row in enumerate(rows):
        col = SIGNAL if (row_accents and row_accents[i]) else INK
        for j, cell in enumerate(row):
            x = xs[j] + 0.09 if align[j] == "left" else xs[j + 1] - 0.09
            mono = j > 0 and any(ch.isdigit() for ch in str(cell))
            ax.text(x, y - row_h / 2, str(cell), fontsize=10.5 if j == 0 else 10,
                    va="center", ha=align[j], color=col,
                    fontweight="bold" if j == 0 else "normal",
                    family="DejaVu Sans Mono" if mono else "DejaVu Sans")
        y -= row_h
        ax.plot([0, width], [y, y], color=RULE, lw=0.7, zorder=0)
    if note_lines:
        y -= 0.16
        for line in note_lines:
            ax.text(0, y, line, fontsize=9, va="top", ha="left", color=MUTED)
            y -= 0.20
    fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005)
    return fig


def fig1_definitions():
    rows = [
        ["cosine_M", "cosine similarity", "r ∝ M⁺ᵧ v", "summed circuit M", "338"],
        ["dot_M", "dot product", "r ∝ Mᵀ v", "summed circuit M", "41"],
        ["cosine_perhead", "cosine similarity", "r ∝ Σₕ (WₒWᵥ)⁺ᵧ hⱼ", "per head, then sum", "232"],
        ["dot_perhead", "dot product", "r ∝ Σₕ (WₒWᵥ)ᵀ hⱼ", "per head, then sum", "139"],
    ]
    fig = table_fig(
        ["bracket", "objective (Lever 1)", "closed form", "aggregation (Lever 3)", "per-prompt 90%"],
        rows, [1.45, 1.5, 1.8, 1.55, 1.0],
        "Four read-direction definitions",
        "GPT-J, 37-head canonical set, 69 tasks × 150 prompts. Lever 2 fixed at the τ-truncated "
        "pseudo-inverse (cumulative σ² ≥ 0.90); Lever 4 stored both ways.",
        note="Per-head brackets sum the unnormalised per-head solutions and normalise once at the end "
             "(sub-choice 3′).  Per-prompt 90% = principal components needed for 90% of the variance "
             "of all 8,250 per-prompt read directions (55 train tasks × 150 prompts), CENTRED PCA — "
             "as in every table and matrix here.",
        row_accents=[True, False, False, False], width=13.6,
        align=["left", "left", "left", "left", "right"])
    fig.savefig(OUT / "1_definitions.png", bbox_inches="tight")
    plt.close(fig)


def fig2_dimensionality():
    rows = []
    accents = []
    for b in BRACKETS:
        for norm in ("unit", "natural"):
            d = {}
            for r in csv.DictReader(open(RD / f"{b}__{norm}" / "summary.csv")):
                a = r["analysis"]
                key = ("task" if a.endswith("_r_task") else
                       "pooled" if a.endswith("_pooled_perprompt") else
                       "within" if a.endswith("_within_task_median") else None)
                # cosine_M__unit carries the retired literal-inverse rows too; keep rank90
                if key and not a.startswith("literal"):
                    d[key] = r
            rows.append([f"{b}", norm, d["pooled"]["n90_pcs"],
                         f'{float(d["pooled"]["stable_rank_raw"]):.2f}',
                         f'{float(d["pooled"]["stable_rank_centered"]):.2f}',
                         d["within"]["n90_pcs"]])
            accents.append(b == "cosine_M")
    fig = table_fig(
        ["bracket", "Lever 4", "90% PCs (all prompts)", "stable rank (raw)",
         "stable rank (centred)", "90% PCs (one task)"],
        rows, [1.3, 0.8, 1.45, 1.2, 1.5, 1.35],
        "Dimensionality of each definition",
        "Centred PCA, float64 SVD. Every direction here is per-prompt: 8,250 of them, "
        "55 train tasks × 150 prompts. The last column is one task's 150 prompts, median over tasks.",
        note="Lever 4 barely moves any column — the objective (Lever 1) dominates, aggregation "
             "(Lever 3) is second.",
        row_accents=accents, width=12.6,
        align=["left", "left", "right", "right", "right", "right"])
    fig.savefig(OUT / "2_dimensionality.png", bbox_inches="tight")
    plt.close(fig)


def fig3_overlap():
    ks, sym, cont = {}, np.eye(4), np.eye(4)
    idx = {b: i for i, b in enumerate(BRACKETS)}
    for r in csv.DictReader(open(RD / "cross_bracket_overlap" / "overlap_summary.csv")):
        a, b = idx[r["bracket_a"]], idx[r["bracket_b"]]
        ks[r["bracket_a"]], ks[r["bracket_b"]] = int(r["k_a"]), int(r["k_b"])
        sym[a, b] = sym[b, a] = float(r["symmetric_overlap"])
        cont[a, b] = float(r["weighted_containment_a_in_b"])
        cont[b, a] = float(r["weighted_containment_b_in_a"])

    fig, ax = plt.subplots(figsize=(8.6, 7.8), dpi=220)
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.16, right=0.88)
    shown = sym.copy()
    np.fill_diagonal(shown, np.nan)          # a subspace's overlap with itself is not a finding
    cmap = TEAL.copy()
    cmap.set_bad("#F6F8F7")
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    labels = [f"{b}\n(k={ks[b]})" for b in BRACKETS]
    ax.set_xticks(range(4), labels, fontsize=9.5, color=INK)
    ax.set_yticks(range(4), labels, fontsize=9.5, color=INK)
    for i in range(4):
        for j in range(4):
            if i == j:
                txt, col = "—", MUTED
            else:
                txt = f"{sym[i, j]:.2f}\n{cont[i, j]:.2f} in col"
                col = "#FFFFFF" if sym[i, j] > 0.62 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color=col,
                    family="DejaVu Sans Mono", linespacing=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("symmetric overlap   ‖Vₐᵀ V_b‖²_F / min(k)", fontsize=9.5, color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=0, colors=MUTED)
    fig.text(0.03, 0.955, "Do the definitions find the same subspace?",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.03, 0.905, "90% principal subspaces (centred PCA) of the 8,250 unit-norm per-prompt "
             "read directions, 55 train tasks.", fontsize=9.5, color=MUTED, ha="left", va="top")
    fig.text(0.03, 0.075,
             "Upper number: symmetric overlap, 1 = one subspace nested in the other.\n"
             "Lower number: variance-weighted containment of the row's subspace inside the column's.\n"
             "Random-subspace chance ≤ 0.083, so every pair is far above chance and far below nesting.",
             fontsize=9.5, color=MUTED, ha="left", va="top", linespacing=1.6)
    fig.savefig(OUT / "3_overlap.png", bbox_inches="tight")
    plt.close(fig)


def fig4_containment():
    rows, accents = [], []
    for b in BRACKETS:
        r = next(csv.DictReader(open(RD / "unit_vs_natural_containment" / f"{b}_summary.csv")))
        rows.append([b, r["k_unit"], r["k_nat"],
                     f'{float(r["weighted_containment_nat_in_unit"]):.3f}',
                     f'{float(r["weighted_containment_unit_in_nat"]):.3f}',
                     f'{float(r["min_pc_containment"]):.3f}'])
        accents.append(b == "cosine_M")
    fig = table_fig(
        ["bracket", "k (unit)", "k (natural)", "natural in unit", "unit in natural",
         "worst single PC"],
        rows, [1.3, 0.8, 0.9, 1.1, 1.05, 1.1],
        "Does the magnitude choice change the subspace?",
        "Variance-weighted containment of each Lever-4 variant's 90% per-prompt subspace inside "
        "the other's.",
        note="No. Both directions sit above 0.976, so the two variants essentially share one "
             "subspace — unit norm simply spreads the same variance over a few more components.",
        row_accents=accents, width=10.5,
        align=["left", "right", "right", "right", "right", "right"])
    fig.savefig(OUT / "4_containment.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_definitions()
    fig2_dimensionality()
    fig3_overlap()
    fig4_containment()
    for f in sorted(OUT.glob("*.png")):
        print(f"{f.name}  {f.stat().st_size // 1024} KB")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
