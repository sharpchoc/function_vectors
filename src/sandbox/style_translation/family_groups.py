"""Two categories of convention families (user decision 2026-09-10) and a grouped figure grid.

LEXICAL  — lexically diverse: the convention shows up across many different words, so the model
           has to generalise a rule (British spelling, -ise/-ize, -t pasts, contractions, number and
           ordinal words).
FIXED    — lexically identical: one fixed marker or a pure formatting choice, the same every time
           (whilst/while, &/and, %/percent, dashes, ellipses, quotes, spacing, case, serial comma).

`grouped_grid` lays the two groups out side by side: left block = LEXICAL, right block = FIXED, each
with its own title and background tint, so every per-family figure in results/style_translation
separates the two at a glance.
"""
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LEXICAL = ["us_uk", "ise_ize", "brit_t_past", "contractions", "num_words", "ordinal_words"]
FIXED = ["whilst", "ampersand", "percent_sign", "em_dash", "ellipsis", "curly_quotes", "quote_punct",
         "double_space", "oxford_comma", "sentence_caps", "all_caps"]
GROUPS = [
    ("Lexically diverse conventions — the rule applies across many different words", LEXICAL, "#e6edf7"),
    ("Lexically identical conventions — one fixed marker or formatting choice", FIXED, "#f9f0e3"),
]
GROUP_OF = {f: g for g, (_, fams, _) in enumerate(GROUPS) for f in fams}


def grouped_order(fams):
    """Families in group order (LEXICAL then FIXED), keeping only those present."""
    return [f for _, group, _ in GROUPS for f in group if f in fams]


def grouped_grid(fams, slots_per_family=1, fam_cols=(2, 3), nrows=None, panel_w=3.9, panel_h=3.0,
                 top=0.84, bottom=0.06, sharex=False, sharey=True, gap=0.05):
    """Two side-by-side blocks of panels. Returns (fig, axes) with axes[(fam, slot)].

    fam_cols   families per row in the (left, right) block; each family takes `slots_per_family`
               adjacent columns (e.g. 2 for a nat/alt pair).
    nrows      rows per block (default: enough for the larger block).
    The block titles and tinted backgrounds are drawn in figure coordinates; call fig.suptitle /
    fig.legend above `top` yourself. Unused slots are switched off. `axes_meta` on the figure holds
    {"bottom": [...], "left": [...]} axes per block for axis labels.
    """
    groups = [[f for f in group if f in fams] for _, group, _ in GROUPS]
    if nrows is None:
        nrows = max(math.ceil(len(g) / c) for g, c in zip(groups, fam_cols))
    ncols = [c * slots_per_family for c in fam_cols]
    width = panel_w * sum(ncols) + panel_w * 0.6
    fig = plt.figure(figsize=(width, panel_h * nrows + 2.2))
    # horizontal extents of the two blocks (figure fraction)
    lm, rm = 0.045, 0.995
    usable = rm - lm - gap
    w_left = usable * ncols[0] / sum(ncols); w_right = usable - w_left
    extents = [(lm, lm + w_left), (rm - w_right, rm)]
    axes, first = {}, None
    meta = {"bottom": [], "left": []}
    for gi, ((title, _, tint), group, (x0, x1)) in enumerate(zip(GROUPS, groups, extents)):
        gs = fig.add_gridspec(nrows, ncols[gi], left=x0, right=x1, top=top, bottom=bottom, wspace=0.28, hspace=0.75)
        fig.patches.append(FancyBboxPatch((x0 - 0.012, bottom - 0.05), x1 - x0 + 0.024, top - bottom + 0.085,
                                          boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                                          facecolor=tint, edgecolor="none", zorder=-5))
        fig.text((x0 + x1) / 2, top + 0.018, title, ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")
        n_slots = nrows * ncols[gi]
        for si in range(n_slots):
            r, c = divmod(si, ncols[gi])
            ax = fig.add_subplot(gs[r, c], sharex=first if sharex else None, sharey=first if sharey else None)
            if first is None:
                first = ax
            fi, slot = divmod(si, slots_per_family)
            if fi < len(group):
                axes[(group[fi], slot)] = ax
                if c == 0:
                    meta["left"].append(ax)
                last_row_for_col = (len(group) * slots_per_family - 1 - c) // ncols[gi]
                if r == last_row_for_col:
                    meta["bottom"].append(ax)
            else:
                ax.axis("off")
            if sharey and c != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
    fig.axes_meta = meta
    return fig, axes
