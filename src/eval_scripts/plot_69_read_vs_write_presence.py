#!/usr/bin/env python
"""Poster visual: where the READ and WRITE features become present in the residual stream.

Mean cos by token type against depth, for both feature families:
  WRITE = the task function vector v_A                      (direct_FV_presence)
  READ  = --read_source read_dir   -> r_A = M^+ v_A         (read_dir_presence)
          --read_source label_mean -> m_A(L6), the L6 label-token mean that steers
                                                            (label_mean_L6_presence)
Colour encodes token type (input / cue "A:" / label-target, same mapping as the other
FV_location posters); line style encodes family (solid = write, dashed = read). The two
headline curves — read@label and write@cue — are drawn at full weight with their peaks
marked; the other four are held back so the dissociation reads at poster distance.

--layout controls how the two families share the plot:
  single  one shared cos axis (default; honest, but a low-amplitude family looks flat)
  dual    twin y-axes, read left / write right — each family filling its own scale.
          NOTE: both families are measured in the SAME unit (cosine), so a second scale
          makes the apparent crossing point an artefact of the chosen limits. Use only
          when the point being made is about the LOCATION of each peak, never about which
          curve is "higher"; the figure is annotated to that effect.
  stacked two panels sharing the layer axis — the non-distorting way to give each family
          its own scale.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

CAT = {"cue": "#2a78d6", "label": "#eb6834", "input": "#1baf7a"}
PRETTY = {"cue": "cue “A:”", "label": "target", "input": "input"}
INK, INK_MUTED, GRID = "#1a1a19", "#6b6b68", "#e6e5e2"
WRITE_DIR = "direct_FV_presence"
# Two different objects can play the "read" role; they are NOT on a comparable scale:
#   read_dir   r_A = M^+ v_A  — glossary read direction, unit, no shared-mean bulk
#   label_mean m_A(L6)        — raw mean residual at the demo label token (the vector that
#                               actually steers at L6); a raw mean, so it carries the large
#                               shared residual-stream component and reads high everywhere.
READ_SOURCES = {
    "read_dir": ("top_down_read_dir_presence", "read"),
    "label_mean": ("bottom_up_label_mean_L6_presence", "read"),
}
WRITE_LABEL = "write"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=TASK69_RUN_DIR / "feature_locations")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--read_source", choices=sorted(READ_SOURCES), default="read_dir")
    p.add_argument("--layout", choices=["single", "dual", "stacked"], default="single")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.root / "poster_visuals"
    return args


def load_profiles(root, read_dir_name):
    profs = {}
    for folder in (WRITE_DIR, read_dir_name):
        z = np.load(root / folder / "fv_location.npz", allow_pickle=False)
        cols = [str(c) for c in z["columns"]]
        mat = z["cos"].mean(0)
        for k in CAT:
            idx = [i for i, c in enumerate(cols) if c.endswith("_" + k)]
            profs[(folder, k)] = mat[:, idx].mean(axis=1)
    return profs


def draw_family(ax, profs, folder, n_layers, ls, headline_kind):
    """Plot one family's three token-type curves; return its (peak_layer, peak_value)."""
    for k in ("cue", "label", "input"):
        head = k == headline_kind
        ax.plot(range(n_layers), profs[(folder, k)], color=CAT[k], ls=ls,
                lw=3.4 if head else 1.7, alpha=1.0 if head else 0.42,
                zorder=4 if head else 2, solid_capstyle="round", dash_capstyle="round")
    prof = profs[(folder, headline_kind)]
    return int(np.argmax(prof)), float(prof.max())


def mark_peak(ax, L, y, k, kind, tx, ty):
    ax.plot([L], [y], "o", ms=11, color=CAT[k], mec="white", mew=2.2, zorder=6)
    short = {"label": "target tokens", "cue": "cue tokens"}[k]
    ax.annotate(f"{kind} peaks · L{L}  ({short})", xy=(L, y), xytext=(tx, ty),
                ha="left", va="bottom", fontsize=12.5, color=CAT[k], fontweight="bold",
                zorder=7, arrowprops=dict(arrowstyle="-", color=CAT[k], lw=1.2, alpha=0.65,
                                          shrinkA=4, shrinkB=8,
                                          connectionstyle="arc3,rad=0.12"))


def style_axis(ax, n_layers, xlabel=True):
    ax.set_xlim(-0.4, n_layers - 0.6)
    ax.set_xticks(range(0, n_layers, 2))
    if xlabel:
        ax.set_xlabel("layer", fontsize=13, color=INK, labelpad=8)
    ax.tick_params(labelsize=11, colors=INK_MUTED)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main():
    args = parse_args()
    read_dir_name, read_label = READ_SOURCES[args.read_source]
    profs = load_profiles(args.root, read_dir_name)
    n_layers = len(next(iter(profs.values())))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    r_max = max(profs[(read_dir_name, k)].max() for k in CAT)
    w_max = max(profs[(WRITE_DIR, k)].max() for k in CAT)

    title = "Read and Write Feature Locations"

    if args.layout == "stacked":
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.4), sharex=True)
        fig.subplots_adjust(left=0.095, right=0.975, top=0.895, bottom=0.075, hspace=0.16)
        for ax, folder, ls, kind, kk, mx in (
                (axes[0], read_dir_name, "--", "READ", "label", r_max),
                (axes[1], WRITE_DIR, "-", "WRITE", "cue", w_max)):
            L, y = draw_family(ax, profs, folder, n_layers, ls, kk)
            if kind == "READ":
                Lr, yr = L, y
            else:
                Lw, yw = L, y
            ax.set_ylim(0, mx * 1.30)
            # READ peaks early, where the legend sits — swing its callout right
            mark_peak(ax, L, y, kk, kind, L + (8.5 if kind == "READ" else 1.2),
                      mx * 1.02)
            style_axis(ax, n_layers, xlabel=(folder == WRITE_DIR))
            ax.set_ylabel(("READ" if kind == "READ" else "WRITE") + "\nmean cos",
                          fontsize=12.5, color=INK, labelpad=8)
        leg_ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        fig.subplots_adjust(left=0.085, right=0.912 if args.layout == "dual" else 0.975,
                            top=0.885,
                            bottom=0.115)
        if args.layout == "dual":
            axw = ax.twinx()
            Lr, yr = draw_family(ax, profs, read_dir_name, n_layers, "--", "label")
            Lw, yw = draw_family(axw, profs, WRITE_DIR, n_layers, "-", "cue")
            ax.set_ylim(0, r_max * 1.52)
            axw.set_ylim(0, w_max * 1.52)
            mark_peak(ax, Lr, yr, "label", "READ", 1.0, r_max * 1.13)
            mark_peak(axw, Lw, yw, "cue", "WRITE", 15.4, w_max * 1.13)
            ax.set_ylabel("READ feature   mean cos   (dashed lines)", fontsize=12.5,
                          color=INK, labelpad=8)
            axw.set_ylabel("WRITE feature   mean cos   (solid lines)", fontsize=12.5,
                           color=INK, labelpad=12, rotation=270, va="bottom")
            axw.tick_params(labelsize=11, colors=INK_MUTED)
            for s in ("top", "left"):
                axw.spines[s].set_visible(False)
            axw.spines["right"].set_color(GRID)
            axw.spines["bottom"].set_color(GRID)
            style_axis(ax, n_layers)
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_color(GRID)
        else:
            Lr, yr = draw_family(ax, profs, read_dir_name, n_layers, "--", "label")
            Lw, yw = draw_family(ax, profs, WRITE_DIR, n_layers, "-", "cue")
            hi = max(r_max, w_max) * 1.62
            ax.set_ylim(0, hi)
            mark_peak(ax, Lr, yr, "label", "READ", 4.9, 0.760 * hi)
            mark_peak(ax, Lw, yw, "cue", "WRITE", 15.0, 0.715 * hi)
            ax.set_ylabel("mean cos with the task feature", fontsize=13, color=INK,
                          labelpad=8)
            style_axis(ax, n_layers)
        leg_ax = ax

    fig.text(0.085, 0.962, title, fontsize=19, color=INK, fontweight="bold", va="top")

    cat_handles = [Line2D([], [], color=CAT[k], lw=3.2) for k in ("cue", "label", "input")]
    fam_handles = [Line2D([], [], color=INK_MUTED, lw=2.4, ls=ls) for ls in ("-", "--")]
    leg1 = leg_ax.legend(cat_handles, [PRETTY[k] for k in ("cue", "label", "input")],
                         title="token type", loc="upper left", bbox_to_anchor=(0.007, 0.99),
                         frameon=False, fontsize=11.5, title_fontsize=11.5, labelcolor=INK)
    leg1._legend_box.align = "left"
    leg_ax.add_artist(leg1)
    leg2 = leg_ax.legend(fam_handles, [WRITE_LABEL, read_label], title="feature",
                         loc="upper left", bbox_to_anchor=(0.20, 0.99), frameon=False,
                         fontsize=11.5, title_fontsize=11.5, labelcolor=INK)
    leg2._legend_box.align = "left"

    parts = ["read_vs_write_presence"]
    if args.read_source != "read_dir":
        parts.append(args.read_source)
    if args.layout != "single":
        parts.append(args.layout)
    stem = "_".join(parts)
    fig.savefig(args.out_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)

    with open(args.out_dir / f"{stem}.csv", "w") as f:
        keys = [(fo, k) for fo in (WRITE_DIR, read_dir_name) for k in ("cue", "label", "input")]
        f.write("layer," + ",".join(f"{'write' if fo == WRITE_DIR else 'read'}_{k}"
                                    for fo, k in keys) + "\n")
        for l in range(n_layers):
            f.write(f"{l}," + ",".join(f"{profs[key][l]:.5f}" for key in keys) + "\n")
    print(f"wrote {args.out_dir}/{stem}.png  "
          f"(read peak L{Lr} {yr:.3f} | write peak L{Lw} {yw:.3f})")


if __name__ == "__main__":
    main()
