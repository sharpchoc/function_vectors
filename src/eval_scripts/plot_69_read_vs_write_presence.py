#!/usr/bin/env python
"""Poster visual: where the READ and WRITE features become present in the residual stream.

One panel, mean cos by token type against depth, overlaying both feature families:
  WRITE  = the task function vector v_A            (direct_FV_presence)
  READ   = the task read direction r_A = M^+ v_A   (read_dir_presence, cosine_perhead)
Colour encodes token type (input / cue "A:" / label-target, same mapping as the other
FV_location posters); line style encodes family (solid = write, dashed = read). The two
headline curves — read@label and write@cue — are drawn at full weight with their peaks
marked; the other four are held back so the dissociation reads at poster distance.

Writes RESULTS/69_task_run/FV_location/poster_visuals/read_vs_write_presence[_labelmean].{png,csv}
plus the plotted curves as csv.
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
PRETTY = {"cue": "cue “A:”", "label": "label / target", "input": "input"}
INK, INK_MUTED, GRID = "#1a1a19", "#6b6b68", "#e6e5e2"
WRITE_DIR = "direct_FV_presence"
# Two different objects can play the "read" role; they are NOT on a comparable scale:
#   read_dir   r_A = M^+ v_A  — glossary read direction, unit, no shared-mean bulk
#   label_mean m_A(L6)        — raw mean residual at the demo label token (the vector that
#                               actually steers at L6); a raw mean, so it carries the large
#                               shared residual-stream component and reads high everywhere.
READ_SOURCES = {
    "read_dir": ("read_dir_presence", "read  (read direction r_A = M⁺v_A)"),
    "label_mean": ("label_mean_L6_presence", "read  (L6 label-token mean m_A(L6))"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=TASK69_RUN_DIR / "FV_location")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--read_source", choices=sorted(READ_SOURCES), default="read_dir",
                   help="which object plays the READ role (see READ_SOURCES)")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.root / "poster_visuals"
    return args


def main():
    args = parse_args()
    read_dir_name, read_label = READ_SOURCES[args.read_source]
    FAMILIES = {WRITE_DIR: ("write  (function vector v_A)", "-"),
                read_dir_name: (read_label, "--")}
    HEADLINE = {(WRITE_DIR, "cue"), (read_dir_name, "label")}
    profs = {}
    for folder in FAMILIES:
        z = np.load(args.root / folder / "fv_location.npz", allow_pickle=False)
        cols = [str(c) for c in z["columns"]]
        mat = z["cos"].mean(0)
        for k in CAT:
            idx = [i for i, c in enumerate(cols) if c.endswith("_" + k)]
            profs[(folder, k)] = mat[:, idx].mean(axis=1)
    n_layers = len(next(iter(profs.values())))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.80, bottom=0.135)

    for (folder, k), prof in profs.items():
        head = (folder, k) in HEADLINE
        ax.plot(range(n_layers), prof, color=CAT[k], ls=FAMILIES[folder][1],
                lw=3.4 if head else 1.7, alpha=1.0 if head else 0.42,
                zorder=4 if head else 2, solid_capstyle="round",
                dash_capstyle="round")

    # Peaks of the two headline curves. Both callouts sit in the headroom band below the
    # legends and are tied to their peak by a connector, so neither can foul the y-axis
    # label, the legends, or each other.
    hi_y = max(p.max() for p in profs.values()) * 1.62   # must match set_ylim below
    ax.set_ylim(0, hi_y)
    for folder, k, tx, ty in ((read_dir_name, "label", 4.9, 0.760),
                              (WRITE_DIR, "cue", 15.0, 0.715)):
        prof = profs[(folder, k)]
        L = int(np.argmax(prof))
        ax.plot([L], [prof[L]], "o", ms=11, color=CAT[k], mec="white", mew=2.2, zorder=6)
        short = {"label": "label tokens", "cue": "cue tokens"}[k]
        ax.annotate(f"{'WRITE' if folder == WRITE_DIR else 'READ'} peaks · L{L}  ({short})",
                    xy=(L, prof[L]), xytext=(tx, ty * hi_y), ha="left", va="bottom",
                    fontsize=12.5, color=CAT[k], fontweight="bold", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=CAT[k], lw=1.2, alpha=0.65,
                                    shrinkA=4, shrinkB=8,
                                    connectionstyle="arc3,rad=0.12"))

    ax.set_xlim(-0.4, n_layers - 0.6)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xlabel("layer  (residual stream, block output)", fontsize=13, color=INK,
                  labelpad=8)
    ax.set_ylabel("mean cos with the task feature", fontsize=13, color=INK, labelpad=8)
    ax.tick_params(labelsize=11, colors=INK_MUTED)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    fig.text(0.085, 0.955, "Read and write features appear at different places",
             fontsize=18, color=INK, fontweight="bold", va="top")
    sub = ("The read feature is picked up early, at the demo label tokens; the function "
           "vector it produces shows up\nlater, at the cue tokens where the answer is "
           "generated.   Mean over 69 tasks × 150 clean 10-shot prompts.")
    if args.read_source == "label_mean":
        sub += ("\nNote: m_A(L6) is a raw mean activation, so it carries the shared "
                "residual-stream component and sits high at every position;\nits "
                "label-vs-other margin (1.4×), not its absolute level, is what is "
                "comparable with the function vector.")
    fig.text(0.085, 0.895, sub, fontsize=12, color=INK_MUTED, va="top", linespacing=1.5)

    cat_handles = [Line2D([], [], color=CAT[k], lw=3.2) for k in ("cue", "label", "input")]
    fam_handles = [Line2D([], [], color=INK_MUTED, lw=2.4, ls=FAMILIES[f][1]) for f in FAMILIES]
    leg1 = ax.legend(cat_handles, [PRETTY[k] for k in ("cue", "label", "input")],
                     title="token type", loc="upper left", bbox_to_anchor=(0.007, 0.99),
                     frameon=False, fontsize=11.5, title_fontsize=11.5, labelcolor=INK)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    leg2 = ax.legend(fam_handles, [FAMILIES[f][0] for f in FAMILIES], title="feature",
                     loc="upper left", bbox_to_anchor=(0.20, 0.99), frameon=False,
                     fontsize=11.5, title_fontsize=11.5, labelcolor=INK)
    leg2._legend_box.align = "left"

    suffix = "" if args.read_source == "read_dir" else f"_{args.read_source}"
    for ext in ("png",):
        fig.savefig(args.out_dir / f"read_vs_write_presence{suffix}.{ext}", dpi=300,
                    facecolor="white")
    plt.close(fig)

    with open(args.out_dir / f"read_vs_write_presence{suffix}.csv", "w") as f:
        keys = [(fo, k) for fo in FAMILIES for k in ("cue", "label", "input")]
        f.write("layer," + ",".join(f"{'write' if 'direct' in fo else 'read'}_{k}"
                                    for fo, k in keys) + "\n")
        for l in range(n_layers):
            f.write(f"{l}," + ",".join(f"{profs[key][l]:.5f}" for key in keys) + "\n")
    print(f"wrote {args.out_dir}/read_vs_write_presence{suffix}.png")
    for key in HEADLINE:
        print(f"  {key}: peak {profs[key].max():.3f} at layer {int(np.argmax(profs[key]))}")


if __name__ == "__main__":
    main()
