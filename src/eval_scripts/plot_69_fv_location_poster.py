#!/usr/bin/env python
"""Poster visual for direct_FV_presence: cosine only, fitted sequential scale.

Message the figure must carry: the task FV is most present in the MID-DEPTH layers,
concentrated at the cue ("A:") tokens and partially at the demo label/target tokens,
while input tokens stay low throughout.

Left panel: layer x token-position heatmap of cos(z_l, v_A), single-hue sequential ramp
(dataviz reference blue 100->700) with the scale fitted to the data range instead of the
symmetric diverging scale of the analysis figure. A category strip under the x-axis marks
input / cue / label so the repeating triples are readable without dense tick labels. Right
panel: the same data collapsed to one line per token category, sharing the layer axis —
this is what makes the "cue >> label > input, peaking mid-depth" ordering explicit.

Reads results/69_task_run/FV_location/direct_FV_presence/fv_location.npz; writes
poster_visuals/fv_presence_poster.{png,pdf} + the plotted profile as csv.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR  # noqa: E402

# dataviz reference palette: sequential blue ramp (steps 100..700) and categorical slots 1-3
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CAT = {"cue": "#2a78d6", "label": "#eb6834", "input": "#1baf7a"}
INK, INK_MUTED = "#1a1a19", "#6b6b68"
BAND = (9, 15)          # the mid-depth band the poster calls out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", type=Path,
                   default=TASK69_RUN_DIR / "FV_location" / "direct_FV_presence")
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.in_dir / "poster_visuals"
    return args


def main():
    args = parse_args()
    z = np.load(args.in_dir / "fv_location.npz", allow_pickle=False)
    cols = [str(c) for c in z["columns"]]
    mat = z["cos"].mean(0)                       # (28, 32) mean over 69 tasks
    n_layers = mat.shape[0]
    kinds = [c.split("_", 1)[1] for c in cols]
    cat_idx = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in CAT}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("fv_blue", BLUE_RAMP)
    vmin, vmax = float(mat.min()), float(mat.max())

    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[3.05, 1.0, 0.045],
                          height_ratios=[1, 0.052], hspace=0.06, wspace=0.10,
                          left=0.055, right=0.95, top=0.815, bottom=0.155)
    ax = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[1, 0], sharex=ax)      # category strip
    axl = fig.add_subplot(gs[0, 1], sharey=ax)      # per-category layer profile
    axc = fig.add_subplot(gs[0, 2])                 # colorbar

    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest",
                   extent=(-0.5, len(cols) - 0.5, -0.5, n_layers - 0.5))
    ax.add_patch(Rectangle((-0.5, BAND[0] - 0.5), len(cols), BAND[1] - BAND[0] + 1,
                           fill=False, ec="#ffffff", lw=2.2, zorder=4))
    ax.add_patch(Rectangle((-0.5, BAND[0] - 0.5), len(cols), BAND[1] - BAND[0] + 1,
                           fill=False, ec=INK, lw=0.9, ls=(0, (5, 3)), zorder=5))
    pk_l, pk_c = np.unravel_index(mat.argmax(), mat.shape)
    ax.annotate(f"peak  cos = {mat[pk_l, pk_c]:.2f}\nlayer {pk_l}, query cue",
                xy=(pk_c - 0.4, pk_l), xytext=(pk_c - 7.6, pk_l + 8.4), fontsize=10.5,
                color=INK, ha="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                connectionstyle="arc3,rad=-0.18"))
    for x in np.arange(2.5, len(cols) - 2, 3):
        ax.axvline(x, color="#ffffff", lw=0.6, alpha=0.45, zorder=3)
    ax.set_ylabel("layer  (residual stream, block output)", fontsize=12, color=INK)
    ax.set_yticks(range(0, n_layers, 3))
    ax.tick_params(labelsize=10, colors=INK_MUTED)
    fig.text(0.055, 0.935, "Where the function vector lives in the residual stream",
             fontsize=17, color=INK, fontweight="bold", va="top")
    fig.text(0.055, 0.877,
             "cos(residual stream, task function vector) — mean over 69 tasks × 150 clean "
             "10-shot prompts", fontsize=11.5, color=INK_MUTED, va="top")
    plt.setp(ax.get_xticklabels(), visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- category strip + demo-group labels ---
    strip = np.zeros((1, len(cols), 3))
    for k, ids in cat_idx.items():
        rgb = matplotlib.colors.to_rgb(CAT[k])
        for i in ids:
            strip[0, i] = rgb
    axs.imshow(strip, aspect="auto", origin="lower",
               extent=(-0.5, len(cols) - 0.5, 0, 1), interpolation="nearest")
    groups = [(1.0, "demo 1"), (4.0, "2"), (7.0, "3"), (10.0, "4"), (13.0, "5"), (16.0, "6"),
              (19.0, "7"), (22.0, "8"), (25.0, "9"), (28.0, "10"), (30.5, "query")]
    axs.set_xticks([g[0] for g in groups])
    axs.set_xticklabels([g[1] for g in groups], fontsize=10, color=INK_MUTED)
    axs.set_yticks([])
    axs.tick_params(length=0)
    for s in axs.spines.values():
        s.set_visible(False)
    axs.set_xlabel("token position   (each demo contributes input → cue → label tokens; "
                   "multi-token spans averaged)", fontsize=11, color=INK_MUTED, labelpad=8)
    handles = [Rectangle((0, 0), 1, 1, fc=CAT[k]) for k in ("input", "cue", "label")]
    axs.legend(handles, ["input tokens", "cue tokens  “A:”", "label / target tokens"],
               loc="upper center", bbox_to_anchor=(0.5, -2.4), ncol=3, frameon=False,
               fontsize=11, handlelength=1.1, handleheight=1.1, columnspacing=2.6,
               labelcolor=INK)

    # --- per-category layer profile (shared y) ---
    axl.axhspan(BAND[0] - 0.5, BAND[1] + 0.5, color="#f0efec", zorder=0)
    for k in ("cue", "label", "input"):
        prof = mat[:, cat_idx[k]].mean(axis=1)
        axl.plot(prof, range(n_layers), color=CAT[k], lw=2.4, zorder=3)
        top = int(np.argmax(prof))
        axl.annotate({"cue": "cue “A:”", "label": "label", "input": "input"}[k],
                     xy=(prof[top], top), xytext=(5, 6), textcoords="offset points",
                     fontsize=11, color=CAT[k], fontweight="bold")
    axl.set_xlabel("mean cos by token type", fontsize=11, color=INK_MUTED)
    axl.set_xlim(0, max(0.36, mat.max() * 1.06))
    axl.set_xticks([0, 0.1, 0.2, 0.3])
    axl.tick_params(labelsize=10, colors=INK_MUTED)
    plt.setp(axl.get_yticklabels(), visible=False)
    axl.grid(axis="x", color="#e6e5e2", lw=0.8, zorder=1)
    axl.set_axisbelow(True)
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)
    axl.text(0.006, BAND[1] + 0.35, f"layers {BAND[0]}–{BAND[1]}", va="bottom", ha="left",
             fontsize=10, color=INK_MUTED)

    cb = fig.colorbar(im, cax=axc)
    cb.set_label("cos(z, v_A)", fontsize=11, color=INK_MUTED)
    cb.ax.tick_params(labelsize=10, colors=INK_MUTED)
    cb.outline.set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fv_presence_poster.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    with open(args.out_dir / "layer_profile_by_token_type.csv", "w") as f:
        f.write("layer,input,cue,label,query_cue\n")
        q = cols.index("query_cue")
        for l in range(n_layers):
            f.write(f"{l}," + ",".join(f"{mat[l, cat_idx[k]].mean():.5f}"
                                       for k in ("input", "cue", "label"))
                    + f",{mat[l, q]:.5f}\n")
    print(f"wrote {args.out_dir} (scale fitted to {vmin:.3f}-{vmax:.3f}; "
          f"peak cos {mat.max():.3f} at layer {pk_l}, {cols[pk_c]})")


if __name__ == "__main__":
    main()
