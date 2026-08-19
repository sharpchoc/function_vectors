#!/usr/bin/env python
"""Poster visuals for the FV_location presence studies (direct_FV_presence,
label_mean_L6_presence, ...): one metric per figure, fitted sequential scale.

What a figure must carry: WHERE in the residual stream the reference direction is present —
which layer band, and which token type (input / cue "A:" / label-target).

Left panel: layer x token-position heatmap, single-hue sequential ramp (dataviz reference
blue 100->700) with the scale fitted to the data range instead of the symmetric diverging
scale of the analysis figures. A category strip under the x-axis marks input / cue / label
so the repeating triples read without dense tick labels. Right panel: the same data
collapsed to one line per token category, sharing the layer axis — this is what makes the
ordering between token types explicit at poster distance.

Examples:
  # FV direction, cosine only (the original poster)
  plot_69_fv_location_poster.py --in_dir .../direct_FV_presence --metric cos --band 9 15
  # L6 raw-mean label-token steering vector, both metrics
  plot_69_fv_location_poster.py --in_dir .../label_mean_L6_presence --metric cos --band 5 8
  plot_69_fv_location_poster.py --in_dir .../label_mean_L6_presence --metric dot --band 5 8
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

# per-study wording; keyed by the presence folder name
STUDIES = {
    "direct_FV_presence": dict(
        vec="task function vector",
        title="Where the function vector lives in the residual stream",
        band=(9, 15)),
    "label_mean_L6_presence": dict(
        vec="L6 label-token task mean  m_A(L6)",
        title="Where the best steering vector lives in the residual stream",
        band=(5, 8)),
    "read_dir_presence": dict(
        vec="task read direction",
        title="Where the read direction lives in the residual stream",
        band=(6, 16)),
}
METRICS = {
    "cos": dict(sym="cos(z, v)", axis="mean cos by token type",
                blurb="cos(residual stream, {vec})"),
    "dot": dict(sym="z · v / ||v||", axis="mean projection by token type",
                blurb="projection magnitude of the residual stream onto {vec}"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", type=Path,
                   default=TASK69_RUN_DIR / "feature_locations" / "direct_FV_presence")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--metric", choices=sorted(METRICS), default="cos")
    p.add_argument("--band", type=int, nargs=2, default=None,
                   help="layer band to call out; default from the study table")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--band_note", type=str, default=None,
                   help="extra text after the band label, e.g. 'steering peak L6-L7'")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.in_dir / "poster_visuals"
    return args


def main():
    args = parse_args()
    study = STUDIES.get(args.in_dir.name, dict(vec="reference direction",
                                               title="Presence in the residual stream",
                                               band=(9, 15)))
    band = tuple(args.band) if args.band else study["band"]
    title = args.title or study["title"]
    met = METRICS[args.metric]

    z = np.load(args.in_dir / "fv_location.npz", allow_pickle=False)
    cols = [str(c) for c in z["columns"]]
    mat = z[args.metric].mean(0)                 # (28, 32) mean over 69 tasks
    n_layers = mat.shape[0]
    kinds = [c.split("_", 1)[1] for c in cols]
    cat_idx = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in CAT}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("fv_blue", BLUE_RAMP)
    vmin, vmax = float(mat.min()), float(mat.max())
    fmt = (lambda v: f"{v:.2f}") if args.metric == "cos" else (lambda v: f"{v:.0f}")

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
    ax.add_patch(Rectangle((-0.5, band[0] - 0.5), len(cols), band[1] - band[0] + 1,
                           fill=False, ec="#ffffff", lw=2.2, zorder=4))
    ax.add_patch(Rectangle((-0.5, band[0] - 0.5), len(cols), band[1] - band[0] + 1,
                           fill=False, ec=INK, lw=0.9, ls=(0, (5, 3)), zorder=5))
    pk_l, pk_c = np.unravel_index(mat.argmax(), mat.shape)
    pretty = cols[pk_c].replace("_", " ").replace("demo", "demo ")
    dx, dy = (-7.6, 8.4) if pk_c > len(cols) / 2 else (7.6, 8.4)
    if pk_l > n_layers - 8:
        dy = -8.4
    ax.annotate(f"peak  {fmt(mat[pk_l, pk_c])}\nlayer {pk_l}, {pretty}",
                xy=(pk_c - 0.4, pk_l), xytext=(pk_c + dx, pk_l + dy), fontsize=10.5,
                color=INK, ha="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                connectionstyle="arc3,rad=-0.18"))
    for x in np.arange(2.5, len(cols) - 2, 3):
        ax.axvline(x, color="#ffffff", lw=0.6, alpha=0.45, zorder=3)
    ax.set_ylabel("layer  (residual stream, block output)", fontsize=12, color=INK)
    ax.set_yticks(range(0, n_layers, 3))
    ax.tick_params(labelsize=10, colors=INK_MUTED)
    fig.text(0.055, 0.935, title, fontsize=17, color=INK, fontweight="bold", va="top")
    fig.text(0.055, 0.877,
             met["blurb"].format(vec=study["vec"])
             + " — mean over 69 tasks × 150 clean 10-shot prompts",
             fontsize=11.5, color=INK_MUTED, va="top")
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
    axl.axhspan(band[0] - 0.5, band[1] + 0.5, color="#f0efec", zorder=0)
    profs = {k: mat[:, cat_idx[k]].mean(axis=1) for k in CAT}
    hi = max(p.max() for p in profs.values())
    # the band note is anchored first (bottom-right, always empty) so line labels avoid it
    note = f"shaded: layers {band[0]}–{band[1]}" + (f"\n{args.band_note}" if args.band_note else "")
    axl.text(hi * 1.16, 0.2, note, va="bottom", ha="right", fontsize=10, color=INK_MUTED,
             linespacing=1.35)
    placed = [(hi * 1.05, 1.5)]
    for k in ("cue", "label", "input"):
        axl.plot(profs[k], range(n_layers), color=CAT[k], lw=2.4, zorder=3)
        # Direct-label a line only where it is clearly separated from the other two AND
        # the label will not sit on top of one already placed; curves that run together
        # (e.g. cue vs input for the label-mean vector) rely on the shared category
        # legend below instead, which is present in every figure.
        others = [profs[o] for o in CAT if o != k]
        sep = np.minimum.reduce([np.abs(profs[k] - o) for o in others])
        best = int(np.argmax(sep))
        if sep[best] <= 0.09 * hi:
            continue
        x_here = profs[k][best]
        if any(abs(x_here - px) < 0.20 * hi and abs(best - py) < 6 for px, py in placed):
            continue
        placed.append((x_here, best))
        side = 5 if x_here >= max(o[best] for o in others) else -5
        axl.annotate({"cue": "cue “A:”", "label": "label", "input": "input"}[k],
                     xy=(x_here, best), xytext=(side, 6), textcoords="offset points",
                     fontsize=11, color=CAT[k], fontweight="bold",
                     ha="left" if side > 0 else "right")
    axl.set_xlabel(met["axis"], fontsize=11, color=INK_MUTED)
    axl.set_xlim(0, hi * 1.18)
    axl.tick_params(labelsize=10, colors=INK_MUTED)
    plt.setp(axl.get_yticklabels(), visible=False)
    axl.grid(axis="x", color="#e6e5e2", lw=0.8, zorder=1)
    axl.set_axisbelow(True)
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)

    cb = fig.colorbar(im, cax=axc)
    cb.set_label(met["sym"], fontsize=11, color=INK_MUTED)
    cb.ax.tick_params(labelsize=10, colors=INK_MUTED)
    cb.outline.set_visible(False)

    stem = f"presence_poster_{args.metric}"
    for ext in ("png",):
        fig.savefig(args.out_dir / f"{stem}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    with open(args.out_dir / f"layer_profile_by_token_type_{args.metric}.csv", "w") as f:
        f.write("layer,input,cue,label,query_cue\n")
        q = cols.index("query_cue")
        for l in range(n_layers):
            f.write(f"{l}," + ",".join(f"{profs[k][l]:.5f}" for k in ("input", "cue", "label"))
                    + f",{mat[l, q]:.5f}\n")
    print(f"wrote {args.out_dir}/{stem}.png (scale fitted {vmin:.3f}-{vmax:.3f}; "
          f"peak {fmt(mat.max())} at layer {pk_l}, {cols[pk_c]})")


if __name__ == "__main__":
    main()
