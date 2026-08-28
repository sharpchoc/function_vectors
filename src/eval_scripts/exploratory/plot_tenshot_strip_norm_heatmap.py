"""
Plot the 10-shot intervene-token STRIP norm study (reads saved grids; CPU-only, no GPU).

Companion to steer_tenshot_strip_norm_heatmap.py. Per --metric (rel/raw/ngrow; default rel).
ngrow (= ||steered||/||clean|| - 1 at qfinal) is SIGNED: strips use a diverging map centred on 0
and the lines reduce each grid to its extreme value (max |cell|, sign kept). For rel/raw:
  - strip_{metric}_alpha{a}.png    : per α, cols = 4 combos, rows = 30 intervene tokens; each cell
    = that (token, combo) 29×29 layer heatmap (x=intervene layer, y=read layer), one global
    SEQUENTIAL colorbar (values are ≥0 magnitudes, unlike the diverging cos shifts).
  - scalar_lines_{metric}.png      : line summary in the scalar_lines format (x = 30 intervene
    tokens, hue = direction, style = α, y = per-grid reduction: max, or mean of top-k cells via
    --top_k). --labels_only restricts x to the 10 demo label tokens.
Output to <root>/figures/.
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

# categorical slots 1-4 (fixed order, matches plot_tenshot_strip_lines.py); α = line style
COMBO_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
ALPHA_STYLES = {2.0: "-", 4.0: "--", 8.0: ":"}

METRIC_LABEL = {"rel": "‖Δ‖/‖clean‖ at qfinal", "raw": "‖Δ‖ at qfinal",
                "ngrow": "‖steered‖/‖clean‖ − 1 at qfinal"}
SIGNED = {"ngrow"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "tenshot_strip_intervention_norm_heatmap"))
    p.add_argument("--metric", choices=["rel", "raw", "ngrow"], default="rel")
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--top_k", type=int, default=1,
                   help="lines: reduce each grid to the mean of its top-k cells (1 = plain max)")
    p.add_argument("--labels_only", action="store_true",
                   help="lines: plot only the 10 demo label tokens")
    p.add_argument("--skip_strips", action="store_true", help="only render the lines figure")
    return p.parse_args()


def load_grid(root, task_pair, dir_name, tkey, a, metric):
    p = Path(root) / task_pair / f"{dir_name}__{tkey}_alpha{a:g}_{metric}_grid.npy"
    return np.load(p) if p.exists() else None


def global_vmax(root, alphas, metric):
    vmax = 0.0
    for task_pair, dir_name, _ in COMBOS:
        for tkey in IKEYS:
            for a in alphas:
                g = load_grid(root, task_pair, dir_name, tkey, a, metric)
                if g is not None and np.isfinite(g).any():
                    v = np.nanmax(np.abs(g)) if metric in SIGNED else np.nanmax(g)
                    vmax = max(vmax, float(v))
    return vmax or 1e-6


def reduce_grid(g, top_k, signed=False):
    vals = g[np.isfinite(g)]
    if vals.size == 0:
        return np.nan
    if signed:  # extreme value: largest |cell|, sign preserved (top_k ignored)
        return float(vals.ravel()[np.argmax(np.abs(vals))])
    return float(np.sort(vals.ravel())[-top_k:].mean())


def make_strip(root, alpha, vmax, metric, out_path):
    nr, nc = len(IKEYS), len(COMBOS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc + 1.4, 1.15 * nr + 0.8),
                             squeeze=False, constrained_layout=True)
    im = None
    n_layers = 29
    for r, tkey in enumerate(IKEYS):
        for cj, (task_pair, dir_name, clab) in enumerate(COMBOS):
            ax = axes[r][cj]
            g = load_grid(root, task_pair, dir_name, tkey, alpha, metric)
            if g is None:
                ax.set_axis_off()
                continue
            n_layers = g.shape[0]
            if metric in SIGNED:
                im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
                diag_c = "k"
            else:
                im = ax.imshow(g, origin="lower", cmap="magma", vmin=0.0, vmax=vmax, aspect="equal")
                diag_c = "w"
            ax.plot([0, n_layers - 1], [0, n_layers - 1], color=diag_c, lw=0.4, ls=":", alpha=0.4)
            ax.set_xticks([0, n_layers - 1]); ax.set_yticks([0, n_layers - 1])
            ax.tick_params(labelsize=4, length=1.5, pad=1)
            if cj == 0:
                ax.set_ylabel(tkey, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
            if r == 0:
                ax.set_title(clab, fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("intervene layer", fontsize=5)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.4, label=f"{METRIC_LABEL[metric]} (global scale)")
    fig.suptitle(f"10-shot intervene-token strip, {metric} magnitude — α={alpha:g}  (read fixed at "
                 f"query-final; shared global scale vmax={vmax:.3f})\nrows = intervene token · cols = "
                 f"combo · each cell x=intervene layer, y=read layer (0–28), lower-tri≡0", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_lines(root, alphas, metric, top_k, labels_only, out_path):
    ikeys = [f"d{i}_lab" for i in range(1, N_SHOTS + 1)] if labels_only else IKEYS
    x = np.arange(len(ikeys))
    fig, ax = plt.subplots(figsize=(7.5, 5.2) if labels_only else (13.5, 5.2))
    for (task_pair, dir_name, clab), color in zip(COMBOS, COMBO_COLORS):
        for a in alphas:
            y = [np.nan if g is None else reduce_grid(g, top_k, metric in SIGNED)
                 for g in (load_grid(root, task_pair, dir_name, tk, a, metric) for tk in ikeys)]
            ax.plot(x, np.array(y, dtype=float), ALPHA_STYLES.get(a, "-"), color=color, linewidth=2,
                    marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)

    stat = ("signed extreme" if metric in SIGNED
            else "peak" if top_k == 1 else f"top-{top_k} mean")
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    if labels_only:
        ax.set_xticklabels([str(d) for d in range(1, N_SHOTS + 1)])
        ax.set_xlabel("demo index (label token of demo d)")
    else:
        for d in range(1, N_SHOTS):
            ax.axvline(3 * d - 0.5, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_xticklabels([k.split("_")[1] for k in ikeys], fontsize=7, color="#555555")
        for d in range(N_SHOTS):
            ax.text(3 * d + 1, -0.085, f"demo {d + 1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#333333")
        ax.set_xlabel("intervene token (position in the 10-shot prompt)", labelpad=22)
    ax.set_title(f"10-shot strip, {metric} magnitude: {stat} per intervene token")
    ax.set_xlim(-0.5, len(ikeys) - 0.5)
    ax.set_ylabel(f"{METRIC_LABEL[metric]} ({stat} over intervene × read cells)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)

    combo_handles = [Line2D([], [], color=c, linewidth=2, label=lab)
                     for (_, _, lab), c in zip(COMBOS, COMBO_COLORS)]
    alpha_handles = [Line2D([], [], color="#666666", linewidth=2, linestyle=ALPHA_STYLES[a],
                            label=f"α={a:g}") for a in alphas]
    leg1 = ax.legend(handles=combo_handles, title="direction", loc="upper left", fontsize=8,
                     title_fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=alpha_handles, title="α", loc="upper left",
              bbox_to_anchor=(0.27, 1.0) if labels_only else (0.13, 1.0),
              fontsize=8, title_fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_strips:
        vmax = global_vmax(root, args.alphas, args.metric)
        print(f"global vmax ({args.metric}) = {vmax:.5f}")
        for a in args.alphas:
            make_strip(root, a, vmax, args.metric, fig_dir / f"strip_{args.metric}_alpha{a:g}.png")
            print(f"saved strip_{args.metric}_alpha{a:g}.png")
    name = f"scalar_lines_{args.metric}"
    if args.labels_only:
        name += "_labels_only"
    if args.top_k > 1:
        name += f"_top{args.top_k}"
    make_lines(root, args.alphas, args.metric, args.top_k, args.labels_only, fig_dir / f"{name}.png")
    print(f"saved {name}.png")
    print(f"DONE -> {fig_dir}")


if __name__ == "__main__":
    main()
