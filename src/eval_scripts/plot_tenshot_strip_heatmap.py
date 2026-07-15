"""
Plot the 10-shot intervene-token STRIP study (reads saved grids; CPU-only, no GPU).

Because the read token is fixed (query-final), the token×token matrix collapses to a vertical strip
over the 30 intervene tokens. Two views, both on ONE global colour scale across all grids:
  - scalar_overview.png : rows = 30 intervene tokens, cols = 12 (4 combos × 3 α); each cell = that
    (token, combo, α) peak Δcos, annotated. The headline comparable summary.
  - strip_alpha{a}.png  : per α, cols = 4 combos, rows = 30 intervene tokens; each cell = that
    (token, combo) 29×29 layer heatmap (x=intervene layer, y=read layer), one global diverging colorbar.
See steer_tenshot_strip_cos_heatmap.py for how grids are produced.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import LABEL_GEOMETRY_DIR

N_SHOTS = 10
IKEYS = [f"d{i}_{s}" for i in range(1, N_SHOTS + 1) for s in ("in", "pre", "lab")]  # 30, sequence order

COMBOS = [
    ("antonym_synonym", "antonym_to_synonym", "ant→syn"),
    ("antonym_synonym", "synonym_to_antonym", "syn→ant"),
    ("next_number_digits_prev_number_digits", "prev_number_digits_to_next_number_digits", "prev→next"),
    ("next_number_digits_prev_number_digits", "next_number_digits_to_prev_number_digits", "next→prev"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "tenshot_strip_intervention_cos_heatmap"))
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    return p.parse_args()


def gpath(root, task_pair, dir_name, tkey, a):
    return Path(root) / task_pair / f"{dir_name}__{tkey}_alpha{a:g}_grid.npy"


def load_grid(root, task_pair, dir_name, tkey, a):
    p = gpath(root, task_pair, dir_name, tkey, a)
    return np.load(p) if p.exists() else None


def global_vmax(root, alphas):
    vmax = 0.0
    for task_pair, dir_name, _ in COMBOS:
        for tkey in IKEYS:
            for a in alphas:
                g = load_grid(root, task_pair, dir_name, tkey, a)
                if g is not None:
                    vmax = max(vmax, float(np.nanmax(np.abs(g))))
    return vmax or 1e-6


def make_scalar_overview(root, alphas, vmax, out_path):
    ncol = len(COMBOS) * len(alphas)
    M = np.full((len(IKEYS), ncol), np.nan)
    col_labels = []
    c = 0
    for task_pair, dir_name, clab in COMBOS:
        for a in alphas:
            for r, tkey in enumerate(IKEYS):
                g = load_grid(root, task_pair, dir_name, tkey, a)
                if g is not None:
                    M[r, c] = float(np.nanmax(g))
            col_labels.append(f"{clab}\nα{a:g}")
            c += 1
    fig, ax = plt.subplots(figsize=(0.62 * ncol + 3.0, 0.32 * len(IKEYS) + 1.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(ncol)); ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(IKEYS))); ax.set_yticklabels(IKEYS, fontsize=6)
    ax.set_ylabel("intervene token", fontsize=9)
    for r in range(len(IKEYS)):
        for cc in range(ncol):
            v = M[r, cc]
            if np.isfinite(v):
                ax.text(cc, r, f"{v:.3f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="peak dircos (global scale)")
    ax.set_title("10-shot: peak dircos per intervene token → query-final  (read fixed at qfinal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_strip(root, alpha, vmax, out_path):
    nr, nc = len(IKEYS), len(COMBOS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc + 1.4, 1.15 * nr + 0.8),
                             squeeze=False, constrained_layout=True)
    im = None
    n_layers = 29
    for r, tkey in enumerate(IKEYS):
        for cj, (task_pair, dir_name, clab) in enumerate(COMBOS):
            ax = axes[r][cj]
            g = load_grid(root, task_pair, dir_name, tkey, alpha)
            if g is None:
                ax.set_axis_off()
                continue
            n_layers = g.shape[0]
            im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            ax.plot([0, n_layers - 1], [0, n_layers - 1], color="k", lw=0.4, ls=":", alpha=0.4)
            ax.set_xticks([0, n_layers - 1]); ax.set_yticks([0, n_layers - 1])
            ax.tick_params(labelsize=4, length=1.5, pad=1)
            if cj == 0:
                ax.set_ylabel(tkey, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
            if r == 0:
                ax.set_title(clab, fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("intervene layer", fontsize=5)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.4, label="dircos: cos(counterfactual Δ, steering Δ) (global scale)")
    fig.suptitle(f"10-shot intervene-token strip — α={alpha:g}  (read fixed at query-final; shared global "
                 f"scale vmax={vmax:.3f})\nrows = intervene token · cols = combo · each cell x=intervene "
                 f"layer, y=read layer (0–28), lower-tri≡0", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vmax = global_vmax(root, args.alphas)
    print(f"global vmax = {vmax:.5f}")
    make_scalar_overview(root, args.alphas, vmax, fig_dir / "scalar_overview.png")
    print("saved scalar_overview.png")
    for a in args.alphas:
        make_strip(root, a, vmax, fig_dir / f"strip_alpha{a:g}.png")
        print(f"saved strip_alpha{a:g}.png")
    print(f"DONE -> {fig_dir}")


if __name__ == "__main__":
    main()
