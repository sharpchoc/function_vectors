"""
Combine the two-shot token-pair cosine-shift grids into the 15 per-token-pair figures (one per
ordered token-pair), plus the 120 individual single-panel heatmaps. Pure plotting -- reads the saved
grids (no model / GPU). See steer_twoshot_tokenpair_cos_heatmap.py for how the grids are produced.

Each of the 15 combined figures has 8 panels = 4 steering combinations (rows) × 2 α (cols), on a
shared symmetric diverging scale (within the figure) for direct comparison:
    rows: antonym→synonym, synonym→antonym, prev→next (digits), next→prev (digits)
    cols: α=2, α=4
The demo-2 input token (input2 / t2) is the one token that differs across functions; figures whose
source OR read token is input2 are flagged in the suptitle (steer dir mixes lexical+function; the read
baseline cosine is < 1, unlike the 5 clean tokens).
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

TOKENS = ["label1", "input2", "prelabel2", "label2", "qinput", "qfinal"]
TOK_LABEL = {"label1": "label-1", "input2": "input-2", "prelabel2": "pre-label-2",
             "label2": "label-2", "qinput": "query-input", "qfinal": "query-final"}
SHORT = {"label1": "L1", "input2": "in2", "prelabel2": "pre2", "label2": "L2",
         "qinput": "qIn", "qfinal": "qFin"}

# 4 steering combinations (figure rows): (task_pair, direction_name, row label)
COMBOS = [
    ("antonym_synonym", "antonym_to_synonym", "antonym→synonym"),
    ("antonym_synonym", "synonym_to_antonym", "synonym→antonym"),
    ("next_number_digits_prev_number_digits", "prev_number_digits_to_next_number_digits", "prev→next (digits)"),
    ("next_number_digits_prev_number_digits", "next_number_digits_to_prev_number_digits", "next→prev (digits)"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "twoshot_tokenpair_intervention_cos_heatmap"))
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    return p.parse_args()


def grid_path(root, task_pair, dir_name, src_t, read_t, alpha):
    return Path(root) / task_pair / f"{dir_name}__{src_t}_to_{read_t}_alpha{alpha:g}_grid.npy"


def peak(grid):
    fk, fi = np.unravel_index(np.nanargmax(grid), grid.shape)
    return float(grid[fk, fi]), int(fi), int(fk)


def token_pairs():
    return [(TOKENS[i], TOKENS[j]) for i in range(len(TOKENS) - 1) for j in range(i + 1, len(TOKENS))]


def load_grid(root, task_pair, dir_name, src_t, read_t, alpha):
    p = grid_path(root, task_pair, dir_name, src_t, read_t, alpha)
    return np.load(p) if p.exists() else None


def global_vmax(root, alphas):
    """Single |dircos| ceiling across ALL grids (every combo × α × token-pair) -> one shared scale."""
    vmax = 0.0
    for task_pair, dir_name, _ in COMBOS:
        for src_t, read_t in token_pairs():
            for a in alphas:
                g = load_grid(root, task_pair, dir_name, src_t, read_t, a)
                if g is not None:
                    vmax = max(vmax, float(np.nanmax(np.abs(g))))
    return vmax or 1e-6


# rows = intervention/source tokens t1..t5; cols = read tokens t2..t6 (drops the always-empty
# col=label1 / row=qfinal). Cell (r,c) is filled iff read token (c+1) comes after source token r.
SRC_ROWS = list(range(len(TOKENS) - 1))   # 0..4  -> label1..qinput
READ_COLS = list(range(1, len(TOKENS)))   # 1..5  -> input2..qfinal


def make_matrix_figure(root, task_pair, dir_name, dir_label, alpha, vmax, out_path):
    """6-token×6-token grid (5×5 after trimming empties): each cell a 29×29 layer×layer heatmap,
    all on one shared scale, single colorbar. The canonical 'every token-pair' view."""
    nr, nc = len(SRC_ROWS), len(READ_COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc + 1.2, 2.05 * nr + 0.6),
                             squeeze=False, constrained_layout=True)
    im = None
    n_layers = 29
    for r in SRC_ROWS:
        for ci, c in enumerate(READ_COLS):
            ax = axes[r][ci]
            src_t, read_t = TOKENS[r], TOKENS[c]
            g = load_grid(root, task_pair, dir_name, src_t, read_t, alpha) if c > r else None
            if g is None:
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():     # keep axis alive (so edge labels render) but invisible
                    s.set_visible(False)
            else:
                n_layers = g.shape[0]
                im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
                ax.plot([0, n_layers - 1], [0, n_layers - 1], color="k", lw=0.5, ls=":", alpha=0.4)
                # per-cell LAYER ticks so each cell's orientation (x=intervene, y=read) is unambiguous
                ax.set_xticks([0, n_layers - 1]); ax.set_yticks([0, n_layers - 1])
                ax.tick_params(labelsize=5, length=2, pad=1)
                ax.set_xlabel("intervene layer", fontsize=6, labelpad=1)
                ax.set_ylabel("read layer", fontsize=6, labelpad=1)
            if ci == 0:                          # left-edge: intervention/source TOKEN (every row)
                ax.annotate(f"intervene\n{TOK_LABEL[src_t]}", xy=(-0.5, 0.5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=8, annotation_clip=False, fontweight="bold")
            if r == 0:                           # top-edge: read TOKEN (row 0 is fully filled)
                ax.set_title(f"read {TOK_LABEL[read_t]}", fontsize=8, fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="dircos: cos(counterfactual Δ, steering Δ) (global scale)")
    # explicit per-cell axis KEY in the empty lower-left triangle (clip off so it can spill across cells)
    key = ("Each cell = 29×29 layer grid:\n"
           "  x → intervention layer (0–28)\n"
           "  y ↑ read layer (0–28)\n"
           "signal only ABOVE the dotted diagonal\n"
           "(read layer > intervention layer)")
    axes[len(SRC_ROWS) - 1][0].annotate(key, xy=(0.1, 1.3), xycoords="axes fraction", ha="left",
                                        va="center", fontsize=8.5, annotation_clip=False,
                                        bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#999"))
    fig.suptitle(f"Token×token intervention map — {dir_label}, α={alpha:g}  (2-shot ICL, shared global "
                 f"scale, vmax={vmax:.3f})\nrows = intervention token · columns = read token · "
                 f"each cell x=intervene layer / y=read layer", fontsize=10)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_scalar_overview(root, alphas, vmax, out_path):
    """One figure: for every (combo, α) a small 5×5 matrix whose cells = peak dircos of that token-pair,
    annotated, on one shared 0..vmax scale. At-a-glance 'which token drives which'."""
    nr, nc = len(COMBOS), len(alphas)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc + 1.0, 3.0 * nr),
                             squeeze=False, constrained_layout=True)
    im = None
    row_labels = [SHORT[TOKENS[r]] for r in SRC_ROWS]
    col_labels = [SHORT[TOKENS[c]] for c in READ_COLS]
    for ri, (task_pair, dir_name, dir_label) in enumerate(COMBOS):
        for ai, a in enumerate(alphas):
            ax = axes[ri][ai]
            M = np.full((len(SRC_ROWS), len(READ_COLS)), np.nan)
            for r in SRC_ROWS:
                for ci, c in enumerate(READ_COLS):
                    if c > r:
                        g = load_grid(root, task_pair, dir_name, TOKENS[r], TOKENS[c], a)
                        if g is not None:
                            M[r, ci] = float(np.nanmax(g))
            im = ax.imshow(M, cmap="Reds", vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=7)
            ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_title(f"{dir_label}, α={a:g}", fontsize=8)
            if ai == 0:
                ax.set_ylabel("intervene", fontsize=8)
            if ri == nr - 1:
                ax.set_xlabel("read", fontsize=8)
            for r in range(len(SRC_ROWS)):
                for c in range(len(READ_COLS)):
                    v = M[r, c]
                    if np.isfinite(v):
                        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=6,
                                color="white" if v > 0.55 * vmax else "black")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.7, label="peak dircos (global scale)")
    fig.suptitle("Peak dircos per token-pair — scalar overview (2-shot ICL, shared global scale)", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_combined = n_individual = 0
    for src_t, read_t in token_pairs():
        # load the 8 grids (4 combos × 2 α); track which are present
        loaded = {}      # (combo_idx, alpha) -> grid
        vmax = 0.0
        n_layers = None
        for ci, (task_pair, dir_name, _) in enumerate(COMBOS):
            for a in args.alphas:
                p = grid_path(root, task_pair, dir_name, src_t, read_t, a)
                if not p.exists():
                    continue
                g = np.load(p)
                loaded[(ci, a)] = g
                vmax = max(vmax, float(np.nanmax(np.abs(g))))
                n_layers = g.shape[0]
                # also write the individual single-panel heatmap
                tp_dir = root / task_pair / "figures"
                tp_dir.mkdir(parents=True, exist_ok=True)
                ps, pi, pk = peak(g)
                fig, ax = plt.subplots(figsize=(5.2, 4.6))
                vm = float(np.nanmax(np.abs(g))) or 1e-6
                im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="equal")
                ax.plot([0, n_layers - 1], [0, n_layers - 1], color="k", lw=0.6, ls=":", alpha=0.5)
                ax.set_xlabel(f"intervention layer ({TOK_LABEL[src_t]})")
                ax.set_ylabel(f"read layer ({TOK_LABEL[read_t]})")
                ax.set_title(f"{dir_name}  α={a:g}\npeak {ps:+.3f} @ i{pi}/k{pk}", fontsize=9)
                fig.colorbar(im, ax=ax, label="dircos: cos(counterfactual Δ, steering Δ)")
                fig.tight_layout()
                fig.savefig(tp_dir / f"{dir_name}__{src_t}_to_{read_t}_alpha{a:g}_heatmap.png", dpi=130)
                plt.close(fig)
                n_individual += 1

        if not loaded:
            print(f"  (skip {src_t}->{read_t}: no grids found yet)")
            continue
        vmax = vmax or 1e-6

        # combined 8-panel figure: rows = combos, cols = alphas
        nrows, ncols = len(COMBOS), len(args.alphas)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.7 * nrows),
                                 squeeze=False, constrained_layout=True)
        im = None
        for ci, (task_pair, dir_name, row_label) in enumerate(COMBOS):
            for cj, a in enumerate(args.alphas):
                ax = axes[ci][cj]
                g = loaded.get((ci, a))
                if g is None:
                    ax.set_axis_off()
                    continue
                im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
                ax.plot([0, n_layers - 1], [0, n_layers - 1], color="k", lw=0.6, ls=":", alpha=0.5)
                ps, pi, pk = peak(g)
                ax.set_title(f"{row_label}, α={a:g}  peak {ps:+.3f} @ i{pi}/k{pk}", fontsize=8)
                if ci == nrows - 1:
                    ax.set_xlabel(f"intervention layer ({TOK_LABEL[src_t]})")
                if cj == 0:
                    ax.set_ylabel(f"read layer ({TOK_LABEL[read_t]})")

        caveat = ""
        if src_t == "input2" or read_t == "input2":
            caveat = ("\n(input-2 differs across functions: steer dir mixes lexical+function; "
                      "read baseline cos < 1)")
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.85, label="dircos: cos(counterfactual Δ, steering Δ) (shared scale)")
        fig.suptitle(f"Steer {TOK_LABEL[src_t]} → read {TOK_LABEL[read_t]}  "
                     f"(2-shot ICL, shared colour scale){caveat}", fontsize=12)
        out = fig_dir / f"{src_t}_to_{read_t}_combined.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        n_combined += 1
        print(f"saved {out.name}  (shared vmax={vmax:.4f}, {len(loaded)}/8 panels)")

    # ---- comparable views on ONE global scale ----
    gvmax = global_vmax(root, args.alphas)
    print(f"\nglobal vmax across all grids = {gvmax:.5f}")
    n_matrix = 0
    for task_pair, dir_name, dir_label in COMBOS:
        for a in args.alphas:
            out = fig_dir / f"matrix__{dir_name}_alpha{a:g}.png"
            make_matrix_figure(root, task_pair, dir_name, dir_label, a, gvmax, out)
            n_matrix += 1
            print(f"saved {out.name}")
    make_scalar_overview(root, args.alphas, gvmax, fig_dir / "scalar_overview.png")
    print("saved scalar_overview.png")

    print(f"\nDONE: {n_combined} per-pair combined + {n_matrix} token×token matrix + 1 scalar overview"
          f" + {n_individual} individual panels -> {fig_dir}")


if __name__ == "__main__":
    main()
