"""
Plot the six-token interval-patching study (pure plotting, no GPU). Reads the arrays written by
patch_interval_sixtoken.py and renders:

  1. combined_logit_shift_heatmap.png -- the 6x6 logit-flip shift grid for both task pairs side by
     side with a SHARED symmetric colour scale (rows = i = token switched to target; cols = j = token
     pinned to original; cell = mean steered - baseline logit_diff). Upper triangle (j>i) only.

  2. <pair>_downstream_propagation.png -- for one task pair, a 6x6 upper-triangle facet grid; panel
     (i,j) is a heatmap of downstream cosine change dcos(k, L) for each downstream token k>j (rows)
     across residual entries L=7..28 (cols). Shared per-task diverging scale. Positive => the patch
     pushed token k toward the target prompt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import LABEL_GEOMETRY_DIR

TASKS = ["antonym_synonym", "next_number_digits_prev_number_digits"]
TASK_LABEL = {"antonym_synonym": "antonym→synonym",
              "next_number_digits_prev_number_digits": "prev→next (digits)"}
TOKEN_NAMES = ["demo1 label", "demo2 input", "demo2 pre label", "demo2 label",
               "query input", "query pre label"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "twoshot" / "interval_patch_sixtoken"))
    p.add_argument("--regime", type=str, default="L6_and_above",
                   help="layer-regime subfolder under --root (e.g. L6_and_above, all_layers)")
    return p.parse_args()


def plot_combined_logit(root, summaries):
    grids = {t: np.load(root / f"{t}_logit_shift_grid.npy") for t in TASKS}
    vmax = max(float(np.nanmax(np.abs(g))) for g in grids.values()) or 1e-6
    fig, axes = plt.subplots(1, len(TASKS), figsize=(5.8 * len(TASKS), 5.4),
                             squeeze=False, constrained_layout=True)
    im = None
    for c, t in enumerate(TASKS):
        ax = axes[0][c]
        g = grids[t]
        im = ax.imshow(g, origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(6)); ax.set_xticklabels(TOKEN_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(6))
        ax.set_xlabel("j  (token pinned to original)")
        if c == 0:
            ax.set_yticklabels(TOKEN_NAMES)
            ax.set_ylabel("i  (token switched to target)")
        else:
            ax.set_yticklabels([])
        base = summaries[t]["baseline_logit_diff"]
        ax.set_title(f"{TASK_LABEL[t]}  (n={summaries[t]['n_pairs']})\n"
                     f"baseline logit_diff {base:+.2f}", fontsize=9)
        for i in range(6):
            for j in range(6):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85,
                        label="mean (steered − baseline) logit(tgt_gold) − logit(src_gold)  [shared]")
    fig.suptitle("Interval patch: output logit flip per (switch-on i, pin j) token pair", fontsize=12)
    out = root / "figures" / "combined_logit_shift_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (shared vmax={vmax:.3f})")


def plot_downstream(root, t, summary):
    down = np.load(root / f"{t}_downstream_dcos.npy")          # [6,6,6,29]
    n_layers = down.shape[-1]
    first = summary["patch_from_entry"] + 1   # entry == patch onset of k is ~0 by construction
    read_entries = list(range(first, n_layers))
    sub = down[..., first:]
    vmax = float(np.nanmax(np.abs(sub))) or 1e-6

    fig, axes = plt.subplots(6, 6, figsize=(22, 14), squeeze=False, constrained_layout=True)
    im = None
    for i in range(6):
        for j in range(6):
            ax = axes[i][j]
            ks = list(range(j + 1, 6))                          # downstream tokens
            if i >= j or not ks:                                # only upper triangle with a downstream k
                ax.axis("off")
                continue
            mat = down[i, j, ks, :][:, first:]                  # [n_k, n_read_entries]
            im = ax.imshow(mat, origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_yticks(range(len(ks))); ax.set_yticklabels([TOKEN_NAMES[k] for k in ks], fontsize=7)
            ax.set_xticks([0, len(read_entries) - 1]); ax.set_xticklabels([read_entries[0], read_entries[-1]], fontsize=7)
            ax.set_title(f"i = {TOKEN_NAMES[i]}\nj = {TOKEN_NAMES[j]}", fontsize=8)
    for j in range(6):
        axes[5][j].set_xlabel("read entry (7..28)", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Δcos(steered_k, tgt_k) − cos(base_k, tgt_k)")
    # figure-level axes: panel ROW = i (switched to target), panel COL = j (pinned to original)
    fig.supylabel("panel row  =  i : token switched to target  (earlier → later in prompt, top → bottom)",
                  fontsize=12)
    fig.supxlabel("panel column  =  j : token pinned to original  (earlier → later in prompt, left → right)",
                  fontsize=12)
    fig.suptitle(f"{TASK_LABEL[t]}: downstream cosine propagation per (switch-on i, pin j) token pair\n"
                 f"within each panel:  x = read entry (7..28),  y = downstream token k>j,  "
                 f"colour = Δcos pushed toward the target prompt", fontsize=13)
    out = root / "figures" / f"{t}_downstream_propagation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}  (vmax={vmax:.3f})")


def main():
    args = parse_args()
    root = Path(args.root) / args.regime
    (root / "figures").mkdir(parents=True, exist_ok=True)
    summaries = {t: json.load(open(root / f"{t}_summary.json")) for t in TASKS}
    plot_combined_logit(root, summaries)
    for t in TASKS:
        plot_downstream(root, t, summaries[t])


if __name__ == "__main__":
    main()
