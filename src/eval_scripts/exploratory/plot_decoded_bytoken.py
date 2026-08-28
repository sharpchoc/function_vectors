"""
By-token scatter set for the DECODED FVs: at a fixed layer, one subplot per token position
(demo1-4 x {pre,first,last} + query), decoded-FV projection onto mag FV (x) vs id FV (y),
colored by task. Reads the cached projections from decode_fv_map_and_project.py.
"""
import os, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
PROJ = torch.load(REPO / "results" / "magid_decoded_varicl40" / "projections.pt", weights_only=False)
FIG = REPO / "figures"
COLORS = {"magnitude": "tab:red", "identity": "tab:blue"}
LAYERS = [13, 20]


def main():
    res, positions = PROJ["res"], PROJ["positions"]
    for L in LAYERS:
        allx = np.concatenate([res[p][t]["x"][:, L] for p in positions for t in COLORS])
        ally = np.concatenate([res[p][t]["y"][:, L] for p in positions for t in COLORS])
        xlim = (allx.min(), allx.max()); ylim = (ally.min(), ally.max())
        ncols = 4; nrows = int(np.ceil(len(positions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
        for k, p in enumerate(positions):
            ax = axes[k // ncols][k % ncols]
            for t in COLORS:
                ax.scatter(res[p][t]["x"][:, L], res[p][t]["y"][:, L], s=9, alpha=0.5,
                           color=COLORS[t], edgecolors="none", label=f"{t} task" if k == 0 else None)
            lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
            ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=0.7)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(p + (" *diff*" if p.startswith("demo4") else ""), fontsize=9)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("dec·mag FV", fontsize=6); ax.set_ylabel("dec·id FV", fontsize=6)
        for j in range(len(positions), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, markerscale=2, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Layer {L}: DECODED FV (varicl_top40 icl10 map) projected onto mag/id FVs, "
                     f"by token position — colored by task", fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0.03, 1, 0.99])
        out = FIG / f"decoded_varicl40_bytoken_L{L}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
