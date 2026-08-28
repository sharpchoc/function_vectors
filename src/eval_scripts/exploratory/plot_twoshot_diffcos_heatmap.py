#!/usr/bin/env python
"""Heatmap of MEAN pairwise cosine among function-difference vectors D = act(f1) - act(f2),
across token position (the 5 two-shot capture roles) x layer, per task pair.

Generalizes plot_pairwise_cos_hist_byjudge.py: that script histograms the pairwise cosine of D at
a single layer and two positions; here we collapse each histogram to its mean and sweep the full
layer x position grid. For each prompt-key k = (label1, label2, query) we form D[k] = act_f1(k) -
act_f2(k) at a given (role, layer), unit-normalize the rows, and take the mean over all pairwise
cosines (upper triangle of U @ U.T). All prompts (no judge split). Independent color scale per pair.

Reads ARTIFACTS_ROOT/twoshot_paired_graded/<pair>/shard_*.pt (+ index.json for f1/f2 names).
Writes RESULTS direction2_label_geometry/twoshot_diffcos_heatmap/<pair>_meancos_heatmap.png and a
combined meancos_grid.json.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR

ROLES = ["demo1_prelabel", "demo1_label", "demo2_prelabel", "demo2_label", "query_final"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_root", type=Path, default=ARTIFACTS_ROOT / "twoshot_paired_graded")
    p.add_argument("--pairs", nargs="+",
                   default=["antonym_synonym", "next_number_digits_prev_number_digits"])
    p.add_argument("--out_dir", type=Path, default=LABEL_GEOMETRY_DIR / "twoshot" / "diffcos_heatmap")
    return p.parse_args()


def mean_pairwise_cos(M):
    """Mean of the upper-triangle pairwise cosine among rows of M (unit-normalized)."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = M / n
    C = U @ U.T
    iu = np.triu_indices(C.shape[0], k=1)
    return float(C[iu].mean())


def load_pair(graded_dir):
    cfg = json.loads((graded_dir / "index.json").read_text())["config"]
    f1, f2 = cfg["function_tasks"]["f1"], cfg["function_tasks"]["f2"]
    # acts[(role, function_task, (label1,label2,query))] -> (n_layers, hidden) float32
    acts, n_layers = {}, None
    for sp in sorted(glob.glob(str(graded_dir / "shard_*.pt"))):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        A = d["activations"].to(torch.float32).numpy()
        n_layers = A.shape[1]
        for i, m in enumerate(d["metadata"]):
            key = (m["role"], m["function_task"], (m["label1"], m["label2"], m["query_word"]))
            acts[key] = A[i]
    return f1, f2, acts, n_layers


def build_grid(f1, f2, acts, n_layers):
    """Return [len(ROLES) x n_layers] mean-pairwise-cos grid and the per-role n_keys."""
    grid = np.full((len(ROLES), n_layers), np.nan)
    n_keys = {}
    for ri, role in enumerate(ROLES):
        keys = sorted({k for (r, f, k) in acts if r == role and f == f1}
                      & {k for (r, f, k) in acts if r == role and f == f2})
        n_keys[role] = len(keys)
        # stack all D vectors once per layer
        d1 = np.stack([acts[(role, f1, k)] for k in keys], axis=0)  # (n_keys, n_layers, hidden)
        d2 = np.stack([acts[(role, f2, k)] for k in keys], axis=0)
        D = d1 - d2
        for L in range(n_layers):
            grid[ri, L] = mean_pairwise_cos(D[:, L, :])
    return grid, n_keys


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = {}

    for pair in args.pairs:
        f1, f2, acts, n_layers = load_pair(args.graded_root / pair)
        grid, n_keys = build_grid(f1, f2, acts, n_layers)

        fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.34), 3.6))
        im = ax.imshow(grid, aspect="auto", cmap="viridis",
                       vmin=np.nanmin(grid), vmax=np.nanmax(grid))  # independent scale per pair
        ax.set_yticks(range(len(ROLES)))
        ax.set_yticklabels(ROLES)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(range(n_layers), fontsize=7)
        ax.set_xlabel("layer")
        ax.set_title(f"{pair} — mean pairwise cosine of D = act({f1}) − act({f2})  (all prompts, "
                     f"n={n_keys[ROLES[0]]})")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("mean pairwise cosine")
        # annotate values
        for ri in range(len(ROLES)):
            for L in range(n_layers):
                ax.text(L, ri, f"{grid[ri, L]:.2f}", ha="center", va="center",
                        fontsize=4.5, color="white")
        fig.tight_layout()
        out_png = args.out_dir / f"{pair}_meancos_heatmap.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"wrote {out_png}")

        out_json[pair] = {"f1": f1, "f2": f2, "roles": ROLES, "layers": list(range(n_layers)),
                          "n_keys": n_keys, "mean_cos_grid": grid.tolist()}

    (args.out_dir / "meancos_grid.json").write_text(json.dumps(out_json, indent=2))
    print(f"wrote {args.out_dir / 'meancos_grid.json'}")
    # quick console summary: peak cell per pair
    for pair, o in out_json.items():
        g = np.array(o["mean_cos_grid"])
        ri, L = np.unravel_index(np.nanargmax(g), g.shape)
        print(f"  {pair}: peak mean cos {g[ri, L]:.3f} at role={o['roles'][ri]} L{L}")


if __name__ == "__main__":
    main()
