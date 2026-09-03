#!/usr/bin/env python
"""Heatmaps for the (token position x layer) ridge sweep (token_layer_fv_regression.py).

Reads artifacts/69_task_run/token_layer_regressions/layer<L>.json and writes to
results/69_task_run/token_layer_regressions/:
  heldout_r2_heatmap.png   token position (rows) x layer (cols), colour = mean per-task R^2
                           on the 14 HELD-OUT tasks (target = task FV, split-pool-mean
                           denominator) — the headline
  train_r2_heatmap.png     same metric in-sample on the 55 train tasks
  r2_grid.csv              long-format table: layer, position, heldout R^2, train R^2,
                           chosen alpha, alpha_pinned
  best_cells.txt           top cells by held-out R^2 and the per-layer/per-position maxima
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "token_layer_regressions"
OUT = TASK69_RUN_DIR / "FV_linear_decodability" / "token_layer_regressions"


def main():
    files = sorted((f for f in AR.glob("layer*.json") if f.stem[5:].isdigit()),   # skip later per-position files
                   key=lambda p: int(p.stem[5:]))
    assert files, f"no layer jsons in {AR}"
    layers = [int(f.stem[5:]) for f in files]
    data = {l: json.load(open(f)) for l, f in zip(layers, files)}
    positions = data[layers[0]]["positions"]
    print(f"{len(layers)} layers x {len(positions)} positions")

    H = np.full((len(positions), len(layers)), np.nan)
    T = np.full_like(H, np.nan)
    rows = []
    for li, l in enumerate(layers):
        for pi, pname in enumerate(positions):
            r = data[l]["results"][pname]
            H[pi, li] = r["r2_heldout_mean"]
            T[pi, li] = r["r2_train_mean"]
            rows.append([l, pname, r["r2_heldout_mean"], r["r2_train_mean"],
                         r["best_alpha"], r["alpha_pinned"]])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "r2_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "position", "r2_heldout_mean", "r2_train_mean",
                    "best_alpha", "alpha_pinned"])
        w.writerows(rows)

    def heat(M, title, fname, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=150)
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(layers)), [str(l) for l in layers], fontsize=7.5)
        ax.set_yticks(range(len(positions)), positions, fontsize=7)
        ax.set_xlabel("layer")
        ax.set_ylabel("token position")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, label="held-out $R^2$")
        bi = np.unravel_index(np.nanargmax(M), M.shape)   # reported, not drawn
        fig.tight_layout()
        fig.savefig(OUT / fname, bbox_inches="tight")
        return bi

    vmax = float(np.nanmax(T))
    bh = heat(H, "Where the write feature is linearly decodable (held-out $R^2$)",
              "heldout_r2_heatmap.png", vmin=0.0, vmax=vmax)
    bt = heat(T, "Same map scored on the 55 train tasks ($R^2$)",
              "train_r2_heatmap.png", vmin=0.0, vmax=vmax)

    order = np.argsort(H, axis=None)[::-1]
    with open(OUT / "best_cells.txt", "w") as f:
        f.write("top 15 cells by held-out R^2 (target = task FV)\n")
        for k in order[:15]:
            pi, li = np.unravel_index(k, H.shape)
            f.write(f"  L{layers[li]:<2d} {positions[pi]:12s} heldout {H[pi, li]:.4f}  "
                    f"train {T[pi, li]:.4f}\n")
        f.write(f"\nbest held-out cell: L{layers[bh[1]]} {positions[bh[0]]} "
                f"= {H[bh]:.4f}\n")
        f.write(f"best train cell:    L{layers[bt[1]]} {positions[bt[0]]} = {T[bt]:.4f}\n")
        f.write("\nbest position per layer (held-out):\n")
        for li, l in enumerate(layers):
            pi = int(np.nanargmax(H[:, li]))
            f.write(f"  L{l:<2d} {positions[pi]:12s} {H[pi, li]:.4f}\n")
        f.write("\nbest layer per position (held-out):\n")
        for pi, pname in enumerate(positions):
            li = int(np.nanargmax(H[pi]))
            f.write(f"  {pname:12s} L{layers[li]:<2d} {H[pi, li]:.4f}\n")
    print(open(OUT / "best_cells.txt").read()[:900])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
