#!/usr/bin/env python
"""Poster-ready version of the held-out (token x layer) R^2 heatmap.

Differences from the analysis figure (plot_token_layer_regressions.py):
  * rows grouped into one block per ICL example, three rows each in reading order
      input  = last token of the demo's input word
      cue    = the ':' immediately before the label
      target = the demo's label (LAST token)
    with white separators between examples and a bracket label per block;
  * only the first --n_shots examples are drawn (default 6) — the pattern repeats and the
    full 10 wastes poster space; the omitted examples stay in r2_grid.csv;
  * colour scale clipped to the range the data actually occupies (nothing reaches 0.8);
  * a highlight box around the cue/target sawtooth;
  * snappy title, large fonts, no chart junk.

Inputs: artifacts/69_task_run/token_layer_regressions/layer<L>.json (label positions) and
layer<L>_input21.json (input positions).
Output: results/69_task_run/token_layer_regressions/poster_visuals/heldout_r2_poster.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "token_layer_regressions"
OUT = TASK69_RUN_DIR / "token_layer_regressions" / "poster_visuals"
ROLE_ROWS = ("input", "cue", "target")


def main(n_shots=4, box_l0=6, box_l1=9):
    layers = sorted(int(f.stem[5:]) for f in AR.glob("layer*.json")
                    if "_" not in f.stem[5:])
    lab = {l: json.load(open(AR / f"layer{l}.json"))["results"] for l in layers}
    inp_files = {l: AR / f"layer{l}_input21.json" for l in layers}
    have_input = all(p.exists() for p in inp_files.values())
    inp = ({l: json.load(open(p))["results"] for l, p in inp_files.items()}
           if have_input else {})
    if not have_input:
        missing = [l for l, p in inp_files.items() if not p.exists()]
        print(f"WARNING: input positions missing for layers {missing}; drawing cue/target only")

    rows, row_labels, block_of = [], [], []
    for n in range(1, n_shots + 1):
        for role in ROLE_ROWS:
            if role == "input":
                if not have_input:
                    continue
                vals = [inp[l][f"d{n}_inp_last"]["r2_heldout_mean"] for l in layers]
            elif role == "cue":
                vals = [lab[l][f"d{n}_pre"]["r2_heldout_mean"] for l in layers]
            else:
                vals = [lab[l][f"d{n}_last"]["r2_heldout_mean"] for l in layers]
            rows.append(vals); row_labels.append(role); block_of.append(n)
    # query block
    if have_input:
        rows.append([inp[l]["query_inp_last"]["r2_heldout_mean"] for l in layers])
        row_labels.append("input"); block_of.append("query")
    rows.append([lab[l]["query_cue"]["r2_heldout_mean"] for l in layers])
    row_labels.append("cue"); block_of.append("query")

    M = np.array(rows)
    # data-fitted scale: nothing reaches 0.8, and the input rows sit at/below zero
    vmin = np.floor(float(np.nanmin(M)) * 20) / 20
    vmax = np.ceil(float(np.nanmax(M)) * 20) / 20
    print(f"grid {M.shape}, colour range {vmin:.2f}-{vmax:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 0.42 * len(rows) + 3.0), dpi=200)
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(layers)), [str(l) for l in layers], fontsize=12)
    ax.set_xlabel("layer", fontsize=16, labelpad=8)
    ax.set_yticks(range(len(rows)), row_labels, fontsize=13)
    ax.tick_params(axis="y", length=0)

    # white separators + block brackets
    blocks = {}
    for i, b in enumerate(block_of):
        blocks.setdefault(b, []).append(i)
    for b, idxs in blocks.items():
        if min(idxs) > 0:
            ax.axhline(min(idxs) - 0.5, color="white", lw=3)
        name = f"example {b}" if b != "query" else "query"
        ax.text(-3.4, np.mean(idxs), name, ha="right", va="center",
                fontsize=13.5, fontweight="bold")
        ax.annotate("", xy=(-2.6, min(idxs) - 0.42), xytext=(-2.6, max(idxs) + 0.42),
                    xycoords=("data", "data"), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=2.2, color="0.25"))

    # Sawtooth highlight: a VERTICAL band over layers box_l0..box_l1, spanning the ICL
    # example blocks only (the query block has no target row, so it carries no tooth).
    # This is where the cue-below-target tooth is clearest: the cue row only catches its
    # target row after several examples (ex1 -0.02 vs 0.46, ex2 0.35 vs 0.53 at L6).
    x0 = layers.index(box_l0) - 0.5
    x1 = layers.index(box_l1) + 0.5
    ex_rows = [i for i, b in enumerate(block_of) if isinstance(b, int)]
    y0, y1 = min(ex_rows) - 0.5, max(ex_rows) + 0.5
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fill=False, edgecolor="#00e5ff", lw=3.4, zorder=5))
    # label inside the box over the dark example-1 input/cue rows (avoids the title)
    ax.text((x0 + x1) / 2, 0.5, "sawtooth", color="#00e5ff", fontsize=16,
            fontweight="bold", ha="center", va="center", zorder=6)

    cb = fig.colorbar(im, ax=ax, pad=0.13, shrink=0.85)
    cb.set_label("held-out $R^2$", fontsize=15)
    cb.ax.tick_params(labelsize=12)
    ax.set_title("Where the function vector is linearly readable",
                 fontsize=21, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "heldout_r2_poster.png", bbox_inches="tight")
    print(f"wrote {OUT / 'heldout_r2_poster.png'}")


if __name__ == "__main__":
    main()
