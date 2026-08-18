#!/usr/bin/env python
"""Line-plot recut of the held-out (token x layer) R^2 poster heatmap.

Same data as plot_token_layer_poster.py (artifacts/69_task_run/token_layer_regressions/
layer<L>.json + layer<L>_input21.json), redrawn in the style of the FV_location
read-vs-write dual figure: layer on x, held-out R^2 on y, one colour per token role
    input  = last token of the demo's input word
    cue    = the ':' immediately before the label
    target = the demo's label (LAST token)
and one line per ICL example (1..n_shots), light-to-bold with example index, so the
sawtooth ratchet reads as a fan of curves instead of striped heatmap rows. The query
cue (the canonical FV site) is the dashed line.

Output: results/69_task_run/token_layer_regressions/poster_visuals/
        heldout_r2_lines_6shot.png (+ .csv with the plotted values)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "token_layer_regressions"
OUT = TASK69_RUN_DIR / "token_layer_regressions" / "poster_visuals"

ROLE_COLOR = {"cue": "#2f7fe0", "target": "#e8623d", "input": "#2fae82"}
ROLE_LABEL = {"cue": 'cue ":"', "target": "target / label", "input": "input"}


def main(n_shots=6, out_stem="heldout_r2_lines_6shot"):
    layers = sorted(int(f.stem[5:]) for f in AR.glob("layer*.json")
                    if "_" not in f.stem[5:])
    lab = {l: json.load(open(AR / f"layer{l}.json"))["results"] for l in layers}
    inp = {l: json.load(open(AR / f"layer{l}_input21.json"))["results"] for l in layers}

    def series(role, n):
        key = {"cue": f"d{n}_pre", "target": f"d{n}_last", "input": f"d{n}_inp_last"}[role]
        src = inp if role == "input" else lab
        return np.array([src[l][key]["r2_heldout_mean"] for l in layers])

    query_cue = np.array([lab[l]["query_cue"]["r2_heldout_mean"] for l in layers])

    fig, ax = plt.subplots(figsize=(13.5, 8.0), dpi=200)
    for role in ("input", "target", "cue"):  # cue drawn last so the bold line sits on top
        for n in range(1, n_shots + 1):
            frac = (n - 1) / max(n_shots - 1, 1)
            last = n == n_shots
            ax.plot(layers, series(role, n), color=ROLE_COLOR[role],
                    alpha=0.18 + 0.42 * frac if not last else 1.0,
                    lw=1.2 + 0.8 * frac if not last else 4.0,
                    solid_capstyle="round", zorder=3 + (2 if last else frac))
    ax.plot(layers, query_cue, color=ROLE_COLOR["cue"], lw=3.0, ls=(0, (5, 2.2)),
            alpha=0.9, zorder=4)

    # peak of the bold cue line
    cue_last = series("cue", n_shots)
    pk = int(np.argmax(cue_last))
    ax.scatter([layers[pk]], [cue_last[pk]], s=130, color=ROLE_COLOR["cue"], zorder=7)
    ax.annotate(f"peaks · L{layers[pk]}  (cue, example {n_shots})",
                xy=(layers[pk], cue_last[pk]), xytext=(layers[pk] + 2.2, cue_last[pk] + 0.06),
                fontsize=17, fontweight="bold", color=ROLE_COLOR["cue"],
                arrowprops=dict(arrowstyle="-", lw=1.4, color=ROLE_COLOR["cue"],
                                connectionstyle="arc3,rad=-0.25"))
    # direct-label the dead example-1 cue line: before any label exists there is
    # nothing to read at the cue, so the fan starts at example 2
    cue_1 = series("cue", 1)
    ax.annotate("example 1 cue  (no label seen yet)",
                xy=(5.8, cue_1[layers.index(6)]), xytext=(7.6, 0.20),
                fontsize=14, color=ROLE_COLOR["cue"], alpha=0.85,
                arrowprops=dict(arrowstyle="-", lw=1.2, color=ROLE_COLOR["cue"],
                                alpha=0.5, connectionstyle="arc3,rad=0.25"))

    ax.set_xlabel("layer", fontsize=17, labelpad=8)
    ax.set_ylabel("held-out $R^2$  (activation → FV ridge)", fontsize=17, labelpad=10)
    ax.set_xticks(range(0, len(layers), 2))
    ax.tick_params(labelsize=13)
    ax.set_xlim(layers[0] - 0.4, layers[-1] + 0.4)
    ax.set_ylim(-0.32, 0.78)
    ax.axhline(0, color="0.75", lw=1.0, zorder=1)
    ax.grid(axis="y", color="0.90", lw=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], color=ROLE_COLOR[r], lw=3.6, label=ROLE_LABEL[r])
               for r in ("cue", "target", "input")]
    leg1 = ax.legend(handles=handles, title="token type", loc="upper left",
                     bbox_to_anchor=(0.015, 0.99), fontsize=14, title_fontsize=14,
                     frameon=True, framealpha=0.92, edgecolor="none", alignment="left")
    ax.add_artist(leg1)
    handles2 = [Line2D([], [], color="0.35", lw=1.3, alpha=0.4,
                       label="example 1  (light)"),
                Line2D([], [], color="0.35", lw=4.0, label=f"example {n_shots}  (bold)"),
                Line2D([], [], color="0.35", lw=3.0, ls=(0, (5, 2.2)), label="query cue")]
    ax.legend(handles=handles2, title="line", loc="lower right",
              bbox_to_anchor=(0.99, 0.02), fontsize=14, title_fontsize=14,
              frameon=True, framealpha=0.92, edgecolor="none", alignment="left")

    ax.set_title("Where the function vector is linearly readable",
                 fontsize=24, fontweight="bold", pad=18, loc="left")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{out_stem}.png", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / f"{out_stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = [f"{r}_ex{n}" for r in ("input", "cue", "target")
                for n in range(1, n_shots + 1)] + ["query_cue"]
        w.writerow(["layer"] + cols)
        data = {f"{r}_ex{n}": series(r, n) for r in ("input", "cue", "target")
                for n in range(1, n_shots + 1)}
        data["query_cue"] = query_cue
        for i, l in enumerate(layers):
            w.writerow([l] + [f"{data[c][i]:.4f}" for c in cols])
    print(f"wrote {OUT / out_stem}.png/.csv  (peak L{layers[pk]} = {cue_last[pk]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shots", type=int, default=6)
    ap.add_argument("--out_stem", default="heldout_r2_lines_6shot")
    a = ap.parse_args()
    main(n_shots=a.n_shots, out_stem=a.out_stem)
