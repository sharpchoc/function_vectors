#!/usr/bin/env python
"""Line-plot recut of the held-out (token x layer) R^2 poster heatmap.

Same data as plot_token_layer_poster.py (artifacts/69_task_run/token_layer_regressions/
layer<L>.json + layer<L>_input21.json), redrawn in the style of the FV_location
read-vs-write dual figure: layer on x, held-out R^2 on y, one colour per token role
    input  = last token of the demo's input word
    cue    = the ':' immediately before the label
    target = the demo's label (LAST token)

Layer numbering (user decision 2026-08-19): when the embedding-only baseline
(ridge_embedding_baseline.py -> artifacts .../embedding.json) is present, it is plotted
as LAYER 0 (wte lookup, no blocks) and the 28 transformer block outputs are renumbered
1..28; without it the axis is the block index 0..27 as in the heatmap posters.

Two modes:
  * --mode avg (default): ONE line per role, averaged over examples --avg_from..--avg_to
    (default 6-10, where decodability has saturated) — the headline three-line figure.
  * --mode fan: one line per ICL example (1..n_shots), light-to-bold with example index,
    so the sawtooth ratchet reads as a fan of curves; the query cue is the dashed line.

Output: results/69_task_run/token_layer_regressions/poster_visuals/<out_stem>.png (+ .csv)
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
ROLE_LABEL = {"cue": 'cue ":"', "target": "target", "input": "input"}


def load():
    layers = sorted(int(f.stem[5:]) for f in AR.glob("layer*.json")
                    if "_" not in f.stem[5:])
    lab = {l: json.load(open(AR / f"layer{l}.json"))["results"] for l in layers}
    inp = {l: json.load(open(AR / f"layer{l}_input21.json"))["results"] for l in layers}

    def series(role, n):
        key = {"cue": f"d{n}_pre", "target": f"d{n}_last", "input": f"d{n}_inp_last"}[role]
        src = inp if role == "input" else lab
        return np.array([src[l][key]["r2_heldout_mean"] for l in layers])

    query_cue = np.array([lab[l]["query_cue"]["r2_heldout_mean"] for l in layers])

    # embedding-only baseline (ridge_embedding_baseline.py): X = wte[token_id], no blocks
    emb_series = None
    if (AR / "embedding.json").exists() and (AR / "embedding_input21.json").exists():
        elab = json.load(open(AR / "embedding.json"))["results"]
        einp = json.load(open(AR / "embedding_input21.json"))["results"]

        def emb_series(role, n=None):
            if role == "query_cue":
                return elab["query_cue"]["r2_heldout_mean"]
            key = {"cue": f"d{n}_pre", "target": f"d{n}_last",
                   "input": f"d{n}_inp_last"}[role]
            return (einp if role == "input" else elab)[key]["r2_heldout_mean"]

    return layers, series, query_cue, emb_series


def style_axes(ax, xs, emb=False):
    """xs = the x values actually plotted. With the embedding baseline present the axis
    is renumbered: layer 0 = the embedding (wte, no blocks), layers 1..28 = the outputs
    of GPT-J's 28 transformer blocks (user decision 2026-08-19)."""
    ax.set_xlabel("layer (0 = embedding)" if emb else "layer", fontsize=17, labelpad=8)
    ax.set_ylabel("held-out $R^2$", fontsize=17, labelpad=10)
    ax.set_xticks(range(0, len(xs), 2))
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    ax.tick_params(labelsize=13)
    ax.axhline(0, color="0.75", lw=1.0, zorder=1)
    ax.grid(axis="y", color="0.90", lw=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Where the function vector is linearly readable",
                 fontsize=24, fontweight="bold", pad=18, loc="left")


def write_csv(path, layers, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + list(cols))
        for i, l in enumerate(layers):
            w.writerow([l] + [f"{cols[c][i]:.4f}" for c in cols])


def main_avg(avg_from=6, avg_to=10, out_stem="heldout_r2_lines"):
    layers, series, _, emb_series = load()
    avg = {role: np.mean([series(role, n) for n in range(avg_from, avg_to + 1)], axis=0)
           for role in ("input", "cue", "target")}
    # embedding = layer 0, blocks = layers 1..28
    have_emb = emb_series is not None
    xs = list(range(len(layers) + 1)) if have_emb else list(layers)
    if have_emb:
        avg = {role: np.concatenate(
                   [[np.mean([emb_series(role, n) for n in range(avg_from, avg_to + 1)])],
                    avg[role]]) for role in avg}

    fig, ax = plt.subplots(figsize=(13.5, 8.0), dpi=200)
    for role in ("input", "target", "cue"):
        ax.plot(xs, avg[role], color=ROLE_COLOR[role], lw=4.2,
                solid_capstyle="round", zorder=4 if role == "cue" else 3)

    pk = int(np.argmax(avg["cue"]))

    style_axes(ax, xs, emb=have_emb)
    ax.set_ylim(min(float(min(a.min() for a in avg.values())), 0.0) - 0.03, 0.78)
    handles = [Line2D([], [], color=ROLE_COLOR[r], lw=4.2, label=ROLE_LABEL[r])
               for r in ("cue", "target", "input")]
    ax.legend(handles=handles, title="token type", loc="upper left",
              bbox_to_anchor=(0.015, 0.99), fontsize=15, title_fontsize=15,
              frameon=True, framealpha=0.92, edgecolor="none", alignment="left")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{out_stem}.png", bbox_inches="tight")
    plt.close(fig)
    write_csv(OUT / f"{out_stem}.csv", xs, avg)
    print(f"wrote {OUT / out_stem}.png/.csv  (examples {avg_from}-{avg_to}; "
          f"cue peak x={xs[pk]} = {avg['cue'][pk]:.3f}"
          + (f"; emb cue {avg['cue'][0]:.3f} target {avg['target'][0]:.3f} "
             f"input {avg['input'][0]:.3f}" if have_emb else "") + ")")


def main_fan(n_shots=6, out_stem="heldout_r2_lines_6shot"):
    layers, series, query_cue, emb_series = load()
    # embedding = layer 0, blocks = layers 1..28
    have_emb = emb_series is not None
    xs = list(range(len(layers) + 1)) if have_emb else list(layers)

    def full(role, n):
        s = series(role, n)
        return np.concatenate([[emb_series(role, n)], s]) if have_emb else s

    fig, ax = plt.subplots(figsize=(13.5, 8.0), dpi=200)
    for role in ("input", "target", "cue"):  # cue drawn last so the bold line sits on top
        for n in range(1, n_shots + 1):
            frac = (n - 1) / max(n_shots - 1, 1)
            last = n == n_shots
            ax.plot(xs, full(role, n), color=ROLE_COLOR[role],
                    alpha=0.18 + 0.42 * frac if not last else 1.0,
                    lw=1.2 + 0.8 * frac if not last else 4.0,
                    solid_capstyle="round", zorder=3 + (2 if last else frac))
    qc = (np.concatenate([[emb_series("query_cue")], query_cue])
          if have_emb else query_cue)
    ax.plot(xs, qc, color=ROLE_COLOR["cue"], lw=3.0, ls=(0, (5, 2.2)),
            alpha=0.9, zorder=4)

    # peak of the bold cue line
    cue_last = full("cue", n_shots)
    pk = int(np.argmax(cue_last))
    ax.scatter([xs[pk]], [cue_last[pk]], s=130, color=ROLE_COLOR["cue"], zorder=7)
    ax.annotate(f"peaks · L{xs[pk]}  (cue, example {n_shots})",
                xy=(xs[pk], cue_last[pk]), xytext=(xs[pk] + 2.2, cue_last[pk] + 0.06),
                fontsize=17, fontweight="bold", color=ROLE_COLOR["cue"],
                arrowprops=dict(arrowstyle="-", lw=1.4, color=ROLE_COLOR["cue"],
                                connectionstyle="arc3,rad=-0.25"))
    # direct-label the dead example-1 cue line: before any label exists there is
    # nothing to read at the cue, so the fan starts at example 2
    cue_1 = full("cue", 1)
    x_lab = 6 + (1 if have_emb else 0)
    ax.annotate("example 1 cue  (no label seen yet)",
                xy=(x_lab - 0.2, cue_1[x_lab]), xytext=(x_lab + 1.8, 0.20),
                fontsize=14, color=ROLE_COLOR["cue"], alpha=0.85,
                arrowprops=dict(arrowstyle="-", lw=1.2, color=ROLE_COLOR["cue"],
                                alpha=0.5, connectionstyle="arc3,rad=0.25"))

    style_axes(ax, xs, emb=have_emb)
    ax.set_ylim(-0.32, 0.78)
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

    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{out_stem}.png", bbox_inches="tight")
    plt.close(fig)
    cols = {f"{r}_ex{n}": full(r, n) for r in ("input", "cue", "target")
            for n in range(1, n_shots + 1)}
    cols["query_cue"] = qc
    write_csv(OUT / f"{out_stem}.csv", xs, cols)
    print(f"wrote {OUT / out_stem}.png/.csv  (peak L{xs[pk]} = {cue_last[pk]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("avg", "fan"), default="avg")
    ap.add_argument("--avg_from", type=int, default=6)
    ap.add_argument("--avg_to", type=int, default=10)
    ap.add_argument("--n_shots", type=int, default=6)
    ap.add_argument("--out_stem", default=None)
    a = ap.parse_args()
    if a.mode == "avg":
        main_avg(avg_from=a.avg_from, avg_to=a.avg_to,
                 out_stem=a.out_stem or "heldout_r2_lines")
    else:
        main_fan(n_shots=a.n_shots, out_stem=a.out_stem or "heldout_r2_lines_6shot")
