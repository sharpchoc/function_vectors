#!/usr/bin/env python
"""Scatter: cos(f1, f2) at the LABEL token vs at the FINAL query token, per shared word,
colored by GPT-4 judge correctness.

Each point is one shared output word w (the demo label). For a given layer:
    x = cosine( antonym activation, synonym activation )  at the demo LABEL token
    y = cosine( antonym activation, synonym activation )  at the FINAL query token
The two prompts share the same label token w and the same query q, differing only in the
demo input, so the cosine measures how similar the two functions' representations are.

Judge correctness is per (function, w): each word carries an antonym verdict and a synonym
verdict (judge_top1, the GPT-4 full-answer grade), so points are colored 4 ways:
    both correct / antonym-only / synonym-only / neither.

Reads the graded+tagged capture results/oneshot_paired_graded/<pair>/shard_*.pt. Overwrites
plots under results/oneshot_paired_analysis/<pair>/.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_dir", type=Path, default=Path("results/oneshot_paired_graded/antonym_synonym"))
    p.add_argument("--out_dir", type=Path, default=Path("results/oneshot_paired_analysis/antonym_synonym"))
    p.add_argument("--layers", type=int, nargs="+", default=[11])
    p.add_argument("--f1", type=str, default="antonym")
    p.add_argument("--f2", type=str, default="synonym")
    return p.parse_args()


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    args = parse_args()
    # acts[(role, function_task, word)] = [n_layers, hidden]; judge[(function_task, word)] = bool
    acts, judge = {}, {}
    for sp in sorted(glob.glob(str(args.graded_dir / "shard_*.pt"))):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        A = d["activations"].to(torch.float32).numpy()
        for i, m in enumerate(d["metadata"]):
            acts[(m["role"], m["function_task"], m["output_word"])] = A[i]
            if "judge_top1" in m:
                judge[(m["function_task"], m["output_word"])] = bool(m["judge_top1"])
    print(f"loaded {len(acts)} activation rows; {len(judge)} judge verdicts")

    words = sorted({w for (r, f, w) in acts if r == "source" and f == args.f1}
                   & {w for (r, f, w) in acts if r == "source" and f == args.f2})
    print(f"{len(words)} shared words")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    CATS = [("both correct", "#2ca02c", "o"),
            (f"{args.f1} only", "#1f77b4", "^"),
            (f"{args.f2} only", "#ff7f0e", "v"),
            ("neither", "#d62728", "x")]

    for L in args.layers:
        x, y, cat = [], [], []
        for w in words:
            xl = cos(acts[("source", args.f1, w)][L], acts[("source", args.f2, w)][L])
            yf = cos(acts[("target", args.f1, w)][L], acts[("target", args.f2, w)][L])
            c1 = judge.get((args.f1, w)); c2 = judge.get((args.f2, w))
            if c1 is None or c2 is None:
                continue
            k = 0 if (c1 and c2) else 1 if c1 else 2 if c2 else 3
            x.append(xl); y.append(yf); cat.append(k)
        x, y, cat = np.array(x), np.array(y), np.array(cat)

        fig, ax = plt.subplots(figsize=(7, 6.5))
        # draw "neither" first (background), correct categories on top so they're visible.
        for k in (3, 2, 1, 0):
            lab, col, mk = CATS[k]
            sel = cat == k
            bg = (k == 3)
            ax.scatter(x[sel], y[sel], s=18 if bg else 40, c=col, marker=mk,
                       alpha=0.30 if bg else 0.9, label=f"{lab} (n={int(sel.sum())})",
                       zorder=1 if bg else 3, linewidths=0.6 if mk == "x" else 0)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], ls="--", c="0.6", lw=1, label="y = x")
        ax.set_xlabel(f"cos({args.f1}, {args.f2})  at LABEL token  (layer {L})")
        ax.set_ylabel(f"cos({args.f1}, {args.f2})  at FINAL query token  (layer {L})")
        ax.set_title(f"{args.f1}–{args.f2}: f1–f2 activation cosine, label vs final (L{L})\n"
                     f"colored by GPT-4 judge top-1 correctness  (n={len(x)} words)")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = args.out_dir / f"fig_cos_label_vs_final_L{L:02d}_byjudge.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

        # also dump the underlying points
        with open(args.out_dir / f"cos_label_vs_final_L{L:02d}_byjudge.json", "w") as f:
            json.dump({"layer": L, "f1": args.f1, "f2": args.f2,
                       "points": [{"cos_label": float(xi), "cos_final": float(yi),
                                   "category": CATS[k][0]} for xi, yi, k in zip(x, y, cat)]}, f, indent=2)


if __name__ == "__main__":
    main()
