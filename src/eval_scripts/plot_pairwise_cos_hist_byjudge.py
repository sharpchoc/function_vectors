#!/usr/bin/env python
"""Histograms of PAIRWISE cosine similarity among the per-word function-difference vectors
D = act(f1) - act(f2), per TASK PAIR and per token position, overlaying ALL words vs the
GPT-4-judge-correct subset.

For a task pair (f1,f2) at a token position (label / final query), each shared word w gives a
difference vector D[w] = act_f1(w) - act_f2(w) at the chosen layer (this is the "function axis"
contribution: the two prompts share the label token and query, differing only in the demo input).
We unit-normalize the D rows and histogram the pairwise cosine over all word pairs. Overlaid: the
subset of words the GPT-4 judge marked top-1 correct under BOTH functions of the pair (judge_top1),
i.e. words the model handles correctly both ways.

4 panels: rows = {antonym_synonym, next_number_prev_number}, cols = {label, final query}.

Reads the graded+tagged captures ARTIFACTS_ROOT/oneshot_paired_graded/<pair>/shard_*.pt (+ index.json
for the f1/f2 task names). Writes LABEL_GEOMETRY_DIR/oneshot_paired_analysis/fig_pairwise_diffcos_hist_L<L>_byjudge.png.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_root", type=Path, default=ARTIFACTS_ROOT / "oneshot_paired_graded")
    p.add_argument("--pairs", nargs="+", default=["antonym_synonym", "next_number_prev_number"])
    p.add_argument("--out_dir", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_paired_analysis")
    p.add_argument("--layer", type=int, default=11)
    return p.parse_args()


ROLES = [("source", "label token"), ("target", "final query token")]


def load_pair(graded_dir):
    cfg = json.loads((graded_dir / "index.json").read_text())["config"]
    f1, f2 = cfg["function_tasks"]["f1"], cfg["function_tasks"]["f2"]
    acts, judge = {}, {}
    for sp in sorted(glob.glob(str(graded_dir / "shard_*.pt"))):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        A = d["activations"].to(torch.float32).numpy()
        for i, m in enumerate(d["metadata"]):
            acts[(m["role"], m["function_task"], m["output_word"])] = A[i]
            if "judge_top1" in m:
                judge[(m["function_task"], m["output_word"])] = bool(m["judge_top1"])
    return f1, f2, acts, judge


def pairwise_cos(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = M / n
    C = U @ U.T
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def main():
    args = parse_args()
    L = args.layer
    fig, axes = plt.subplots(len(args.pairs), len(ROLES), figsize=(13, 5 * len(args.pairs)))
    axes = np.atleast_2d(axes)
    stats = {}

    for ri, pair in enumerate(args.pairs):
        f1, f2, acts, judge = load_pair(args.graded_root / pair)
        for ci, (role, role_label) in enumerate(ROLES):
            words = sorted({w for (r, f, w) in acts if r == role and f == f1}
                           & {w for (r, f, w) in acts if r == role and f == f2})
            correct = [w for w in words if judge.get((f1, w)) and judge.get((f2, w))]

            def diff_stack(ws):
                return np.stack([acts[(role, f1, w)][L] - acts[(role, f2, w)][L] for w in ws], axis=0)

            cos_all = pairwise_cos(diff_stack(words))
            cos_cor = pairwise_cos(diff_stack(correct)) if len(correct) > 1 else np.array([])

            ax = axes[ri, ci]
            lo = min(cos_all.min(), cos_cor.min() if cos_cor.size else cos_all.min())
            bins = np.linspace(lo, 1.0, 60)
            ax.hist(cos_all, bins=bins, density=True, alpha=0.5, color="#7f7f7f",
                    label=f"all words (n={len(words)}, {len(cos_all)} pairs)")
            if cos_cor.size:
                ax.hist(cos_cor, bins=bins, density=True, alpha=0.55, color="#2ca02c",
                        label=f"GPT-4 correct, both fns (n={len(correct)}, {len(cos_cor)} pairs)")
                ax.axvline(cos_cor.mean(), color="#2ca02c", ls="--", lw=1.2)
            ax.axvline(cos_all.mean(), color="#7f7f7f", ls="--", lw=1.2)
            ax.axvline(0, color="0.3", lw=0.8)
            ax.set_title(f"{pair} — {role_label} (L{L})\n"
                         f"D = {f1} - {f2};  mean cos: all={cos_all.mean():.3f}"
                         + (f", correct={cos_cor.mean():.3f}" if cos_cor.size else ""))
            ax.set_xlabel("pairwise cosine of difference vectors")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            stats[f"{pair}/{role}"] = {"n_words": len(words), "n_correct_both": len(correct),
                                       "mean_all": float(cos_all.mean()),
                                       "mean_correct": float(cos_cor.mean()) if cos_cor.size else None}

    fig.suptitle(f"Pairwise cosine of function-difference vectors by task pair & token position (L{L})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"fig_pairwise_diffcos_hist_L{L:02d}_byjudge.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    with open(args.out_dir / f"pairwise_diffcos_hist_L{L:02d}_byjudge.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
