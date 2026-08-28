#!/usr/bin/env python
"""Stable rank (and mean pairwise cosine) per layer of the stacked function-difference matrix
D = act(f1) - act(f2), per task pair and per token position, with ALL-words and GPT-4-correct lines.

For each task pair (f1,f2), token position (label/final), and layer L: build the matrix whose rows
are the per-word difference vectors D[w] = act_f1(w) - act_f2(w) at layer L, unit-normalize the rows,
and report:
    stable rank  = Σσ² / σ₁²   ( = W/σ₁² for unit rows; low => one dominant axis)
    mean pairwise cosine of the unit rows.
Two line groups per pair: ALL words (solid) and GPT-4-judge-correct-under-BOTH-functions (dashed).

Two figures (label token, final query token), each with [stable rank | mean pairwise cosine] subplots.
Reads ARTIFACTS_ROOT/oneshot_paired_graded/<pair>/. Overwrites the per-position compare figures.
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

ROLES = [("source", "label token", "label"), ("target", "final query token", "final")]
COLORS = {"antonym_synonym": "#1f77b4", "next_number_prev_number": "#d62728"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_root", type=Path, default=ARTIFACTS_ROOT / "oneshot_paired_graded")
    p.add_argument("--pairs", nargs="+", default=["antonym_synonym", "next_number_prev_number"])
    p.add_argument("--out_dirs", type=Path, nargs="+",
                   default=[LABEL_GEOMETRY_DIR / "oneshot_paired_analysis", LABEL_GEOMETRY_DIR / "oneshot_paired_diff_geometry"])
    return p.parse_args()


def load_pair(graded_dir):
    cfg = json.loads((graded_dir / "index.json").read_text())["config"]
    f1, f2 = cfg["function_tasks"]["f1"], cfg["function_tasks"]["f2"]
    acts, judge, n_layers = {}, {}, None
    for sp in sorted(glob.glob(str(graded_dir / "shard_*.pt"))):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        A = d["activations"].to(torch.float32).numpy()
        n_layers = A.shape[1]
        for i, m in enumerate(d["metadata"]):
            acts[(m["role"], m["function_task"], m["output_word"])] = A[i]
            if "judge_top1" in m:
                judge[(m["function_task"], m["output_word"])] = bool(m["judge_top1"])
    return f1, f2, acts, judge, n_layers


def metrics(D):
    """unit-normalize rows; return (stable_rank, mean_pairwise_cos)."""
    n = np.linalg.norm(D, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = D / n
    sv = np.linalg.svd(U, compute_uv=False)
    sv2 = sv ** 2
    sr = float(sv2.sum() / sv2[0]) if sv2[0] > 0 else 0.0
    C = U @ U.T
    iu = np.triu_indices(C.shape[0], k=1)
    pc = float(C[iu].mean()) if iu[0].size else 0.0
    return sr, pc


def main():
    args = parse_args()
    # gather data
    data = {}
    n_layers = None
    for pair in args.pairs:
        f1, f2, acts, judge, n_layers = load_pair(args.graded_root / pair)
        data[pair] = (f1, f2, acts, judge)

    series = {}  # series[(pair, role, subset)] = dict(layers, sr[], pc[])
    for pair in args.pairs:
        f1, f2, acts, judge = data[pair]
        for role, _, _ in ROLES:
            words = sorted({w for (r, f, w) in acts if r == role and f == f1}
                           & {w for (r, f, w) in acts if r == role and f == f2})
            correct = [w for w in words if judge.get((f1, w)) and judge.get((f2, w))]
            for subset, ws in (("all", words), ("correct", correct)):
                srs, pcs = [], []
                for L in range(n_layers):
                    D = np.stack([acts[(role, f1, w)][L] - acts[(role, f2, w)][L] for w in ws], axis=0)
                    sr, pc = metrics(D)
                    srs.append(sr); pcs.append(pc)
                series[(pair, role, subset)] = {"n": len(ws), "sr": srs, "pc": pcs}

    layers = list(range(n_layers))
    for role, role_label, role_tag in ROLES:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for pair in args.pairs:
            col = COLORS.get(pair, None)
            for subset, ls, lw in (("all", "-", 1.8), ("correct", "--", 1.8)):
                s = series[(pair, role, subset)]
                lbl = f"{pair} ({subset}, n={s['n']})"
                axes[0].plot(layers, s["sr"], ls=ls, lw=lw, color=col, marker="o", ms=2.5, label=lbl)
                axes[1].plot(layers, s["pc"], ls=ls, lw=lw, color=col, marker="o", ms=2.5, label=lbl)
        axes[0].set_title(f"Stable rank (unit-normalised diffs) — {role_label}")
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("stable rank  Σσ²/σ₁²")
        axes[1].set_title(f"Mean pairwise cosine of diffs — {role_label}")
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("mean pairwise cosine")
        axes[1].axhline(0, color="0.6", lw=0.8)
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"Function-difference geometry per layer — {role_label} "
                     f"(corrected shared-input capture; all vs GPT-4-correct)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        for od in args.out_dirs:
            od.mkdir(parents=True, exist_ok=True)
            fig.savefig(od / f"fig_compare_{role_tag}.png", dpi=150)
        plt.close(fig)
        print(f"wrote fig_compare_{role_tag}.png to {[str(o) for o in args.out_dirs]}")

    # dump series
    dump = {f"{p}|{r}|{s}": series[(p, r, s)] for (p, r, s) in series}
    with open(args.out_dirs[0] / "stable_rank_by_layer_byjudge.json", "w") as f:
        json.dump(dump, f, indent=2)


if __name__ == "__main__":
    main()
