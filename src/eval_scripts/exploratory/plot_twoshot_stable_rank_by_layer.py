#!/usr/bin/env python
"""Stable rank per layer of the stacked function-difference matrix D = act(f1) - act(f2), for the
TWO-shot paired captures, per task pair and token position.

For each task pair (f1,f2), token position (the 5 two-shot roles), and layer L: stack the per-prompt
difference vectors D[k] = act_f1(k)[L] - act_f2(k)[L] over prompt-keys k=(label1,label2,query),
UNIT-NORMALIZE the rows, and report stable rank = Σσ²/σ₁² (= W/σ₁² for unit rows; low => one dominant
axis). All prompts (no judge split).

One figure, one panel per task pair (x=layer, one line per role). The mean-pairwise-cosine view is
intentionally NOT plotted here -- it is identical to the cosine heatmap (twoshot/diffcos_heatmap/).

Reads ARTIFACTS_ROOT/twoshot_paired_graded/<pair>/. Writes
RESULTS direction2_label_geometry/twoshot/stable_rank/stable_rank.png + stable_rank_by_layer.json.
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
ROLE_COLORS = {"demo1_prelabel": "#9467bd", "demo1_label": "#1f77b4", "demo2_prelabel": "#17becf",
               "demo2_label": "#2ca02c", "query_final": "#d62728"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_root", type=Path, default=ARTIFACTS_ROOT / "twoshot_paired_graded")
    p.add_argument("--pairs", nargs="+",
                   default=["antonym_synonym", "next_number_digits_prev_number_digits"])
    p.add_argument("--out_dir", type=Path, default=LABEL_GEOMETRY_DIR / "twoshot" / "stable_rank")
    return p.parse_args()


def load_pair(graded_dir):
    cfg = json.loads((graded_dir / "index.json").read_text())["config"]
    f1, f2 = cfg["function_tasks"]["f1"], cfg["function_tasks"]["f2"]
    acts, n_layers = {}, None
    for sp in sorted(glob.glob(str(graded_dir / "shard_*.pt"))):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        A = d["activations"].to(torch.float32).numpy()
        n_layers = A.shape[1]
        for i, m in enumerate(d["metadata"]):
            acts[(m["role"], m["function_task"], (m["label1"], m["label2"], m["query_word"]))] = A[i]
    return f1, f2, acts, n_layers


def stable_rank(D):
    """unit-normalize rows; return stable rank Σσ²/σ₁²."""
    n = np.linalg.norm(D, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = D / n
    sv = np.linalg.svd(U, compute_uv=False)
    sv2 = sv ** 2
    return float(sv2.sum() / sv2[0]) if sv2[0] > 0 else 0.0


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump = {}

    fig, axes = plt.subplots(1, len(args.pairs), figsize=(7 * len(args.pairs), 5))
    axes = np.atleast_1d(axes)
    for pi, pair in enumerate(args.pairs):
        f1, f2, acts, n_layers = load_pair(args.graded_root / pair)
        layers = list(range(n_layers))
        series = {}
        for role in ROLES:
            keys = sorted({k for (r, f, k) in acts if r == role and f == f1}
                          & {k for (r, f, k) in acts if r == role and f == f2})
            d1 = np.stack([acts[(role, f1, k)] for k in keys], axis=0)
            d2 = np.stack([acts[(role, f2, k)] for k in keys], axis=0)
            D = d1 - d2
            srs = [stable_rank(D[:, L, :]) for L in range(n_layers)]
            series[role] = {"n": len(keys), "sr": srs}

        ax = axes[pi]
        for role in ROLES:
            ax.plot(layers, series[role]["sr"], color=ROLE_COLORS[role], marker="o", ms=2.5,
                    label=f"{role} (n={series[role]['n']})")
        ax.set_title(f"{pair}\nD = {f1} − {f2}")
        ax.set_xlabel("layer")
        ax.set_ylabel("stable rank  Σσ²/σ₁²  (low => one dominant axis)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        dump[pair] = {"f1": f1, "f2": f2, "layers": layers, "roles": ROLES, "series": series}

    fig.suptitle("Two-shot function-difference stable rank (unit-normalised diffs) per layer & token position "
                 "(all prompts)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = args.out_dir / "stable_rank.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")

    (args.out_dir / "stable_rank_by_layer.json").write_text(json.dumps(dump, indent=2))
    print(f"wrote {args.out_dir / 'stable_rank_by_layer.json'}")
    for pair, o in dump.items():
        print(f"  {pair}: stable rank @ L9 — "
              + ", ".join(f"{r}={o['series'][r]['sr'][9]:.2f}" for r in ROLES))


if __name__ == "__main__":
    main()
