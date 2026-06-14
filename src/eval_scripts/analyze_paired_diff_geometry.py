"""
Difference-geometry of paired 1-shot activations (capture_oneshot_paired_tasks.py).

For each task pair, each shared OUTPUT word w contributes two activations that differ
only in the ICL demo input: f1 and f2. We form the per-word difference D = act_f1 -
act_f2 and, per layer, characterise the stack D [W, 4096] with the two metrics from the
Stream-E geometry analysis:

  * STABLE RANK  sr(M) = (Σ σ_i²) / σ_1²  = ‖M‖_F² / ‖M‖_2²
      - raw (on D) and after unit-normalising each row (the reported "after normalisation").
        After row-normalisation ‖M‖_F² = W, so sr = W / σ_1²: low => one dominant axis.
  * COSINE SIMILARITY among the unit difference vectors:
      - pairwise mean / std / |mean|  (alignment of the per-word difference directions)
      - alignment with the mean axis: mean over rows of <u_i, unit(mean_i u_i)>
      - first singular-value variance share σ_1²/Σσ² (the dominant-axis energy).

Computed at two token roles:
  label  = last_label_token (the demo label ` w`, the function fingerprint position)
  final  = last_prompt_token (the final query token)

Pure numpy/matplotlib; no model, no baukit, no function vectors. Outputs under
results/oneshot_paired_diff_geometry/: per-pair <pair>_diff_geometry.{json,csv} and
combined comparison figures across pairs.

Run:  python src/eval_scripts/analyze_paired_diff_geometry.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PAIRS = ["antonym_synonym", "synonym_rhyme", "antonym_rhyme", "next_number_prev_number"]
ROLE_MAP = {"label": "last_label_token", "final": "last_prompt_token"}


def load_capture(capture_dir):
    """-> dict[(role, function, word)] = [n_layers, hidden] float32, plus n_layers."""
    index = json.load(open(capture_dir / "index.json"))
    acts, n_layers = {}, None
    for shard in index["shards"]:
        sp = Path(shard)
        if not sp.is_absolute():
            sp = capture_dir / sp.name
        data = torch.load(sp, map_location="cpu", weights_only=False)
        arr = data["activations"].to(torch.float32).numpy()
        if n_layers is None:
            n_layers = arr.shape[1]
        for i, m in enumerate(data["metadata"]):
            acts[(m["role"], m["function"], m["output_word"])] = arr[i]
    return acts, n_layers


def diff_matrix(acts, role, layer):
    """D = A_f1 - A_f2 over output words present under both functions at this role."""
    w1 = {w for (r, f, w) in acts if r == role and f == "f1"}
    w2 = {w for (r, f, w) in acts if r == role and f == "f2"}
    words = sorted(w1 & w2)
    a1 = np.stack([acts[(role, "f1", w)][layer] for w in words], axis=0).astype(np.float64)
    a2 = np.stack([acts[(role, "f2", w)][layer] for w in words], axis=0).astype(np.float64)
    return a1 - a2, len(words)


def stable_rank(M):
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv ** 2
    return float(sv2.sum() / sv2[0]) if sv2[0] > 0 else 0.0, sv2


def cosine_stats(unit_rows):
    """Pairwise cosine summary + alignment with the mean axis, for unit-norm rows."""
    C = unit_rows @ unit_rows.T
    iu = np.triu_indices(C.shape[0], k=1)
    pc = C[iu] if iu[0].size else np.array([0.0])
    mean_axis = unit_rows.mean(axis=0)
    n = np.linalg.norm(mean_axis)
    mean_axis = mean_axis / n if n > 0 else mean_axis
    align = unit_rows @ mean_axis  # signed alignment of each diff with the average diff
    return {
        "pairwise_cos_mean": float(pc.mean()),
        "pairwise_cos_std": float(pc.std()),
        "pairwise_cos_abs_mean": float(np.abs(pc).mean()),
        "align_with_mean_axis_mean": float(align.mean()),
        "mean_axis_resultant_len": float(n),  # |mean of unit vectors|: 1=perfectly aligned, 0=isotropic
    }


def analyze_pair(pair, capture_root):
    acts, n_layers = load_capture(Path(capture_root) / pair)
    out = {role: {} for role in ROLE_MAP}
    W_by_role = {}
    for role_key, role_name in ROLE_MAP.items():
        for layer in range(n_layers):
            D, W = diff_matrix(acts, role_name, layer)
            W_by_role[role_key] = W
            sr_raw, sv2_raw = stable_rank(D)
            norms = np.linalg.norm(D, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Dn = D / norms
            sr_norm, sv2_norm = stable_rank(Dn)
            cs = cosine_stats(Dn)
            out[role_key][str(layer)] = {
                "W": W,
                "stable_rank_raw": sr_raw,
                "stable_rank_norm": sr_norm,
                "first_sv_share_raw": float(sv2_raw[0] / sv2_raw.sum()),
                "first_sv_share_norm": float(sv2_norm[0] / sv2_norm.sum()),
                "row_norm_mean": float(np.linalg.norm(D, axis=1).mean()),
                **cs,
            }
    return out, n_layers, W_by_role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--capture_root", default="results/oneshot_paired_tasks")
    ap.add_argument("--output_root", default="results/oneshot_paired_diff_geometry")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = {}
    n_layers = None
    for pair in args.pairs:
        res, n_layers, W_by_role = analyze_pair(pair, args.capture_root)
        all_results[pair] = res
        with open(out_root / f"{pair}_diff_geometry.json", "w") as f:
            json.dump({"pair": pair, "W_by_role": W_by_role, "roles": res}, f, indent=2)
        # per-pair CSV (label + final stacked)
        with open(out_root / f"{pair}_diff_geometry.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["role", "layer", "W", "stable_rank_raw", "stable_rank_norm",
                         "first_sv_share_norm", "pairwise_cos_mean", "pairwise_cos_std",
                         "align_with_mean_axis_mean", "mean_axis_resultant_len"])
            for role_key in ROLE_MAP:
                for layer in range(n_layers):
                    r = res[role_key][str(layer)]
                    wr.writerow([role_key, layer, r["W"], f"{r['stable_rank_raw']:.4f}",
                                 f"{r['stable_rank_norm']:.4f}", f"{r['first_sv_share_norm']:.4f}",
                                 f"{r['pairwise_cos_mean']:.4f}", f"{r['pairwise_cos_std']:.4f}",
                                 f"{r['align_with_mean_axis_mean']:.4f}", f"{r['mean_axis_resultant_len']:.4f}"])
        L11 = res["label"]["11"]
        print(f"[{pair}] W(label)={W_by_role['label']}  L11 label: "
              f"stable_rank norm={L11['stable_rank_norm']:.2f} raw={L11['stable_rank_raw']:.2f}  "
              f"pairwise_cos_mean={L11['pairwise_cos_mean']:.3f}  align_mean={L11['align_with_mean_axis_mean']:.3f}")

    # ----- combined comparison figures (label token) -----
    layers = list(range(n_layers))
    colors = {"antonym_synonym": "#4C72B0", "synonym_rhyme": "#DD8452",
              "antonym_rhyme": "#55A868", "next_number_prev_number": "#C44E52"}
    for role_key in ROLE_MAP:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for pair in args.pairs:
            c = colors.get(pair)
            sr = [all_results[pair][role_key][str(l)]["stable_rank_norm"] for l in layers]
            pc = [all_results[pair][role_key][str(l)]["pairwise_cos_mean"] for l in layers]
            axes[0].plot(layers, sr, marker="o", ms=3, label=pair, color=c)
            axes[1].plot(layers, pc, marker="o", ms=3, label=pair, color=c)
        axes[0].set_title(f"Stable rank (unit-normalised diffs) — {role_key} token")
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("stable rank  Σσ²/σ₁²")
        axes[1].set_title(f"Mean pairwise cosine of diffs — {role_key} token")
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("mean pairwise cosine")
        axes[1].axhline(0, color="0.6", lw=0.8)
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_root / f"fig_compare_{role_key}.png", dpi=150)
        plt.close(fig)

    print(f"wrote outputs -> {out_root}")


if __name__ == "__main__":
    main()
