"""
Analyze paired 1-shot ICL activations captured by capture_oneshot_paired.py.

Pure numpy / torch / sklearn / scipy / matplotlib — NO model, NO baukit. The only
repo dependency is the (baukit-free) low-rank/ridge helpers in
regress_activation_to_fv_pca_ridge.py (torch_pca, project, reconstruct,
ridge_eig_prep, ridge_predict), imported directly.

For a task pair we have, per output word w and function f in {f1,f2}, two captured
activation stacks (28 layers x 4096):
    source = demo label token (" w")
    target = final query token (trailing "A:")

Per layer L (all 28) we compute:
  (1) Label-token difference geometry:  D_label = A1 - A2  (source acts, f1 - f2).
  (2) Final query-token difference geometry: D_final = Y1 - Y2 (target acts).
  (3) FV-space projection of final-token acts: orthonormal 2-D basis of
      span{fv_f1, fv_f2} (Gram-Schmidt), difference-axis (fv_f1 - fv_f2) separation
      (AUC / Cohen's d / threshold accuracy); 2-D scatter PNGs for layers 0,6,11,16,22,27.
  (4) Bonus: cosine between the top singular vector of D_final and (fv_f1 - fv_f2).
  (5) Source->target predictive map M: dX -> dY, ridge with held-out R^2 (LOO/k-fold),
      SVD effective rank, reduced-rank R^2(rank) for rank in {1,2,4,8,16,32}, and
      rotation-like tests (||M^T M - I||_F on the active subspace, ||M dx||/||dx||
      mean/std, orthogonal Procrustes residual vs ridge residual).

Outputs under results/oneshot_paired_analysis/<task_pair>/:
    label_geometry.json, final_geometry.json, fv_projection.json,
    source_target_map.json, summary.csv, fig_*.png
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# baukit-free helpers (verified: only argparse/csv/json/time/pathlib/numpy/torch).
from regress_activation_to_fv_pca_ridge import (
    ridge_eig_prep,
    ridge_predict,
)


TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "landmark_park": ("landmark-country", "park-country"),
}
SCATTER_LAYERS = [0, 6, 11, 16, 22, 27]
RR_RANKS = [1, 2, 4, 8, 16, 32]


def parse_args():
    p = argparse.ArgumentParser(description="Geometry analysis of paired 1-shot ICL activations.")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--capture_root", type=str, default="results/oneshot_paired")
    p.add_argument("--fv_root", type=str, default="results/function_vectors/gpt-j/train_selected")
    p.add_argument("--output_root", type=str, default="results/oneshot_paired_analysis")
    p.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge lambda for the source->target map.")
    p.add_argument("--n_folds", type=int, default=5,
                   help="K-fold for held-out R^2 of the source->target map; <=0 or >=W uses leave-one-out.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_capture(capture_dir):
    """Load all shards into a dict keyed by (role, function, w) -> [n_layers, hidden] np.float32.

    Returns (acts, n_layers, config).
    """
    index = json.load(open(capture_dir / "index.json"))
    config = index["config"]
    acts = {}
    n_layers = None
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = capture_dir / shard_path.name
        data = torch_load_trusted(shard_path, map_location="cpu")
        activations = data["activations"]  # [rows, n_layers, hidden]
        metadata = data["metadata"]
        if len(metadata) != activations.shape[0]:
            raise ValueError(f"Metadata/activation mismatch in {shard_path}")
        arr = activations.to(torch.float32).numpy()
        if n_layers is None:
            n_layers = arr.shape[1]
        for i, meta in enumerate(metadata):
            key = (meta["role"], meta["function"], meta["output_word"])
            acts[key] = arr[i]
    return acts, n_layers, config


def build_layer_matrices(acts, role, layer):
    """For a role/layer, return (W_list, F1[W,hidden], F2[W,hidden]) over words present in BOTH functions."""
    f1_words = {w for (r, f, w) in acts if r == role and f == "f1"}
    f2_words = {w for (r, f, w) in acts if r == role and f == "f2"}
    words = sorted(f1_words.intersection(f2_words))
    a1 = np.stack([acts[(role, "f1", w)][layer] for w in words], axis=0)
    a2 = np.stack([acts[(role, "f2", w)][layer] for w in words], axis=0)
    return words, a1, a2


def load_function_vector(fv_root, task):
    fv_path = Path(fv_root) / task / f"{task}_function_vector.pt"
    if not fv_path.exists():
        raise FileNotFoundError(fv_path)
    data = torch_load_trusted(fv_path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return np.asarray(fv.detach().float().cpu().reshape(-1).numpy(), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Geometry metrics
# --------------------------------------------------------------------------- #
def diff_geometry(D):
    """SVD-based geometry summary of a difference matrix D [W, hidden].

    Returns a dict with spectrum, variance@k, participation ratio, effective rank,
    pairwise-cosine summary, rank-1/colinearity fraction, 1st-SV variance share, and
    the mean-centered first-SV share (common direction vs spread).
    """
    D = np.asarray(D, dtype=np.float64)
    W = D.shape[0]
    sv = np.linalg.svd(D, compute_uv=False)
    sv2 = sv ** 2
    total = float(sv2.sum())

    def var_at(k):
        k = min(k, len(sv2))
        return float(sv2[:k].sum() / total) if total > 0 else 0.0

    pr = float((sv2.sum() ** 2) / (np.sum(sv2 ** 2))) if np.sum(sv2 ** 2) > 0 else 0.0
    # entropy-based effective rank (Roy & Vetterli).
    p = sv2 / total if total > 0 else np.zeros_like(sv2)
    nz = p[p > 0]
    eff_rank = float(np.exp(-np.sum(nz * np.log(nz)))) if nz.size else 0.0

    norms = np.linalg.norm(D, axis=1)
    sum_sq_norms = float(np.sum(norms ** 2))
    mean_vec = D.mean(axis=0)
    rank1_fraction = float((np.dot(mean_vec, mean_vec) * W) / sum_sq_norms) if sum_sq_norms > 0 else 0.0
    first_sv_share = var_at(1)

    # mean-centered: how much is a shared common direction vs per-word spread.
    Dc = D - mean_vec
    svc = np.linalg.svd(Dc, compute_uv=False)
    svc2 = svc ** 2
    centered_total = float(svc2.sum())
    centered_first_sv_share = float(svc2[0] / centered_total) if centered_total > 0 else 0.0

    # pairwise cosine distribution among rows (subsample for very large W).
    rng = np.random.default_rng(0)
    idx = np.arange(W)
    if W > 200:
        idx = rng.choice(W, size=200, replace=False)
    Dn = D[idx]
    nrm = np.linalg.norm(Dn, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    U = Dn / nrm
    C = U @ U.T
    iu = np.triu_indices(C.shape[0], k=1)
    cos_vals = C[iu] if iu[0].size else np.array([0.0])

    # top singular vector (right) of D for cross-analysis (4).
    if total > 0:
        _, _, vt = np.linalg.svd(D, full_matrices=False)
        top_sv_vec = vt[0]
    else:
        top_sv_vec = np.zeros(D.shape[1])

    return {
        "W": int(W),
        "singular_values": [float(x) for x in sv[: min(64, len(sv))]],
        "variance_at_k": {str(k): var_at(k) for k in (1, 2, 4, 8, 16)},
        "participation_ratio": pr,
        "effective_rank": eff_rank,
        "first_sv_variance_share": first_sv_share,
        "rank1_colinearity_fraction": rank1_fraction,
        "centered_first_sv_share": centered_first_sv_share,
        "mean_diff_norm": float(np.linalg.norm(mean_vec)),
        "row_norm_mean": float(norms.mean()),
        "pairwise_cosine_mean": float(np.mean(cos_vals)),
        "pairwise_cosine_std": float(np.std(cos_vals)),
        "pairwise_cosine_abs_mean": float(np.mean(np.abs(cos_vals))),
        "_top_sv_vec": top_sv_vec,  # internal; popped before json dump
    }


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def gram_schmidt_basis(fv1, fv2):
    """Orthonormal 2-D basis (e1, e2) of span{fv1, fv2}; e1 along fv1."""
    e1 = unit(fv1)
    r = fv2 - np.dot(fv2, e1) * e1
    e2 = unit(r)
    return e1, e2


def separation_stats(s1, s2):
    """Separation of two 1-D score arrays: AUC, Cohen's d, best threshold accuracy."""
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    n1, n2 = len(s1), len(s2)
    # AUC via Mann-Whitney rank statistic (P(s1 > s2)); oriented so >=0.5.
    allv = np.concatenate([s1, s2])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    r1 = ranks[:n1].sum()
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2) if n1 and n2 else 0.5
    auc = max(auc, 1.0 - auc)
    m1, m2 = s1.mean(), s2.mean()
    v1, v2 = s1.var(ddof=1) if n1 > 1 else 0.0, s2.var(ddof=1) if n2 > 1 else 0.0
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)) if (n1 + n2) > 2 else 0.0
    cohen_d = float((m1 - m2) / pooled) if pooled > 0 else 0.0
    # best threshold accuracy (1-D).
    thresholds = np.unique(allv)
    best_acc = 0.5
    for t in thresholds:
        acc_a = (np.mean(s1 > t) * n1 + np.mean(s2 <= t) * n2) / (n1 + n2)
        acc_b = (np.mean(s1 <= t) * n1 + np.mean(s2 > t) * n2) / (n1 + n2)
        best_acc = max(best_acc, acc_a, acc_b)
    return float(auc), float(cohen_d), float(best_acc)


# --------------------------------------------------------------------------- #
# Source -> target predictive map
# --------------------------------------------------------------------------- #
def make_folds(W, n_folds, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(W)
    if n_folds <= 0 or n_folds >= W:
        return [np.array([i]) for i in range(W)]  # leave-one-out
    return [perm[i::n_folds] for i in range(n_folds)]


def ridge_fit_predict(x_fit, y_fit, x_eval, alpha):
    xf = torch.from_numpy(np.asarray(x_fit, dtype=np.float64))
    yf = torch.from_numpy(np.asarray(y_fit, dtype=np.float64))
    xe = torch.from_numpy(np.asarray(x_eval, dtype=np.float64))
    xbar, ybar, evals, evecs, c = ridge_eig_prep(xf, yf)
    pred = ridge_predict(xe, xbar, ybar, evals, evecs, c, alpha)
    return pred.numpy()


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_full_ridge_map(dX, dY, alpha):
    """Closed-form ridge weight M (hidden x hidden) mapping centered dX -> centered dY:
       M = (Xc^T Xc + alpha I)^-1 Xc^T Yc  (returns M as [in, out]); plus intercept b."""
    xbar = dX.mean(axis=0)
    ybar = dY.mean(axis=0)
    Xc = dX - xbar
    Yc = dY - ybar
    d = Xc.shape[1]
    G = Xc.T @ Xc + alpha * np.eye(d)
    M = np.linalg.solve(G, Xc.T @ Yc)  # [in, out]
    return M, xbar, ybar


def source_target_map(dX, dY, alpha, n_folds, seed):
    """Characterize the ridge map dX -> dY. Returns metric dict."""
    dX = np.asarray(dX, dtype=np.float64)
    dY = np.asarray(dY, dtype=np.float64)
    W, d = dX.shape

    # The map acts only within the (<= 2(W-1))-dim span of the centered data, so project
    # BOTH sides onto an orthonormal basis B of that span. Every metric below (held-out
    # R^2, M's singular spectrum, reduced-rank R^2, Procrustes residual, norm preservation)
    # is invariant under this orthonormal change of basis applied to both input and output;
    # cost drops from O(d^3) to O(m^3) with m = rank << d=4096. Centering downstream
    # commutes with the projection, so projecting the uncentered data here is exact.
    stack = np.vstack([dX - dX.mean(0), dY - dY.mean(0)])
    _, s_stack, vt_stack = np.linalg.svd(stack, full_matrices=False)
    tol = max(stack.shape) * (float(s_stack[0]) if s_stack.size else 0.0) * np.finfo(np.float64).eps
    m = max(int((s_stack > tol).sum()), 1)
    B = vt_stack[:m].T  # [d, m] orthonormal columns
    dX = dX @ B
    dY = dY @ B
    d = m

    # Held-out R^2 (per-fold ridge), plus held-out norm-preservation ||M dx||/||dx||.
    folds = make_folds(W, n_folds, seed)
    preds = np.zeros_like(dY)
    norm_ratios = []
    for test_idx in folds:
        mask = np.ones(W, dtype=bool)
        mask[test_idx] = False
        x_fit, y_fit = dX[mask], dY[mask]
        x_eval = dX[test_idx]
        pred = ridge_fit_predict(x_fit, y_fit, x_eval, alpha)
        preds[test_idx] = pred
        # norm preservation on the held-out predictions of the (centered-around-train) map.
        for j, ti in enumerate(test_idx):
            dn = np.linalg.norm(dX[ti])
            if dn > 0:
                norm_ratios.append(np.linalg.norm(pred[j] - dY[mask].mean(axis=0)) / dn)
    heldout_r2 = r2_score(dY, preds)
    norm_ratios = np.asarray(norm_ratios) if norm_ratios else np.array([0.0])

    # Full-data ridge map for structural characterization.
    M, xbar, ybar = fit_full_ridge_map(dX, dY, alpha)  # M: [d, d]
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv ** 2
    total = float(sv2.sum())
    p = sv2 / total if total > 0 else np.zeros_like(sv2)
    nz = p[p > 0]
    map_eff_rank = float(np.exp(-np.sum(nz * np.log(nz)))) if nz.size else 0.0
    map_part_ratio = float((sv2.sum() ** 2) / np.sum(sv2 ** 2)) if np.sum(sv2 ** 2) > 0 else 0.0

    # Reduced-rank regression R^2(rank): truncate M to top-r via SVD, evaluate IN-SAMPLE on centered data.
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Xc = dX - xbar
    Yc = dY - ybar
    rr_r2 = {}
    for r in RR_RANKS:
        rr = min(r, len(S))
        Mr = (U[:, :rr] * S[:rr]) @ Vt[:rr]
        pred_c = Xc @ Mr
        rr_r2[str(r)] = r2_score(Yc, pred_c)

    # Rotation-like tests.
    # (i) ||M^T M - I||_F on the active subspace: restrict M to its top-k right/left singular subspace.
    k_active = max(1, min(int(round(map_eff_rank)), len(S)))
    # M restricted: in-basis Vt[:k] -> out-basis U[:,:k]; the in/out k x k block is diag(S[:k]).
    Sk = S[:k_active]
    block = np.diag(Sk)
    # rescale so the block has unit average gain before testing orthogonality (rotation up to scale).
    scale = np.mean(Sk) if np.mean(Sk) > 0 else 1.0
    block_n = block / scale
    mtm_minus_i = float(np.linalg.norm(block_n.T @ block_n - np.eye(k_active), ord="fro"))

    # (ii) norm preservation summarized above (held-out).
    # (iii) orthogonal Procrustes residual vs ridge residual on centered data.
    R, _ = orthogonal_procrustes(Xc, Yc)  # R: [d, d] orthogonal, minimizes ||Xc R - Yc||_F
    proc_pred = Xc @ R
    proc_res = float(np.linalg.norm(Yc - proc_pred, ord="fro"))
    ridge_pred_in = Xc @ M
    ridge_res = float(np.linalg.norm(Yc - ridge_pred_in, ord="fro"))
    yc_norm = float(np.linalg.norm(Yc, ord="fro"))
    procrustes_gap = (proc_res - ridge_res) / yc_norm if yc_norm > 0 else 0.0

    return {
        "W": int(W),
        "alpha": float(alpha),
        "n_folds_effective": len(folds),
        "map_R2": float(heldout_r2),
        "map_singular_values": [float(x) for x in sv[: min(64, len(sv))]],
        "map_eff_rank": map_eff_rank,
        "map_participation_ratio": map_part_ratio,
        "reduced_rank_R2": rr_r2,
        "active_subspace_k": int(k_active),
        "mtm_minus_I_fro": mtm_minus_i,
        "norm_preservation_mean": float(norm_ratios.mean()),
        "norm_preservation_std": float(norm_ratios.std()),
        "procrustes_residual_fro": proc_res,
        "ridge_residual_fro": ridge_res,
        "procrustes_gap": float(procrustes_gap),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    task_f1, task_f2 = TASK_PAIRS[args.task_pair]

    capture_dir = Path(args.capture_root) / args.task_pair
    acts, n_layers, _config = load_capture(capture_dir)
    print(f"[{args.task_pair}] loaded capture: n_layers={n_layers}, rows={len(acts)}")

    fv_f1 = load_function_vector(args.fv_root, task_f1)
    fv_f2 = load_function_vector(args.fv_root, task_f2)
    fv1u, fv2u = unit(fv_f1), unit(fv_f2)
    e1, e2 = gram_schmidt_basis(fv1u, fv2u)
    diff_axis = unit(fv1u - fv2u)

    out_dir = Path(args.output_root) / args.task_pair
    out_dir.mkdir(parents=True, exist_ok=True)

    label_geo = {}
    final_geo = {}
    fv_proj = {}
    st_map = {}
    summary_rows = []

    for layer in range(n_layers):
        # (1) label-token difference geometry (source).
        words_s, A1, A2 = build_layer_matrices(acts, "source", layer)
        D_label = A1 - A2
        lg = diff_geometry(D_label)
        top_sv_label = lg.pop("_top_sv_vec")
        label_geo[str(layer)] = lg

        # (2) final query-token difference geometry (target).
        words_t, Y1, Y2 = build_layer_matrices(acts, "target", layer)
        D_final = Y1 - Y2
        fg = diff_geometry(D_final)
        top_sv_final = fg.pop("_top_sv_vec")
        final_geo[str(layer)] = fg

        # (3) FV-space projection of final-token acts.
        all_final = np.concatenate([Y1, Y2], axis=0)
        coords1 = np.stack([Y1 @ e1, Y1 @ e2], axis=1)
        coords2 = np.stack([Y2 @ e1, Y2 @ e2], axis=1)
        score1 = Y1 @ diff_axis
        score2 = Y2 @ diff_axis
        auc, cohen_d, thr_acc = separation_stats(score1, score2)
        fv_proj[str(layer)] = {
            "f1_mean_coord": [float(coords1[:, 0].mean()), float(coords1[:, 1].mean())],
            "f2_mean_coord": [float(coords2[:, 0].mean()), float(coords2[:, 1].mean())],
            "diff_axis_auc": auc,
            "diff_axis_cohen_d": cohen_d,
            "diff_axis_threshold_acc": thr_acc,
            "f1_diff_axis_mean": float(score1.mean()),
            "f2_diff_axis_mean": float(score2.mean()),
        }

        # (4) bonus: cosine(top SV of D_final, diff_axis).
        dfinal_fv_cos = float(abs(np.dot(unit(top_sv_final), diff_axis)))
        label_fv_cos = float(abs(np.dot(unit(top_sv_label), diff_axis)))

        # (5) source -> target map on function-contrastive differences.
        # Match words present in both roles.
        common_words = [w for w in words_s if w in set(words_t)]
        s_index = {w: i for i, w in enumerate(words_s)}
        t_index = {w: i for i, w in enumerate(words_t)}
        dX = np.stack([D_label[s_index[w]] for w in common_words], axis=0)
        dY = np.stack([D_final[t_index[w]] for w in common_words], axis=0)
        sm = source_target_map(dX, dY, args.ridge_alpha, args.n_folds, args.seed)
        st_map[str(layer)] = sm

        # scatter PNGs for representative layers.
        if layer in SCATTER_LAYERS:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(coords1[:, 0], coords1[:, 1], s=14, alpha=0.6, label=f"f1 ({task_f1})", color="#4C72B0")
            ax.scatter(coords2[:, 0], coords2[:, 1], s=14, alpha=0.6, label=f"f2 ({task_f2})", color="#DD8452")
            ax.set_xlabel("FV basis e1 (along fv_f1)")
            ax.set_ylabel("FV basis e2 (orth. component of fv_f2)")
            ax.set_title(f"{args.task_pair} L{layer}: final-token acts in span(fv_f1,fv_f2)\n"
                         f"diff-axis AUC={auc:.3f} d={cohen_d:.2f} acc={thr_acc:.3f}")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"fig_fv_scatter_L{layer:02d}.png", dpi=150)
            plt.close(fig)

        summary_rows.append({
            "task_pair": args.task_pair,
            "layer": layer,
            "eff_rank_label": lg["effective_rank"],
            "eff_rank_final": fg["effective_rank"],
            "rank1_fraction_label": lg["rank1_colinearity_fraction"],
            "rank1_fraction_final": fg["rank1_colinearity_fraction"],
            "fv_separation": auc,
            "dfinal_fv_cosine": dfinal_fv_cos,
            "map_R2": sm["map_R2"],
            "map_eff_rank": sm["map_eff_rank"],
            "map_rank1_R2": sm["reduced_rank_R2"]["1"],
            "map_rank2_R2": sm["reduced_rank_R2"]["2"],
            "map_rank4_R2": sm["reduced_rank_R2"]["4"],
            "map_rank8_R2": sm["reduced_rank_R2"]["8"],
            "map_rank16_R2": sm["reduced_rank_R2"]["16"],
            "procrustes_gap": sm["procrustes_gap"],
            "norm_preservation_mean": sm["norm_preservation_mean"],
        })
        print(f"[{args.task_pair}] L{layer:02d}: eff_rank(label/final)="
              f"{lg['effective_rank']:.2f}/{fg['effective_rank']:.2f} "
              f"fv_sep_auc={auc:.3f} dfinal_fv_cos={dfinal_fv_cos:.3f} "
              f"map_R2={sm['map_R2']:.3f} map_eff_rank={sm['map_eff_rank']:.2f} "
              f"proc_gap={sm['procrustes_gap']:.3f}")

    # write JSON outputs.
    with open(out_dir / "label_geometry.json", "w") as f:
        json.dump(label_geo, f, indent=2)
    with open(out_dir / "final_geometry.json", "w") as f:
        json.dump(final_geo, f, indent=2)
    with open(out_dir / "fv_projection.json", "w") as f:
        json.dump({"task_f1": task_f1, "task_f2": task_f2, "layers": fv_proj}, f, indent=2)
    with open(out_dir / "source_target_map.json", "w") as f:
        json.dump(st_map, f, indent=2)

    csv_fields = [
        "task_pair", "layer", "eff_rank_label", "eff_rank_final",
        "rank1_fraction_label", "rank1_fraction_final", "fv_separation",
        "dfinal_fv_cosine", "map_R2", "map_eff_rank",
        "map_rank1_R2", "map_rank2_R2", "map_rank4_R2", "map_rank8_R2", "map_rank16_R2",
        "procrustes_gap", "norm_preservation_mean",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r[k] for k in csv_fields})

    print(f"[{args.task_pair}] wrote outputs -> {out_dir}")


if __name__ == "__main__":
    main()
