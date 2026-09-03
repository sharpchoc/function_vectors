#!/usr/bin/env python
"""Claim 6 on the NEW read feature: the map from the task-unique part u_A to the write feature.

USER DECISION 2026-09-03: the read feature is c + u_A with u_A = mean_{l=5..7} of the
carrier-projected-out task means. After train-mean centering the carrier c drops out, so the
read->write map is a map from u_A to the task FV v_A.

Same protocol as understanding_read_write_linear_map/rotation_vs_ridge.py (55 train / 14
held-out tasks, train-mean centering, ridge = dual with LOO-CV lambda, orthogonal Procrustes,
optional single global scale, pooled held-out R^2 with the test-mean reference), applied to
X = u_A. Two additions:

  * per-prompt evaluation: the task-level map is applied to each held-out prompt's own
    u_A^j (same construction per prompt) and scored against the task FV;
  * a k-degrees-of-freedom sweep: PCA on the 55 train u_A, keep the top-k read components
    (k = 1, 2, 4, ...), and fit (a) an unconstrained linear map from those k coordinates
    and (b) a Procrustes rotation of the rank-k projected read features, scored on the
    held-out tasks. Reference: the held-out task FVs projected onto their OWN top-k train
    write PCs (the ceiling for any rank-k predictor) and the read-side PCA reconstruction.

Writes results/69_task_run/understanding_read_write_linear_map/meanresid_map/
  fits_summary.csv           full-rank fits (baseline / rotation / rotation+scale / ridge)
  congruence.csv             family-centered geometry of u_A vs v_A
  kdim_sweep.csv             held-out R^2 vs k for the rank-k fits
  linear_map_simple.png      SIMPLE main-text figure: per held-out task cos(predicted, true FV)
  rotation_simple.png        SIMPLE 4-bar appendix figure (mean shift / rotation / +scale / ridge)
  kdim_sweep.png             SIMPLE held-out R^2 vs k
  rotation_detail.png        3-panel appendix figure (congruence / bars / ridge spectrum)
  spectra.npz
CPU, fp64.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

PP = ARTIFACTS_ROOT / "69_task_run" / "label_resid_perprompt"
RM = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
FV = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT = TASK69_RUN_DIR / "understanding_read_write_linear_map" / "meanresid_map"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
LAYERS = (5, 6, 7)
LAMBDA_GRID = np.logspace(-2, 6, 17)
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 54)

TEAL, PURPLE, INK, MUTED, ORANGE, BLUE = "#0e7c6b", "#7c3aad", "#181c1e", "#5d6771", "#c2410c", "#2a78d6"


def r2(pred, true, ref):
    """Pooled R^2 of pred vs true with reference point ref (broadcast)."""
    return 1.0 - ((true - pred) ** 2).sum() / ((true - ref) ** 2).sum()


def mean_cos(a, b):
    return float(np.mean(np.einsum("ij,ij->i", a, b)
                         / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


def ridge_dual(Xc, Yc):
    """Dual ridge on centered data; lambda by LOO-CV. Returns (A, lam) with pred = Xte Xc^T A."""
    K = Xc @ Xc.T
    n = len(Xc)
    best = (None, np.inf)
    for lam in LAMBDA_GRID:
        H = K @ np.linalg.inv(K + lam * np.eye(n))
        E = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
        m = (E ** 2).mean()
        if m < best[1]:
            best = (lam, m)
    lam = best[0]
    A = np.linalg.solve(K + lam * np.eye(n), Yc)
    return A, lam


def procrustes(Xc, Yc, tol=1e-10):
    """R = argmin ||Xc R - Yc||_F over orthogonal R (thin route, rank <= n); also the
    trace-formula scale and the correlation spectrum. R is returned as a rank-limited
    partial isometry acting on span(Xc) (components outside that span are dropped)."""
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = S > tol * S[0]
    U, S, Vt = U[:, keep], S[keep], Vt[keep]
    Uy, Sy, Vty = np.linalg.svd(Yc, full_matrices=False)
    small = (S[:, None] * (U.T @ Uy)) * Sy[None, :]
    Us, Ss, Vts = np.linalg.svd(small, full_matrices=False)
    R = (Vt.T @ Us) @ (Vts @ Vty)
    scale = Ss.sum() / (S ** 2).sum()
    return R, scale, Ss


def pairwise_cos(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    iu = np.triu_indices(len(X), k=1)
    return (Xn @ Xn.T)[iu]


def gram_cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return np.linalg.norm(Yc.T @ Xc, "fro") ** 2 / (np.linalg.norm(Xc @ Xc.T, "fro") * np.linalg.norm(Yc @ Yc.T, "fro"))


def principal_cos(X, Y, energy=0.90):
    out = []
    for Z in (X, Y):
        Zc = Z - Z.mean(0)
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), energy)) + 1
        out.append(Vt[:k].T)
    return np.linalg.svd(out[0].T @ out[1], compute_uv=False), out[0].shape[1], out[1].shape[1]


def style(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c9ccc7")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
    ax.set_facecolor("white")


def load():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    is_train = np.array([t in set(split["train_tasks"]) for t in tasks])
    M = torch.stack([torch.load(RM / f"{t}.pt", map_location="cpu", weights_only=False)["resid_means"][list(LAYERS)].double()
                     for t in tasks])                                      # (69, 3, d)
    cd = M.mean(0)
    cd = cd / cd.norm(dim=1, keepdim=True)                                 # per-layer carrier directions
    U_task, U_pp, Y_task = [], {}, []
    for t in tasks:
        z = torch.load(PP / f"{t}.pt", map_location="cpu", weights_only=False)["acts"][:, list(LAYERS)].double()  # (150,3,d)
        r = z - (z * cd).sum(-1, keepdim=True) * cd                        # carrier projected out per layer
        u = r.mean(dim=1)                                                  # (150, d) per-prompt u_A^j
        U_pp[t] = u.numpy()
        U_task.append(u.mean(0).numpy())
        Y_task.append(torch.load(FV / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double().mean(0).numpy())
    return tasks, is_train, np.stack(U_task), U_pp, np.stack(Y_task)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, is_train, X, Xpp, Y = load()
    te_tasks = [t for t, tr in zip(tasks, is_train) if not tr]
    Xtr, Ytr, Xte, Yte = X[is_train], Y[is_train], X[~is_train], Y[~is_train]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc, Xtec = Xtr - xm, Ytr - ym, Xte - xm
    Xpp_te = np.concatenate([Xpp[t] - xm for t in te_tasks])
    Ypp_te = np.concatenate([np.repeat(Y[tasks.index(t)][None], len(Xpp[t]), 0) for t in te_tasks])
    ref_te = Yte.mean(0)
    spectra = {}

    # ---- congruence of u_A vs v_A (all 69, family-centered) ----
    cx, cy = pairwise_cos(X - X.mean(0)), pairwise_cos(Y - Y.mean(0))
    nx, ny = np.linalg.norm(X - X.mean(0), axis=1), np.linalg.norm(Y - Y.mean(0), axis=1)
    pac, kx, ky = principal_cos(X, Y)
    Xn = (X - X.mean(0)) / np.linalg.norm(X - X.mean(0), axis=1, keepdims=True)
    Yn = (Y - Y.mean(0)) / np.linalg.norm(Y - Y.mean(0), axis=1, keepdims=True)
    C = Xn @ Yn.T
    matched, mism = np.diag(C), C[~np.eye(len(X), dtype=bool)]
    with open(OUT / "congruence.csv", "w") as fh:
        fh.write("pairwise_cos_pearson,pairwise_cos_spearman,centered_norm_pearson,gram_cka,"
                 "max_principal_cos,median_principal_cos,k90_read,k90_fv,matched_cos_mean,"
                 "mismatched_cos_mean,n_matched_above_mism_p95,read_centered_norm_median,fv_centered_norm_median\n")
        fh.write(f"{pearsonr(cx, cy)[0]:.4f},{spearmanr(cx, cy)[0]:.4f},{pearsonr(nx, ny)[0]:.4f},"
                 f"{gram_cka(X, Y):.4f},{pac.max():.4f},{np.median(pac):.4f},{kx},{ky},"
                 f"{matched.mean():.4f},{mism.mean():.4f},{(matched > np.percentile(mism, 95)).sum()},"
                 f"{np.median(nx):.2f},{np.median(ny):.2f}\n")
    spectra["pairwise_cos_read"], spectra["pairwise_cos_write"] = cx, cy

    # ---- full-rank fits ----
    A, lam = ridge_dual(Xc, Yc)
    R, scale, Ss = procrustes(Xc, Yc)
    U_, S_, _ = np.linalg.svd(Xc, full_matrices=False)
    spectra["ridge_sv"] = np.linalg.svd(S_[:, None] * (U_.T @ A), compute_uv=False)
    spectra["procrustes_corr"] = Ss
    preds = {
        "trainmean_baseline": (np.repeat(ym[None], len(Yte), 0), np.repeat(ym[None], len(Ypp_te), 0), np.repeat(ym[None], len(Ytr), 0)),
        "rotation": (Xtec @ R + ym, Xpp_te @ R + ym, Xc @ R + ym),
        "rotation+scale": (scale * (Xtec @ R) + ym, scale * (Xpp_te @ R) + ym, scale * (Xc @ R) + ym),
        "ridge": (Xtec @ Xc.T @ A + ym, Xpp_te @ Xc.T @ A + ym, Xc @ Xc.T @ A + ym),
    }
    fit_rows = []
    for name, (p_te, p_pp, p_tr) in preds.items():
        fit_rows.append({"method": name, "lambda": lam if name == "ridge" else "",
                         "scale": round(scale, 3) if name == "rotation+scale" else "",
                         "heldout_r2_testmean": round(r2(p_te, Yte, ref_te), 4),
                         "heldout_r2_trainmean": round(r2(p_te, Yte, ym), 4),
                         "heldout_perprompt_r2": round(r2(p_pp, Ypp_te, ref_te), 4),
                         "heldout_centered_cos": round(mean_cos(p_te - ym, Yte - ym), 4) if name != "trainmean_baseline" else 0.0,
                         "train_r2": round(r2(p_tr, Ytr, ym), 4)})
    with open(OUT / "fits_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fit_rows[0].keys())); w.writeheader(); w.writerows(fit_rows)
    for r_ in fit_rows:
        print(f"{r_['method']:>20}: heldout R2 testmean {r_['heldout_r2_testmean']:.3f} | trainmean "
              f"{r_['heldout_r2_trainmean']:.3f} | per-prompt {r_['heldout_perprompt_r2']:.3f} | "
              f"train {r_['train_r2']:.3f} lam={r_['lambda']} s={r_['scale']}")

    full_ridge_val = next(r_["heldout_r2_testmean"] for r_ in fit_rows if r_["method"] == "ridge")

    # ---- k-degrees-of-freedom sweep ----
    _, Sx, Vx = np.linalg.svd(Xc, full_matrices=False)
    _, Sy, Vy = np.linalg.svd(Yc, full_matrices=False)
    krows = []
    for k in KS:
        Pk, Qk = Vx[:k], Vy[:k]
        Xk_tr, Xk_te, Xk_pp = Xc @ Pk.T @ Pk, Xtec @ Pk.T @ Pk, Xpp_te @ Pk.T @ Pk
        Ztr, Zte, Zpp = Xc @ Pk.T, Xtec @ Pk.T, Xpp_te @ Pk.T
        Ak, lamk = ridge_dual(Ztr, Yc)
        p_lin, p_lin_pp = Zte @ Ztr.T @ Ak + ym, Zpp @ Ztr.T @ Ak + ym
        Rk, sk, _ = procrustes(Xk_tr, Yc)
        p_rot, p_rot_pp = Xk_te @ Rk + ym, Xk_pp @ Rk + ym
        p_rs, p_rs_pp = sk * (Xk_te @ Rk) + ym, sk * (Xk_pp @ Rk) + ym
        p_wceil = (Yte - ym) @ Qk.T @ Qk + ym
        read_recon = r2(Xk_te + xm, Xte, xm)
        krows.append({"k": k, "lambda_linear": lamk, "scale_rot": round(sk, 3),
                      "linear_r2": round(r2(p_lin, Yte, ref_te), 4), "linear_perprompt_r2": round(r2(p_lin_pp, Ypp_te, ref_te), 4),
                      "rotation_r2": round(r2(p_rot, Yte, ref_te), 4), "rotation_perprompt_r2": round(r2(p_rot_pp, Ypp_te, ref_te), 4),
                      "rotscale_r2": round(r2(p_rs, Yte, ref_te), 4), "rotscale_perprompt_r2": round(r2(p_rs_pp, Ypp_te, ref_te), 4),
                      "write_pca_ceiling_r2": round(r2(p_wceil, Yte, ref_te), 4),
                      "read_pca_recon_r2": round(read_recon, 4),
                      "read_train_var_frac": round(float((Sx[:k] ** 2).sum() / (Sx ** 2).sum()), 4),
                      "linear_train_r2": round(r2(Ztr @ Ztr.T @ Ak + ym, Ytr, ym), 4),
                      "rotscale_train_r2": round(r2(sk * (Xk_tr @ Rk) + ym, Ytr, ym), 4)})
        print(f"k={k:>2}: linear {krows[-1]['linear_r2']:.3f} | rotation {krows[-1]['rotation_r2']:.3f} | "
              f"rot+scale {krows[-1]['rotscale_r2']:.3f} (s={sk:.2f}) | write-PCA ceiling {krows[-1]['write_pca_ceiling_r2']:.3f} "
              f"| read recon {read_recon:.3f} | per-prompt lin/rot+s {krows[-1]['linear_perprompt_r2']:.3f}/{krows[-1]['rotscale_perprompt_r2']:.3f}")
    with open(OUT / "kdim_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(krows[0].keys())); w.writeheader(); w.writerows(krows)
    np.savez_compressed(OUT / "spectra.npz", **spectra)

    # ---- SIMPLE figure 1: 4 bars ----
    order = ["trainmean_baseline", "rotation", "rotation+scale", "ridge"]
    labels = ["mean shift only\n($\\bar v$)", "rotation\n(Procrustes)", "rotation\n+ one scalar", "unconstrained\nlinear (ridge)"]
    vals = [next(r_["heldout_r2_testmean"] for r_ in fit_rows if r_["method"] == m) for m in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150); fig.patch.set_facecolor("white"); style(ax)
    bars = ax.bar(range(4), vals, color=["0.72", PURPLE, PURPLE, TEAL], width=0.62, zorder=3)
    bars[1].set_alpha(0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(4), labels, fontsize=10.5)
    ax.set_ylabel("held-out $R^2$ (14 tasks)", fontsize=11, color=INK)
    ax.set_ylim(min(-0.15, min(vals) - 0.05), max(vals) + 0.12)
    ax.set_title("Predicting a held-out task's write feature from its read feature", loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout(); fig.savefig(OUT / "rotation_simple.png", facecolor="white"); plt.close(fig)

    # ---- SIMPLE figure 0 (main text): per held-out task, cos(predicted FV, true FV) for the linear map ----
    p_ridge = preds["ridge"][0]
    cos_map = np.einsum("ij,ij->i", p_ridge, Yte) / (np.linalg.norm(p_ridge, axis=1) * np.linalg.norm(Yte, axis=1))
    cos_gen = (Yte @ ym) / (np.linalg.norm(Yte, axis=1) * np.linalg.norm(ym))
    ordr = np.argsort(-cos_map)
    with open(OUT / "linear_map_per_task.csv", "w", newline="") as fh:
        fh.write("task,cos_pred_true,cos_trainmeanFV_true\n")
        for i in ordr:
            fh.write(f"{te_tasks[i]},{cos_map[i]:.4f},{cos_gen[i]:.4f}\n")
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=150); fig.patch.set_facecolor("white"); style(ax)
    xs = np.arange(len(ordr))
    ax.bar(xs, cos_map[ordr], width=0.62, color=TEAL, zorder=3, label=f"linear map from read feature, mean cos {cos_map.mean():.2f}")
    ax.axhline(cos_gen.mean(), color="0.45", lw=1.4, ls=(0, (5, 3)), zorder=4,
               label=f"generic FV (train mean) as predictor, mean cos {cos_gen.mean():.2f}")
    ax.set_xticks(xs, [te_tasks[i] for i in ordr], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("cos(predicted FV, true FV)", fontsize=11, color=INK)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title(f"Held-out tasks: write feature predicted from the read feature by one linear map (held-out $R^2$ = {full_ridge_val:.2f})",
                 loc="left", fontsize=11.5, color=INK, pad=10)
    fig.tight_layout(); fig.savefig(OUT / "linear_map_simple.png", facecolor="white"); plt.close(fig)
    print(f"linear map per held-out task: mean cos {cos_map.mean():.3f} (min {cos_map.min():.3f}) vs generic FV {cos_gen.mean():.3f}")

    # ---- SIMPLE figure 2: R^2 vs k ----
    ks = [r_["k"] for r_ in krows]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150); fig.patch.set_facecolor("white"); style(ax)
    ax.plot(ks, [r_["write_pca_ceiling_r2"] for r_ in krows], color="0.55", lw=1.5, ls=(0, (4, 3)), marker="o", ms=3.5,
            label="ceiling: write feature's own top-$k$ PCs")
    ax.plot(ks, [r_["linear_r2"] for r_ in krows], color=TEAL, lw=2.0, marker="o", ms=4.5, label="unconstrained linear from $k$ read PCs")
    ax.plot(ks, [r_["rotscale_r2"] for r_ in krows], color=PURPLE, lw=2.2, marker="o", ms=4.5, label="rotation + one scalar of $k$ read PCs")
    full_ridge = next(r_["heldout_r2_testmean"] for r_ in fit_rows if r_["method"] == "ridge")
    ax.axhline(full_ridge, color=TEAL, lw=1.0, ls=":", alpha=0.8)
    ax.text(ks[-1], full_ridge + 0.012, f"full ridge {full_ridge:.2f}", ha="right", fontsize=9, color=TEAL)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("$k$ = number of read-feature principal components used", fontsize=11, color=INK)
    ax.set_ylabel("held-out $R^2$ (14 tasks)", fontsize=11, color=INK)
    ax.set_ylim(min(0, min(r_["rotscale_r2"] for r_ in krows) - 0.05), 1.02)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("How many read dimensions does the read→write map need?", loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout(); fig.savefig(OUT / "kdim_sweep.png", facecolor="white"); plt.close(fig)

    # ---- detail figure (appendix) ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), dpi=150); fig.patch.set_facecolor("white")
    ax = axes[0]; style(ax)
    ax.plot([-0.6, 1], [-0.6, 1], color="0.7", lw=0.8)
    ax.scatter(cx, cy, s=4, alpha=0.25, color=TEAL, edgecolors="none")
    ax.set_xlabel("centered pairwise cos, read $u_A$"); ax.set_ylabel("centered pairwise cos, write $v_A$")
    ax.set_title(f"Pairwise congruence ({len(cx)} pairs)\nPearson {pearsonr(cx, cy)[0]:.3f} · gram-CKA {gram_cka(X, Y):.3f}", fontsize=10)
    ax = axes[1]; style(ax)
    v1 = [next(r_["heldout_r2_testmean"] for r_ in fit_rows if r_["method"] == m) for m in order]
    v2 = [next(r_["heldout_perprompt_r2"] for r_ in fit_rows if r_["method"] == m) for m in order]
    ax.bar(np.arange(4) - 0.19, v1, width=0.38, color=TEAL, label="task centroids")
    ax.bar(np.arange(4) + 0.19, v2, width=0.38, color=BLUE, label="per-prompt read features")
    ax.set_xticks(range(4), ["mean shift", "rotation", "rotation\n+scale", "ridge"], fontsize=8.5)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_ylabel("held-out $R^2$ (test-mean ref.)"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Held-out prediction of task FVs (14 tasks)", fontsize=10)
    ax = axes[2]; style(ax)
    sv = spectra["ridge_sv"]; ax.plot(np.arange(1, len(sv) + 1), sv / sv[0], color=TEAL, label="ridge map, read $u_A$")
    ax.set_yscale("log"); ax.set_xlabel("singular value index"); ax.set_ylabel("$\\sigma_i/\\sigma_1$")
    ax.set_title("Ridge-map spectrum on the train span\n(flat = scaled rotation)", fontsize=10); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "rotation_detail.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
