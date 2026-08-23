"""Is the read->write linear map just a rotation?

Three tests on the task-level data (69 tasks; X = m_A(L), Y = task FV):

1. Geometry congruence BEFORE any map: pair-by-pair correlation of the centered
   pairwise cosines between families, centered-norm correlation, and linear CKA
   between the centered Gram matrices. High congruence = the point clouds are
   already the same shape, so a rotation could align them.
2. Orthogonal Procrustes (pure rotation, and rotation + single global scale) fit
   on the 55 train tasks vs the unconstrained ridge (dual, intercept via
   train-centering, lambda by LOO-CV on train) — held-out R^2 (testmean and
   trainmean conventions) and mean cos(pred, true).
3. Singular-value spectrum of the fitted ridge map restricted to the train span:
   flat spectrum = scaled rotation; decaying = anisotropic, direction-dependent gain.

Same captures as centered_cossim_hists.py. CPU, fp64.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402

READ_ROOT = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
FV_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT = RESULTS_ROOT / "69_task_run" / "understanding_read_write_linear_map"
SPLIT = json.load(open(REPO / "task_splits" / "extended_steerable_69_prunedfail.json"))
READ_LAYERS = [6, 13]
LAMBDA_GRID = np.logspace(-2, 6, 17)

tasks = sorted(p.stem for p in READ_ROOT.glob("*.pt"))
train_set = set(SPLIT["train_tasks"])
assert len(train_set) == 55, "unexpected split format"
is_train = np.array([t in train_set for t in tasks])
assert is_train.sum() == 55 and len(tasks) == 69

reads = {L: [] for L in READ_LAYERS}
fvs = []
for t in tasks:
    r = torch.load(READ_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
    for L in READ_LAYERS:
        reads[L].append(r["resid_means"][L].double().numpy())
    f = torch.load(FV_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
    fvs.append(f["fv"].double().mean(0).numpy())
Y = np.stack(fvs)


def pairwise_cos(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    iu = np.triu_indices(len(X), k=1)
    return (Xn @ Xn.T)[iu]


def gram_cka(X, Y):
    """Standard linear CKA (sample-side Gram alignment): do the two families
    place the 69 tasks in the same relative configuration?"""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    hsic = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    return hsic / (np.linalg.norm(Xc @ Xc.T, "fro") * np.linalg.norm(Yc @ Yc.T, "fro"))


def feature_alignment(X, Y):
    """Feature-side analogue (variance-weighted subspace overlap in the 4096-d
    activation space): do the two families occupy the same DIRECTIONS?"""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    num = np.linalg.norm(Yc @ Xc.T, "fro") ** 2  # = <Xc^T Xc, Yc^T Yc>_F
    return num / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro"))


def principal_angle_cosines(X, Y, energy=0.90):
    """Cosines of principal angles between the two families' centered
    variance-90% subspaces in activation space."""
    out = []
    for Z in (X, Y):
        Zc = Z - Z.mean(0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), energy)) + 1
        out.append(Vt[:k].T)
    return np.linalg.svd(out[0].T @ out[1], compute_uv=False), out[0].shape[1], out[1].shape[1]


def r2(pred, true, denom_center):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - denom_center) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def mean_cos(pred, true):
    return np.mean(np.einsum("ij,ij->i", pred, true)
                   / (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)))


geom_rows, fit_rows, cross_rows, spectra = [], [], [], {}
for L in READ_LAYERS:
    X = np.stack(reads[L])

    # --- 0. cross-family cosines: matched cos(m_A, v_A) vs mismatched cos(m_A, v_B) ---
    for variant, Xv, Yv in [("uncentered", X, Y),
                            ("centered", X - X.mean(0), Y - Y.mean(0))]:
        Xn = Xv / np.linalg.norm(Xv, axis=1, keepdims=True)
        Yn = Yv / np.linalg.norm(Yv, axis=1, keepdims=True)
        C = Xn @ Yn.T
        matched = np.diag(C)
        mism = C[~np.eye(69, dtype=bool)]
        spectra[f"L{L}_crossfam_matched_{variant}"] = matched
        spectra[f"L{L}_crossfam_mismatched_{variant}"] = mism
        cross_rows.append((L, variant, matched.mean(), np.median(matched),
                           matched.min(), matched.max(), mism.mean(),
                           (matched > np.percentile(mism, 95)).sum()))

    # --- 1. congruence (all 69, all-69 centering, matching centered_cossim_hists) ---
    cx, cy = pairwise_cos(X - X.mean(0)), pairwise_cos(Y - Y.mean(0))
    nx = np.linalg.norm(X - X.mean(0), axis=1)
    ny = np.linalg.norm(Y - Y.mean(0), axis=1)
    pac, kx, ky = principal_angle_cosines(X, Y)
    geom_rows.append((L, pearsonr(cx, cy)[0], spearmanr(cx, cy)[0],
                      pearsonr(nx, ny)[0], gram_cka(X, Y), feature_alignment(X, Y),
                      pac.max(), np.median(pac), kx, ky))
    spectra[f"L{L}_principal_angle_cos"] = pac

    # --- fits: train-centered ---
    Xtr, Ytr = X[is_train], Y[is_train]
    Xte, Yte = X[~is_train], Y[~is_train]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - xm, Ytr - ym
    Xtec = Xte - xm

    # ridge (dual, LOO-CV lambda)
    K = Xc @ Xc.T
    best = (None, np.inf)
    for lam in LAMBDA_GRID:
        Kinv = np.linalg.inv(K + lam * np.eye(55))
        H = K @ Kinv
        E = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
        mse = (E ** 2).mean()
        if mse < best[1]:
            best = (lam, mse)
    lam = best[0]
    A = np.linalg.solve(K + lam * np.eye(55), Yc)          # 55 x 4096 dual coefs
    pred_ridge = Xtec @ Xc.T @ A + ym

    # ridge-map singular values on the train span: W = Xc^T A = V (S U^T A)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    sv_ridge = np.linalg.svd(S[:, None] * (U.T @ A), compute_uv=False)
    spectra[f"L{L}_ridge"] = sv_ridge

    # Procrustes: R = argmin ||Xc R - Yc||_F over orthogonal R
    M = Xc.T @ Yc
    # thin route: M = (V S U^T)(U' S' V'^T) — work in the joint span
    Uy, Sy, Vty = np.linalg.svd(Yc, full_matrices=False)
    small = (S[:, None] * (U.T @ Uy)) * Sy[None, :]        # V^T M V' in small coords
    Us, Ss, Vts = np.linalg.svd(small)
    R = (Vt.T @ Us) @ (Vts @ Vty)                          # 4096 x 4096 rank<=55
    scale = Ss.sum() / (S ** 2).sum()
    pred_rot = Xtec @ R + ym
    pred_rotscale = scale * (Xtec @ R) + ym
    spectra[f"L{L}_procrustes_corr"] = Ss                   # correlation spectrum

    for name, pred in [("ridge", pred_ridge), ("rotation", pred_rot),
                       ("rotation+scale", pred_rotscale),
                       ("trainmean_baseline", np.repeat(ym[None, :], len(Yte), 0))]:
        fit_rows.append((L, name, lam if name == "ridge" else "",
                         r2(pred, Yte, Yte.mean(0)), r2(pred, Yte, ym),
                         mean_cos(pred - ym, Yte - ym) if name != "trainmean_baseline" else 0.0,
                         mean_cos(pred, Yte)))

with open(OUT / "rotation_vs_ridge_summary.csv", "w") as fh:
    fh.write("# geometry congruence (all 69 tasks, family-centered)\n")
    fh.write("layer,pairwise_cos_pearson,pairwise_cos_spearman,centered_norm_pearson,"
             "gram_cka,feature_alignment,max_principal_cos,median_principal_cos,k90_read,k90_fv\n")
    for r_ in geom_rows:
        fh.write(f"{r_[0]}," + ",".join(f"{v:.4f}" for v in r_[1:8]) + f",{r_[8]},{r_[9]}\n")
    fh.write("# cross-family cosines: matched cos(m_A, v_A) vs mismatched cos(m_A, v_B), all 69 tasks\n")
    fh.write("layer,variant,matched_mean,matched_median,matched_min,matched_max,mismatched_mean,n_matched_above_mism_p95\n")
    for r_ in cross_rows:
        fh.write(f"{r_[0]},{r_[1]}," + ",".join(f"{v:.4f}" for v in r_[2:-1]) + f",{r_[-1]}\n")
    fh.write("# held-out fits (14 tasks; centered_cos = cos of train-mean-centered pred vs true)\n")
    fh.write("layer,method,lambda,heldout_r2_testmean,heldout_r2_trainmean,heldout_centered_cos,heldout_cos\n")
    for r_ in fit_rows:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]}," + ",".join(f"{v:.4f}" for v in r_[3:]) + "\n")
np.savez_compressed(OUT / "rotation_vs_ridge_spectra.npz", **spectra)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
ax = axes[0]
X6c = np.stack(reads[6]) - np.stack(reads[6]).mean(0)
cx6, cy6 = pairwise_cos(X6c), pairwise_cos(Y - Y.mean(0))
ax.plot([-0.6, 1], [-0.6, 1], color="0.7", lw=0.8)
ax.scatter(cx6, cy6, s=4, alpha=0.25, color="#0e7c6b", edgecolors="none")
ax.set_xlabel("centered pairwise cos, read $m_A$(L6)")
ax.set_ylabel("centered pairwise cos, write $v_A$")
gr = geom_rows[0]
ax.set_title(f"Pairwise congruence (2346 pairs)\nPearson {gr[1]:.3f} · gram-CKA {gr[4]:.3f} · feat-align {gr[5]:.3f}", fontsize=10)

ax = axes[1]
methods = ["ridge", "rotation", "rotation+scale", "trainmean_baseline"]
labels = ["ridge", "rotation\n(Procrustes)", "rotation\n+scale", "train-mean\nbaseline"]
w = 0.38
for off, L, color in [(-w / 2, 6, "#0e7c6b"), (w / 2, 13, "#2a5fd1")]:
    vals = [next(r_[3] for r_ in fit_rows if r_[0] == L and r_[1] == m) for m in methods]
    ax.bar(np.arange(4) + off, vals, width=w, color=color, label=f"read L{L}")
ax.set_xticks(range(4), labels, fontsize=8)
ax.set_ylabel("held-out $R^2$ (testmean)")
ax.axhline(0, color="0.5", lw=0.8)
ax.legend(frameon=False, fontsize=9)
ax.set_title("Held-out prediction of task FVs (14 tasks)", fontsize=10)

ax = axes[2]
for L, color in [(6, "#0e7c6b"), (13, "#2a5fd1")]:
    sv = spectra[f"L{L}_ridge"]
    ax.plot(np.arange(1, len(sv) + 1), sv / sv[0], color=color, label=f"ridge map, read L{L}")
ax.set_yscale("log")
ax.set_xlabel("singular value index")
ax.set_ylabel("$\\sigma_i / \\sigma_1$")
ax.set_title("Ridge-map spectrum on the train span\n(flat = scaled rotation)", fontsize=10)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "rotation_vs_ridge.png", dpi=180)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
bins = np.linspace(-1, 1, 81)
for ax, L in zip(axes2, READ_LAYERS):
    for variant, color in [("uncentered", "0.55"), ("centered", "#a8742a")]:
        mism = spectra[f"L{L}_crossfam_mismatched_{variant}"]
        ax.hist(mism, bins=bins, density=True, color=color, alpha=0.3,
                label=f"mismatched, {variant}")
    for variant, color in [("uncentered", "k"), ("centered", "#c0392b")]:
        matched = spectra[f"L{L}_crossfam_matched_{variant}"]
        ax.hist(matched, bins=bins, density=True, color=color, alpha=0.75,
                histtype="step", lw=1.8, label=f"matched cos($m_A$,$v_A$), {variant}")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel(f"cos(read $m_A$(L{L}), write $v_B$)")
    ax.set_title(f"read L{L} vs task FV", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
axes2[0].set_ylabel("density")
fig2.suptitle("Cross-family cosines: each task's read feature vs its own / other tasks' FVs", fontsize=11)
fig2.tight_layout()
fig2.savefig(OUT / "crossfamily_cos_hists.png", dpi=180)
print(open(OUT / "rotation_vs_ridge_summary.csv").read())
