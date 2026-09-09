#!/usr/bin/env python
"""Appendix G follow-up: is the read->write rotation a rotation in only a FEW PLANES?

The k-dimensional sweep in claim6_meanresid_map.py restricts the INPUT (top-k read PCs); it
tests the rank of the map. This script restricts the ROTATION instead: R = exp(A) with A a
skew-symmetric generator of rank 2m, i.e. a genuine rotation of R^d that turns exactly m
planes and is the identity elsewhere. Predictor as in Appendix G,

    v_hat_A = v_bar + s * R (u_A - u_bar),   means from the 55 train tasks, one scalar s,

fitted by Adam on the train tasks (exp(A) = I + Q (exp(S) - I) Q^T with Q (d x 2m) orthonormal
and S (2m x 2m) skew, so no d x d exponential is formed). Three inits per m (one seeded from the
full Procrustes pairing of the top-m read PCs, two random); the fit with the lowest TRAIN loss
is kept. Held-out R^2 on the 14 tasks, test-mean reference, is reported two ways:

  projected  : held-out u_A projected onto the 54-dim train read span before the map. This is
               what the thin Procrustes fit of Appendix G does implicitly (its R is a rank-54
               partial isometry, not an orthogonal matrix); m = 54 reproduces its 0.588.
  fullspace  : the same fitted R applied as a true rotation of R^d. The 38% of held-out read
               variance outside the train span passes through unrotated (orthogonal to the FVs).

Figure (user decision 2026-09-09): the genuine full-space rotation only, against the unconstrained
linear map (ridge, meanresid_map/fits_summary.csv) as the reference. Ridge is a legitimate
full-space linear map (it annihilates the out-of-span component; a rotation cannot), so the two are
on the same footing. The projected-input column stays in the CSV. `--plot_only` redraws from the CSV.

Writes results/69_task_run/understanding_read_write_linear_map/meanresid_map/
  mplane_sweep.csv, mplane_sweep.png
CPU, fp64, ~8 min (plot only: seconds).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src", _BOOT / "src" / "eval_scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from utils.paper_style import apply_paper_style, C  # noqa: E402
import claim6_meanresid_map as CM  # noqa: E402  (load / procrustes / r2 / OUT)

apply_paper_style()
torch.set_default_dtype(torch.float64)
OUT = CM.OUT
MS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 54)
STEPS, LR, SEED = 2500, 0.02, 0


def apply_rot(M, L, s, X):
    """s * X exp(A) with A = Q (L - L^T) Q^T, Q = qr(M). Row-vector convention as in Appendix G."""
    Q, _ = torch.linalg.qr(M)
    S = L - L.T
    return s * (X + (X @ Q) @ (torch.linalg.matrix_exp(S) - torch.eye(S.shape[0])) @ Q.T)


def fit(m, init, Xc, Yc, Vx, Rfull, sfull):
    d = Xc.shape[1]
    if init == "random":
        M = torch.randn(d, 2 * m) / d ** 0.5
        L = 0.1 * torch.randn(2 * m, 2 * m)
    else:  # plane i = (read PC i, its image under the full Procrustes map), 90 degrees
        P = Vx[:m]
        img = torch.tensor(P.numpy() @ Rfull)
        img = img - (img @ P.T) @ P
        img = img / img.norm(dim=1, keepdim=True)
        M = torch.cat([P, img], 0).T.clone()
        L = torch.zeros(2 * m, 2 * m)
        for i in range(m):
            L[i, m + i] = torch.pi / 2
    M.requires_grad_(); L.requires_grad_()
    s = torch.tensor(float(sfull), requires_grad=True)
    opt = torch.optim.Adam([M, L, s], lr=LR)
    denom = (Yc ** 2).sum()
    best = (np.inf, None)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((apply_rot(M, L, s, Xc) - Yc) ** 2).sum() / denom
        loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), (M.detach().clone(), L.detach().clone(), s.item()))
    return best


def main():
    torch.manual_seed(SEED)
    tasks, is_train, U, _, Y = CM.load()
    Xtr, Xte, Ytr, Yte = U[is_train], U[~is_train], Y[is_train], Y[~is_train]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    B = np.linalg.svd(Xtr - xm, full_matrices=False)[2][:len(Xtr) - 1]      # train read span
    Xc, Yc = torch.tensor(Xtr - xm), torch.tensor(Ytr - ym)
    Xte_full = torch.tensor(Xte - xm)
    Xte_proj = torch.tensor(((Xte - xm) @ B.T) @ B)
    Yte_t, ref, ym_t = torch.tensor(Yte), torch.tensor(Yte.mean(0)), torch.tensor(ym)
    Rfull, sfull, _ = CM.procrustes(Xc.numpy(), Yc.numpy())
    _, _, Vx = torch.linalg.svd(Xc, full_matrices=False)
    full_rs = CM.r2(sfull * (Xte_proj.numpy() @ Rfull) + ym, Yte, Yte.mean(0))
    print(f"held-out read variance inside the train read span: "
          f"{np.linalg.norm(Xte_proj.numpy(), 'fro') ** 2 / np.linalg.norm(Xte_full.numpy(), 'fro') ** 2:.3f}")
    print(f"full Procrustes rotation+scale (Appendix G): held-out R2 {full_rs:.3f}, s={sfull:.3f}")

    rows = []
    for m in MS:
        fits = [fit(m, init, Xc, Yc, Vx, Rfull, sfull) for init in ("procrustes", "random", "random")]
        _, (M, L, sv) = min(fits, key=lambda f: f[0])
        with torch.no_grad():
            rows.append({"m_planes": m, "scale": round(sv, 3),
                         "train_r2": round(CM.r2(apply_rot(M, L, sv, Xc).numpy(), Yc.numpy(), 0.0), 4),
                         "heldout_r2_projected_input": round(CM.r2((apply_rot(M, L, sv, Xte_proj) + ym_t).numpy(), Yte, Yte.mean(0)), 4),
                         "heldout_r2_fullspace": round(CM.r2((apply_rot(M, L, sv, Xte_full) + ym_t).numpy(), Yte, Yte.mean(0)), 4)})
        r = rows[-1]
        print(f"m={m:>2}: train {r['train_r2']:.3f} | held-out projected {r['heldout_r2_projected_input']:.3f} "
              f"| full-space {r['heldout_r2_fullspace']:.3f} | s={r['scale']:.2f}", flush=True)
    with open(OUT / "mplane_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    plot(rows)
    print("wrote", OUT / "mplane_sweep.csv", OUT / "mplane_sweep.png")


def plot(rows):
    ridge = next(float(r["heldout_r2_testmean"]) for r in csv.DictReader(open(OUT / "fits_summary.csv"))
                 if r["method"] == "ridge")
    ms = [int(r["m_planes"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(ms, [float(r["heldout_r2_fullspace"]) for r in rows], color=C.red, lw=1.6, marker="o",
            label="rotation in $m$ planes + one scalar")
    ax.axhline(ridge, color=C.map, lw=0.9, ls=":", alpha=0.9)
    ax.text(ms[0], ridge + 0.012, f"unconstrained linear map (ridge) {ridge:.2f}", ha="left", fontsize=8.5, color=C.map)
    ax.set_xscale("log", base=2); ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("$m$ = number of planes the rotation acts in")
    ax.set_ylabel("held-out $R^2$ (14 tasks)")
    ax.set_ylim(-0.05, 1.02); ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Can a rotation in a few planes carry the read feature to the write feature?")
    fig.tight_layout(); fig.savefig(OUT / "mplane_sweep.png"); plt.close(fig)


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        plot(list(csv.DictReader(open(OUT / "mplane_sweep.csv"))))
        print("wrote", OUT / "mplane_sweep.png")
    else:
        main()
