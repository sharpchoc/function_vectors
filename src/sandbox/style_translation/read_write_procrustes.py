#!/usr/bin/env python
"""Constrained read→write maps (user request 2026-09-12): orthogonal Procrustes (+ scalar scale) and rank-restricted
Procrustes (rotation between the top-k PCs of read and write) fit on the 11 lexically identical families' per-prompt
rows (train-mean centred), scored on the 6 lexically diverse families. Controls: shuffled pairing. Reference: ridge λ=10."""
import sys
from pathlib import Path
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.read_write_ridge import DualRidge, evaluate, load, ORIGINAL_LEX
from src.sandbox.style_translation.family_groups import FIXED

class Lin:
    """pred = ym + (X - xm) @ M ; evaluate() only needs .predict/.ym/.lam"""
    def __init__(self, xm, ym, M, lam="-"): self.xm, self.ym, self.M, self.lam = xm, ym, M, lam
    def predict(self, X): return self.ym + (X - self.xm) @ self.M

def procrustes(Xc, Yc, k=None, scale=True):
    """min ||s Xc P R Q^T - Yc||: P,Q = top-k PC bases of Xc,Yc (identity when k None); R orthogonal."""
    if k is None:
        A = Xc; B = Yc; P = Q = None
    else:
        _, _, Vx = np.linalg.svd(Xc, full_matrices=False); _, _, Vy = np.linalg.svd(Yc, full_matrices=False)
        P, Q = Vx[:k].T, Vy[:k].T; A, B = Xc @ P, Yc @ Q
    U, S, Vt = np.linalg.svd(A.T @ B, full_matrices=False); R = U @ Vt
    s = S.sum() / (A ** 2).sum() if scale else 1.0
    M = s * R if k is None else s * (P @ R @ Q.T)
    return M, s

def main(lr=12, lw=24):
    Xtr, Ytr, ftr, ptr = load(list(FIXED), lr, lw); Xte, Yte, fte, pte = load(list(ORIGINAL_LEX), lr, lw)
    xm, ym = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - xm, Ytr - ym
    rng = np.random.default_rng(0); Xsh = Xc[rng.permutation(len(Xc))]
    def report(name, model):
        o, per = evaluate(model, Xte, Yte, fte, pte, name)
        print(f"{name:44s} R²(train-mean) {o['r2_trainmean']:+.3f} R²(test-mean) {o['r2_testmean']:+.3f} within {o['r2_within']:+.3f} | "
              f"centroid cos {o['cos_centroid_mean']:.2f} | convention cos {o['cos_diff_mean']:.2f} R² {o['r2_diff']:+.3f} | "
              + " ".join(f"{x['family'][:6]} {x['cos_diff']:.2f}" for x in per), flush=True)
    print(f"read L{lr} → write L{lw}; train = 11 identical families ({len(Xtr)} prompts), test = 6 diverse ({len(Xte)})")
    report("ridge λ=10 (reference)", DualRidge(Xtr, Ytr).fit(10))
    for k in (None, 1000, 200, 50, 11):
        M, s = procrustes(Xc, Yc, k); report(f"Procrustes + scale, rank {'full' if k is None else k} (s={s:.2f})", Lin(xm, ym, M))
    M, s = procrustes(Xc, Yc, None, scale=False); report("Procrustes, no scale, full rank", Lin(xm, ym, M))
    M, s = procrustes(Xsh, Yc, None); report("CONTROL shuffled pairing, Procrustes full", Lin(xm, ym, M))
    M, s = procrustes(Xsh, Yc, 200); report("CONTROL shuffled pairing, Procrustes rank 200", Lin(xm, ym, M))
    # Procrustes fit on the 11 CENTROID DIFFERENCES only (the convention-level map, 11 samples)
    Dx = np.array([Xc[(ftr == f) & (ptr == "nat")].mean(0) - Xc[(ftr == f) & (ptr == "alt")].mean(0) for f in FIXED])
    Dy = np.array([Yc[(ftr == f) & (ptr == "nat")].mean(0) - Yc[(ftr == f) & (ptr == "alt")].mean(0) for f in FIXED])
    M, s = procrustes(Dx, Dy, None); report(f"Procrustes on the 11 centroid differences (s={s:.2f})", Lin(xm, ym, M))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, 24)
