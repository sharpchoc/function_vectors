#!/usr/bin/env python
"""Why does the read -> write map stop near R^2 .25? Diagnostics on the paper recipe (GPU, 2026-11-01).

  ceiling   oracle: held-out write centroids projected onto the span of the TRAINING write centroids (minus their mean), and onto the
            top-k principal directions of the training per-prompt write contrasts; R^2 vs the training mean = the best any map whose
            predictions live in that space can do
  curve     learning curve: R^2 (ridge, lambda 10) on a fixed held-out family set as the number of training families grows (random subsets)
  docs      held-out DOCUMENTS instead of families: fit on the non-held-out documents (write_split.heldout) of every family, score on the
            held-out documents' family centroids (same scoring formula, reference = mean of the training-document centroids)
  identity  no map: the held-out family's own X centroid used as its Y prediction (same dimension required), and X centroid rescaled
            onto the training mean (mean + (x - mean_x)); how much of an X -> Y R^2 is just X ~ Y
  pooled    original-prompt map trained on its training families' rows from BOTH prompt sets (original + translation), scored on original
            held-out family centroids
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.map_layer_sweep import Data, centroids, score, unit
from src.sandbox.style_translation.write_split import heldout


def ridge_pred(Xtr, Ytr, Xq, lam=10.0):
    dt = torch.float64; Xtr, Ytr, Xq = Xtr.to(dt), Ytr.to(dt), Xq.to(dt); mx, my = Xtr.mean(0), Ytr.mean(0); Xc = Xtr - mx
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=dt, device=Xc.device), Xc.T @ (Ytr - my))
    return (Xq - mx) @ W + my


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ceiling", "curve", "docs", "pooled", "identity"]); ap.add_argument("--model", required=True)
    ap.add_argument("--x", default="ev_mean@8"); ap.add_argument("--y", default="cue@24"); ap.add_argument("--splits", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--draws", type=int, default=10)
    a = ap.parse_args(); MP = model_paths(a.model); D = Data(MP, "cuda"); F = D.F; X, Y = D.feat(a.x), D.feat(a.y); fh = open(a.out, "a")
    dt = torch.float64
    if a.mode == "ceiling":
        for s in a.splits:
            sp = json.load(open(s)); tr, te = sp["train"], sp["test"]
            Yte, Ytrc = centroids(Y, F, te).to(dt), centroids(Y, F, tr).to(dt); mu = Ytrc.mean(0)
            Q, _ = torch.linalg.qr((Ytrc - mu).T); P = mu + ((Yte - mu) @ Q) @ Q.T
            rec = dict(mode="ceiling", split=Path(s).stem, span="train_centroids", k=Q.shape[1], r2=score(P, Yte, Ytrc)[0])
            fh.write(json.dumps(rec) + "\n")
            Ytr = Y[torch.as_tensor(np.isin(F, tr), device=Y.device)].to(dt); _, _, Vh = torch.linalg.svd(Ytr - Ytr.mean(0), full_matrices=False)
            for k in (36, 100, 300, 1000):
                B = Vh[:k].T; P = mu + ((Yte - mu) @ B) @ B.T
                fh.write(json.dumps(dict(mode="ceiling", split=Path(s).stem, span="train_prompt_pcs", k=k, r2=score(P, Yte, Ytrc)[0])) + "\n")
    elif a.mode == "curve":
        fams = sorted(set(F)); rng = np.random.default_rng(0)
        for d in range(a.draws):
            perm = rng.permutation(fams); te = list(perm[:8]); rest = list(perm[8:])
            for n in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
                if n > len(rest):
                    continue
                tr = list(rng.permutation(rest)[:n]); m = torch.as_tensor(np.isin(F, tr), device=X.device)
                P = ridge_pred(X[m], Y[m], centroids(X, F, te)); r2 = score(P, centroids(Y, F, te).to(dt), centroids(Y, F, tr).to(dt))
                fh.write(json.dumps(dict(mode="curve", draw=d, n_train=n, r2=r2[0], cos=r2[2])) + "\n")
            fh.flush()
    elif a.mode == "docs":
        # need doc ids per pair row: rebuild from prompt_pairs_sites row order (prep used sorted (doc, k) keys per family)
        S = MP["prompt_pairs"].parent / "prompt_pairs_sites"; doc = []
        for f in dict.fromkeys(F):
            z = np.load(S / f"{f}.npz", mmap_mode="r"); idx = {(d_, p, int(k)) for d_, p, k in zip(z["doc_id"], z["pole"], z["k"])}
            keys = sorted({(d_, k) for (d_, p, k) in idx if (d_, "nat", k) in idx and (d_, "alt", k) in idx}); doc += [d_ for d_, _ in keys]
        ho = np.array([heldout(d_) for d_ in doc]); fams = sorted(set(F)); assert len(doc) == len(F)
        mtr = torch.as_tensor(~ho, device=X.device); Ftr, Fte = F[~ho], F[ho]
        Xte, Yte = centroids(X[torch.as_tensor(ho, device=X.device)], Fte, fams), centroids(Y[torch.as_tensor(ho, device=X.device)], Fte, fams).to(dt)
        Ytrc = centroids(Y[mtr], Ftr, fams).to(dt)
        for lam in (1.0, 10.0, 100.0):
            r2 = score(ridge_pred(X[mtr], Y[mtr], Xte, lam), Yte, Ytrc)
            fh.write(json.dumps(dict(mode="docs", lam=lam, n_fam=len(fams), r2_train_mean=r2[0], r2_test_mean=r2[1], cos=r2[2])) + "\n")
        # reference: a family's own training-document centroid as the "prediction" (how stable a family's write centroid is across documents)
        r2 = score(Ytrc, Yte, Ytrc); fh.write(json.dumps(dict(mode="docs", lam=None, note="own training-document centroid", r2_train_mean=r2[0], cos=r2[2])) + "\n")
    elif a.mode == "identity":
        for s in a.splits:
            sp = json.load(open(s)); tr, te = sp["train"], sp["test"]
            Xte, Xtrc = centroids(X, F, te).to(dt), centroids(X, F, tr).to(dt); Yte, Ytrc = centroids(Y, F, te).to(dt), centroids(Y, F, tr).to(dt)
            r_id = score(Xte, Yte, Ytrc); r_sh = score(Ytrc.mean(0) + (Xte - Xtrc.mean(0)), Yte, Ytrc)
            fh.write(json.dumps(dict(mode="identity", x=a.x, y=a.y, split=Path(s).stem, r2_identity=r_id[0], cos_identity=r_id[2], r2_shifted=r_sh[0], cos_shifted=r_sh[2])) + "\n")
    elif a.mode == "pooled":
        D2 = Data(model_paths("qwen25_code_tr"), "cuda"); F2 = D2.F; X2, Y2 = D2.feat(a.x), D2.feat(a.y)
        for s in a.splits:
            sp = json.load(open(s)); tr, te = sp["train"], sp["test"]
            m1 = torch.as_tensor(np.isin(F, tr), device=X.device); m2 = torch.as_tensor(np.isin(F2, tr), device=X.device)
            Yte, Ytrc, Xte = centroids(Y, F, te).to(dt), centroids(Y, F, tr).to(dt), centroids(X, F, te)
            base = score(ridge_pred(X[m1], Y[m1], Xte), Yte, Ytrc)
            pooled = score(ridge_pred(torch.cat([X[m1], X2[m2]]), torch.cat([Y[m1], Y2[m2]]), Xte), Yte, Ytrc)
            fh.write(json.dumps(dict(mode="pooled", split=Path(s).stem, r2_base=base[0], r2_pooled=pooled[0], cos_base=base[2], cos_pooled=pooled[2])) + "\n")
    fh.flush(); print("done", a.mode, flush=True)


if __name__ == "__main__":
    main()
