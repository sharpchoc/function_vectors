#!/usr/bin/env python
"""Read -> write map: map TYPES beyond plain ridge, same data/scoring as map_layer_sweep.py (GPU, 2026-11-01).

Rows / centroids / scoring exactly as the paper map (unit pair contrasts, held-out family centroids, R^2 vs the training-family mean).
Methods (all fitted on TRAINING families only):
  ridge<l>          plain ridge, lambda l (the paper map at l = 10)
  rrr<l>_<r>        reduced-rank ridge: ridge W projected onto the top-r right singular vectors of the fitted training predictions
  kern<b>           family analogy: prediction = softmax(b * cos(x, X_f)) over training-family read centroids X_f, applied to their write centroids
  ridgekern<l>_<b>  average of ridge<l> and kern<b>
  shrink<l>_<s>     ridge<l> prediction pulled towards the training-mean centroid: mean + s * (pred - mean)
Output: one JSON line per (feature pair, method, split).
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


def run(X, Y, F, train, test, methods):
    dt = torch.float64; mtr = torch.as_tensor(np.isin(F, train), device=X.device)
    Xtr, Ytr = X[mtr].to(dt), Y[mtr].to(dt); mx, my = Xtr.mean(0), Ytr.mean(0); Xc = Xtr - mx
    s_, V = torch.linalg.eigh(Xc.T @ Xc); C = V.T @ (Xc.T @ (Ytr - my))
    Xte, Yte = centroids(X, F, test).to(dt), centroids(Y, F, test).to(dt); Xtrc, Ytrc = centroids(X, F, train).to(dt), centroids(Y, F, train).to(dt)
    W = {}
    def ridgeW(l):
        if l not in W:
            W[l] = (V / (s_ + l)) @ C
        return W[l]
    def kern(b):
        w = torch.softmax(b * (unit(Xte) @ unit(Xtrc).T), 1); return w @ Ytrc
    out = {}
    for m in methods:
        if m.startswith("ridgekern"):
            l, b = map(float, m[9:].split("_")); P = 0.5 * (((Xte - mx) @ ridgeW(l) + my) + kern(b))
        elif m.startswith("ridge"):
            P = (Xte - mx) @ ridgeW(float(m[5:])) + my
        elif m.startswith("rrr"):
            l, r = m[3:].split("_"); Wl = ridgeW(float(l)); _, _, Vh = torch.linalg.svd(Xc @ Wl, full_matrices=False); B = Vh[: int(r)].T
            P = (Xte - mx) @ (Wl @ B @ B.T) + my
        elif m.startswith("kern"):
            P = kern(float(m[4:]))
        elif m.startswith("shrink"):
            l, s = map(float, m[6:].split("_")); P = Ytrc.mean(0) + s * (((Xte - mx) @ ridgeW(l) + my) - Ytrc.mean(0))
        out[m] = score(P, Yte, Ytrc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--pairs", nargs="+", required=True, help="'<Xspec> <Yspec>' e.g. 'ev_mean@8 cue@24'")
    ap.add_argument("--methods", nargs="+", required=True); ap.add_argument("--splits", nargs="+", required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    MP = model_paths(a.model); D = Data(MP, "cuda"); F = D.F; fh = open(a.out, "a")
    for pr in a.pairs:
        xs, ys = pr.split(); X, Y = D.feat(xs), D.feat(ys)
        for s in a.splits:
            sp = json.load(open(s))
            for m, (r2tr, r2te, cos) in run(X, Y, F, sp["train"], sp["test"], a.methods).items():
                fh.write(json.dumps(dict(pair=pr, method=m, split=Path(s).stem, r2_train_mean=r2tr, r2_test_mean=r2te, cos=cos)) + "\n")
        fh.flush(); print("done", pr, flush=True)


if __name__ == "__main__":
    main()
