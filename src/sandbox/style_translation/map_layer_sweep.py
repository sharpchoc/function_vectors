#!/usr/bin/env python
"""Read -> write map: where to take the identification (read) feature from (GPU, 2026-11-01).

Same recipe as the paper map (v11): rows = (document, k) pair contrasts nat - alt over all k = 3/4 prompts, each unit-normalised; ridge
with intercept; scored on held-out family centroids (mean of unit pair contrasts, re-normalised), R^2 against the mean TRAINING-family
centroid (paper headline), R^2 against the held-out mean, mean cos(pred, true). Features from prompt_pairs_sites/ (capture_pair_sites_layers.py).

  prep     per dataset: pair contrasts at every layer for each site -> <artifacts>/map_sweep/diffs_<site>.npy [P, 29, D] fp16, fams.npy
  screen   every config in --configs at a FIXED lambda grid on the given split files (exploration only; selection happens here)
  confirm  chosen configs with the full protocol: lambda by leave-one-family-out on the training families (grid 1e-4..1e3, pooled R^2),
           refit, score; run on split files NOT used for screening
Config syntax: X=<site>@<layers>[|pca<k>] Y=<site>@<layer>; layers '8' (one layer), '6-10' (mean of the unit contrasts of layers 6..10,
re-normalised), '6+8+10' (concatenation of unit contrasts);
'ev_mean@8&cue@16' concatenates features from different sites. Example: "X=ev_mean@8 Y=cue@24" = the paper map.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths

LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
SITES = ("ev_mean", "ev_last", "cue")


def unit(a, dim=-1):
    return a / a.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def prep(MP, fams):
    S = MP["prompt_pairs"].parent / "prompt_pairs_sites"; OUT = MP["prompt_pairs"].parent / "map_sweep"; OUT.mkdir(exist_ok=True)
    rows = {s: [] for s in SITES}; F = []
    for f in fams:
        z = np.load(S / f"{f}.npz")
        idx = {(d, p, int(k)): i for i, (d, p, k) in enumerate(zip(z["doc_id"], z["pole"], z["k"]))}
        keys = sorted({(d, k) for (d, p, k) in idx if (d, "nat", k) in idx and (d, "alt", k) in idx})
        a = np.array([idx[(d, "nat", k)] for d, k in keys]); b = np.array([idx[(d, "alt", k)] for d, k in keys])
        for s in SITES:
            rows[s].append((z[s][a].astype(np.float32) - z[s][b].astype(np.float32)).astype(np.float16))
        F += [f] * len(keys); print(f"prep {f}: {len(keys)} pairs", flush=True)
    for s in SITES:
        np.save(OUT / f"diffs_{s}.npy", np.concatenate(rows[s]))
    np.save(OUT / "fams.npy", np.array(F)); print("prep done", OUT, flush=True)


class Data:
    def __init__(self, MP, dev):
        D = MP["prompt_pairs"].parent / "map_sweep"; self.F = np.load(D / "fams.npy"); self.dev = dev
        self.raw = {s: torch.from_numpy(np.load(D / f"diffs_{s}.npy", mmap_mode="r")[:]) for s in SITES}   # CPU fp16 [P, 29, D]
        self.cache = {}

    def feat(self, spec):
        """unit-normalised pair features [P, d] on the GPU (fp32) for 'site@layers'."""
        if spec in self.cache:
            return self.cache[spec]
        if "&" in spec:                                   # mixed sites: concatenation of unit features, rescaled to unit norm
            parts = spec.split("&"); return torch.cat([self.feat(q) for q in parts], 1) / np.sqrt(len(parts))
        site, L = spec.split("@")
        if "+" in L:
            x = torch.cat([unit(self.raw[site][:, int(l)].to(self.dev).float()) for l in L.split("+")], 1) / np.sqrt(len(L.split("+")))
        elif "-" in L:
            lo, hi = map(int, L.split("-")); x = unit(torch.stack([unit(self.raw[site][:, l].to(self.dev).float()) for l in range(lo, hi + 1)]).mean(0))
        else:
            x = unit(self.raw[site][:, int(L)].to(self.dev).float())
        if len(self.cache) > 24:
            self.cache.pop(next(iter(self.cache)))
        self.cache[spec] = x; return x


def centroids(x, F, fams):
    return torch.stack([unit(x[torch.as_tensor(F == f, device=x.device)].mean(0)) for f in fams])


def score(P, Yte, Ytr_c):
    r2_tr = 1 - ((Yte - P) ** 2).sum() / ((Yte - Ytr_c.mean(0)) ** 2).sum()
    r2_te = 1 - ((Yte - P) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    cos = ((P * Yte).sum(1) / (P.norm(dim=1) * Yte.norm(dim=1))).mean()
    return float(r2_tr), float(r2_te), float(cos)


def parse(cfg):
    d = dict(t.split("=", 1) for t in cfg.split()); x = d["X"]; pca = None
    if "|pca" in x:
        x, k = x.split("|pca"); pca = int(k)
    return x, d["Y"], pca


def fit_eval(X, Y, F, train, test, lams, lofo=False):
    """ridge per lambda (one eigendecomposition); with lofo=True lambda is chosen by leave-one-family-out on train (pooled R^2)."""
    dt = torch.float64; mtr = torch.as_tensor(np.isin(F, train), device=X.device)
    Xtr, Ytr, Ftr = X[mtr].to(dt), Y[mtr].to(dt), F[np.isin(F, train)]
    cvr = None
    if lofo:
        preds = {l: torch.zeros_like(Ytr) for l in lams}
        for f in train:
            m = torch.as_tensor(Ftr != f, device=X.device); Xi, Yi = Xtr[m], Ytr[m]; mx, my = Xi.mean(0), Yi.mean(0); Xc = Xi - mx
            s_, V = torch.linalg.eigh(Xc.T @ Xc); C = V.T @ (Xc.T @ (Yi - my)); Xo = (Xtr[~m] - mx) @ V
            for l in lams:
                preds[l][~m] = (Xo / (s_ + l)) @ C + my
        sst = ((Ytr - Ytr.mean(0)) ** 2).sum(); cvr = {l: float(1 - ((Ytr - preds[l]) ** 2).sum() / sst) for l in lams}
        lams = [max(cvr, key=cvr.get)]
    mx, my = Xtr.mean(0), Ytr.mean(0); Xc = Xtr - mx
    s_, V = torch.linalg.eigh(Xc.T @ Xc); C = V.T @ (Xc.T @ (Ytr - my))
    Xc_te = centroids(X, F, test).to(dt); Yc_te = centroids(Y, F, test).to(dt); Yc_tr = centroids(Y, F, train).to(dt)
    out = {}
    for l in lams:
        P = ((Xc_te - mx) @ V / (s_ + l)) @ C + my; out[l] = score(P, Yc_te, Yc_tr)
    return out, cvr


def pca_project(X, F, train, k):
    Xtr = X[torch.as_tensor(np.isin(F, train), device=X.device)]; mu = Xtr.mean(0)
    _, _, Vh = torch.linalg.svd((Xtr - mu).double(), full_matrices=False); B = Vh[:k].T.float()
    return (X - mu) @ B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prep", "screen", "confirm"]); ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="*", default=[]); ap.add_argument("--configs_file", default=None)
    ap.add_argument("--splits", nargs="*", default=[]); ap.add_argument("--lams", nargs="*", type=float, default=[1.0, 10.0, 100.0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    MP = model_paths(args.model); pool = json.load(open(MP["results"] / "code_pool.json"))["pool"]
    if args.mode == "prep":
        return prep(MP, pool)
    dev = "cuda"; Dd = Data(MP, dev); F = Dd.F
    cfgs = args.configs + ([l.strip() for l in open(args.configs_file) if l.strip()] if args.configs_file else [])
    splits = [(Path(s).stem, json.load(open(s))) for s in args.splits]
    for _, sp in splits:
        assert set(sp["train"]) | set(sp["test"]) == set(pool) and not set(sp["train"]) & set(sp["test"]), "split must partition the pool"
    done = set()
    if args.out and args.out.exists():
        done = {(r["config"], r["split"]) for r in map(json.loads, open(args.out))}
    fh = open(args.out, "a") if args.out else None
    for cfg in cfgs:
        t0 = time.time(); xs, ys, pca = parse(cfg); X0 = Dd.feat(xs); Y = Dd.feat(ys)
        for name, sp in splits:
            if (cfg, name) in done:
                continue
            X = X0 if pca is None else pca_project(X0, F, sp["train"], pca)
            res, cvr = fit_eval(X, Y, F, sp["train"], sp["test"], args.lams, lofo=args.mode == "confirm")
            for l, (r2tr, r2te, cos) in res.items():
                rec = dict(config=cfg, split=name, lam=l, r2_train_mean=r2tr, r2_test_mean=r2te, cos=cos, n_train=len(sp["train"]), n_test=len(sp["test"]),
                           lofo_r2=None if cvr is None else cvr[l])
                if fh:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"{cfg}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
