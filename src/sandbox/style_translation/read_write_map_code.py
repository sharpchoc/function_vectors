#!/usr/bin/env python
"""Read -> write map for the code-convention families (CPU).

Data: prompt_pairs/<family>.npz (capture_prompt_pairs_code.py): per prompt read = L8 mean over evidence tokens, write = L24 cue-token
residual, k in {3,4}, correct completions, both poles. Family-level 80/20 split (seeded): 45 train / 11 test families.
Map A (prompt -> centroid): X = per-prompt read features, Y = the write CENTROID of the prompt's (family, pole) (mean write feature
over that family-pole's prompts). Map B (centroid -> centroid): X = read centroids, Y = write centroids (2 rows per family).
Ridge with intercept (train-mean centering); lambda chosen by leave-one-family-out CV on the train families (variance-weighted R^2),
refit on all train families, evaluated on the test families. LOFO R^2 pools the held-out predictions of all folds. R^2 = 1 - ||Y - Yhat||_F^2 / ||Y - mean(Y_test)||_F^2 (variance-weighted
over dimensions), plus the mean per-dimension R^2. Controls: predicting the train-mean write centroid (R^2 <= 0 by construction) and a
family-shuffled target (Map A). Outputs -> <results>/read_write_map/: split.json, cv.csv, results.json, cv_curve.png, test_scatter.png.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths


def r2(Y, Yh):
    res = float(((Y - Yh) ** 2).sum()); tot = max(float(((Y - Y.mean(0)) ** 2).sum()), 1e-12)
    per = 1 - ((Y - Yh) ** 2).sum(0) / np.maximum(((Y - Y.mean(0)) ** 2).sum(0), 1e-12)
    return 1 - res / tot, float(per.mean())


def ridge_fit(X, Y, lam):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; Xt = torch.as_tensor(X, dtype=torch.float64, device=dev); Yt = torch.as_tensor(Y, dtype=torch.float64, device=dev)
    mx, my = Xt.mean(0), Yt.mean(0); Xc, Yc = Xt - mx, Yt - my
    G = Xc.T @ Xc; W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=dev), Xc.T @ Yc)
    return W.cpu().numpy(), mx.cpu().numpy(), my.cpu().numpy()


def ridge_predict(model, X):
    W, mx, my = model; return (X - mx) @ W + my


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--seed", type=int, default=43); ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--lams", nargs="*", type=float, default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--skip_cv", action="store_true", help="reuse cv.csv from a previous run (no LOFO recomputation)")
    args = ap.parse_args()
    MP = model_paths(args.model); R = MP["results"]; OUT = R / "read_write_map"; OUT.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(R / "code_pool.json"))["pool"]; rng = np.random.default_rng(args.seed)
    perm = list(rng.permutation(pool)); n_test = int(round(args.test_frac * len(pool))); test = sorted(perm[:n_test]); train = sorted(perm[n_test:])
    json.dump(dict(seed=args.seed, train=train, test=test), open(OUT / "split.json", "w"), indent=1)
    D = {}
    for f in pool:
        z = np.load(MP["prompt_pairs"] / f"{f}.npz"); X = z["read"].astype(np.float32); W = z["write"].astype(np.float32); pole = z["pole"]
        cen = {p: (X[pole == p].mean(0), W[pole == p].mean(0)) for p in ("nat", "alt")}
        D[f] = dict(X=X, W=W, pole=pole, cen=cen, Yc=np.stack([cen[p][1] for p in pole]))       # Yc: each prompt's family-pole write centroid
    def stackA(fams):
        return np.concatenate([D[f]["X"] for f in fams]), np.concatenate([D[f]["Yc"] for f in fams]), np.concatenate([[f] * len(D[f]["X"]) for f in fams])
    def stackB(fams):
        return np.stack([D[f]["cen"][p][0] for f in fams for p in ("nat", "alt")]), np.stack([D[f]["cen"][p][1] for f in fams for p in ("nat", "alt")])
    # --- LOFO CV on train families
    cv = []
    XA, YA, FA = stackA(train); XB, YB = stackB(train)
    FB = np.array([f for f in train for _ in ("nat", "alt")])
    # fast LOFO: per fold ONE eigendecomposition of the centred Gram, then every lambda is a cheap product
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    def lofo_preds(X_, Y_, F_, lams):
        Xt, Yt = T(X_), T(Y_); preds = {lam: np.zeros_like(Y_) for lam in lams}
        for f in sorted(set(F_)):
            m = torch.as_tensor(F_ != f, device=dev); Xi, Yi = Xt[m], Yt[m]; mx, my = Xi.mean(0), Yi.mean(0); Xc = Xi - mx
            s_, V = torch.linalg.eigh(Xc.T @ Xc); C = V.T @ (Xc.T @ (Yi - my)); Xo = (Xt[~m] - mx) @ V
            for lam in lams:
                preds[lam][(F_ == f)] = ((Xo / (s_ + lam)) @ C + my).cpu().numpy()
        return preds
    if args.skip_cv and (OUT / "cv.csv").exists():
        cv = [dict(lam=float(r["lam"]), A_lofo_r2=float(r["A_lofo_r2"]), B_lofo_r2=float(r["B_lofo_r2"])) for r in csv.DictReader(open(OUT / "cv.csv"))]
    else:
        PA = lofo_preds(XA, YA, FA, args.lams); PB = lofo_preds(XB, YB, FB, args.lams)
        for lam in args.lams:
            cv.append(dict(lam=lam, A_lofo_r2=r2(YA, PA[lam])[0], B_lofo_r2=r2(YB, PB[lam])[0])); print(cv[-1], flush=True)
    with open(OUT / "cv.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cv[0])); w.writeheader(); w.writerows(cv)
    lamA = max(cv, key=lambda c: c["A_lofo_r2"])["lam"]; lamB = max(cv, key=lambda c: c["B_lofo_r2"])["lam"]
    # --- refit and test
    XT, YT, FT = stackA(test); XBt, YBt = stackB(test)
    mA = ridge_fit(XA, YA, lamA); predA = ridge_predict(mA, XT); r2A, r2A_dim = r2(YT, predA)
    mB = ridge_fit(XB, YB, lamB); predB = ridge_predict(mB, XBt); r2B, r2B_dim = r2(YBt, predB)
    # per-family R^2 for map A (each test family's prompts vs its two centroids)
    perfam = {f: r2(YT[FT == f], predA[FT == f])[0] for f in test}
    # controls
    base = np.tile(YA.mean(0), (len(YT), 1)); r2_mean = r2(YT, base)[0]
    sh = rng.permutation(len(train)); shuffled = {f: train[sh[i]] for i, f in enumerate(train)}
    YA_sh = np.concatenate([np.stack([D[shuffled[f]]["cen"][p][1] for p in D[f]["pole"]]) for f in train]); mS = ridge_fit(XA, YA_sh, lamA); r2_sh = r2(YT, ridge_predict(mS, XT))[0]
    # map A also scored centroid-wise: average the per-prompt predictions of a test family-pole and compare with its write centroid
    predA_cen = np.stack([predA[(FT == f) & (np.concatenate([D[g]["pole"] for g in test]) == p)].mean(0) for f in test for p in ("nat", "alt")]); r2A_cen = r2(YBt, predA_cen)[0]
    # cosine of the predicted vs true nat-alt write DIFFERENCE per test family (direction of the style-level write vector)
    cosd, cosdA = [], []
    for i, f in enumerate(test):
        dt = YBt[2 * i] - YBt[2 * i + 1]; dp = predB[2 * i] - predB[2 * i + 1]; cosd.append(float(dt @ dp / (np.linalg.norm(dt) * np.linalg.norm(dp) + 1e-9)))
        dA = predA_cen[2 * i] - predA_cen[2 * i + 1]; cosdA.append(float(dt @ dA / (np.linalg.norm(dt) * np.linalg.norm(dA) + 1e-9)))
    np.savez_compressed(OUT / "test_predictions.npz", predA=predA.astype(np.float32), YT=YT.astype(np.float32), FT=FT, predB=predB.astype(np.float32), YBt=YBt.astype(np.float32), test=np.array(test))
    res = dict(seed=args.seed, n_train_families=len(train), n_test_families=len(test), n_train_prompts=int(len(XA)), n_test_prompts=int(len(XT)), dim=int(XA.shape[1]),
               lambda_A=lamA, lambda_B=lamB, A_prompt_to_centroid=dict(r2_variance_weighted=r2A, r2_mean_per_dim=r2A_dim, r2_of_averaged_predictions_vs_centroids=r2A_cen, cos_pred_vs_true_nat_minus_alt_per_family=dict(zip(test, cosdA)), cos_mean=float(np.mean(cosdA)), note="per-family R^2 is degenerate here (a family's targets take only two values) and is not reported"),
               B_centroid_to_centroid=dict(r2_variance_weighted=r2B, r2_mean_per_dim=r2B_dim, cos_pred_vs_true_nat_minus_alt_per_family=dict(zip(test, cosd)), cos_mean=float(np.mean(cosd))),
               controls=dict(train_mean_prediction_r2=r2_mean, family_shuffled_targets_r2=r2_sh), cv=cv)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.semilogx([c["lam"] for c in cv], [c["A_lofo_r2"] for c in cv], "o-", label="map A: prompt → write centroid"); ax.semilogx([c["lam"] for c in cv], [c["B_lofo_r2"] for c in cv], "s-", label="map B: read centroid → write centroid")
    ax.set_xlabel("ridge λ"); ax.set_ylabel("leave-one-family-out R² (train families)"); ax.grid(alpha=.3); ax.legend(); ax.set_title("λ selection"); fig.tight_layout(); fig.savefig(OUT / "cv_curve.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(test)); w = .38
    ax.bar(x - w / 2, cosdA, w, color="#1f4f7a", label=f"map A (prompt → centroid, predictions averaged per pole), mean {np.mean(cosdA):.2f}")
    ax.bar(x + w / 2, cosd, w, color="#2b7a4b", label=f"map B (centroid → centroid), mean {np.mean(cosd):.2f}")
    ax.set_xticks(x); ax.set_xticklabels(test, rotation=45, ha="right", fontsize=8); ax.set_ylim(-1, 1); ax.axhline(0, color="k", lw=.6); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    ax.set_ylabel("cos(predicted, true) natural − alternative write difference"); ax.set_title("Held-out test families: does the mapped read feature point along the family's write direction?")
    fig.tight_layout(); fig.savefig(OUT / "test_scatter.png", dpi=130); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k != "cv"}, indent=1)); print("->", OUT)


if __name__ == "__main__":
    main()
