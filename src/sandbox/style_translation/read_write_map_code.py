#!/usr/bin/env python
"""Read -> write map for the code-convention families: ONE linear map from per-prompt read features to per-prompt write features.

Data: prompt_pairs/<family>.npz (capture_prompt_pairs_code.py): per prompt read = L8 residual mean over the evidence tokens,
write = L24 residual at the final cue token; k in {3, 4}, both poles, all documents; rows carry style_ok / judge_ok / correct flags
(--select correct uses the correct ones, the default).
Split: families 80/20 by seed (45 train / 11 test). Fit: ridge regression with intercept (train-mean centring), X = read, Y = write,
on all prompts of the train families; lambda chosen by leave-one-family-out CV on the train families (held-out predictions of all folds
pooled, R^2 variance-weighted over dimensions). The chosen map is refit on all train families and saved; evaluation on the test
families is a separate step. Outputs -> <results>/read_write_map/: split.json, cv.csv, cv_curve.png, fit.json;
model -> <artifacts>/read_write_map/ridge_L8_to_L24.npz (W [D, D], x_mean, y_mean, lambda).
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
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths


def r2(Y, Yh):
    """variance-weighted R^2 over all dimensions, plus the mean per-dimension R^2"""
    res = float(((Y - Yh) ** 2).sum()); tot = max(float(((Y - Y.mean(0)) ** 2).sum()), 1e-12)
    per = 1 - ((Y - Yh) ** 2).sum(0) / np.maximum(((Y - Y.mean(0)) ** 2).sum(0), 1e-12)
    return 1 - res / tot, float(per.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--seed", type=int, default=43); ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--lams", nargs="*", type=float, default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--split_file", default=None, help="JSON with 'train' and 'test' family lists (overrides the seeded 80/20 split)")
    ap.add_argument("--tag", default=None, help="write results to read_write_map/sandbox/<tag>/ and the model to ridge_L8_to_L24_<tag>.npz")
    ap.add_argument("--select", choices=["correct", "all"], default="correct", help="prompt_pairs rows to use: correct = convention followed AND judge OK (the fitted map), all = every k = 3, 4 prompt")
    args = ap.parse_args()
    MP = model_paths(args.model); R = MP["results"]; OUT = R / "read_write_map" / ("sandbox/" + args.tag if args.tag else ""); OUT.mkdir(parents=True, exist_ok=True)
    MOD = MP["prompt_pairs"].parent / "read_write_map"; MOD.mkdir(parents=True, exist_ok=True); MODEL_FILE = MOD / (f"ridge_L8_to_L24_{args.tag}.npz" if args.tag else "ridge_L8_to_L24.npz")
    pool = json.load(open(R / "code_pool.json"))["pool"]; rng = np.random.default_rng(args.seed)
    if args.split_file:
        sf = json.load(open(args.split_file)); train, test = sorted(sf["train"]), sorted(sf["test"]); assert set(train) | set(test) == set(pool) and not set(train) & set(test)
        json.dump(dict(split_file=args.split_file, train=train, test=test), open(OUT / "split.json", "w"), indent=1)
    else:
        perm = list(rng.permutation(pool)); n_test = int(round(args.test_frac * len(pool))); test = sorted(perm[:n_test]); train = sorted(perm[n_test:])
        json.dump(dict(seed=args.seed, train=train, test=test), open(OUT / "split.json", "w"), indent=1)
    X, Y, F = [], [], []
    for f in train:
        z = np.load(MP["prompt_pairs"] / f"{f}.npz"); m = z["correct"] if args.select == "correct" else np.ones(len(z["k"]), bool)
        X.append(z["read"][m].astype(np.float32)); Y.append(z["write"][m].astype(np.float32)); F += [f] * int(m.sum())
    X, Y, F = np.concatenate(X), np.concatenate(Y), np.array(F)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; Xt = torch.as_tensor(X, dtype=torch.float64, device=dev); Yt = torch.as_tensor(Y, dtype=torch.float64, device=dev)
    # leave-one-family-out: one eigendecomposition per fold, every lambda a cheap product
    preds = {lam: np.zeros_like(Y) for lam in args.lams}
    for f in train:
        m = torch.as_tensor(F != f, device=dev); Xi, Yi = Xt[m], Yt[m]; mx, my = Xi.mean(0), Yi.mean(0); Xc = Xi - mx
        s_, V = torch.linalg.eigh(Xc.T @ Xc); C = V.T @ (Xc.T @ (Yi - my)); Xo = (Xt[~m] - mx) @ V
        for lam in args.lams:
            preds[lam][F == f] = ((Xo / (s_ + lam)) @ C + my).cpu().numpy()
        print(f"fold {f} done", flush=True)
    cv = [dict(lam=lam, lofo_r2=r2(Y, preds[lam])[0], lofo_r2_per_dim=r2(Y, preds[lam])[1]) for lam in args.lams]
    with open(OUT / "cv.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cv[0])); w.writeheader(); w.writerows(cv)
    best = max(cv, key=lambda c: c["lofo_r2"]); lam = best["lam"]
    # refit on all train families
    mx, my = Xt.mean(0), Yt.mean(0); Xc, Yc = Xt - mx, Yt - my
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=Xc.dtype, device=dev), Xc.T @ Yc)
    fit_r2 = r2(Y, ((Xc @ W) + my).cpu().numpy())
    np.savez_compressed(MODEL_FILE, W=W.cpu().numpy().astype(np.float32), x_mean=mx.cpu().numpy().astype(np.float32), y_mean=my.cpu().numpy().astype(np.float32),
                        lam=lam, train_families=np.array(train), read_layer=8, write_layer=24)
    json.dump(dict(seed=args.seed, train_families=train, test_families=test, n_train_prompts=int(len(X)), dim=int(X.shape[1]), lambda_grid=args.lams, lambda_selected=lam,
                   lofo_r2_at_selected=best["lofo_r2"], lofo_r2_per_dim_at_selected=best["lofo_r2_per_dim"], train_fit_r2=fit_r2[0], cv=cv,
                   model_file=str(MODEL_FILE), select=args.select), open(OUT / "fit.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.semilogx([c["lam"] for c in cv], [c["lofo_r2"] for c in cv], "o-"); ax.axvline(lam, color="r", ls="--", label=f"selected λ = {lam:g}")
    ax.set_xlabel("ridge λ"); ax.set_ylabel("leave-one-family-out R² (per-prompt write features, pooled)"); ax.grid(alpha=.3); ax.legend(); ax.set_title("λ selection on the 45 train families")
    fig.tight_layout(); fig.savefig(OUT / "cv_curve.png", dpi=130); plt.close(fig)
    print(json.dumps({k: v for k, v in json.load(open(OUT / "fit.json")).items() if k not in ("cv", "train_families")}, indent=1)); print("->", OUT)


if __name__ == "__main__":
    main()
