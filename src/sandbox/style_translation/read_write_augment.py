#!/usr/bin/env python
"""Quick test (user request 2026-09-16): does adding the lexically diverse TEXT families to the training set improve the read→write map on the
held-out CODE families? Per-prompt ridge (read L_r → write L24, train-mean centring, λ fixed at the value the CV picked for the code pool = 10),
train = code training families of the 80/20 split [+ the 16 text families], test = the 11 held-out code families, scored on their (family,
pole) centroids: centroid cos (train-mean centred), centroid R² (around the train mean), implied convention cos, and the in-span fraction of
the held-out centroids w.r.t. the training centroids."""
import argparse, json, sys
from pathlib import Path
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.read_write_ridge import load, configure, make_ridge, r2, cos
from src.sandbox.style_translation.models import paths as model_paths


def centroid_scores(pred, Yte, Fte, Pte, test, ym, train_cents):
    ct, cp, labs = [], [], []
    for f in test:
        for s in ("nat", "alt"):
            m = (Fte == f) & (Pte == s); ct.append(Yte[m].mean(0)); cp.append(pred[m].mean(0)); labs.append((f, s))
    ct, cp = np.array(ct), np.array(cp)
    cc = float(np.mean([cos(p - ym, t - ym) for p, t in zip(cp, ct)]))
    conv = [cos(cp[2 * i] - cp[2 * i + 1], ct[2 * i] - ct[2 * i + 1]) for i in range(len(test))]
    Q, _ = np.linalg.qr((train_cents - ym).T); H = ct - ym
    inspan = float((np.linalg.norm(Q.T @ H.T, axis=0) ** 2).sum() / (H ** 2).sum())
    return dict(centroid_cos=cc, centroid_r2=r2(cp, ct, ym), centroid_r2_testmean=r2(cp, ct, ct.mean(0)), conv_cos=float(np.mean(conv)), inspan=inspan,
                per_family={f: c for f, c in zip(test, conv)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--read_layer", type=int, default=10); ap.add_argument("--write_layer", type=int, default=24)
    ap.add_argument("--lam", type=float, default=10.0); ap.add_argument("--split", default="80_20")
    args = ap.parse_args()
    configure(args.model, "code"); MP = model_paths(args.model)
    code = json.load(open(MP["results"] / "code_pool_full.json"))["pool"]; text = json.load(open(MP["results"] / "pool.json"))["pool"]
    sp = json.load(open(MP["results"] / "code" / "read_write_map" / f"fixed_split_{args.split}.json")); train_c, test = sp["train"], sp["test"]
    lr, lw = args.read_layer, args.write_layer
    Xc, Yc, Fc, Pc = load(code, lr, lw); Xt, Yt, Ft, Pt = load(text, lr, lw)
    mte = np.isin(Fc, test); Xte, Yte, Fte, Pte = Xc[mte], Yc[mte], Fc[mte], Pc[mte]
    out = {}
    for name, (X, Y, F, P) in {"code train only (44)": (Xc[~mte], Yc[~mte], Fc[~mte], Pc[~mte]),
                               "code train + 16 text families (60)": (np.vstack([Xc[~mte], Xt]), np.vstack([Yc[~mte], Yt]), np.concatenate([Fc[~mte], Ft]), np.concatenate([Pc[~mte], Pt])),
                               "16 text families only": (Xt, Yt, Ft, Pt)}.items():
        R = make_ridge(X, Y).fit(args.lam); pred = R.predict(Xte)
        cents = np.array([Y[(F == f) & (P == s)].mean(0) for f in sorted(set(F)) for s in ("nat", "alt")])
        o = centroid_scores(pred, Yte, Fte, Pte, test, R.ym, cents); out[name] = o
        print(f"read L{lr} → write L{lw}, λ={args.lam:g} | train = {name:36s}: held-out centroid cos {o['centroid_cos']:.3f} | centroid R² {o['centroid_r2']:+.3f} (test-mean {o['centroid_r2_testmean']:+.3f}) | convention cos {o['conv_cos']:.3f} | in-span fraction {o['inspan']:.3f}", flush=True)
    print("per held-out family, convention cos (code only → code + text):")
    for f in test:
        print(f"    {f:18s} {out['code train only (44)']['per_family'][f]:.2f} → {out['code train + 16 text families (60)']['per_family'][f]:.2f}")
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, open(MP["results"] / "code" / "read_write_map" / f"augment_text_L{lr}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
