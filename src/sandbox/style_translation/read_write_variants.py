#!/usr/bin/env python
"""Write-up table + figure: every read→write map variant tried on the CODE pool, on one metric — held-out centroid R² around the training
mean (prediction of each held-out (family, pole) mean write activation at L24). All per-prompt ridges use λ = 10 (the CV choice on the code pool);
the centroid map uses λ = 0.32 (its CV choice). Variants: split 36/19 (category-stratified, seed 2026), split 44/11 (80/20, seed 2026), 44/11
with rust_question moved to train, five random 44/11 splits (seeds 1–5), read layers 5–10 on the 80/20 split, the centroid-level map, and
training augmented with the 16 text families. Text-pool and GPT-J numbers are taken from their stored tables.
Output → <results>/writeup/read_write_variants.{csv,png}"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.read_write_ridge import load, configure, make_ridge, DualRidge, r2, code_strata
from src.sandbox.style_translation.read_write_augment import centroid_scores


def cent_r2(X, Y, F, P, train, test, lam=10.0):
    tr, te = np.isin(F, train), np.isin(F, test)
    R = make_ridge(X[tr], Y[tr]).fit(lam); pred = R.predict(X[te])
    cents = np.array([Y[(F == f) & (P == s)].mean(0) for f in train for s in ("nat", "alt")])
    return centroid_scores(pred, Y[te], F[te], P[te], test, R.ym, cents)


def plot(d, OUT):
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(d) + 1.6))
    cols = {"code": "#1f6c80", "text→code": "#b8860b", "text": "#8e44ad", "text (GPT-J)": "#9e9e9e"}
    y = np.arange(len(d))[::-1]; sd = d["r2_sd"].fillna(0) if "r2_sd" in d else pd.Series([0.0] * len(d))
    ax.barh(y, d.r2, color=[cols[p] for p in d.pool], edgecolor="black", linewidth=0.5)
    has = sd > 0
    if has.any():
        ax.errorbar(d.r2[has], y[has.values], xerr=sd[has], fmt="none", ecolor="black", capsize=3, lw=1)
    for yi, (_, r), e in zip(y, d.iterrows(), sd):
        txt = f"{r.r2:.2f}" + (f" ± {e:.2f}" if e > 0 else "")
        ax.text(max(r.r2, 0) + e + 0.012, yi, txt, va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(d.variant, fontsize=8.5); ax.set_xlim(-0.12, 0.8); ax.axvline(0, color="grey", lw=0.6); ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("held-out centroid R² (mean write activation of each held-out family and pole, around the training mean)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in cols.items()], loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("Read→write linear map, every variant tried: held-out centroid R²\n(write feature = cue-token activation at L24; Qwen2.5-7B base unless stated)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "read_write_variants.png", dpi=150); plt.close(fig)


def main():
    configure("qwen25_base", "code"); MP = model_paths("qwen25_base"); R = MP["results"]; OUT = R / "writeup"; RW = R / "code" / "read_write_map"
    if "--plot_only" in sys.argv:
        plot(pd.read_csv(OUT / "read_write_variants.csv"), OUT); print("->", OUT / "read_write_variants.png"); return
    pool = json.load(open(R / "code_pool_full.json"))["pool"]; text = json.load(open(R / "pool.json"))["pool"]; cat = code_strata(pool)
    sp = json.load(open(RW / "fixed_split_80_20_v2.json")); train, test = sp["train"], sp["test"]   # the 80/20 split (45 train / 10 test; rust_question, the only Rust family, in train)
    rows = []
    data = {lr: load(pool, lr, 24) for lr in (5, 6, 7, 8, 9, 10)}
    X8, Y8, F8, P8 = data[8]
    # 1. read-layer sweep on the 80/20 split
    for lr in (5, 6, 7, 8, 9, 10):
        X, Y, F, P = data[lr]; o = cent_r2(X, Y, F, P, train, test)
        rows.append(dict(variant=f"per-prompt ridge, 80/20 split, read L{lr}", pool="code", n_train=len(train), n_test=len(test), r2=o["centroid_r2"], inspan=o["inspan"]))
    # 2. five random 80/20 draws at L8 (10 test families, category-stratified, seeds 1-5)
    rs = []
    for seed in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(seed); te_ = []
        for c in sorted(set(cat.values())):
            fs = sorted(f for f in pool if cat[f] == c); k = max(1, round(len(fs) * len(test) / len(pool))); te_ += [str(x) for x in rng.choice(fs, size=min(k, len(fs)), replace=False)]
        te_ = sorted(te_); tr_ = [f for f in pool if f not in te_]
        rs.append(cent_r2(X8, Y8, F8, P8, tr_, te_)["centroid_r2"])
    rows.append(dict(variant="per-prompt ridge, five random 80/20 draws, read L8 (mean ± sd)", pool="code", n_train=len(pool) - len(test), n_test=len(test), r2=float(np.mean(rs)), r2_sd=float(np.std(rs, ddof=1)), inspan=np.nan))
    # 3. centroid-level map on the 80/20 split at L8
    lab = [(f, s_) for f in pool for s_ in ("nat", "alt")]
    Rc = np.array([X8[(F8 == f) & (P8 == s_)].mean(0) for f, s_ in lab]); Wc = np.array([Y8[(F8 == f) & (P8 == s_)].mean(0) for f, s_ in lab])
    mtr = np.array([f in train for f, _ in lab]); M = DualRidge(Rc[mtr], Wc[mtr]).fit(0.316); pc = M.predict(Rc[~mtr])
    rows.append(dict(variant=f"centroid-level ridge ({int(mtr.sum())} training means), 80/20 split, read L8", pool="code", n_train=len(train), n_test=len(test), r2=r2(pc, Wc[~mtr], M.ym), inspan=np.nan))
    # 4. training augmented with the 16 text families
    Xt, Yt, Ft, Pt = load(text, 8, 24)
    te = np.isin(F8, test); trm = np.isin(F8, train)
    Xa, Ya, Fa, Pa = np.vstack([X8[trm], Xt]), np.vstack([Y8[trm], Yt]), np.concatenate([F8[trm], Ft]), np.concatenate([P8[trm], Pt])
    Rg = make_ridge(Xa, Ya).fit(10.0); pred = Rg.predict(X8[te])
    cents = np.array([Ya[(Fa == f) & (Pa == s_)].mean(0) for f in sorted(set(Fa)) for s_ in ("nat", "alt")])
    o = centroid_scores(pred, Y8[te], F8[te], P8[te], test, Rg.ym, cents)
    rows.append(dict(variant="per-prompt ridge, 80/20 split + 16 text families in training, read L8", pool="code", n_train=len(train) + 16, n_test=len(test), r2=o["centroid_r2"], inspan=o["inspan"]))
    Rg = make_ridge(Xt, Yt).fit(10.0); pred = Rg.predict(X8[te])
    cents = np.array([Yt[(Ft == f) & (Pt == s_)].mean(0) for f in text for s_ in ("nat", "alt")])
    o = centroid_scores(pred, Y8[te], F8[te], P8[te], test, Rg.ym, cents)
    rows.append(dict(variant="per-prompt ridge trained on the 16 text families only, tested on the 10 code test families, read L8", pool="text→code", n_train=16, n_test=len(test), r2=o["centroid_r2"], inspan=o["inspan"]))
    # 6. stored text-pool and GPT-J numbers
    for f, lr in (("pool_summary_L12_L24.csv", 12), ("pool_summary_L0_L24.csv", 0)):
        d = pd.read_csv(R / "read_write_map" / f); fx = d[d.protocol == "fixed"].iloc[0]; lo = d[d.protocol == "lofo"]
        rows.append(dict(variant=f"text pool (12 families), fixed 8/4 split stratified by axis, read L{lr}", pool="text", n_train=8, n_test=4, r2=fx.r2_centroid_trainmean, inspan=np.nan))
        rows.append(dict(variant=f"text pool (12 families), leave-one-family-out, read L{lr} (mean of 12)", pool="text", n_train=11, n_test=1, r2=lo.r2_centroid_trainmean.mean(), inspan=np.nan))
    g = pd.read_csv(R.parent / "read_write_map" / "ridge_summary.csv"); m = g[g.setting == "identical→diverse|cv"].iloc[0]
    rows.append(dict(variant="GPT-J: 11 lexically identical → 6 lexically diverse, read L0", pool="text (GPT-J)", n_train=11, n_test=6, r2=m.r2_centroid_trainmean, inspan=np.nan))
    d = pd.DataFrame(rows); d.to_csv(OUT / "read_write_variants.csv", index=False)
    plot(d, OUT)
    print(d.to_string(index=False)); print("->", OUT / "read_write_variants.png")


if __name__ == "__main__":
    main()
