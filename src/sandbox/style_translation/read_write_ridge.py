#!/usr/bin/env python
"""Read→write linear map for the style-translation study: per-prompt ridge from the evidence-token
read activation (layer L_r, default 0 = embeddings) to the cue-token write activation (layer L_w,
default 24), trained on the LEXICALLY IDENTICAL families and scored on the LEXICALLY DIVERSE ones.

Data: artifacts/style_translation/prompt_pairs/<family>.npz (capture_prompt_pairs.py): 200 texts × 2
poles per family at k = 4.  Conventions follow results/69_task_run/read_write_relationship/linear_mapping:
dual ridge with intercept, features/targets centred on TRAIN statistics, λ chosen by leave-one-FAMILY-out
CV on the training families (grid 1e-2 … 1e8 in units of mean kernel diagonal), reported at the CV λ and
at harsher penalties (×10, ×100, ×1000).

R² conventions (held-out prompts):
  r2_testmean   standard R² (denominator = variance around the held-out mean)
  r2_trainmean  denominator = variance around the TRAIN mean write vector (the "predict the average" baseline)
  r2_within     per-family centred: both prediction and target have their held-out FAMILY mean removed —
                asks whether the map predicts which text / which pole a prompt is, not which family
  centroid      the 12 (family, pole) held-out centroids: R² and cos(pred, true) after train-mean centring
  diff          the 6 held-out CONVENTION vectors (nat centroid − alt centroid): cos(pred diff, true diff)
                and R² of the difference — the read→write map of the convention itself
Controls: shuffled pairing (X rows permuted within the training set), reverse split (train diverse →
test identical), leave-one-family-out over all 17, read-layer sweep {0,2,4,8,12,24} → write L24, write-layer
sweep read L0 → {12,16,20,24}.
Outputs → results/style_translation/read_write_map/: ridge_summary.csv, lambda_curve.csv, per_family.csv,
read_write_map.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT
from src.sandbox.style_translation.family_groups import LEXICAL, FIXED

PAIRS = ARTIFACTS_ROOT / "style_translation" / "prompt_pairs"
OUT = RESULTS_ROOT / "style_translation" / "read_write_map"
LAMBDA_GRID = np.logspace(-2, 8, 21)
ORIGINAL_LEX = ("us_uk", "ise_ize", "brit_t_past", "contractions", "num_words", "ordinal_words")   # the 17-family study


def load(fams, lr, lw):
    X, Y, fam, pole = [], [], [], []
    for f in fams:
        d = np.load(PAIRS / f"{f}.npz")
        X.append(d[f"read_L{lr}"].astype(np.float64)); Y.append(d[f"write_L{lw}"].astype(np.float64))
        fam += [f] * len(d["pole"]); pole += list(d["pole"])
    return np.vstack(X), np.vstack(Y), np.array(fam), np.array(pole)


def r2(pred, true, ref):
    """1 - SS_res / SS_tot with SS_tot around `ref` (a vector or an array broadcastable to true)."""
    return float(1 - ((pred - true) ** 2).sum() / ((true - ref) ** 2).sum())


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class DualRidge:
    """pred(Xte) = ym + (Xte - xm) Xc^T (K + λ·s·I)^-1 Yc, K = Xc Xc^T, s = mean(diag K) so λ is dimensionless."""

    def __init__(self, Xtr, Ytr):
        self.xm, self.ym = Xtr.mean(0), Ytr.mean(0)
        self.Xc, self.Yc = Xtr - self.xm, Ytr - self.ym
        self.K = self.Xc @ self.Xc.T
        self.scale = float(np.trace(self.K) / len(self.K))

    def fit(self, lam):
        n = len(self.K)
        self.A = np.linalg.solve(self.K + lam * self.scale * np.eye(n), self.Yc)
        self.lam = lam
        return self

    def predict(self, X):
        return self.ym + (X - self.xm) @ (self.Xc.T @ self.A)


def group_cv(Xtr, Ytr, gtr, grid):
    """leave-one-group-out CV: mean held-out R² (train-mean denominator) per λ."""
    scores = np.zeros(len(grid)); groups = np.unique(gtr)
    for g in groups:
        m = gtr == g
        R = DualRidge(Xtr[~m], Ytr[~m])
        for i, lam in enumerate(grid):
            scores[i] += r2(R.fit(lam).predict(Xtr[m]), Ytr[m], R.ym) / len(groups)
    return scores


def evaluate(R, Xte, Yte, fte, pte, name):
    p = R.predict(Xte)
    out = dict(setting=name, lam=R.lam, r2_testmean=r2(p, Yte, Yte.mean(0)), r2_trainmean=r2(p, Yte, R.ym))
    pw, yw = p.copy(), Yte.copy()
    for f in np.unique(fte):
        m = fte == f
        pw[m] -= p[m].mean(0); yw[m] -= Yte[m].mean(0)
    out["r2_within"] = r2(pw, yw, 0.0)
    cp, ct, dp, dt, per = [], [], [], [], []
    for f in np.unique(fte):
        c = {s: (p[(fte == f) & (pte == s)].mean(0), Yte[(fte == f) & (pte == s)].mean(0)) for s in ("nat", "alt")}
        for s in ("nat", "alt"):
            cp.append(c[s][0]); ct.append(c[s][1])
        dpred, dtrue = c["nat"][0] - c["alt"][0], c["nat"][1] - c["alt"][1]
        dp.append(dpred); dt.append(dtrue)
        per.append(dict(setting=name, family=f, cos_diff=cos(dpred, dtrue), r2_diff=r2(dpred, dtrue, 0.0),
                        norm_ratio=float(np.linalg.norm(dpred) / np.linalg.norm(dtrue)),
                        cos_centroid_nat=cos(c["nat"][0] - R.ym, c["nat"][1] - R.ym),
                        r2_within_fam=r2(pw[fte == f], yw[fte == f], 0.0)))
    cp, ct, dp, dt = map(np.array, (cp, ct, dp, dt))
    out["r2_centroid_trainmean"] = r2(cp, ct, R.ym)
    out["cos_centroid_mean"] = float(np.mean([cos(a - R.ym, b - R.ym) for a, b in zip(cp, ct)]))
    out["cos_diff_mean"] = float(np.mean([x["cos_diff"] for x in per]))
    out["r2_diff"] = r2(dp, dt, 0.0)
    return out, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read_layer", type=int, default=0)
    ap.add_argument("--write_layer", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LEX = [f for f in LEXICAL if f in ORIGINAL_LEX and (PAIRS / f"{f}.npz").exists()]
    FIX = [f for f in FIXED if (PAIRS / f"{f}.npz").exists()]
    print(f"train (lexically identical): {FIX}\ntest (lexically diverse): {LEX}")
    lr, lw = args.read_layer, args.write_layer
    rows, per_rows, curve = [], [], []

    def run(train, test, name, lr=lr, lw=lw, shuffle=False, seed=0, curve_out=False):
        nonlocal per_rows
        Xtr, Ytr, ftr, _ = load(train, lr, lw)
        Xte, Yte, fte, pte = load(test, lr, lw)
        if shuffle:
            rng = np.random.default_rng(seed); Xtr = Xtr[rng.permutation(len(Xtr))]
        cv = group_cv(Xtr, Ytr, ftr, LAMBDA_GRID)
        lam_cv = LAMBDA_GRID[int(np.argmax(cv))]
        R = DualRidge(Xtr, Ytr)
        if curve_out:
            for lam, s in zip(LAMBDA_GRID, cv):
                o, _ = evaluate(R.fit(lam), Xte, Yte, fte, pte, name)
                curve.append(dict(lam=lam, cv_r2_trainmean=s, **{k: v for k, v in o.items() if k not in ("setting", "lam")}))
        for tag, lam in (("cv", lam_cv), ("x10", lam_cv * 10), ("x100", lam_cv * 100), ("x1000", lam_cv * 1000)):
            o, per = evaluate(R.fit(lam), Xte, Yte, fte, pte, f"{name}|{tag}")
            o.update(read_layer=lr, write_layer=lw, lam_tag=tag, cv_r2=float(cv.max()), n_train=len(Xtr), n_test=len(Xte))
            rows.append(o); per_rows.extend(dict(read_layer=lr, write_layer=lw, lam_tag=tag, **x) for x in per)
            print(f"{o['setting']:40s} λ={lam:9.3g} R²(test-mean) {o['r2_testmean']:+.3f} | R²(train-mean) {o['r2_trainmean']:+.3f} | "
                  f"within-family {o['r2_within']:+.3f} | centroid R² {o['r2_centroid_trainmean']:+.3f} cos {o['cos_centroid_mean']:.2f} | "
                  f"convention diff: cos {o['cos_diff_mean']:.2f} R² {o['r2_diff']:+.3f}", flush=True)
            if tag == "cv":
                for x in per:
                    print(f"    {x['family']:14s} diff cos {x['cos_diff']:.2f} R² {x['r2_diff']:+.3f} |pred|/|true| {x['norm_ratio']:.2f} within-family R² {x['r2_within_fam']:+.3f}")
        return lam_cv

    print("\n== main: train identical → test diverse ==")
    run(FIX, LEX, "identical→diverse", curve_out=True)
    print("\n== controls ==")
    run(FIX, LEX, "identical→diverse|shuffled-X", shuffle=True)
    run(LEX, FIX, "diverse→identical")
    Xa, Ya, fa, pa = load(FIX + LEX, lr, lw)
    lofo = []
    for f in FIX + LEX:
        m = fa == f
        grid = LAMBDA_GRID[::2]
        cv = group_cv(Xa[~m], Ya[~m], fa[~m], grid); lam = grid[int(np.argmax(cv))]
        R = DualRidge(Xa[~m], Ya[~m]).fit(lam)
        o, per = evaluate(R, Xa[m], Ya[m], fa[m], pa[m], f"lofo|{f}")
        lofo.append(o); per_rows.extend(dict(read_layer=lr, write_layer=lw, lam_tag="lofo", **x) for x in per)
        print(f"LOFO {f:14s} λ={lam:8.3g} R²(train-mean) {o['r2_trainmean']:+.3f} within {o['r2_within']:+.3f} diff cos {o['cos_diff_mean']:.2f}")
    rows.append(dict(setting="lofo-all17|mean", read_layer=lr, write_layer=lw, lam_tag="cv",
                     **{k: float(np.mean([o[k] for o in lofo])) for k in lofo[0] if k not in ("setting", "lam")}))
    print("\n== layer sweeps (train identical → test diverse, CV λ) ==")
    for lr_ in (0, 2, 4, 8, 12, 24):
        if lr_ != lr:
            run(FIX, LEX, f"identical→diverse|read L{lr_}", lr=lr_, lw=lw)
    for lw_ in (12, 16, 20):
        run(FIX, LEX, f"identical→diverse|write L{lw_}", lr=lr, lw=lw_)

    with open(OUT / "ridge_summary.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "setting", k))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT / "per_family.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_rows[0])); w.writeheader(); w.writerows(per_rows)
    with open(OUT / "lambda_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    plot(rows, per_rows, curve, lr, lw)


def plot(rows, per_rows, curve, lr, lw):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = [c["lam"] for c in curve]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].semilogx(lam, [c["cv_r2_trainmean"] for c in curve], "k--", label="CV on identical families (leave-one-family-out)")
    ax[0].semilogx(lam, [c["r2_trainmean"] for c in curve], "-o", ms=3, label="held-out diverse: R² vs train-mean")
    ax[0].semilogx(lam, [c["r2_testmean"] for c in curve], "-o", ms=3, label="held-out diverse: R² vs test-mean")
    ax[0].semilogx(lam, [c["r2_within"] for c in curve], "-o", ms=3, label="held-out diverse: within-family R²")
    ax[0].set_ylim(-0.5, 1); ax[0].axhline(0, color="grey", lw=.5); ax[0].set_xlabel("ridge λ (× mean kernel diagonal)"); ax[0].set_ylabel("R²")
    ax[0].set_title("Prompt-level R² vs penalty"); ax[0].legend(fontsize=7)
    ax[1].semilogx(lam, [c["cos_diff_mean"] for c in curve], "-o", ms=3, label="cos(predicted, true) convention vector")
    ax[1].semilogx(lam, [c["cos_centroid_mean"] for c in curve], "-o", ms=3, label="cos(predicted, true) (family, pole) centroid")
    ax[1].semilogx(lam, [c["r2_diff"] for c in curve], "-o", ms=3, label="R² of the convention vector")
    ax[1].set_ylim(-0.5, 1); ax[1].axhline(0, color="grey", lw=.5); ax[1].set_xlabel("ridge λ"); ax[1].set_title("Held-out convention vectors (nat − alt)"); ax[1].legend(fontsize=7)
    main = [r for r in per_rows if r["setting"] == "identical→diverse|cv" and r["read_layer"] == lr and r["write_layer"] == lw]
    shuf = {r["family"]: r for r in per_rows if r["setting"] == "identical→diverse|shuffled-X|cv"}
    fams = [r["family"] for r in main]; x = np.arange(len(fams))
    ax[2].bar(x - .2, [r["cos_diff"] for r in main], .4, label="ridge (CV λ)")
    ax[2].bar(x + .2, [shuf[f]["cos_diff"] for f in fams], .4, color="lightgrey", label="shuffled pairing control")
    ax[2].set_xticks(x); ax[2].set_xticklabels(fams, rotation=30, ha="right"); ax[2].set_ylabel("cos(predicted, true) convention vector")
    ax[2].set_title("Per held-out family"); ax[2].axhline(0, color="grey", lw=.5); ax[2].legend(fontsize=7)
    fig.suptitle(f"Read→write ridge: evidence-token read activation (L{lr}) → cue-token write activation (L{lw}); "
                 f"trained on 11 lexically identical families, scored on 6 lexically diverse families", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "read_write_map.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
