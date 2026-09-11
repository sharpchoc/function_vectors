#!/usr/bin/env python
"""Mixed-training variant of read_write_ridge.py (user request 2026-09-11): train on the 11 lexically
identical families PLUS 3 of the 6 lexically diverse families (per-prompt rows), test on the remaining
3 diverse families; all 20 splits. λ by leave-one-family-out CV over the 3 diverse training families
(grid 1e-1 … 1e5). For each split the identical-only map (λ = 31.6, the CV choice of the main run) is
scored on the same 3 test families, so the gain from adding diverse families is read directly.
Outputs → results/style_translation/read_write_map/mixed_splits.csv (one row per split × model) and a
per-family summary printed to stdout.
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.read_write_ridge import (DualRidge, evaluate, group_cv, load, OUT, ORIGINAL_LEX)
from src.sandbox.style_translation.family_groups import FIXED

GRID = np.logspace(-1, 5, 7)
LR, LW = 0, 24


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIX = list(FIXED); LEX = list(ORIGINAL_LEX)
    Xf, Yf, ff, pf = load(FIX, LR, LW)
    Xl, Yl, fl, pl = load(LEX, LR, LW)
    rows, per = [], []
    for test in itertools.combinations(LEX, 3):
        train_lex = [f for f in LEX if f not in test]
        mtr, mte = np.isin(fl, train_lex), np.isin(fl, list(test))
        Xtr = np.vstack([Xf, Xl[mtr]]); Ytr = np.vstack([Yf, Yl[mtr]]); ftr = np.concatenate([ff, fl[mtr]])
        Xte, Yte, fte, pte = Xl[mte], Yl[mte], fl[mte], pl[mte]
        # λ by CV over the 3 diverse training families only (held-out family = one diverse family at a time)
        cv = np.zeros(len(GRID))
        for g in train_lex:
            m = ftr == g
            R = DualRidge(Xtr[~m], Ytr[~m])
            for i, lam in enumerate(GRID):
                p = R.fit(lam).predict(Xtr[m]); cv[i] += (1 - ((p - Ytr[m]) ** 2).sum() / ((Ytr[m] - R.ym) ** 2).sum()) / 3
        lam = GRID[int(np.argmax(cv))]
        name = "+".join(test)
        o_mix, per_mix = evaluate(DualRidge(Xtr, Ytr).fit(lam), Xte, Yte, fte, pte, f"mixed|{name}")
        o_fix, per_fix = evaluate(DualRidge(Xf, Yf).fit(31.6), Xte, Yte, fte, pte, f"identical-only|{name}")
        for o, model in ((o_mix, "identical+3diverse"), (o_fix, "identical-only")):
            o.update(model=model, test=name, cv_r2=float(cv.max())); rows.append(o)
        for x, model in ([(x, "identical+3diverse") for x in per_mix] + [(x, "identical-only") for x in per_fix]):
            x.update(model=model, test=name); per.append(x)
        print(f"test={name:40s} λ={lam:7.3g} | mixed: R²(train-mean) {o_mix['r2_trainmean']:+.3f} within {o_mix['r2_within']:+.3f} "
              f"diff cos {o_mix['cos_diff_mean']:.2f} R² {o_mix['r2_diff']:+.3f} | identical-only: R² {o_fix['r2_trainmean']:+.3f} "
              f"diff cos {o_fix['cos_diff_mean']:.2f} R² {o_fix['r2_diff']:+.3f}", flush=True)
    with open(OUT / "mixed_splits.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT / "mixed_splits_per_family.csv", "w", newline="") as fh:
        keys = sorted({k for r in per for k in r}); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(per)
    print("\n== mean over the 20 splits ==")
    for model in ("identical+3diverse", "identical-only"):
        rr = [r for r in rows if r["model"] == model]
        print(f"{model:20s} R²(test-mean) {np.mean([r['r2_testmean'] for r in rr]):+.3f} | R²(train-mean) {np.mean([r['r2_trainmean'] for r in rr]):+.3f} | "
              f"within-family {np.mean([r['r2_within'] for r in rr]):+.3f} | centroid cos {np.mean([r['cos_centroid_mean'] for r in rr]):.2f} | "
              f"convention diff cos {np.mean([r['cos_diff_mean'] for r in rr]):.2f} R² {np.mean([r['r2_diff'] for r in rr]):+.3f}")
    print("\n== per held-out family (each held out in 10 splits) ==")
    print(f"{'family':14s} {'mixed cos':>9s} {'mixed R²diff':>12s} {'|p|/|t|':>7s} {'ident cos':>9s} {'ident R²diff':>12s}")
    for f in LEX:
        a = [x for x in per if x["family"] == f and x["model"] == "identical+3diverse"]
        b = [x for x in per if x["family"] == f and x["model"] == "identical-only"]
        print(f"{f:14s} {np.mean([x['cos_diff'] for x in a]):9.2f} {np.mean([x['r2_diff'] for x in a]):+12.3f} {np.mean([x['norm_ratio'] for x in a]):7.2f} "
              f"{np.mean([x['cos_diff'] for x in b]):9.2f} {np.mean([x['r2_diff'] for x in b]):+12.3f}")


if __name__ == "__main__":
    main()
