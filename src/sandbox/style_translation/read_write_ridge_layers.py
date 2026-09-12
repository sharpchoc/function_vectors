#!/usr/bin/env python
"""Read-layer / write-layer sweep of the read→write ridge (identical → diverse), reduced λ grid.
Appends rows to results/style_translation/read_write_map/layer_sweep.csv."""
import csv, sys
from pathlib import Path
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.read_write_ridge import DualRidge, evaluate, group_cv, load, OUT, ORIGINAL_LEX
from src.sandbox.style_translation.family_groups import FIXED
GRID = np.logspace(-1, 5, 7)
SETTINGS = [(2, 24), (4, 24), (8, 24), (12, 24), (24, 24), (4, 12), (4, 16), (4, 20), (8, 12), (8, 16), (8, 20), (0, 12), (0, 16), (0, 20)]
rows = []
for lr, lw in SETTINGS:
    Xtr, Ytr, ftr, _ = load(list(FIXED), lr, lw); Xte, Yte, fte, pte = load(list(ORIGINAL_LEX), lr, lw)
    cv = group_cv(Xtr, Ytr, ftr, GRID); lam = GRID[int(np.argmax(cv))]
    R = DualRidge(Xtr, Ytr).fit(lam); o, per = evaluate(R, Xte, Yte, fte, pte, f"read L{lr} → write L{lw}")
    o.update(read_layer=lr, write_layer=lw, cv_r2=float(cv.max())); rows.append(o)
    print(f"read L{lr:2d} → write L{lw:2d}  λ={lam:7.3g} | R²(test-mean) {o['r2_testmean']:+.3f} R²(train-mean) {o['r2_trainmean']:+.3f} "
          f"within {o['r2_within']:+.3f} | centroid cos {o['cos_centroid_mean']:.2f} | convention diff cos {o['cos_diff_mean']:.2f} R² {o['r2_diff']:+.3f} | "
          + " ".join(f"{x['family'][:6]} {x['cos_diff']:.2f}" for x in per), flush=True)
    with open(OUT / "layer_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
