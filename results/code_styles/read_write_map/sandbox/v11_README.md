# v11 — the v10 recipe on the 53-family pool (2026-09-23)

Same protocol as v10 (pair differences, all k = 3/4 prompts, unit norm, ridge with λ by leave-one-family-out, read L8 → write L24,
i.e. zero-indexed blocks 7 → 23) on the same ten partitions `../split_v10_{66,80}_s{1..5}.json`, which already exclude js_hungarian
(dropped from the pool by decision 2026-09-23); v10 fits still contained it. Scored on held-out family centroids with
`src/eval_scripts/score_code_map_centroids.py --prefix v11_pairdiff_unitnorm` (the paper-audit formula; reproduces v10 to 4 d.p.).

| split | families (train/test) | training pairs | R² train-mean ref | R² test-mean ref | cos(pred, true) |
|---|---|---|---|---|---|
| 66/34 ×5 | 34–37 / 16–19 | 13,600–14,800 | **.237 ± .030** (per split .225 .246 .263 .190 .260) | .163 | .426 |
| 80/20 ×5 | 42–44 / 9–11 | — | .222 ± .075 | .104 | .414 |
| v10 66/34 (54 families, for reference) | | | .232 ± .028 | .161 | .424 |

Summary table: `v11_pairdiff_unitnorm_summary.csv` (SD = sample SD over splits).
