# v10 — the v7 recipe on the regen8 corpus, ten random splits (2026-09-23)

Same protocol as v8 (sep_18_runs): the v7 recipe (pair differences, all k = 3/4 prompts, unit norm, ridge with λ by leave-one-family-out)
on the SAME ten random partitions as v8 (`../split_v10_{66,80}_s{1..5}.json` = the v8 partitions minus py_ternary and py_abbrev, which left the
pool), evaluated on unit-normalised centroid differences. `v10_summary.csv`; folders `v10_pairdiff_unitnorm_split{66,80}_s{1..5}/`.

| split | data | R² train-mean denom | R² test-mean denom | cos(pred, true) |
|---|---|---|---|---|
| 66/34 ×5 | original (v8, 09-18) | .244 ± .033 | .178 ± .028 | .453 ± .034 |
| 66/34 ×5 | regen8 (v10) | .232 ± .028 | .161 ± .023 | .424 ± .025 |
| 80/20 ×5 | original (v8) | .230 ± .082 | .126 ± .088 | .444 ± .084 |
| 80/20 ×5 | regen8 (v10) | .215 ± .073 | .102 ± .075 | .410 ± .064 |

The regen8 corpus is consistently a little lower on every split (about .01–.03 in R², .03 in cos) — the same direction as the fixed test18
split (.263 → .219 train-mean R²). Not split-specific.
