# v8 — the v7 setup on 10 random family splits (5 × 66/34, 5 × 80/20)

Setup per run = v7: per (document, k) pair differences (read_nat − read_alt at L8, write_nat − write_alt at L24), all k = 3/4 prompts,
each pair's read and write difference unit-normalised; ridge with intercept, λ by leave-one-family-out CV (λ = 10 in all ten runs).
Splits: `../split_v8_{66,80}_s{1..5}.json` (random, seeds 1001–1005, 19 or 11 test families). Runs: `v8_pairdiff_unitnorm_split{66,80}_s{1..5}/`.
Evaluation as v7: per family the mean of the unit-normalised pair differences, re-normalised; read → map vs the write analogue;
R² over the test families. Summary `v8_summary.csv`:

| split | held-out R², train-mean denominator | held-out R², test-mean denominator | direction cos |
|---|---|---|---|
| 66/34 (19 test), 5 splits | .244 ± .033 (range .21–.29) | .178 ± .028 | .45 ± .03 |
| 80/20 (11 test), 5 splits | .230 ± .082 (range .13–.35) | .126 ± .088 | .44 ± .08 |

Same transfer level as the fixed 2026-09-15 split (v7: .238 / .167 / .44); the 80/20 splits are noisier because 11 test families is few.
