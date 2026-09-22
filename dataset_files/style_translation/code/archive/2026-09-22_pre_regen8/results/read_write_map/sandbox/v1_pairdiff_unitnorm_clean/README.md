# v1 (clean data) — the v7 recipe refit on the cleaned corpus (2026-09-22)

- Data: per-prompt activations captured after the corpus-wide padding clean-up (49 families regenerated, py_private redefined as
  underscore-prefixed vs plain private members). Rows = per (document, k) pair differences, X = read_nat − read_alt (L8 evidence means),
  Y = write_nat − write_alt (L24 cue), all k = 3/4 prompts (no correctness filter), 14,794 train pairs; both differences unit-normalised.
- Split: `../../split_2026-09-22_test18.json` = the 2026-09-15 test19 split minus py_ternary (dropped from the pool): 37 train / 18 test.
- Fit: ridge with intercept, λ by leave-one-family-out (grid 1e-4..1e3): λ = 10, LOFO R² .088 (per-dim .039); train-fit R² .545.
  Model: `artifacts/.../read_write_map/ridge_L8_to_L24_v1_pairdiff_unitnorm_clean.npz`.
- Evaluation on unit-normalised centroid DIFFERENCE vectors (`centroid_diff_eval.json`, `tmp/eval_clean_map.py`), 18 test families:
  R² .186 (test-mean denominator), .263 (train-mean denominator); constant train-mean −.104; cos(pred, true) mean .465 (constant .08);
  norm ratio .41; train in-sample R² .924.
- Comparison with sep_18_runs/v7 (old data, 19 test families incl. py_ternary): R² .175 / .248, cos .44. Per family: py_private .27 → .49
  (redefined), py_not_in .12 → .04, others within ±.06.
