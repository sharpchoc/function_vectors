# v7 — v3 with unit-normalised per-prompt DIFFERENCE vectors

- Rows as v3: per (document, k) pair differences, X = read_nat − read_alt (L8 evidence means), Y = write_nat − write_alt (L24 cue),
  all k = 3/4 prompts (no correctness filter), 14,794 train pairs; each pair's read difference and write difference scaled to unit
  L2 norm before fitting.
- Split as v3: `../../split_2026-09-15_test19.json` (19 test families, 37 train).
- Fit: ridge with intercept, λ by leave-one-family-out CV (grid 1e-4..1e3): λ = 10, LOFO R² .086 (per-dim .037); train-fit R² .551.
  Model: `artifacts/.../read_write_map/ridge_L8_to_L24_v7_pairdiff_ridge_all_prompts_unitnorm.npz` (apply to unit-normalised read differences).
- Evaluation on unit-normalised centroid DIFFERENCE vectors (per family: the mean of the unit-normalised pair differences, re-normalised;
  read → map → compared with the write analogue; `centroid_diff_eval.json`), 19 test families: R² .167 (test-mean denominator),
  .238 (train-mean denominator); constant train-mean −.093; cos(pred, true) mean .44 (constant .08); per family from .88 (py_snake_camel),
  .85 (js_quotes), .82 (py_fstring) to .11 (py_not_in), .15 (py_ternary, rust_question); predicted norm .40; train in-sample R² .93.
  Normalising the raw mean difference instead of averaging unit differences gives the same numbers to two decimals.
