# v3 — ridge map on per-prompt DIFFERENCES: (read_nat − read_alt) → (write_nat − write_alt), all prompts

- Rows: one per (document, k) pair of the two poles' prompts (k = 3, 4; no correctness filter): X = read_nat − read_alt (L8 evidence
  means), Y = write_nat − write_alt (L24 cue). 14,794 train pairs from the 37 train families.
- Split: `../../split_2026-09-15_test19.json` (19 test families of the 2026-09-15 split, 37 train).
- Fit: ridge with intercept, λ by leave-one-family-out CV (pooled per-pair predictions, variance-weighted R²): λ = 1e5,
  LOFO R² .015 (per-dim .002); train-fit R² .337. Model: `artifacts/.../read_write_map/ridge_L8_to_L24_v3_pairdiff_ridge_all_prompts_L8_to_L24.npz`.
- Centroid-difference evaluation on the 19 test families (mean read diff → predicted write diff vs true write diff; `centroid_diff_eval.json`):
  R² .031 with the test-mean denominator, .107 with the train-mean denominator (constant train-mean diff: −.085 / 0); cos(pred, true)
  mean .29 (py_fstring .71, operator_spaces .53, py2_except .41; sql_join_style .09, py_ternary .10) vs .07 for the constant train-mean
  diff; predicted norm .44 of the true; train in-sample R² .70.
