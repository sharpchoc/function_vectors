# v1 (clean data) — the v7 recipe on the regen8 corpus (2026-09-23)

- Data: per-prompt activations captured after regen8 (7,271 of 7,567 strict-review failures regenerated; 54 pool families). Rows = per
  (document, k) pair differences, X = read_nat − read_alt (L8 evidence means), Y = write_nat − write_alt (L24 cue), all k = 3/4 prompts
  (no correctness filter), 14,000 train pairs; both differences unit-normalised.
- Split: `../../split_2026-09-23_test18.json` = the 2026-09-15 test19 split minus py_ternary, py_abbrev and js_hungarian (all out of the pool): 35 train / 18 test.
- Fit: ridge with intercept, λ by leave-one-family-out (grid 1e-4..1e3): λ = 10, LOFO R² .084; train-fit R² .577.
  Model: `artifacts/.../read_write_map/ridge_L8_to_L24_v1_pairdiff_unitnorm_clean.npz`.
- Evaluation on unit-normalised centroid DIFFERENCE vectors (`centroid_diff_eval.json`, `tmp/eval_clean_map.py`), 18 test families:
  R² .136 (test-mean denominator), .219 (train-mean denominator); constant train-mean −.107; cos(pred, true) mean .41 (constant .08); identical to the 36-family fit that still had js_hungarian in the training set;
  norm ratio .40; train in-sample R² .933.
- Previous fits of the same recipe: `../sep_22_runs/v1_pairdiff_unitnorm_clean/` (after the first clean-up, 37/18: R² .186 / .263, cos .465)
  and `../sep_18_runs/v7_…` (original data, 37/19: .175 / .248, cos .44). Per family: py_snake_camel .91 and js_quotes .90 unchanged at the
  top; py_fstring .84 → .36, c_comment_style .41 → .32, py_private .49 → .48; rust_question .11 and py_not_in .05 stay at the bottom.
