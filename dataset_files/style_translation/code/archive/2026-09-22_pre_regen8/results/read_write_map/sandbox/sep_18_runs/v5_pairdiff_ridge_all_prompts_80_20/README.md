# v5 — v3 repeated with an 80/20 family split

- Rows: per (document, k) pair differences, X = read_nat − read_alt (L8 evidence means), Y = write_nat − write_alt (L24 cue), all
  k = 3/4 prompts (no correctness filter); 18394 train pairs.
- Split: `../../split_2026-09-15_test10_80_20.json` = the 80/20 test set of 2026-09-15 (`fixed_split_80_20.json`, category-stratified,
  seed 2026) minus py_bool_prefix (not in the pool): 10 test families (comma_space, docstring_style, float_literals, js_camel_snake, py2_iter, py_fstring, py_optional, py_paren_if, py_with_open, rust_question); 46 train.
- Fit: ridge with intercept, λ by leave-one-family-out CV: λ = 100000, LOFO R² 0.035; train-fit R² 0.360.
- Centroid-difference evaluation on the 10 test families (`centroid_diff_eval.json`): R² 0.024 (test-mean denominator),
  0.120 (train-mean denominator); constant train-mean diff -0.109; cos(pred, true) mean 0.37
  (constant 0.10); predicted norm 0.34 of the true; train in-sample R² 0.74.
