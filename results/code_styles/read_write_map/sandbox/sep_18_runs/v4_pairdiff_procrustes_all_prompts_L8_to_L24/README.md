# v4 — scaled Procrustes map on the v3 data (pair differences, all prompts, 66/34 split)

- Rows and split exactly as v3: per (document, k) pair differences, X = read_nat − read_alt (L8), Y = write_nat − write_alt (L24),
  all k = 3/4 prompts; 14794 train pairs; test = the 19 families of `../../split_2026-09-15_test19.json`.
- Fit: W = s · U Vᵀ with U S Vᵀ = SVD(Xcᵀ Yc) on train-mean-centred data, one global scale s = trace(S) / ‖Xc‖²_F (s = 2.33);
  no λ. Leave-one-family-out R² (pooled pair predictions) -0.231; train-fit R² 0.375.
- Centroid-difference evaluation on the 19 test families (`centroid_diff_eval.json`): R² -0.593 (test-mean denominator),
  -0.468 (train-mean denominator); constant train-mean diff -0.085; cos(pred, true) mean 0.31
  (constant 0.07); predicted norm 1.10 of the true; train in-sample R² 0.66.
