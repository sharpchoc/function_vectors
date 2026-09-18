# v6 — v1 with unit-normalised per-prompt features

- Data as v1: per-prompt read (L8 evidence mean) → per-prompt write (L24 cue), correct prompts only (convention followed AND judge OK),
  25,930 train prompts; every prompt's read vector and write vector scaled to unit L2 norm before fitting.
- Split as v1: seed 43, 45 train / 11 test families (comma_space, float_literals, js_strict_eq, py2_except, py_indent, py_loop_vars,
  py_optional, py_private, py_ternary, py_with_open, rust_question).
- Fit: ridge with intercept, λ by leave-one-family-out CV (grid 1e-4..1e3 because unit vectors shrink the Gram): λ = 10,
  LOFO R² .124 (per-dim .082); train-fit R² .471. Model: `artifacts/.../read_write_map/ridge_L8_to_L24_v6_prompt_ridge_correct_only_unitnorm.npz`
  (W, x_mean, y_mean; apply to unit-normalised read vectors).
- Evaluation as v1 on unit-normalised vectors (prompts normalised BEFORE the centroids; `style_centroid_eval.json`):
  style centroids, 22 test styles: R² .176 (test-mean denominator), .263 (train-mean denominator); constant train-mean −.118;
  per-dim mean .078; nearest-centroid identification .45; train in-sample R² .964.
  nat−alt centroid differences, 11 test families: R² −.144 / −.012; cos(pred, true) mean .32 (comma_space .65 … rust_question .07);
  predicted norm .66 of the true.
