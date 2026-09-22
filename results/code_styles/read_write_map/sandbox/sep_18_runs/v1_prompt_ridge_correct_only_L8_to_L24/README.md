# v1 — one ridge map, per-prompt read (L8) → per-prompt write (L24), correct prompts only

- Data: `prompt_pairs/<family>.npz`, rows with `correct` = True (convention followed AND judge OK), k = 3 and 4, both poles,
  all documents; 31,835 prompts over 56 families.
- Split: families 80/20 by seed 43 → 45 train / 11 test (`split.json`; test = comma_space, float_literals, js_strict_eq, py2_except,
  py_indent, py_loop_vars, py_optional, py_private, py_ternary, py_with_open, rust_question).
- Fit: ridge with intercept on all train-family prompts (X = read, Y = write); λ by leave-one-family-out CV on the train families,
  pooled held-out per-prompt predictions, variance-weighted R² (`cv.csv`, `cv_curve.png`): λ = 1e4, LOFO R² .112 (per-dim .069);
  train-fit R² .474 (`fit.json`). Model: `artifacts/style_translation/qwen25_code/read_write_map/ridge_L8_to_L24.npz`.
- Style-centroid evaluation (`style_centroid_eval.json`): map(mean read of a style) vs mean write of the style.
  Train styles (90, in-sample): R² .962, nearest-centroid identification .989, cos(pred, true nat−alt) .936.
  Test styles (22 unseen families): R² .136 (constant −.118), identification .50, cos .32.
- Script: `src/sandbox/style_translation/read_write_map_code.py` (defaults reproduce this run with `--select correct`).
