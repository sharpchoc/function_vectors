# v7 — v3 with unit-normalised per-prompt DIFFERENCE vectors

- Rows as v3: per (document, k) pair differences, X = read_nat − read_alt (L8 evidence means), Y = write_nat − write_alt (L24 cue),
  all k = 3/4 prompts (no correctness filter), 14,794 train pairs; each pair's read difference and write difference scaled to unit
  L2 norm before fitting.
- Split as v3: `../../split_2026-09-15_test19.json` (19 test families, 37 train).
- Fit: ridge with intercept, λ by leave-one-family-out CV (grid 1e-4..1e3): λ = 10, LOFO R² .086 (per-dim .037); train-fit R² .551.
  Model: `artifacts/.../read_write_map/ridge_L8_to_L24_v7_pairdiff_ridge_all_prompts_unitnorm.npz` (apply to unit-normalised read differences).
- Evaluation on the 19 test families: pending the user's instructions.
