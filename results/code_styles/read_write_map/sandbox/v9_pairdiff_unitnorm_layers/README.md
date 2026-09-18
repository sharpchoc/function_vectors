# v9 — the v7 recipe for each read layer (L6, L10, L12, L14, L16, L18; L8 = v7)

Per layer L: rows = per (document, k) pair differences, X = read_nat − read_alt at layer L (mean over evidence tokens, from
`prompt_pairs_layers/`), Y = write_nat − write_alt at L24 (cue token, from `prompt_pairs/`), all k = 3/4 prompts, both unit-normalised;
split `../../split_2026-09-15_test19.json` (19 test / 37 train); ridge with intercept, λ by leave-one-family-out CV (pooled per-pair
predictions, variance-weighted R²). One subfolder per layer (split.json, cv.csv, cv_curve.png, fit.json); `summary.csv` collects the fits.
Models: `artifacts/.../read_write_map/ridge_L{L}_to_L24_v9_pairdiff_unitnorm_layers_L{L}.npz`.
Test-family evaluation: pending the user's instructions.

| read layer | λ | LOFO R² | per-dim | train-fit R² |
|---|---|---|---|---|
| L6 | 10 | 0.077 | 0.033 | 0.550 |
| L8 (v7) | 10 | 0.086 | 0.037 | 0.551 |
| L10 | 10 | 0.084 | 0.032 | 0.551 |
| L12 | 10 | 0.079 | 0.026 | 0.551 |
| L14 | 10 | 0.080 | 0.026 | 0.551 |
| L16 | 10 | 0.081 | 0.027 | 0.552 |
| L18 | 10 | 0.083 | 0.029 | 0.555 |
