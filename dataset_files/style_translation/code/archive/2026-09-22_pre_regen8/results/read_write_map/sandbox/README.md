# Read → write map sandbox (qwen25_code)

- `sep_18_runs/` — every map run made on the per-prompt activations captured BEFORE the corpus-wide padding clean-up (2026-09-15 … 09-19):
  v1–v9 as described in the parent README. Their prompts include documents later found to be padded (24 % of the corpus) and the old,
  invalid py_private definition. Kept for reference only; their model files are under `artifacts/.../read_write_map/sep_18_runs/`.
- `v1_pairdiff_unitnorm_clean/` — the v7 recipe (per-prompt pair differences, all k = 3/4 prompts, unit norm, ridge with intercept, λ by
  leave-one-family-out, test19 split) refit on the cleaned data (2026-09-22 activations: 49 families regenerated, py_private redefined).
