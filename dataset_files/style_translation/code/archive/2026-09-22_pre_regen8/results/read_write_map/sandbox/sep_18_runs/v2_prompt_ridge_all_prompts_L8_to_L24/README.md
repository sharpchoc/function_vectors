# v2 — one ridge map, per-prompt read (L8) → per-prompt write (L24), ALL k = 3/4 prompts (no correctness filter)

- Data: `prompt_pairs/<family>.npz`, every k = 3 and 4 prompt, both poles, all documents (correct and incorrect alike).
- Split: the 2026-09-15 test set (`../../split_2026-09-15_test19.json`): 19 test families = the test half of the old 2/3–1/3
  category-stratified split (seed 2026) behind the .38-cosine result; train = the other 37 pool families; 29,588 train prompts.
- Fit: ridge with intercept, λ by leave-one-family-out CV on the 37 train families (pooled per-prompt predictions, variance-weighted R²):
  λ = 1e4, LOFO R² .118 (per-dim .076). Model: `artifacts/.../qwen25_code/read_write_map/ridge_L8_to_L24_v2_prompt_ridge_all_prompts_L8_to_L24.npz`.
- Centroid → centroid evaluation on the 19 test families (38 styles; centroids over ALL prompts of a style; `centroid_eval.json`):
  R² .115 with the test-mean denominator, .187 with the train-mean denominator (constant train-mean prediction: −.089 / 0 by
  construction); per-dim mean R² −.03; nearest-centroid identification .39 (chance .03); train in-sample R² .975.
