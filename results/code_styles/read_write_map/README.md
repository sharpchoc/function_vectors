# Read → write map (qwen25_code, code-convention families)

Per-prompt activations: `artifacts/style_translation/qwen25_code/prompt_pairs/<family>.npz` (every k = 3/4 prompt, read = L8 mean over
evidence tokens, write = L24 cue-token residual, flags style_ok / judge_ok / correct, k, pole, heldout).

`sandbox/` holds the map variants tried so far, one subfolder each with its own README:
- `v1_prompt_ridge_correct_only_L8_to_L24/` — one ridge map, per-prompt read → per-prompt write, correct prompts only, family split
  45/11, LOFO λ; held-out style-centroid R² .14 (train in-sample .96).
- `v2_prompt_ridge_all_prompts_L8_to_L24/` — same map, every k = 3/4 prompt (no correctness filter), split = the 2026-09-15 test set
  (19 test / 37 train, `split_2026-09-15_test19.json`); LOFO R² .118; test evaluation pending.
- `v3_pairdiff_ridge_all_prompts_L8_to_L24/` — ridge on per-prompt nat−alt DIFFERENCES (paired by document and k), all prompts,
  same split; LOFO R² .015 (λ 1e5), train-fit .34; test evaluation pending.
- `v5_pairdiff_ridge_all_prompts_80_20/` — v3 with the 2026-09-15 80/20 split (10 test / 46 train); centroid-diff R² 0.02 / 0.12, cos 0.37.
- `v4_pairdiff_procrustes_all_prompts_L8_to_L24/` — scaled Procrustes (orthogonal W, one global scale) on the v3 data; centroid-diff R² -0.59 / -0.47, cos 0.31.
- `v6_prompt_ridge_correct_only_unitnorm/` — v1 with unit-normalised per-prompt read and write vectors; λ 10, LOFO R² .124; unit-normalised style-centroid R² .18 / .26 (v1: .14 / .23), difference cos .32.
- `v7_pairdiff_ridge_all_prompts_unitnorm/` — v3 with unit-normalised pair-difference vectors; λ 10, LOFO R² .086 (v3: .015); test evaluation pending.
Nothing in here is a settled result yet.
