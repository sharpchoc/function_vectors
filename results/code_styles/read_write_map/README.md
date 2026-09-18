# Read → write map (qwen25_code, code-convention families)

Per-prompt activations: `artifacts/style_translation/qwen25_code/prompt_pairs/<family>.npz` (every k = 3/4 prompt, read = L8 mean over
evidence tokens, write = L24 cue-token residual, flags style_ok / judge_ok / correct, k, pole, heldout).

`sandbox/` holds the map variants tried so far, one subfolder each with its own README:
- `v1_prompt_ridge_correct_only_L8_to_L24/` — one ridge map, per-prompt read → per-prompt write, correct prompts only, family split
  45/11, LOFO λ; held-out style-centroid R² .14 (train in-sample .96).
- `v2_prompt_ridge_all_prompts_L8_to_L24/` — same map, every k = 3/4 prompt (no correctness filter), split = the 2026-09-15 test set
  (19 test / 37 train, `split_2026-09-15_test19.json`); LOFO R² .118; test evaluation pending.
Nothing in here is a settled result yet.
