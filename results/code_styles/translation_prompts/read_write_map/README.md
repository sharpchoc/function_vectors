# Read → write map on the source-program (translation) prompts (2026-10-30)

Paper (v11) recipe: per (document, k) pair contrasts nat − alt, all k = 3/4 prompts, unit-normalised; X = read (L8 mean over evidence
tokens), Y = write (L24 cue token); ridge with intercept, λ by leave-one-family-out (grid 1e-4..1e3, λ = 10 on every split); scored on
held-out family centroids (`src/eval_scripts/score_code_map_centroids.py --model qwen25_code_tr`).
Activations: `artifacts/style_translation/qwen25_code_tr/prompt_pairs/` (`capture_prompt_pairs_code.py --model qwen25_code_tr --select all`).
Splits: the paper's ten v10 partitions restricted to the 35 translation families (`split_v10_{66,80}_s{1..5}_tr35.json`; 66/34 → 8–12 test
families, 80/20 → only 4–7). Matched baseline = the same recipe and splits on the ORIGINAL prompts' activations
(`../../read_write_map/sandbox/v13_tr35orig_*`, `--pool_file code_pool.json` of this folder).

| 35 families | R² (train-mean ref), 66/34 | 80/20 | cos(pred, true), 66/34 | 80/20 |
|---|---|---|---|---|
| original prompts | .144 ± .039 | .168 ± .066 | .307 | .335 |
| translation prompts | .154 ± .035 | .169 ± .069 | .346 | .369 |

Translation ≥ original R² on 8 of 10 splits (+.010 mean at 66/34), cos higher on 10/10 (+.03–.06). Small, consistent gain; the map stays
weak (paper 53-family v11: .237 / .426). Fits: `sandbox/v13_tr35_pairdiff_unitnorm_split*/`, summary `sandbox/v13_tr35_pairdiff_unitnorm_summary.csv`.
