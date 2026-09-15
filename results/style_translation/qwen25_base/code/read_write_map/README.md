# results/style_translation/qwen25_base/code/read_write_map — read→write linear map on the 55 coding-convention families

Qwen2.5-7B (base). Does a linear map from the read feature (evidence-token mean) to the write feature (cue-token activation), fitted on
some families, predict the write feature of families it never saw? Same estimator and metrics as the text buckets (`../../read_write_map/`,
GPT-J `../../../read_write_map/`), user decisions 2026-09-15:

- **Data**: per k = 4 prompt, one forward pass (`capture_prompt_pairs.py`): read = mean residual over the prompt's evidence tokens (diff-only
  rule, `../read_features/`) at layer L_r; write = residual at the cue token at L24. 55 families × ≈ 400 prompts (both poles) = 21,800 pairs.
- **Split**: ONE fixed random split, 2/3 train – 1/3 test, seed 2026, stratified by convention category (`fixed_split.json`, committed before
  fitting): **36 train / 19 test** families, every category on both sides. No leave-one-family-out.
- **Estimator**: ridge with intercept, features and targets centred on TRAIN statistics, λ by leave-one-family-out CV inside the training set
  (dimensionless grid; the CV picked λ = 10 at every read layer). Solved in feature space (`PrimalRidge`, identical to the dual solver).
- **Read layers**: main L8 (the decided read site) → L24; sweep L5–L10.
- **Scores on the 19 held-out families**: convention-vector cosine = cos(predicted nat − alt centroid difference, true one); its R²;
  prompt-level R² (denominator around the test mean / the train mean / within family). Controls: shuffled read–write pairing in the training
  set (cos ≈ 0 expected); **mean-vector baseline** = cos(true convention vector, mean of the 36 training families' true convention vectors)
  — what "predict the average write direction" achieves without reading anything.
- `subsets.json`: `both29` = families with a gated write feature (headroom) AND a read feature reaching ≥ 50 % of the reference at k = 3;
  `strict22` = strict write gate ∧ read reach. Their test members: 11 and 7.
- Capture sanity: cos(k = 4 cue difference, stored pooled L24 write vector) ≥ .86 for 50 families; bash_test .52, rust_question .68,
  py_paren_if .75, docstring_quotes .76, py_ternary .77 (the stored vector pools k = 0..4 over correct prompts only).

## Result

| read layer → L24 | held-out convention cos (19) | test ∩ both29 (11) | test ∩ strict22 (7) | R² train-mean | R² test-mean | R² within-family | convention R² |
|---|---|---|---|---|---|---|---|
| L5 | 0.35 | 0.40 | 0.37 | +0.061 | +0.038 | +0.012 | -0.17 |
| L6 | 0.36 | 0.40 | 0.39 | +0.055 | +0.032 | +0.011 | -0.22 |
| L7 | 0.37 | 0.41 | 0.39 | +0.059 | +0.036 | +0.013 | -0.15 |
| L8 | 0.38 | 0.43 | 0.41 | +0.065 | +0.042 | +0.015 | -0.10 |
| L9 | 0.39 | 0.43 | 0.42 | +0.066 | +0.043 | +0.018 | -0.08 |
| L10 | 0.39 | 0.43 | 0.42 | +0.071 | +0.049 | +0.024 | -0.06 |

Mean-vector baseline cos = .11 at every layer; shuffled control −.01 … +.02.

- **The map transfers partially.** At L8 the predicted convention vector of a never-seen family has cosine .38 with the true one on average
  (baseline .11, shuffled .00); 13 of 19 test families exceed .3 and 5 exceed .5 (js_quotes .81, py_snake_camel .79, operator_spaces .64,
  docstring_quotes .55, py_fstring .53); the ridge beats the mean-vector baseline in 16 of 19. Prompt-level R² stays small (.04–.07): the
  map captures the family's direction, not the per-prompt variation (within-family R² ≈ .02), and the predicted vectors are shorter than the
  true ones for the well-predicted families (|pred|/|true| .3–.9), so the convention R² is negative for most.
- **Read layer matters little** between L5 and L10 (.35 → .39); the decided L8 is within .01 of the best.
- **Gated families transfer better**: .43 (both29) / .41 (strict22) vs .38 overall.
- **Coverage predicts transfer, as on the text pools.** `coverage.csv` (read L8 / write L24, all 55): the fraction of a family's convention
  direction inside the span of the other 54 families' directions is .56 (write) / .53 (read) on average — the code pool is high-dimensional
  (participation ratio 48 of 55 on both sides), yet read and write similarity structures agree (RSA .78). Across the 19 test families the
  held-out cosine correlates .87 with the write-span fraction and .90 with the read-span fraction: families that share an axis with a
  training family (quotes ↔ quotes, snake ↔ camel, spacing ↔ spacing) are predicted; isolated ones (rust_question, py_not_in, js_semicolons)
  are not.

### By category (test families)

| category | n test | ridge cos | baseline cos |
|---|---|---|---|
| comments / docs | 2 | 0.44 | 0.08 |
| formatting | 2 | 0.54 | 0.09 |
| literals | 3 | 0.56 | 0.10 |
| naming | 3 | 0.48 | 0.01 |
| other languages | 2 | 0.13 | 0.11 |
| syntax / dialect | 7 | 0.27 | 0.16 |

### Per held-out family (read L8 → write L24)

| family | category | ridge cos | baseline | shuffled | convention R² | ‖pred‖/‖true‖ | write span (coverage) | both29 |
|---|---|---|---|---|---|---|---|---|
| docstring_quotes | comments / docs | 0.55 | +0.01 | -0.02 | +0.26 | 0.74 | 0.77 |  |
| c_comment_style | comments / docs | 0.33 | +0.14 | -0.05 | -0.13 | 0.82 | 0.54 | ✓ |
| operator_spaces | formatting | 0.64 | +0.20 | -0.03 | +0.41 | 0.65 | 0.69 | ✓ |
| line_wrap | formatting | 0.44 | -0.02 | +0.06 | +0.12 | 0.69 | 0.51 |  |
| js_quotes | literals | 0.81 | -0.09 | +0.02 | +0.56 | 1.11 | 0.96 | ✓ |
| py_fstring | literals | 0.53 | +0.19 | -0.02 | +0.17 | 0.85 | 0.57 | ✓ |
| float_literals | literals | 0.35 | +0.19 | -0.05 | -0.73 | 1.27 | 0.57 | ✓ |
| py_snake_camel | naming | 0.79 | -0.10 | -0.05 | +0.55 | 0.53 | 0.93 | ✓ |
| py_const_naming | naming | 0.41 | +0.13 | -0.03 | +0.16 | 0.32 | 0.57 |  |
| py_private | naming | 0.23 | +0.01 | -0.03 | -0.60 | 1.05 | 0.45 | ✓ |
| sql_join_style | other languages | 0.29 | +0.10 | -0.03 | -0.00 | 0.58 | 0.44 | ✓ |
| rust_question | other languages | -0.04 | +0.13 | +0.01 | -0.75 | 0.83 | 0.38 |  |
| py_ternary | syntax / dialect | 0.39 | +0.19 | +0.02 | -0.21 | 0.98 | 0.61 |  |
| py2_except | syntax / dialect | 0.38 | +0.18 | -0.01 | -0.03 | 0.81 | 0.57 | ✓ |
| trailing_commas | syntax / dialect | 0.34 | -0.04 | +0.04 | +0.10 | 0.47 | 0.58 |  |
| js_var | syntax / dialect | 0.32 | +0.26 | -0.01 | -0.72 | 1.23 | 0.43 |  |
| py_self_name | syntax / dialect | 0.24 | +0.16 | +0.04 | -0.65 | 1.08 | 0.41 | ✓ |
| js_semicolons | syntax / dialect | 0.15 | +0.19 | +0.08 | -1.39 | 1.34 | 0.56 | ✓ |
| py_not_in | syntax / dialect | 0.06 | +0.17 | +0.03 | -1.65 | 1.34 | 0.47 |  |

## Comparison with the text pools

| pool | protocol | held-out convention cos | baseline / shuffled |
|---|---|---|---|
| GPT-J, 11 lexically identical → 6 lexically diverse (read L0) | fixed | ≈ 0 (see `../../../read_write_map/`) | – |
| Qwen, 12-family headroom text pool (read L12) | fixed 2/3–1/3 | .44 | shuffled ≈ 0 |
| Qwen, 12-family headroom text pool (read L12) | LOFO | .55 within-axis / .21 cross-axis | shuffled ≈ 0 |
| **Qwen, 55 code families (read L8)** | fixed 36/19 | **.38** (.43 on both-feature families) | baseline .11 / shuffled .00 |

Files: `read_write_map_code.png` (per test family + read-layer sweep), `coverage.{png,csv}`, `pool_summary_Lsweep_L24.csv`,
`pool_per_family_Lsweep_L24.csv`, `fixed_split.json`, `subsets.json`.
