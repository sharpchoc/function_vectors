# results/style_translation/qwen25_base/read_write_map — does a linear map from read to write features generalise across conventions? (Qwen2.5-7B base)

Phase 5 of the plan of 2026-09-14. Per-prompt captures (`capture_prompt_pairs.py --model qwen25_base`, k = 4 prompts, 200 texts × 2 poles per
family): X = mean evidence-token activation (read), Y = cue-token activation (write). Dual ridge with intercept, train-mean centring, λ by
leave-one-family-out CV inside the training set (`read_write_ridge.py --pool …`). Pool = families passing the feature gate (`../pool_gated.json`):
strict (both directions significant) → 5 families; headroom-aware (a direction whose unsteered rate is already ≥ .70 is exempt) → 12 families.
The headroom-aware pool is the main result (the strict gate removes us_uk, ise_ize, uk_vocab, unit_abbr, zh_simp_trad, pt_acordo_eu, pt_br_eu
only because their natural pole has no headroom, although their alternative pole lifts by .4–.85).
Metrics as in the GPT-J bucket `../../read_write_map/`: held-out R² (train-mean / test-mean), within-family R², and the convention vector
(held-out nat − alt centroid): cos and R² of prediction vs truth; shuffled-pairing control; axis clusters British {us_uk, ise_ize, uk_vocab},
digits {num_words, ordinal_words, unit_abbr}, Portuguese {pt_acordo_eu, pt_br_eu}, Chinese {zh_simp_trad}.

## Coverage first (`coverage.{png,csv}`, read L12 / write L24)
Fraction of a family's convention direction inside the span of the other 15: British .72–.96, Portuguese .73–.84, digits .53–.80, isolated
(contractions .36/.45, zh_simp_trad .54/.28, register .37/.52). RSA corr(read cos, write cos) = .83; participation ratio 13 of 16 on both sides.

## Leave-one-family-out, 12-family pool, read L12 → write L24 (`pool_summary_L12_L24.csv`, `lofo_convention_cos.png`)

| held-out family | axis (mates in train) | R² (train-mean) | within-family R² | convention cos | shuffled | convention R² |
|---|---|---|---|---|---|---|
| us_uk | british (2) | +0.059 | -0.003 | **0.77** | -0.00 | +0.13 |
| ise_ize | british (2) | +0.061 | +0.004 | **0.69** | -0.07 | +0.36 |
| contractions | - (0) | +0.029 | +0.006 | **0.20** | -0.02 | -0.91 |
| num_words | digits (2) | +0.079 | +0.005 | **0.66** | 0.11 | -1.00 |
| ordinal_words | digits (2) | +0.108 | +0.026 | **0.47** | -0.05 | +0.22 |
| uk_vocab | british (2) | +0.082 | -0.002 | **0.51** | 0.01 | +0.25 |
| unit_abbr | digits (2) | -0.016 | -0.004 | **0.26** | -0.06 | -0.14 |
| title_abbr | - (0) | +0.006 | -0.010 | **0.24** | 0.00 | -1.82 |
| zh_simp_trad | chinese (0) | +0.003 | +0.002 | **0.24** | 0.08 | +0.00 |
| pt_acordo_eu | portuguese (1) | +0.157 | -0.005 | **0.63** | 0.02 | -0.40 |
| pt_br_eu | portuguese (1) | +0.118 | -0.011 | **0.47** | 0.04 | +0.21 |
| es_rae2010 | - (0) | +0.082 | -0.002 | **0.16** | -0.06 | -4.81 |

Mean convention cos 0.44 (shuffled -0.00); **within-axis hold-outs (8) 0.55, cross-axis hold-outs (4) 0.21**. Held-out R² against the train mean is +.00 to +.16 (family location only); within-family R² is 0 in every fold; the convention-vector R² is positive only for ise_ize, uk_vocab, ordinal_words, pt_br_eu, us_uk (predicted norms are too small).

## Fixed stratified split (`fixed_split_headroom.json`, seed 2026, written before fitting)
Train ['us_uk', 'ise_ize', 'contractions', 'ordinal_words', 'unit_abbr', 'zh_simp_trad', 'pt_br_eu', 'es_rae2010']; test ['num_words', 'pt_acordo_eu', 'title_abbr', 'uk_vocab'].
R²(train-mean) +0.063, R²(test-mean) -0.003, within-family +0.001; convention cos 0.44 (shuffled -0.01), R² +0.11. Per test family: num_words cos 0.64; pt_acordo_eu cos 0.44; title_abbr cos 0.24; uk_vocab cos 0.42.

## Strict pool (5 families: contractions, num_words, ordinal_words, title_abbr, es_rae2010; `pool_summary_L12_L24.csv` was overwritten by the 12-pool run — numbers from `logs/qwen_ridge_strict_L12.log`)
LOFO convention cos: num_words .68, ordinal_words .58 (digits axis mates), contractions .20, title_abbr .19, es_rae2010 .16; mean .36 (shuffled .03); R² ≤ +.09.

## Reading
- **The read→write relation transfers along axes, not across them.** A held-out convention that shares an axis with training families is
  predicted at cos .47–.77 (British and Portuguese best); an isolated convention stays at .16–.26, i.e. the in-span fraction. This is the coverage
  ceiling predicted before fitting, now confirmed on Qwen with 12 families instead of GPT-J's 11→6 split (GPT-J: cos .21–.38, no axis-mates).
- **No prompt-level map.** Within-family R² is zero everywhere: the map never predicts which text a cue state belongs to, only which convention
  (partially). Same as on GPT-J and as the 69-task per-prompt result (task-identity readout).
- **What would move it:** more families per axis (the sub-family route), not more isolated conventions; the RSA agreement (.83) and the
  within-axis cosines say the relation is shared where sampled.

Files: `coverage.{png,csv}`, `pool_summary_L12_L24.csv`, `pool_per_family_L12_L24.csv`, `fixed_split_headroom.json`, `fixed_split_strict.json`,
`lofo_convention_cos.png`; read-L0 variant: `pool_summary_L0_L24.csv` (see the L0 line appended below when finished).
