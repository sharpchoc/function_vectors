# results/style_translation/qwen25_base/steering — cue-token (write-feature) steering on Qwen2.5-7B base

Step 4 of the style-translation study on Qwen2.5-7B (base), for the 16-family lexically diverse pool (`../pool.json`), protocol identical to the
GPT-J bucket `../../steering/`: PAIRED cue-token mean-difference vectors pooled over k = 0..4 (`capture_cues.py --model qwen25_base`), screen
layers {2,4,…,24} × α ∈ {0.5,1,2,4} on 50 k = 0 texts (style only, 16 tokens), confirm the top-2 settings per target on all 200 k = 0 texts
(48 tokens, sentence cut, registry classifier ∧ Gemini 2.5 Flash judge), control = another family's vector at the same (L, α).
Relative vector norms ‖v‖/‖resid‖ (.02–.27) match GPT-J's, so the α grid was kept. Hook unit test (dtype-aware, bf16) passed on every pod.
Files as in the GPT-J bucket: `steering_summary.png` (headline), `steering_summary_controls.png`, `screen_layer_alpha.png`, `best_config.csv`,
`steering_summary.csv`, `screen.csv`, `records.npz`.

## Result (accuracy = target convention ∧ faithful, 200 texts; unsteered = same k = 0 prompt scored toward the target)

| family | → alt: unsteered → steered (L, α) | control | → nat: unsteered → steered (L, α) | control |
|---|---|---|---|---|
| us_uk | 0.06 → **0.63** (L24, 4) | 0.70 | 0.76 → **0.78** (L8, 1) | 0.79 |
| ise_ize | 0.01 → **0.68** (L24, 4) | 0.00 | 0.94 → **0.92** (L8, 4) | 0.92 |
| contractions | 0.38 → **0.87** (L20, 4) | 0.40 | 0.43 → **0.71** (L24, 4) | 0.34 |
| num_words | 0.15 → **0.81** (L24, 2) | 0.69 | 0.72 → **0.94** (L24, 4) | 0.94 |
| ordinal_words | 0.66 → **0.94** (L24, 1) | 0.71 | 0.27 → **0.83** (L24, 2) | 0.19 |
| uk_vocab | 0.09 → **0.67** (L24, 4) | 0.20 | 0.76 → **0.81** (L24, 1) | 0.76 |
| register | 0.14 → **0.33** (L20, 4) | 0.17 | 0.48 → **0.48** (L4, 1) | 0.44 |
| diacritics | 0.27 → **0.40** (L24, 4) | 0.33 | 0.56 → **0.58** (L24, 4) | 0.55 |
| unit_abbr | 0.01 → **0.86** (L24, 4) | 0.03 | 0.94 → **0.96** (L24, 0.5) | 0.93 |
| latin_plural | 0.23 → **0.28** (L20, 2) | 0.23 | 0.59 → **0.68** (L24, 4) | 0.61 |
| title_abbr | 0.37 → **0.74** (L24, 4) | 0.10 | 0.41 → **0.48** (L24, 4) | 0.34 |
| zh_simp_trad | 0.01 → **0.53** (L10, 4) | 0.01 | 0.72 → **0.75** (L10, 0.5) | 0.70 |
| pt_acordo_eu | 0.06 → **0.39** (L24, 4) | 0.07 | 0.76 → **0.77** (L12, 4) | 0.73 |
| pt_br_eu | 0.03 → **0.67** (L24, 2) | 0.21 | 0.75 → **0.78** (L6, 0.5) | 0.77 |
| es_rae2010 | 0.12 → **0.18** (L24, 4) | 0.12 | 0.46 → **0.66** (L24, 4) | 0.45 |
| ru_yo | 0.44 → **0.47** (L20, 0.5) | 0.39 | 0.07 → **0.18** (L24, 4) | 0.03 |

Reading: the low-baseline pole lifts strongly in 12 of 16 families (alt: us_uk .06→.63, ise_ize .01→.68, contractions .38→.87, num_words .15→.81,
uk_vocab .09→.67, unit_abbr .01→.86, title_abbr .37→.74, zh_simp_trad .01→.53, pt_br_eu .03→.67, pt_acordo_eu .06→.39; nat: ordinal_words .27→.83,
es_rae2010 .46→.66). Best layer is 24 in most cells (20 for contractions/register; 8–12 for ise_ize/zh_simp_trad/pt_acordo_eu natural).
Weak: register (.14→.33), diacritics (.27→.40), latin_plural (.23→.28), ru_yo (ё pole .07→.18; the model's default is е). Controls stay near
unsteered except on shared axes (us_uk ← ise_ize .70, num_words ← ordinal_words .70). Judge OK .93–.99 (zh_simp_trad .70 at k = 0 — see the
empty-context judge fix below).

Judge fix (2026-09-14): for families whose first opportunity opens the text (zh_simp_trad, all_caps, sentence_caps, some num_words/pt_br_eu/register/
contractions texts) the k = 0 cue is the header newline and the 'so far' context is empty; the judge read the completion as 'repeating the beginning'.
`judge_rollouts.py` now states that the translation has not started; 1,852 affected verdicts (steering confirm + step-3 rollouts) were re-judged.

Gate (`../pool_gated.json`, `feature_gate.py`): strict = Wilson CI of the best steered accuracy excludes the unsteered rate in BOTH directions;
headroom-aware = a direction whose unsteered rate is already ≥ .70 is exempt. Write side passes strict for 5 of 16 families, headroom-aware for
12 of 16.
Pods r968v40ukhmcvb / adeb4l5ustw50v (shards A, B, B2; ~7 pod-h incl. an OOM restart of shard B at batch 25 → 12), terminated.
