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

## Code conventions (2026-09-14): zero-shot steering of the 55 accepted code families

Same protocol on the cheap code corpus (~50 tasks per family, prompts `Task:\n{spec}\n\n{Language}:\n{code cut at cue}`): paired cue-token vectors
pooled over k ∈ {0, 4} (11–46 pairs per pole; split-half cos .2–.5 for the low-pair families, > .8 otherwise), screen {2,…,24} × α on the 50 k = 0 prompts
(style only), confirm the top-2 on all k = 0 prompts with the code judge (continuation cut at the first blank line), control = another family's vector.
Pods 6cxd2gphuk1jz2 / x3upu1gras0qbt / ur8fc5qxwb9dqn / 23cwh23qjmx1m5 (~2.5 h each), terminated. Reference (dashed) = the cheap-check k = 4 accuracy.

| family | → alt: unsteered → steered (L, α) | control | → nat: unsteered → steered (L, α) | control | judge OK | k = 4 alt reference |
|---|---|---|---|---|---|---|
| py_snake_camel | 0.00 → **0.82** (L24, 2) | 0.00 | 0.80 → **0.84** (L2, 0.5) | 0.88 | 0.86 | 0.92 |
| js_camel_snake | 0.00 → **0.76** (L24, 2) | 0.00 | 0.88 → **0.88** (L20, 0.5) | 0.90 | 0.86 | 0.94 |
| py_const_naming | 0.00 → **0.00** (L4, 0.5) ns | 0.00 | 0.02 → **0.06** (L12, 0.5) | 0.00 | 0.74 | 0.68 |
| py_class_naming | 0.28 → **0.42** (L10, 4) | 0.32 | 0.16 → **0.22** (L16, 4) | 0.18 | 0.74 | 0.66 |
| py_private | 0.02 → **0.30** (L24, 4) | 0.00 | 0.08 → **0.22** (L24, 2) | 0.06 | 0.96 | 0.44 |
| py_bool_prefix | 0.04 → **0.08** (L24, 4) ns | 0.10 | 0.02 → **0.30** (L24, 4) | 0.02 | 0.82 | 0.30 |
| py_loop_vars | 0.09 → **0.26** (L24, 4) | 0.07 | 0.30 → **0.50** (L20, 2) | 0.39 | 0.94 | 0.52 |
| js_hungarian | 0.00 → **0.02** (L12, 0.5) | 0.00 | 0.48 → **0.48** (L6, 1) | 0.42 | 0.88 | 0.84 |
| py_abbrev | 0.44 → **0.46** (L24, 4) ns | 0.38 | 0.08 → **0.10** (L24, 2) | 0.08 | 0.90 | 0.74 |
| py_quotes | 0.27 → **0.53** (L24, 2) | 0.16 | 0.37 → **0.49** (L24, 2) | 0.20 | 0.94 | 0.73 |
| js_quotes | 0.63 → **0.61** (L24, 0.5) ns | 0.57 | 0.22 → **0.78** (L24, 2) | 0.16 | 0.88 | 0.92 |
| py_fstring | 0.18 → **0.42** (L24, 4) | 0.20 | 0.16 → **0.80** (L24, 4) | 0.54 | 0.88 | 0.86 |
| js_template | 0.18 → **0.27** (L24, 4) ns | 0.16 | 0.20 → **0.18** (L10, 0.5) | 0.20 | 0.86 | 0.84 |
| num_separators | 0.00 → **0.49** (L24, 4) | 0.03 | 0.20 → **0.26** (L20, 0.5) | 0.26 | 0.71 | 0.66 |
| hex_constants | 0.00 → **0.16** (L24, 4) | 0.00 | 0.35 → **0.40** (L6, 4) | 0.30 | 0.60 | 0.74 |
| float_literals | 0.00 → **0.16** (L24, 4) | 0.05 | 0.47 → **0.63** (L24, 4) | 0.45 | 0.82 | 0.63 |
| sql_bool_case | 0.75 → **0.94** (L24, 2) | 0.22 | 0.19 → **1.00** (L24, 2) | 0.59 | 1.00 | 0.97 |
| py2_print | 0.00 → **0.06** (L8, 4) | 0.02 | 0.10 → **0.28** (L16, 2) | 0.10 | 0.54 | 0.72 |
| py2_iter | 0.00 → **0.71** (L24, 4) | 0.00 | 0.71 → **0.63** (L16, 0.5) | 0.45 | 0.74 | 0.63 |
| py2_except | 0.00 → **0.18** (L24, 2) | 0.00 | 0.13 → **0.20** (L10, 0.5) | 0.13 | 0.93 | 0.93 |
| js_var | 0.04 → **0.32** (L24, 4) | 0.18 | 0.36 → **0.38** (L24, 0.5) | 0.36 | 0.64 | 0.50 |
| js_arrow | 0.06 → **0.84** (L24, 2) | 0.02 | 0.84 → **0.86** (L24, 0.5) | 0.67 | 0.86 | 0.59 |
| js_semicolons | 0.00 → **0.55** (L24, 4) | 0.16 | 0.84 → **0.81** (L24, 0.5) | 0.68 | 0.90 | 0.48 |
| js_strict_eq | 0.11 → **0.79** (L24, 4) | 0.00 | 0.68 → **0.86** (L20, 0.5) | 0.93 | 0.86 | 0.57 |
| trailing_commas | 0.44 → **0.44** (L4, 0.5) ns | 0.38 | 0.06 → **0.21** (L24, 2) | 0.06 | 0.91 | 0.71 |
| py_paren_if | 0.00 → **0.52** (L24, 2) | 0.02 | 0.80 → **0.70** (L20, 4) | 0.68 | 0.84 | 0.54 |
| py_type_hints | 0.63 → **0.90** (L6, 2) | 0.82 | 0.10 → **0.84** (L24, 4) | 0.06 | 0.94 | 0.82 |
| py_optional | 0.00 → **0.00** (L2, 0.5) ns | 0.00 | 0.16 → **0.27** (L20, 4) | 0.18 | 0.71 | 0.45 |
| py_builtin_generics | 0.00 → **0.00** (L8, 0.5) ns | 0.00 | 0.10 → **0.23** (L16, 4) | 0.10 | 0.68 | 0.90 |
| py_comprehension | 0.11 → **0.24** (L16, 4) | 0.17 | 0.39 → **0.41** (L12, 2) | 0.30 | 0.91 | 0.63 |
| py_not_in | 0.02 → **0.33** (L24, 2) | 0.04 | 0.31 → **0.31** (L16, 0.5) | 0.27 | 0.89 | 0.36 |
| py_is_none | 0.03 → **0.16** (L24, 4) | 0.03 | 0.23 → **0.97** (L24, 4) | 0.23 | 0.77 | 0.68 |
| py_ternary | 0.33 → **0.46** (L20, 1) ns | 0.31 | 0.04 → **0.06** (L24, 1) | 0.04 | 0.88 | 0.52 |
| early_return | 0.18 → **0.42** (L24, 4) | 0.22 | 0.36 → **0.28** (L16, 0.5) | 0.28 | 0.90 | 0.84 |
| py_join_concat | 0.09 → **0.20** (L24, 4) | 0.07 | 0.34 → **0.50** (L24, 4) | 0.27 | 0.84 | 0.50 |
| py_with_open | 0.00 → **0.04** (L24, 4) | 0.00 | 0.08 → **0.14** (L24, 4) | 0.02 | 0.74 | 0.84 |
| py_enumerate | 0.22 → **0.45** (L24, 4) | 0.10 | 0.20 → **0.47** (L24, 4) | 0.08 | 0.80 | 0.63 |
| py_self_name | 0.00 → **0.64** (L24, 2) | 0.00 | 0.96 → **0.98** (L4, 0.5) | 0.98 | 0.90 | 0.86 |
| py_indent | 0.00 → **0.80** (L24, 4) | 0.02 | 0.76 → **0.96** (L24, 1) | 0.84 | 0.88 | 0.70 |
| py_tabs | 0.00 → **0.76** (L24, 2) | 0.20 | 0.80 → **0.88** (L4, 4) | 0.82 | 0.86 | 0.96 |
| operator_spaces | 0.02 → **0.53** (L20, 4) | 0.07 | 0.87 → **0.78** (L12, 2) | 0.73 | 0.84 | 0.69 |
| comma_space | 0.02 → **0.45** (L24, 4) | 0.15 | 0.79 → **0.77** (L10, 2) | 0.74 | 0.74 | 0.49 |
| line_wrap | 0.64 → **0.68** (L12, 0.5) ns | 0.56 | 0.02 → **0.76** (L24, 4) | 0.00 | 0.90 | 0.38 |
| blank_lines | 0.00 → **0.03** (L20, 1) | 0.03 | 0.03 → **0.03** (L24, 4) | 0.03 | 0.62 | 0.38 |
| docstring_style | 0.08 → **0.98** (L24, 4) | 0.06 | 0.70 → **0.94** (L24, 4) | 0.68 | 1.00 | 0.92 |
| docstring_quotes | 0.04 → **0.12** (L24, 4) | 0.00 | 0.20 → **0.20** (L10, 1) | 0.32 | 0.88 | 0.96 |
| comment_case | 0.04 → **0.30** (L24, 2) | 0.02 | 0.18 → **0.18** (L8, 1) | 0.20 | 0.60 | 0.62 |
| comment_language | 0.00 → **0.08** (L20, 2) | 0.00 | 0.34 → **0.42** (L24, 1) | 0.34 | 0.46 | 0.54 |
| c_comment_style | 0.02 → **0.22** (L24, 4) | 0.00 | 0.40 → **0.40** (L24, 4) | 0.30 | 0.78 | 0.84 |
| sql_keyword_case | 0.02 → **0.16** (L24, 2) | 0.02 | 0.54 → **0.78** (L24, 4) | 0.52 | 0.66 | 0.80 |
| sql_join_style | 0.00 → **0.46** (L20, 4) | 0.00 | 0.93 → **0.95** (L6, 0.5) | 0.91 | 0.93 | 0.66 |
| r_assignment | 0.06 → **0.72** (L24, 2) | 0.54 | 0.78 → **0.80** (L24, 0.5) | 0.78 | 0.82 | 0.82 |
| rust_question | 0.00 → **0.69** (L24, 4) | 0.05 | 0.74 → **0.74** (L2, 1) | 0.64 | 0.80 | 0.49 |
| bash_test | 0.74 → **0.96** (L24, 4) | 0.70 | 0.14 → **0.96** (L24, 4) | 0.20 | 0.96 | 0.74 |
| php_array | 0.14 → **0.64** (L24, 4) | 0.18 | 0.20 → **0.44** (L24, 4) | 0.06 | 0.76 | 0.60 |

Alternative pole: mean unsteered .13 → steered .43; significant lift in 45 of 55 families (ns = CI includes the unsteered rate); 21 families ≥ .5, 12 ≥ .7. Best layer 24 in most cells.
No effect (≤ .1): py_const_naming, js_hungarian, py_optional, py_builtin_generics, py_with_open, py2_print, py_bool_prefix, blank_lines, comment_language, docstring_quotes —
mostly the families with 11–18 paired prompts (noisy vectors) or weak classifiers; inconclusive rather than negative. Controls stay near unsteered except along
shared axes (py_type_hints ← py_optional .82, js_strict_eq natural ← js_arrow .93). Next: full 200-task corpora for the accepted families, then read features and the map.
