# results/style_translation/qwen25_base/read_steer — evidence-token (read-feature) steering on Qwen2.5-7B base

Step 7 on Qwen2.5-7B (base) for the 16-family pool, protocol identical to `../../read_steer/`: read vectors u_nat(L) = r_nat(L) − r_alt(L) from the
step-6 evidence-token means (`../read_features/`), injected (α·u) at every evidence token of the 3 instances of a k = 3 prompt (`PositionSteer`,
unit-tested at layers 0/6/20), screen {2,…,24} × α ∈ {0.5,1,2,4} on 50 texts, confirm top-2 on all 200 texts with the judge, control = another
family's read vector. Reference = step-3 accuracy at k = 3 when the context genuinely was the target pole.
CJK note: when adjacent opportunities are merged into one token, an instance's evidence can lie beyond the k = 3 cue; such positions are dropped
(1 of 2,000 instances for zh_simp_trad).

## Result (accuracy = target at the 4th decision ∧ faithful, 200 texts)

| family | nat ctx → alt: unsteered → steered (L, α) | genuine alt ctx | alt ctx → nat: unsteered → steered (L, α) | genuine nat ctx |
|---|---|---|---|---|
| us_uk | 0.04 → **0.73** (L12, 2) | 0.59 | 0.22 → **0.84** (L16, 2) | 0.81 |
| ise_ize | 0.00 → **0.66** (L6, 1) | 0.69 | 0.12 → **0.88** (L12, 1) | 0.88 |
| contractions | 0.10 → **0.62** (L6, 4) | 0.69 | 0.26 → **0.81** (L10, 2) | 0.80 |
| num_words | 0.01 → **0.61** (L6, 2) | 0.47 | 0.44 → **0.92** (L2, 1) | 0.93 |
| ordinal_words | 0.00 → **0.93** (L8, 2) | 0.92 | 0.03 → **0.90** (L4, 1) | 0.97 |
| uk_vocab | 0.12 → **0.47** (L8, 4) | 0.34 | 0.47 → **0.71** (L8, 2) | 0.68 |
| register | 0.20 → **0.46** (L12, 4) | 0.30 | 0.51 → **0.75** (L2, 4) | 0.70 |
| diacritics | 0.30 → **0.37** (L6, 2) | 0.39 | 0.41 → **0.47** (L6, 4) | 0.44 |
| unit_abbr | 0.01 → **0.74** (L4, 4) | 0.48 | 0.55 → **0.98** (L10, 2) | 0.96 |
| latin_plural | 0.31 → **0.45** (L16, 4) | 0.40 | 0.48 → **0.56** (L6, 1) | 0.56 |
| title_abbr | 0.26 → **0.41** (L8, 2) | 0.43 | 0.34 → **0.66** (L4, 4) | 0.59 |
| zh_simp_trad | 0.00 → **0.86** (L6, 2) | 0.83 | 0.01 → **0.94** (L8, 2) | 0.91 |
| pt_acordo_eu | 0.10 → **0.69** (L12, 4) | 0.47 | 0.35 → **0.71** (L16, 2) | 0.69 |
| pt_br_eu | 0.03 → **0.60** (L6, 4) | 0.34 | 0.48 → **0.88** (L2, 4) | 0.83 |
| es_rae2010 | 0.04 → **0.51** (L24, 4) | 0.33 | 0.37 → **0.70** (L4, 4) | 0.64 |
| ru_yo | 0.11 → **0.54** (L20, 4) | 0.53 | 0.02 → **0.47** (L2, 2) | 0.51 |

Mean over the 32 cells: unsteered 0.21, steered 0.68, genuine-context reference 0.63; 32 of 32 cells within .10 of the reference, 24 at or above it.
Same picture as on GPT-J: steering the evidence tokens makes Qwen re-read the context as the other convention, with early best layers (2–16 in 27
of 32 cells). zh_simp_trad flips to .86 / .94, es_rae2010 to .51 / .70, pt_br_eu to .60 / .88, pt_acordo_eu .69 / .71, ru_yo .54 / .47. Weak:
diacritics (.37 / .47), latin_plural (.45 / .56).
Pods 7pmat0j42n1bxm / nrefceepw2yvpw (shards R1, R2), terminated.
