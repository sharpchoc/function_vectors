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

## Why the prompt-level R² is small (variance decomposition, all 55 families)

| component of the activation | write L24 (cue token) | read L8 (evidence-token mean) |
|---|---|---|
| between families | .29 | .55 |
| convention pole within a family (nat vs alt) | **.025** | .14 |
| prompt-specific (task, code so far) | .68 | .30 |

Two thirds of the cue-token activation is prompt content that a mean over 1–3 evidence tokens cannot carry, and the convention itself is 2.5 %
of its variance. Ceilings: predicting every test prompt by its TRUE (family, pole) centroid gives R² .31 (family means only: .28). The
held-out ridge reaches prompt-level R² .04, family-mean R² .11 and centroid R² .09 — about a third of the family-level structure of never-seen
families; on a random prompt split WITHIN the same 55 families it reaches R² .34, i.e. the oracle ceiling. The convention-vector cosine is
therefore the informative held-out number; the convention R² is negative mainly because predicted vectors are shorter than the true ones.

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

## Why a linear map cannot interpolate here: the 55 convention vectors are nearly orthogonal

| | mean pairwise cos | mean \|cos\| | pairs with \|cos\| ≥ .5 (of 1,485) | participation ratio | top PC share |
|---|---|---|---|---|---|
| write vectors (L24 cue-token nat − alt, `../steering/vectors`) | +.02 | .06 | 10 | 48 / 55 | .08 |
| read vectors (L8 evidence-token nat − alt) | +.01 | .06 | 8 | 49 / 55 | .07 |

The raw cue-token centroids of different families ARE similar (mean pairwise cos .75; the shared "cue token inside code" state, norm ≈ 173),
but the convention vectors are a ~30 % perturbation on top of it (norm ≈ 52) and point in ~48 independent directions. A ridge fitted on 36
families can only reproduce a held-out direction that lies in the span of those 36; that is why the held-out cosine tracks the span fraction
(.87 / .90) and why the axis-mates (quotes, snake/camel, spacing) transfer while isolated conventions do not.

## Centroid-level map and the 80/20 split (user request 2026-09-15; `read_write_centroid_map.py`)

Second split `fixed_split_80_20.json` (44 train / 11 test, stratified by category, seed 2026, written before fitting). Two maps on it:
the per-prompt ridge (as above) and a **centroid map** fitted on the 88 (family, pole) MEAN activations of the training families (raw
activations, not differences; λ by leave-one-family-out CV → 0.32). Held-out = the 22 centroids of the 11 test families.

| read layer → L24 | centroid map: held-out centroid cos | centroid map: centroid R² (train-mean / test-mean) | centroid map: convention cos | per-prompt map: convention cos | per-prompt map: centroid cos | per-prompt map: centroid R² | per-prompt map: prompt-level R² |
|---|---|---|---|---|---|---|---|
| L5 | .52 | .28 / .21 | .30 | .35 | .58 | .31 | .10 |
| L8 | .53 | .28 / .22 | .32 | .37 | .59 | .33 | .11 |
| L10 | .53 | .28 / .22 | .32 | .38 | .60 | – | .11 |

Shuffled controls: centroid cos −.05 … .10, centroid R² −.08 … −.21, convention cos ≈ 0; mean-vector baseline for the convention .14.

- A held-out family's MEAN write state is predicted well once the shared cue-token component is removed (centroid cos .5–.6, centroid R² ≈ .3
  — the family-identity share of the write variance is .29, so this is near its ceiling). This is the number the prompt-level R² (.04–.11)
  hides: two thirds of the cue-token variance is prompt-specific and unreachable from the evidence tokens.
- The convention offset between a family's two centroids is NOT better predicted at the centroid level (.32 vs .37 for the per-prompt map on the
  same split): fitting on 88 clean means instead of 17,600 noisy prompts does not help, so the limit is coverage of the family-specific
  directions, not noise. The per-prompt map is better on every measure (centroid R² .33 vs .28, centroid cos .59 vs .53, convention cos
  .37 vs .32): the within-family prompt variation teaches the ridge which read directions are informative.
- Per family at L8 (`centroid_map_80_20_per_family.csv`): the axis-mates transfer (js_camel_snake .66 / .73, comma_space .56 / .65,
  py_fstring .45 / .53, float_literals .43 / .49 — centroid map / per-prompt map), the isolated ones do not (rust_question .09 / .05,
  py_optional .14 / .17, py_with_open .15 / .20). Figure `centroid_map_80_20.png`.

### Why the held-out R² tops out near .3: the training span (80/20 split, read L10)

A ridge prediction is a combination of the training targets, so it cannot leave their span. Of the held-out families' centred write
centroids, only **.52** of the variance lies inside the span of the 88 training centroids (read side: .47). The centroid map reaches R² .28 on
the whole and **.54 on the in-span part** (oracle in-span projection .52): it explains what is reachable and nothing else. Per-family in-span
fractions (variance): js_camel_snake .91, py_fstring .84, py_bool_prefix .83, py_paren_if .80 … docstring_style .29, rust_question .17 —
the same ordering as the transfer. The raw similarity of the centroids (cos .72 write, .55 read) is the shared component, which the intercept
absorbs and which carries no family or convention information. Working read layer for further map work = **L10** (user decision).

### The same yardstick on the 69-task function-vector pool (GPT-J, 55 train / 14 held-out)

| pool | what is compared | raw pairwise cosine | centred mean abs cosine | effective dimensions (participation ratio) | held-out variance inside the training span | held-out map result |
|---|---|---|---|---|---|---|
| 69 tasks, GPT-J (55 train / 14 test) | write: function vectors (L13) | .39 | .21 | 39 of 69 | **.73** | R² .68, cos .90 (baseline .64) |
| 69 tasks, GPT-J | read: label means (L6) | .73 | .18 | 44 of 69 | .62 | |
| 55 code families, Qwen (44 train / 11 test) | write: cue-token centroids (L24) | .72 | .10 | 63 of 110 | **.52** | R² .28–.34, cos .5–.6 |
| 55 code families, Qwen | read: evidence-token centroids (L10) | .55 | .10 | 76 of 110 | .47 | |

Raw pairwise cosine measures the shared component (which the intercept absorbs) and is the wrong yardstick: the FVs look *less* similar
than the code centroids (.39 vs .72) but, once centred, task directions are twice as correlated (.21 vs .10), more concentrated (39 of 69 vs
63 of 110), and a held-out task lies .73 inside the span of 55 training tasks vs .52 for a held-out convention inside 44 training families.
In both pools the map's held-out R² ≈ the in-span fraction (.68 vs .73; .28–.34 vs .52 — on the code pool the map explains .54 of the in-span
part): the read→write map generalises exactly as far as the training span reaches. Tasks share factors (morphology, translation,
classification); conventions are closer to one direction each.

## Comparison with the text pools

| pool | protocol | held-out convention cos | baseline / shuffled |
|---|---|---|---|
| GPT-J, 11 lexically identical → 6 lexically diverse (read L0) | fixed | ≈ 0 (see `../../../read_write_map/`) | – |
| Qwen, 12-family headroom text pool (read L12) | fixed 2/3–1/3 | .44 | shuffled ≈ 0 |
| Qwen, 12-family headroom text pool (read L12) | LOFO | .55 within-axis / .21 cross-axis | shuffled ≈ 0 |
| **Qwen, 55 code families (read L8)** | fixed 36/19 | **.38** (.43 on both-feature families) | baseline .11 / shuffled .00 |

Files: `read_write_map_code.png` (per test family + read-layer sweep), `coverage.{png,csv}`, `pool_summary_Lsweep_L24.csv`,
`pool_per_family_Lsweep_L24.csv`, `fixed_split.json`, `subsets.json`.

### Adding the 16 lexically diverse text families to the training set (read L8 → write L24, λ = 10, 80/20 code split; `read_write_augment.py`, `augment_text_L8.json`)

| training set | held-out centroid cos | held-out centroid R² | convention cos | in-span fraction |
|---|---|---|---|---|
| 44 code families | .589 | .333 | .371 | .52 |
| 44 code + 16 text families | .630 | .374 | .359 | .56 |
| 16 text families only | .199 | .034 | .073 | .14 |

The text families add a little generic coverage of the write space (+.04 in centroid cos and R²) and nothing for the convention direction
(every held-out family moves by ≤ .04); on their own they predict almost nothing about code. (L8 because the text captures predate the
L5–L10 layers.)
