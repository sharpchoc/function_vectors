# results/style_translation/qwen25_base — accuracy vs k on Qwen2.5-7B (base)

User request 2026-09-11: repeat step 3 (in-context learning of the 27 writing conventions while translating
Spanish → English) on **Qwen/Qwen2.5-7B, the base model (not Instruct)**. Everything else is the step-3 protocol
of `../README.md`: the same 27 × 200 twin pairs, k = 0..4 rendered instances in context, ONE seeded T = 1
sample per prompt (≤ 48 new tokens, abbreviation-aware sentence cut), registry classifier at the decision,
Gemini 2.5 Flash faithfulness/coherence judge, accuracy = context convention ∧ judge OK, Wilson 95 % CI.

**What is model-specific.** The cue token (last token identical whichever convention follows) depends on the
tokeniser, so cue tokens and prompts were recomputed with Qwen's tokeniser (`cue_tokens.py --model qwen25_base`,
`build_prompts.py --model qwen25_base`; stored in `artifacts/style_translation/qwen25_base/{cues,prompts}` —
the GPT-J cues in `dataset_files/style_translation/pairs/` are untouched). Rollouts: `rollout.py --model qwen25_base`
(bf16, left padding, no BOS — Qwen2.5 has none) → `artifacts/style_translation/qwen25_base/rollouts/`; judge
`judge_rollouts.py --dir …/qwen25_base/rollouts`; `analyze.py --model qwen25_base`. Registry `models.py`.
2 RTX PRO 4500 pods (v9ehfud1icyy93, jz9rfsiwtxqare, ~1 h, terminated); 54,000 completions, 54,000 verdicts, 0 failures.

## Files

| file | contents |
|---|---|
| `accuracy_by_k.png` | 27 panels (LEXICAL 16 / FIXED 11), accuracy vs k per convention, CI bars |
| `unscorable_by_k.png`, `unscorable.csv` | completions with no style decision |
| `summary.csv`, `records.npz` | per (family, style, k) and per completion, same schema as `../summary.csv` |

## Findings — Qwen2.5-7B base vs GPT-J-6B (flipped convention, k = 0 → 4; "style-only" = convention adopted regardless of the judge)

| us_uk | 0.07 → 0.55 | 0.04 → **0.64** | 0.65 | 0.65 | 0.86 / 0.98 |
| ise_ize | 0.03 → 0.55 | 0.01 → **0.69** | 0.59 | 0.70 | 0.85 / 0.98 |
| brit_t_past | 0.07 → 0.20 | 0.03 → **0.27** | 0.26 | 0.28 | 0.77 / 0.96 |
| contractions | 0.41 → 0.61 | 0.42 → **0.68** | 0.70 | 0.69 | 0.83 / 0.98 |
| num_words | 0.06 → 0.28 | 0.13 → **0.48** | 0.34 | 0.49 | 0.81 / 0.97 |
| ordinal_words | 0.42 → 0.65 | 0.71 → **0.89** | 0.79 | 0.94 | 0.76 / 0.96 |
| uk_vocab ¹ | 0.07 → 0.25 | 0.09 → **0.31** | 0.29 | 0.31 | 0.82 / 0.97 |
| register ¹ | 0.14 → 0.22 | 0.15 → **0.33** | 0.27 | 0.34 | 0.83 / 0.98 |
| unit_abbr ¹ | 0.03 → 0.32 | 0.04 → **0.56** | 0.38 | 0.57 | 0.83 / 0.98 |
| diacritics ¹ | 0.28 → 0.30 | 0.33 → **0.45** | 0.35 | 0.45 | 0.82 / 0.97 |
| latin_abbr ¹ | 0.07 → 0.17 | 0.12 → **0.24** | 0.19 | 0.24 | 0.83 / 0.97 |
| hyphen_compound ¹ | 0.18 → 0.20 | 0.26 → **0.21** | 0.23 | 0.23 | 0.87 / 0.98 |
| flat_adverb ¹ | 0.04 → 0.06 | 0.04 → **0.09** | 0.07 | 0.10 | 0.86 / 0.97 |
| irreg_past ¹ | 0.11 → 0.10 | 0.23 → **0.12** | 0.11 | 0.12 | 0.79 / 0.96 |
| latin_plural ¹ | 0.23 → 0.28 | 0.24 → **0.35** | 0.32 | 0.38 | 0.84 / 0.96 |
| title_abbr ¹ | 0.14 → 0.40 | 0.36 → **0.57** | 0.52 | 0.58 | 0.74 / 0.96 |
| whilst | 0.01 → 0.54 | 0.01 → **0.46** | 0.60 | 0.47 | 0.86 / 0.98 |
| ampersand | 0.01 → 0.62 | 0.00 → **0.71** | 0.74 | 0.73 | 0.85 / 0.98 |
| percent_sign | 0.10 → 0.81 | 0.01 → **0.97** | 0.98 | 0.98 | 0.86 / 0.98 |
| em_dash | 0.05 → 0.64 | 0.35 → **0.88** | 0.92 | 0.93 | 0.75 / 0.95 |
| ellipsis | 0.03 → 0.63 | 0.01 → **0.85** | 0.81 | 0.86 | 0.58 / 0.68 |
| curly_quotes | 0.14 → 0.66 | 0.07 → **0.90** | 0.93 | 0.93 | 0.70 / 0.94 |
| quote_punct | 0.42 → 0.83 | 0.32 → **0.93** | 0.89 | 0.93 | 0.88 / 0.99 |
| double_space | 0.00 → 0.74 | 0.01 → **0.87** | 0.93 | 0.90 | 0.72 / 0.98 |
| oxford_comma | 0.64 → 0.70 | 0.21 → **0.78** | 0.79 | 0.78 | 0.87 / 0.99 |
| sentence_caps | 0.00 → 0.68 | 0.00 → **0.90** | 0.92 | 0.92 | 0.68 / 0.92 |
| all_caps | 0.00 → 0.35 | 0.00 → **0.81** | 0.99 | 0.97 | 0.53 / 0.84 |

¹ the ten lexically diverse families added 2026-09-11.

Group means at k = 4 (flipped convention):

| group | GPT-J accuracy | Qwen accuracy | GPT-J style-only | Qwen style-only | Qwen unscorable | Qwen judge OK (all cells) |
|---|---|---|---|---|---|---|
| 6 original lexical families | .47 | .61 | .55 | .62 | .12 | .95 (GPT-J .79) |
| 10 new lexical families | .23 | .32 | .27 | .33 | .21 | |
| 11 fixed-marker families | .66 | .82 | .86 | .85 | .04 | |

1. **Qwen translates far better, which is most of its accuracy gain.** Judge OK is .95 on average vs .79 for
   GPT-J. For the fixed-marker families the *style-only* adoption rate is the same for both models (.85 vs .86):
   the extra accuracy (.82 vs .66) comes from translations that stay faithful and coherent under the convention
   (all_caps .35 → .81, sentence_caps .68 → .90, double_space .74 → .87, ellipsis .63 → .85).
2. **Qwen learns the original lexical conventions somewhat better** (style-only .62 vs .55; num_words spelled-out
   cardinals .34 → .49, ise_ize .59 → .70, ordinal_words .79 → .94), whilst is the one convention it learns less
   (.47 vs .60).
3. **The ten lexically diverse families stay hard for Qwen too.** Style-only adoption at k = 4 is .33 on average
   (GPT-J .27): unit_abbr improves most (.38 → .57), title_abbr .52 → .58, register .27 → .34, diacritics .35 → .45;
   flat_adverb (.10), irreg_past (.12), hyphen_compound (.23), latin_abbr (.24) and uk_vocab (.31) are essentially
   unlearned after four examples, exactly as for GPT-J. Since Qwen's translations are faithful .97 of the time, this
   is not a translation-quality effect: the model sees "lorry", "commence", "kilometers", "pleaded" four times and
   still writes the standard American word. The house-style pole is also weaker for these families (nat accuracy
   .63 vs .85 / .93 for the other groups) and does not improve with k for diacritics, latin_abbr, irreg_past,
   latin_plural, flat_adverb — Qwen, like GPT-J, has no stable default for several of them.
4. **Avoidance persists for the same families**: unscorable .21 on average for the new group (latin_abbr .49,
   irreg_past .44) vs .04 for fixed markers — the model paraphrases around the decision word.

So the conclusion of `../README.md` carries over to a stronger base model: in-context learning of writing
conventions works for formatting choices and systematic orthographic rules applied to the same word, and does
not work in four shots for conventions that require choosing a different lexical item or word form.

## Provenance / caveats
- Cue tokens differ between tokenisers (e.g. Qwen splits capitalised words differently: all_caps word-internal
  cues .14 vs .00 for GPT-J), so per-decision prompts are not byte-identical across models; the decision sites
  (the twins' opportunities) are.
- Same judge, same prompt; judge OK for ellipsis is .68 (bare `...` vs `…` reference artefact, see `../README.md`).
- One sample per prompt at T = 1, as for GPT-J.
