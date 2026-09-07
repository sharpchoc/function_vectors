# results/style_free_text — the in-context convention test WITHOUT the translation scaffold

Variation of `results/style_translation/` (user request 2026-09-07). Same 17 families × 200 texts,
same English twins and cue tokens (`dataset_files/style_translation/pairs/`), same GPT-J sampling
and style scoring — but the prompt is **just the English twin cut right after cue token k**: no
Spanish source, no `Spanish:` / `English:` header. Nothing under `results/style_translation/` or
`src/sandbox/style_translation/` was modified (this variation only imports from that module).

## Protocol (differences from the translation run in bold)

- **Prompt**: `{English twin in style P, cut right after cue token k}`, k = 0..4. Cue tokens were
  **recomputed on the header-free tokenisation** (same last-shared-token rule; the first English
  token can tokenise differently without a preceding newline). **740 of the 34,000 prompts have no
  English context at all** (the first sentence for all_caps / sentence_caps, and texts whose first
  word carries the opportunity): they start from GPT-J's `<|endoftext|>` token.
- Model, sampling, style decision: identical (GPT-J-6B fp16, one seeded T = 1 sample, 48 new tokens,
  cut at the sentence end with a `capped` flag, registry classifiers; scorer unit test 100% on the
  new items).
- **Judge**: Gemini 2.5 Flash, one completion per call, **coherence only** — there is no source to
  be faithful to (user decision). OK = coherent, fluent English that continues the passage sensibly;
  conventions/typography ignored; capped completions not penalised for truncation.
- **Accuracy = P(convention used AND coherent)**; a completion that uses neither convention is
  inaccurate and its share is reported separately (`unscorable.csv`, `unscorable_by_k.png`).
- n = 200 per (family, style, k); Wilson 95% CI.

## Files

| file | contents |
|---|---|
| `accuracy_by_k.png` | 17 panels, accuracy vs k, one line per style |
| `style_only_vs_translation.png` | **the comparison**: convention adoption (style-only, judge-independent) vs k, plain English (solid) vs the same prompts with the Spanish source (dashed, from `results/style_translation/summary.csv`) |
| `unscorable_by_k.png`, `unscorable.csv` | share of completions using neither convention |
| `summary.csv`, `records.npz` | per (family, style, k) breakdown; per-completion arrays |

Scripts: `src/sandbox/style_free_text/{build_prompts_plain,rollout_plain,judge_plain,analyze_plain}.py`.
Artifacts (gitignored): `artifacts/style_free_text/{prompts,rollouts}/<family>.json`.

## Findings
**Headline: without the Spanish source the decision mostly never arises.** In free text the model is
not obliged to reproduce the word, list, quotation, or number that carries the convention, so for
the lexical and number families 60–95 % of completions use neither style ("unscorable"), against
5–30 % with the translation scaffold. Where the decision IS forced by the local context — a sentence
boundary (sentence_caps, all_caps, double_space), an open quotation (quote_punct), a number
awaiting its unit (percent_sign) — in-context learning is still visible, but weaker and noisier than
with the source.

| family | style | acc k=0 | acc k=4 | style-only k=0→4 (plain) | style-only k=0→4 (with Spanish) | unscorable (plain, pooled) | unscorable (with Spanish) | coherent (pooled) |
|---|---|---|---|---|---|---|---|---|
| sentence_caps | nat | 0.37 | 0.38 | 0.72→0.67 | 0.96→0.99 | 0.20 | 0.02 | 0.44 |
| sentence_caps | alt | 0.03 | 0.41 | 0.14→0.65 | 0.00→0.92 | 0.22 | 0.02 | 0.40 |
| all_caps | nat | 0.41 | 0.44 | 0.77→0.70 | 0.98→0.99 | 0.20 | 0.02 | 0.46 |
| all_caps | alt | 0.00 | 0.15 | 0.01→0.70 | 0.01→0.99 | 0.26 | 0.03 | 0.23 |
| double_space | nat | 0.55 | 0.30 | 0.92→0.57 | 0.98→0.99 | 0.26 | 0.01 | 0.42 |
| double_space | alt | 0.00 | 0.28 | 0.00→0.52 | 0.00→0.93 | 0.26 | 0.01 | 0.41 |
| us_uk | nat | 0.12 | 0.16 | 0.14→0.18 | 0.72→0.81 | 0.81 | 0.17 | 0.80 |
| us_uk | alt | 0.02 | 0.14 | 0.02→0.21 | 0.09→0.65 | 0.80 | 0.18 | 0.79 |
| ise_ize | nat | 0.03 | 0.10 | 0.04→0.17 | 0.80→0.83 | 0.87 | 0.16 | 0.75 |
| ise_ize | alt | 0.01 | 0.09 | 0.01→0.14 | 0.04→0.59 | 0.89 | 0.16 | 0.75 |
| brit_t_past | nat | 0.02 | 0.06 | 0.04→0.09 | 0.65→0.59 | 0.93 | 0.30 | 0.74 |
| brit_t_past | alt | 0.00 | 0.04 | 0.01→0.04 | 0.09→0.26 | 0.93 | 0.31 | 0.70 |
| whilst | nat | 0.01 | 0.07 | 0.01→0.10 | 0.76→0.69 | 0.90 | 0.22 | 0.74 |
| whilst | alt | 0.00 | 0.09 | 0.00→0.10 | 0.01→0.60 | 0.89 | 0.21 | 0.72 |
| contractions | nat | 0.04 | 0.26 | 0.04→0.37 | 0.30→0.81 | 0.60 | 0.13 | 0.74 |
| contractions | alt | 0.13 | 0.23 | 0.19→0.29 | 0.48→0.70 | 0.64 | 0.15 | 0.72 |
| ampersand | nat | 0.20 | 0.25 | 0.21→0.28 | 0.91→0.94 | 0.75 | 0.07 | 0.72 |
| ampersand | alt | 0.00 | 0.19 | 0.00→0.21 | 0.01→0.74 | 0.79 | 0.08 | 0.68 |
| oxford_comma | nat | 0.35 | 0.58 | 0.38→0.64 | 0.26→0.93 | 0.36 | 0.05 | 0.88 |
| oxford_comma | alt | 0.30 | 0.48 | 0.32→0.53 | 0.69→0.79 | 0.39 | 0.04 | 0.88 |
| curly_quotes | nat | 0.07 | 0.28 | 0.10→0.37 | 0.43→0.94 | 0.71 | 0.15 | 0.77 |
| curly_quotes | alt | 0.05 | 0.24 | 0.07→0.30 | 0.18→0.93 | 0.73 | 0.16 | 0.77 |
| quote_punct | nat | 0.42 | 0.70 | 0.48→0.81 | 0.25→0.91 | 0.20 | 0.12 | 0.81 |
| quote_punct | alt | 0.17 | 0.77 | 0.20→0.86 | 0.48→0.89 | 0.18 | 0.15 | 0.79 |
| em_dash | nat | 0.01 | 0.10 | 0.01→0.17 | 0.76→0.96 | 0.73 | 0.06 | 0.65 |
| em_dash | alt | 0.01 | 0.10 | 0.01→0.15 | 0.06→0.92 | 0.79 | 0.10 | 0.65 |
| ellipsis | nat | 0.01 | 0.40 | 0.01→0.42 | 0.89→0.99 | 0.77 | 0.03 | 0.54 |
| ellipsis | alt | 0.00 | 0.23 | 0.00→0.41 | 0.04→0.81 | 0.79 | 0.04 | 0.48 |
| num_words | nat | 0.07 | 0.28 | 0.09→0.32 | 0.69→0.93 | 0.72 | 0.12 | 0.79 |
| num_words | alt | 0.04 | 0.18 | 0.07→0.23 | 0.10→0.34 | 0.72 | 0.11 | 0.74 |
| percent_sign | nat | 0.30 | 0.68 | 0.41→0.74 | 0.89→0.99 | 0.28 | 0.01 | 0.79 |
| percent_sign | alt | 0.10 | 0.65 | 0.12→0.73 | 0.12→0.98 | 0.30 | 0.01 | 0.79 |
| ordinal_words | nat | 0.00 | 0.27 | 0.01→0.35 | 0.33→0.97 | 0.69 | 0.05 | 0.65 |
| ordinal_words | alt | 0.15 | 0.17 | 0.19→0.26 | 0.57→0.79 | 0.78 | 0.06 | 0.67 |

Reading `style_only_vs_translation.png` (solid = plain English, dashed = with the Spanish source):

- **Forced decisions learn in context even without a source.** quote_punct 0.20 → 0.86 (plain) vs
  0.48 → 0.89 (with source); percent_sign 0.12 → 0.73 vs 0.10 → 0.98; all_caps 0.01 → 0.70 vs
  0.01 → 0.99; sentence_caps 0.14 → 0.65 vs 0.00 → 0.92; double_space 0.00 → 0.52 vs 0.00 → 0.93;
  oxford_comma 0.32 → 0.53 (no comma) / 0.38 → 0.64 (serial) vs 0.69 → 0.79 / 0.26 → 0.93.
- **Free-choice decisions barely register.** us_uk (0.02 → 0.21 vs 0.09 → 0.65), ise_ize (0.01 → 0.14
  vs 0.04 → 0.59), whilst (0.00 → 0.10 vs 0.01 → 0.60), brit_t_past (≤ 0.09 either way), ampersand
  (0.00 → 0.21 vs 0.01 → 0.74), curly_quotes (0.07 → 0.30 vs 0.18 → 0.93), num_words, ordinal_words,
  em_dash, ellipsis: 60–95 % unscorable because the continuation simply does not contain the
  feature ("…that accumulate **over time**." where the source said "among the components").
- **Frequency learning shows up in the unscorable curve**: for ellipsis the share of completions
  containing an ellipsis at all rises from 1–3 % at k = 0 to 42–44 % at k = 4; for double_space the
  unscorable share instead *rises* with k (9 → 42 %) because deeper in a paragraph the model ends
  the text (newline / end-of-text) rather than starting another sentence — an effect the scaffold
  suppresses (the Spanish still has sentences left to translate).
- **Coherence** (Gemini, no source to compare against): 65–88 % for most families; 40–46 % for the
  sentence families, dominated by empty completions (the paragraph ends) and topic drift; 23 % for
  the ALL CAPS context (capitals degrade generation here too). Punctuation-only completions that
  close the sentence (`."`) are counted coherent by rule — see caveats.
- **k = 0 without any context** (the 740 end-of-text-start prompts): unconditional GPT-J text starts
  lowercase 14 % of the time and in capitals 1 % — the priors the k = 0 points of sentence_caps /
  all_caps reflect.

## Provenance / caveats
- Rollouts 2026-09-07 on three RTX PRO 4500 Blackwell pods (2cwwrbfvfmncw3, 0wc3zcv8j8ctxz,
  vmvx33ugzpvd8e; ≈45 min, all terminated), GPT-J-6B fp16, seed `crc32(f"{family}|plain|{batch}")`;
  34,000 completions. Judge `google/gemini-2.5-flash`, T = 0, one completion per call, 34,000 calls,
  0 unresolved failures. Judge calibration (sentence_caps, 8 verdicts read): rejections were empty
  completions, topic drift, a hashtag, broken grammar — genuine.
- **Rule applied in analysis:** a completion that is only the closing punctuation of the current
  sentence (`."`, `".`, `!”` …) and carries a style decision is counted coherent (718 of the 1,110
  quote_punct rejections were such 2–3-character completions; the sentence cutter stops there and
  the judge, with no reference to compare, called bare punctuation "not a continuation"; in the
  translation run the reference was the same string and the judge accepted them).
- Not comparable one-to-one with `results/style_translation/accuracy_by_k.png`: the judges differ
  (coherence vs faithfulness). `style_only_vs_translation.png` compares the judge-independent
  quantity. Cue positions were recomputed on the header-free tokenisation, so a handful of cues
  differ from the translation run at the first token of the text.
- Nothing under `results/style_translation/` or `src/sandbox/style_translation/` was modified
  (verified with `git diff --stat` before committing).
