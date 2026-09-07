# results/style_translation — does GPT-J pick up a style convention in context while translating?

Step 3 of the style-translation study (2026-09-07). Data: `dataset_files/style_translation/pairs/`
(17 families × 200 Spanish texts, each with two English twins that differ only in the tested
family's style, and cue tokens for every decision point — see that folder's README).

## Protocol

- **Prompt** (one per text × style × k): `Spanish:\n{text_es}\n\nEnglish:\n{twin in style P, cut
  right after cue token k}`. k = number of decision points already rendered in the style before the
  cue (k = 0..4); "cue token" = the last token that is identical whichever style follows (user
  definition; word-internal cues allowed; sentence families use the closing period, or the newline
  after `English:` for the first sentence; curly_quotes: one decision per quotation).
- **Model**: GPT-J-6B fp16, ONE seeded T=1 sample per prompt (repo readout convention), max 48 new
  tokens, cut at the end of the current sentence; `capped` = no sentence end within 48 tokens.
- **Style decision** (deterministic, registry classifiers in `src/sandbox/ext_styleprops/
  properties.py`): the completion (prefixed by the already-generated start of a word-internal cue)
  is classified nat / alt / None. Unit test: every twin's own continuation is classified as its own
  style (100% in all 17 families).
- **Faithfulness / coherence** (Gemini 2.5 Flash via OpenRouter, one completion per call): OK if
  the completion is coherent English that faithfully renders what comes next in the Spanish
  (paraphrase allowed; capped completions are not penalised for truncation); style/typography
  ignored.
- **Accuracy** = P(decision == context style AND judge OK). A completion that avoids the decision
  (decision None) is inaccurate; its share is reported separately (`unscorable.csv`,
  `unscorable_by_k.png`) — user request.
- n = 200 texts per (family, style, k); Wilson 95% CI.

## Files

| file | contents |
|---|---|
| `accuracy_by_k.png` | 17 panels; accuracy vs k (0..4), one line per style (nat = house style, alt = the flipped convention), CI bars |
| `unscorable_by_k.png`, `unscorable.csv` | share of completions with no style decision, per family × style, per k and pooled |
| `summary.csv` | per (family, style, k): accuracy, CI, style_ok, wrong_style, unscorable, style_ok_but_unfaithful, judge_ok, capped, n |
| `records.npz` | per-completion arrays (style, k, decision, judge_ok, capped) to regenerate views |

Scripts: `src/sandbox/style_translation/{build_prompts,scoring,rollout,judge_rollouts,analyze}.py`.
Artifacts (gitignored): `artifacts/style_translation/{prompts,rollouts}/<family>.json` (all prompts,
completions, decisions, judge verdicts, seeds).

## Findings
**Headline: GPT-J picks up most conventions from one or two in-context examples, but the
translation itself is the bottleneck.** Accuracy = convention used AND faithful, coherent
translation; the "style-only" column shows the convention-adoption rate alone.

| family | style | acc k=0 | acc k=1 | acc k=2 | acc k=3 | acc k=4 | style-only k=0→4 | unscorable (pooled) | judge OK (pooled) |
|---|---|---|---|---|---|---|---|---|---|
| sentence_caps | nat | 0.56 | 0.68 | 0.74 | 0.70 | 0.77 | 0.96→0.99 | 0.02 | 0.70 |
| sentence_caps | alt | 0.00 | 0.10 | 0.56 | 0.67 | 0.68 | 0.00→0.92 | 0.02 | 0.66 |
| all_caps | nat | 0.52 | 0.69 | 0.72 | 0.71 | 0.76 | 0.98→0.99 | 0.02 | 0.68 |
| all_caps | alt | 0.00 | 0.32 | 0.34 | 0.36 | 0.35 | 0.01→0.99 | 0.03 | 0.39 |
| double_space | nat | 0.71 | 0.74 | 0.76 | 0.69 | 0.77 | 0.98→0.99 | 0.01 | 0.74 |
| double_space | alt | 0.00 | 0.40 | 0.54 | 0.59 | 0.74 | 0.00→0.93 | 0.01 | 0.70 |
| us_uk | nat | 0.65 | 0.67 | 0.70 | 0.65 | 0.71 | 0.72→0.81 | 0.17 | 0.86 |
| us_uk | alt | 0.07 | 0.41 | 0.49 | 0.58 | 0.55 | 0.09→0.65 | 0.18 | 0.86 |
| ise_ize | nat | 0.68 | 0.70 | 0.73 | 0.68 | 0.76 | 0.80→0.83 | 0.16 | 0.86 |
| ise_ize | alt | 0.03 | 0.52 | 0.54 | 0.60 | 0.55 | 0.04→0.59 | 0.16 | 0.85 |
| brit_t_past | nat | 0.49 | 0.41 | 0.51 | 0.43 | 0.51 | 0.65→0.59 | 0.30 | 0.77 |
| brit_t_past | alt | 0.07 | 0.15 | 0.14 | 0.17 | 0.20 | 0.09→0.26 | 0.31 | 0.76 |
| whilst | nat | 0.72 | 0.63 | 0.65 | 0.58 | 0.58 | 0.76→0.69 | 0.22 | 0.86 |
| whilst | alt | 0.01 | 0.49 | 0.49 | 0.53 | 0.54 | 0.01→0.60 | 0.21 | 0.85 |
| contractions | nat | 0.23 | 0.58 | 0.57 | 0.65 | 0.69 | 0.30→0.81 | 0.13 | 0.83 |
| contractions | alt | 0.41 | 0.47 | 0.56 | 0.61 | 0.61 | 0.48→0.70 | 0.15 | 0.84 |
| ampersand | nat | 0.83 | 0.82 | 0.81 | 0.77 | 0.83 | 0.91→0.94 | 0.07 | 0.86 |
| ampersand | alt | 0.01 | 0.34 | 0.50 | 0.66 | 0.62 | 0.01→0.74 | 0.08 | 0.85 |
| oxford_comma | nat | 0.23 | 0.65 | 0.72 | 0.77 | 0.84 | 0.26→0.93 | 0.05 | 0.87 |
| oxford_comma | alt | 0.64 | 0.69 | 0.72 | 0.70 | 0.70 | 0.69→0.79 | 0.04 | 0.87 |
| curly_quotes | nat | 0.28 | 0.62 | 0.59 | 0.65 | 0.67 | 0.43→0.94 | 0.15 | 0.69 |
| curly_quotes | alt | 0.14 | 0.56 | 0.64 | 0.63 | 0.66 | 0.18→0.93 | 0.16 | 0.71 |
| quote_punct | nat | 0.22 | 0.71 | 0.76 | 0.79 | 0.88 | 0.25→0.91 | 0.12 | 0.87 |
| quote_punct | alt | 0.42 | 0.67 | 0.73 | 0.80 | 0.83 | 0.48→0.89 | 0.15 | 0.88 |
| em_dash | nat | 0.56 | 0.80 | 0.64 | 0.79 | 0.72 | 0.76→0.96 | 0.06 | 0.75 |
| em_dash | alt | 0.05 | 0.71 | 0.57 | 0.70 | 0.64 | 0.06→0.92 | 0.10 | 0.75 |
| ellipsis | nat | 0.45 | 0.56 | 0.60 | 0.60 | 0.71 | 0.89→0.99 | 0.03 | 0.61 |
| ellipsis | alt | 0.03 | 0.55 | 0.61 | 0.63 | 0.63 | 0.04→0.81 | 0.04 | 0.54 |
| num_words | nat | 0.55 | 0.74 | 0.70 | 0.73 | 0.79 | 0.69→0.93 | 0.12 | 0.83 |
| num_words | alt | 0.06 | 0.32 | 0.24 | 0.33 | 0.28 | 0.10→0.34 | 0.11 | 0.79 |
| percent_sign | nat | 0.53 | 0.85 | 0.85 | 0.86 | 0.86 | 0.65→0.97 | 0.06 | 0.86 |
| percent_sign | alt | 0.09 | 0.72 | 0.79 | 0.84 | 0.81 | 0.10→0.98 | 0.09 | 0.85 |
| ordinal_words | nat | 0.24 | 0.65 | 0.74 | 0.72 | 0.78 | 0.33→0.97 | 0.05 | 0.75 |
| ordinal_words | alt | 0.42 | 0.46 | 0.51 | 0.64 | 0.65 | 0.57→0.79 | 0.06 | 0.76 |

Reading the plot (`accuracy_by_k.png`, red = the flipped convention, blue = house style):

- **Fast learners (one example is enough):** all_caps, sentence_caps, double_space, em_dash,
  ellipsis, percent_sign, quote_punct, curly_quotes, oxford_comma, ampersand. The flipped style goes
  from ≈0 at k = 0 to 60–90 % adoption by k = 1–2 (style-only), e.g. all_caps 0.01 → 0.84 → 0.98,
  percent_sign 0.10 → 0.86 → 0.94, sentence_caps 0.00 → 0.12 → 0.82 (this one needs two examples).
- **Partial learners:** us_uk (British 0.09 → 0.65 style-only), ise_ize (0.04 → 0.59–0.69), whilst
  (0.01 → 0.53–0.65), contractions (both directions ≈0.6–0.8 by k = 4), ordinal_words (spelled
  0.57 → 0.79; digits 0.33 → 0.97).
- **Not learned:** brit_t_past (-t pasts 0.09 → 0.26; a third of completions avoid these verbs
  altogether) and num_words (spelled-out cardinals stay at 0.31–0.37 even at k = 4 while digits
  reach 0.93 — the Spanish source shows digits, and the model follows the source form).
- **Priors visible at k = 0** (style-only, before any English example): the model prefers NO serial
  comma (0.69 vs 0.26), spelled ordinals (0.57 vs 0.33), expanded contractions (0.48 vs 0.30),
  punctuation OUTSIDE the closing quote (0.48 vs 0.25); for everything else it starts in the house
  style.
- **Accuracy is capped by translation quality.** Judge OK rates are 0.75–0.88 for most families
  but fall to 0.54–0.61 for ellipsis (odd ellipsis placement inherited from the Spanish) and 0.39
  for the ALL CAPS context (vs 0.68 in standard case): writing in capitals measurably degrades
  GPT-J's translation ("ECOHUMAN METHODS" for "ecological", "HOOPS PLAYER" for "soccer
  enthusiast"). So the all_caps flipped-style accuracy plateaus at ≈0.35 although the style itself
  is adopted 98 % of the time.
- **Unscorable completions** (`unscorable_by_k.png`, `unscorable.csv`): the model avoids the
  decision mostly at k = 0 (curly_quotes 0.40–0.45: it drops the quotation marks (94/170), copies
  the Spanish « » (43) or uses single quotes (33); percent_sign 0.26–0.34; quote_punct 0.23) and
  the rate drops to ≤ 0.10 after one example. Persistently high: brit_t_past 0.21–0.38 (other
  verbs chosen), whilst ≈0.20–0.24, us_uk / ise_ize 0.10–0.23 (synonyms without a spelling split),
  contractions 0.09–0.20. Everything else ≤ 0.10.
- Completions cut by the 48-token cap: 1–5 % except the all_caps flipped context (27–48 %, capitals
  tokenise into many more pieces); the judge is told when a completion was capped.

## Provenance / caveats
- Rollouts 2026-09-07 on three RTX PRO 4500 Blackwell pods (ynswjnas0a608z, f8ajl4h7q85xn0,
  g7t6kx7ugh4p6c; ≈1.5 h each, all terminated), GPT-J-6B fp16, seed `crc32(f"{family}|rollout|{batch}")`,
  token budget 8000 / batch cap 16; 34,000 completions. Judge: `google/gemini-2.5-flash`, T = 0, one
  completion per call, 34,000 calls, 0 unresolved failures.
- Judge calibration (sentence_caps, 14 verdicts read): rejections were genuine mistranslations or
  hallucinated content ("ink" for "pencil lead", "a plan" for "flat"), approvals were faithful
  paraphrases; capped completions were not penalised for truncation.
- Caveats: (1) the Spanish source fixes number/quote/dash forms (digits, « », rayas), which acts as a
  source-form prior — clearest for num_words; (2) house-style artefacts in the English context
  ("At 1st,", capitals after ellipses) may depress the judge's fluency for a few families;
  (3) one T = 1 sample per prompt: per-point CI ≈ ±0.07 at n = 200; (4) T = 1 samples are not
  reproducible across GPU types.
