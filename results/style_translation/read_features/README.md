# results/style_translation/read_features — mean activations at the evidence tokens ("read features")

Step 6 of the style-translation study (2026-09-10). Companion to `../steering/` (the *write* side: a
mean-difference vector at the cue token induces a convention). Here the object is the *read* side: for
each of the 17 families, each pole (nat / alt → 34 styles) and each layer, the mean residual activation
at the **evidence tokens** — the tokens of the English context that realise the convention.

## Definitions (user decisions 2026-09-10)

- **Prompts.** The step-3 **k = 4 translation prompts** (`Spanish:\n{es}\n\nEnglish:\n{twin}` cut at
  the 5th cue token): one per text and pole, 200 texts per family, no success filtering. Each prompt
  contains the first 4 rendered instances of the convention (decision points 0..3).
- **Evidence tokens** (`evidence_tokens.py`). The tokenizer's NATURAL segmentation only — never a
  forced split. For one instance: the style-bearing tokens from the first token where the twin and its
  single-instance flip differ (cue_idx + 1; the cue token is never included) to the last token that
  starts inside the rendered span (trailing whitespace trimmed unless the span is all whitespace).
  Whole-word tokens stay whole; where the tokenizer itself splits, only the differing pieces count;
  if the span is entirely shared with the other twin the divergence token itself is the evidence.
  Worked examples (nat / alt): us_uk ` labor` / ` labour`; brit_t_past ` learned` / ` learnt`;
  contractions `'s` / ` is` (pronoun excluded), ` isn't` / ` is not`; whilst ` while` / ` whilst`,
  ` among` / ` amongst` (` Among`+`st` → `st`; nat `Among` shared → the next word); oxford_comma
  `, and` (2 tokens) / ` and`; double_space ` It` / ` `+` It`; sentence_caps ` This` / ` this`;
  all_caps every token of the sentence (≈ 26 nat / 44 alt tokens per instance — capitals split into
  more pieces); curly_quotes `"`…`"` (opening + closing, 2 tokens) / the 4 byte tokens of “…”;
  quote_punct `."` / `".`; em_dash `—` / ` -`; ellipsis `...` / `…`; num_words ` 3` / ` three`;
  percent_sign `%` / ` percent`; ordinal_words ` 1st` / ` first`. Full audit: `evidence_tokens.csv`.
- **Read feature.** r_pole(F, L) = mean over the 200 prompts of the per-prompt average of
  hidden_states[L] (output of block L, L = 1..28) over ALL evidence tokens of the 4 instances. Stored
  per family in `artifacts/style_translation/read_features/<family>.npz`: `mean_nat`, `mean_alt`
  [28, 4096]; `inst_nat`, `inst_alt` [4, 28, 4096] (per instance index — the k-th instance has k prior
  instances in context); `n_nat`, `n_alt` (200); tokens per prompt; `split_half_cos` (halves by text);
  `norm_diff`, `norm_mean`. The 9 screened layers {2,4,6,8,10,12,16,20,24} are the reporting grid.

## Files

| file | contents |
|---|---|
| `read_feature_summary.png` | per family (grouped: lexically diverse left, lexically identical right), by layer: cos(r_nat − r_alt, v_nat) against the paired cue-token steering vector of the same family and layer; relative norm of the read difference (scaled to its max); split-half reliability (dashed) |
| `read_features.csv` | per family × layer (1..28): rel_norm, split_half_cos, cos_read_write, n, evidence tokens per prompt |
| `evidence_tokens.csv` | per family and pole: the 5 most frequent evidence strings with counts and token counts |

## Findings

1. **The read features are well defined and reliable.** Every family has exactly 200 prompts per pole;
   the split-half cosine of r_nat − r_alt is ≥ 0.99 at every layer for every family (`read_features.csv`).
   The difference is large relative to the mean state — ‖r_nat − r_alt‖/‖mean‖ = 0.25–0.52 at L6,
   0.11–0.50 at L24 — which is expected: these are the very tokens whose identity differs between the
   poles, unlike the cue-token vectors (0.07–0.36) where the token is shared.
2. **Read and write directions mostly differ.** The cosine between the read difference and the
   cue-token steering vector at the same layer is modest for the lexically diverse families and rises
   with depth — us_uk .31 → .51, ise_ize .05 → .41, sentence_caps .11 → .37, whilst −.05 → .35,
   brit_t_past .06 → .27, contractions .06 → .29 (L6 → L24) — and near zero or negative for the fixed-
   marker families: percent_sign ≈ 0, curly_quotes .05, num_words .07, quote_punct −.05, double_space
   −.04 (−.20 at L12), oxford_comma −.30 at L24. The exception is all_caps (.57 → .84): in a k = 4
   all-caps prompt the cue token itself sits in capitalised text, so the read state and the cue state
   overlap. So "the representation of the evidence" and "the direction that makes the model produce
   the convention at the next decision" are largely different objects for these conventions — the
   read → write relation is not the identity, as in the 69-task ICL study.
3. **Token-count asymmetries to keep in mind** (`tokens per prompt`): all_caps averages 104 (nat) vs
   175 (alt) tokens per prompt; double_space 4 vs 8; curly_quotes 8 vs 16; oxford_comma 8 vs 4;
   ordinal_words 8 vs 4 (` 1st` is often 2 tokens). The per-prompt average removes the count from the
   mean, but the alt-pole ALL-CAPS features average over many sub-word pieces.

## Provenance / caveats

- GPT-J-6B fp16, one RunPod RTX PRO 4500 Blackwell pod (9zlp5ccfoluurj, ≈ 25 min incl. model load,
  terminated). `evidence_tokens.py` (CPU: 6,800 prompts × 4 instances, all assertions passed: prompt
  is a prefix of the twin tokenisation, stored cue index reproduced, divergence after the cue, evidence
  inside the prompt and disjoint from the cue) → `capture_evidence.py` (`logs/read_job.sh`) →
  `read_features_analyze.py`.
- Caveats: (i) only k = 4 prompts: every evidence instance has 0–3 prior instances in context, and the
  pooled mean mixes them — `inst_*` in the npz separates the 1st…4th instance; (ii) the translation
  format only (Spanish in context); (iii) all_caps evidence spans whole sentences, so its read feature
  is a sentence-level average, not a token-level one; (iv) the read-vs-write cosine compares against
  the PAIRED cue-token vectors of `../steering/` (2026-09-10).
