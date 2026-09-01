# Style-property study — Stage-0 spec + adjudication memo (2026-09-01)

User decisions already made (2026-09-01, before compute): model = GPT-J-6B only;
pool = ~15–20 candidates behaviorally pre-screened; readout = **sampled adherence only**
(T=1 seeded generation, classified at decision points; no logit-diff); first deliverable
= pipeline + behavioral pre-screen + decodability grid.

## Property spec sheet (17 candidates)

nat = the US-standard / typographically plain pole (base-corpus convention); alt = the
toggled pole. Detectors match BOTH surface forms, so rendered docs are 100%
polarity-consistent regardless of the form the corpus generator produced.

| property | family | nat / alt | confound rating |
|---|---|---|---|
| sentence_caps | case | Capitalized / lowercase sentence starts | low |
| all_caps | case | standard case / ENTIRE SENTENCES IN CAPS | high tokenizer divergence |
| us_uk | spelling | color / colour (curated lexicon, sense-ambiguous pairs excluded) | low |
| ise_ize | spelling | organize / organise (+ -yze, -ization) | low |
| brit_t_past | spelling | learned / learnt | low |
| whilst | spelling | while, among, amid / whilst, amongst, amidst | medium register |
| double_space | typography | one / two spaces after sentence-final period | low |
| oxford_comma | typography | X, Y, and Z / X, Y and Z | low |
| curly_quotes | typography | " / “ ” | low |
| em_dash | typography | word—word / word - word | low |
| ellipsis | typography | ... / … | low; LOW-N (37 docs) |
| quote_punct | typography | punctuation inside / outside closing quote | low |
| num_words | number | 14 / fourteen (cardinals 2–20) | low |
| percent_sign | number | 60% / 60 percent | low |
| ordinal_words | number | 3rd / third (1st–10th) | low |
| contractions | lexical | don't / do not (safe pairs only) | medium register |
| ampersand | lexical | and / & (non-list conjunctions) | medium register |

Corpus: 758 LLM-generated neutral prose docs (~300–500 GPT-J tokens; general batch +
five seeded batches for sparse features), `dataset_files/style_properties/base_corpus.json`.
Generator: google/gemini-2.5-flash via OpenRouter, feature-mix instructions in
`gen_corpus.py`.

## Definitional choices — PROPOSED defaults (user adjudication requested)

1. **Decision-token rule** (new, load-bearing): the decision token pair is the latest
   token end, in shared-prefix coordinates relative to the opportunity span, that is a
   token boundary in BOTH twins at or before the nat/alt divergence char, with the token
   ids asserted equal. This backs off BPE merges that differ between twins (e.g. `,"`
   merging only in the straight-quote twin) instead of dropping the site; the backed-off
   shared chars are regenerated at sampling time and absorbed by the classifier.
   Audit result: ≤1% of sites dropped for every property except all_caps (~9%).
2. **Evidence-token convention**: all tokens overlapping the manifested span (not just
   the first divergent subtoken).
3. **Adherence scoring**: per site, ONE T=1 seeded sample (repo convention; seed
   crc32(f"{prop}|prescreen|{batch}")); classification = property-level loose classifier
   first (e.g. first-alpha case for sentence_caps, any lexicon hit for spelling
   properties), strict expected-continuation prefix match (longer-first) as fallback,
   else unscorable. max_new = expected-continuation tokens + 4, capped 8–16 per property.
4. **Pre-screen gate thresholds** (applied post-hoc; data is stratified adherence, so
   re-adjudication needs no recompute): separation s = P(classified nat | nat context) −
   P(classified nat | alt context) ≥ 0.3 at k ≥ 4; s monotone in k (Spearman > 0 over
   k-bins 0..4); adherence to the prior-DISFAVORED pole ≥ 0.4 at k ≥ 4. If < 8 properties
   survive on GPT-J: STOP and escalate the model question (user decision).
5. **Read-feature headline definition** (Stage C): compute all three candidates from the
   same captures — pooled polarity means / signed difference; paired minimal-pair diff at
   aligned evidence sites; paired diff at matched background tokens — user adjudicates
   the headline before Stage D.
6. **Carrier definition across properties** (Stage C/D3): mean of signed directions vs
   mean of polarity-condition means — to adjudicate with 5.
7. **Head-sum FV analog (W2)**: deferred (most expensive item; only after residual-stream
   write-feature steering works).

## Pre-screen outcome (2026-09-01, GPT-J, corrected run)

**16/17 properties pass** (only `whilst` fails: separation 0.167 at k≥4, 8.5% scorable).
Separation s = P(classified nat | nat ctx) − P(classified nat | alt ctx) at k≥4:
curly_quotes .973, percent_sign .966, all_caps .940, us_uk .909, quote_punct .901,
double_space .855, oxford_comma .850, sentence_caps .792, ordinal_words .784,
brit_t_past .667, ampersand .667, contractions .663, num_words .643, ise_ize .435,
em_dash 1.0 and ellipsis 1.0 (both on thin scorable counts). s at k=0 ≈ 0 everywhere
(identical-prompt control), rising monotonically in k for all passers. Pool artifact:
`task_splits/style_properties_pool.json`. Provisional passes to re-examine with
multi-sample scoring: ellipsis (3.7% scorable), brit_t_past (5.0%); curly_quotes has a
mild k=0 offset (+0.214, ~2.8σ).

**Bug caught by the k=0 control** (first run, discarded): when the decision boundary
fell inside the opportunity span, the inconsistent expected-continuation was built as
the full other rendering instead of its remainder — the strict-fallback classifier could
then only confirm the context's own polarity (contractions showed s_k0 = 0.949 on
IDENTICAL prompts). Fixed in build_datasets (delta-remainder rule) together with curly
apostrophe normalization ('’' → "'"), datasets rebuilt, full rerun. Lesson: always keep
a k=0 identical-prompt cell as a classifier-leak control.

Known limitations to carry forward: ellipsis is low-n (37 docs); oxford_comma, whilst,
brit_t_past have thin k ≥ 4 cells (29–45 sites/polarity); scorable rates at lexicon
properties depend on the loose classifier's hit rate (reported per property in the
pre-screen summary).
