# dataset_files/style_translation — Spanish source corpus (step 1 of the style-translation study)

Built 2026-09-07 after the reset of the style-properties line (DECISIONS 2026-09-07). This
folder holds ONLY Spanish source texts. The later study will prompt GPT-J with
`Spanish: <text>\n\nEnglish: <translation so far>` and observe which of a family's two styles
the model chooses while translating; that step is not built yet.

**Design rule (user):** for every one of the 17 style families, 200 unique, coherent Spanish
paragraphs, each with **at least 5 (aim 5–7) places where the English translation must choose
between the family's two styles**. The Spanish is the same whichever style is later tested, and
is always in ONE fixed, natural Spanish convention — it never carries the tested style itself
(no all-caps Spanish for the all_caps family, etc.).

## Result: 17 × 200 verified texts (`final/<family>.json`)

| family | candidates | pass | kept | Haiku k median | regex k median | words median |
|---|---|---|---|---|---|---|
| sentence_caps | 276 | 263 | 200 | 7 | 7 | 146 |
| all_caps | 276 | 265 | 200 | 7 | 7 | 148 |
| double_space | 276 | 254 | 200 | 6 | 7 | 144 |
| us_uk | 276 | 202 | 200 | 7 | 8 | 135 |
| ise_ize | 276 | 230 | 200 | 8 | 8 | 123 |
| brit_t_past | 627 | 261 | 200 | 6 | 6 | 128 |
| whilst | 276 | 253 | 200 | 7 | 8 | 140 |
| contractions | 336 | 241 | 200 | 8 | 6 | 125 |
| ampersand | 276 | 271 | 200 | 12 | 14 | 140 |
| oxford_comma | 276 | 247 | 200 | 7 | 8 | 145 |
| curly_quotes | 276 | 270 | 200 | 7 | 7 | 141 |
| quote_punct | 276 | 250 | 200 | 6 | 6 | 158 |
| em_dash | 276 | 271 | 200 | 7 | 12 | 138 |
| ellipsis | 276 | 251 | 200 | 6 | 6 | 151 |
| num_words | 276 | 219 | 200 | 6 | 7 | 130 |
| percent_sign | 276 | 247 | 200 | 7 | 7 | 148 |
| ordinal_words | 276 | 226 | 200 | 7 | 7 | 138 |

5,103 candidates generated in total; 3,400 kept. Per-family failure reasons: `spanish_audit.csv`.
(brit_t_past needed two extra rounds: forcing eight specific verbs into arbitrary topics produced
semantically forced sentences that the judge rejected until the instruction asked for situations
where the verbs are natural; contractions needed one small top-up.)

## Fixed Spanish conventions (RAE standard, all numbers as digits — user decision)

| feature | fixed form in every text |
|---|---|
| cardinals | digits: `3 días`, `12 personas` (never words) |
| ordinals | digits with the marker, ONE form: `1.º`, `2.ª`, `3.º` … `10.º`, plurals `1.os` / `1.as` (the RAE apocope `3.er` and word forms incl. plurals primeros/primeras are normalised; the noun «segundos» = seconds of time is left as a word) |
| percentages | `15 %` (digits, space, sign; never «por ciento») |
| quotations | angular quotes `« »` only; a period or comma goes OUTSIDE the closing quote (`«...».`), except an ellipsis that belongs to the quote (`«pero...»`) |
| asides | rayas attached to the aside: `palabra —inciso— palabra`; an aside that ends a sentence keeps its closing raya before the period (`—inciso—.`, RAE) |
| ellipsis | three dots `...` |
| spacing / case / markup | one space after periods, normal sentence capitalisation, no markdown/HTML |
| form | one paragraph, 100–230 words (median ~140), 6–8 sentences |

Enforced by the deterministic post-edit `gen_spanish.fixup` and audited by
`gen_spanish.violations`; a text with any remaining violation cannot pass. **Audit result: 0
violations in all 5,103 candidates.** Known artefact of "all numbers as digits": the count
"one" appears as `1` (`1 o 2 días`) — consistent, slightly unnatural; `un/una` as an article is
untouched.

## What counts as an opportunity (per family)

| family | the Spanish must contain ≥5 … |
|---|---|
| sentence_caps, all_caps, double_space | sentences (≥6 → ≥5 starts/boundaries) |
| us_uk | words whose English has a US/UK spelling variant (color, centro, vecino, favorito, teatro, litro, metro, gris, catálogo, defensa, comportamiento, sabor, honor, humor, labor, viajó, etiquetado, cancelado, joyería, aluminio, pijama, artefacto, puerto …) |
| ise_ize | words whose English ends -ize/-ise (organizar, reconocer, darse cuenta, disculparse, criticar, enfatizar, resumir, minimizar, priorizar, utilizar, analizar …) |
| brit_t_past | preterites of learn/spell/burn/dream/leap/lean/spill/spoil (aprendió, deletreó, quemó, soñó, saltó, se apoyó, derramó, se estropeó / se echó a perder) |
| whilst | mientras (conj.), entre (among), en medio de (amid) |
| contractions | clauses whose English has a contractible auxiliary/copula (no es, no puede, no hay, eso es, estoy …) |
| ampersand | two-noun pairs joined by y |
| oxford_comma | lists of ≥3 items «A, B y C» |
| curly_quotes | quoted spans «…» |
| quote_punct | quotations immediately followed by a period or comma |
| em_dash | asides set off with rayas |
| ellipsis | `...` followed by more text |
| num_words | cardinals 2–20 as plain counts |
| percent_sign | `N %` |
| ordinal_words | `1.º` … `10.º` |

## Files

| path | contents |
|---|---|
| `final/<family>.json` | the 200 accepted texts per family: `doc_id, family, topic, angle, text_es, words, sentences, regex_k, violations (=[]), verify{coherent, fluent, k_found, anchors, notes, judge}, pass` |
| `spanish/<family>.json` | ALL candidates incl. failures (same schema; `pass` null = judge failed to parse, 3 texts) |
| `spanish_audit.csv` | per-family tallies: generated, verified, pass, failure reasons, median words |

## Protocol (scripts in `src/sandbox/style_translation/`)

1. **Generation** — `gen_spanish.py`: Gemini 2.5 Flash via OpenRouter, temperature 1.0, 64
   parallel workers over all 17 families at once; 92 everyday topics × 3 angles (explanation /
   first-person anecdote / advice) = 276 candidates per family (more in top-up rounds); each
   prompt carries the conventions block and the family's concrete instruction (`families.py`),
   asks for ≥6 opportunities spread through the paragraph and none inside the first 8 words.
   Post-edit + audit + regex opportunity count (`families.py` `regex_k`) on every text.
2. **Verification** — `verify_spanish.py`: **Claude Haiku 4.5 via OpenRouter, one text per
   call**, temperature 0, strict JSON: `coherent`, `fluent`, and the list of `anchors` (exact
   Spanish words/constructions where the English must choose NAT vs ALT; per-family counting
   rule in `COUNTING_RULES`); `k_found = len(anchors)`. Length and conventions are NOT judged by
   the model.
3. **Selection** — `collect_verdicts.py --collect --finalize`:
   pass = coherent ∧ fluent ∧ k_found ≥ 5 ∧ regex_k ≥ 5 ∧ 0 audit violations ∧ 100 ≤ words ≤ 230;
   passing texts de-duplicated (identical normalised opening 60 chars, or word-5-gram
   Jaccard > 0.5); the first 200 by doc_id kept.

### Why per-text judge calls rather than one sub-agent per family
The first verification round used 17 Claude Haiku sub-agents (Agent tool), one per family, each
handed its 276 texts. Several did not read: they wrote heuristic scripts (empty `notes`; an
"English-word detector" that flagged 87 % of oxford_comma texts as non-fluent; 241 hallucinated
"numbers written as words" violations in ise_ize where a grep finds zero; length treated as
incoherence). Their verdicts were discarded (kept only under the job's scratch dir) and every
text was re-judged one per call. Calibration of the per-text judge against the regex counts on the
final corpus: |k_haiku − k_regex| ≤ 1 for ≥ 90 % of texts in 10 families; the large gaps are
regex artefacts (em_dash regex double-counts raya pairs, ampersand regex counts adjective
pairs, contractions regex is deliberately narrow), not judge errors — spot-read anchors match
the texts.

## Provenance / caveats
- Generation 2026-09-07, `google/gemini-2.5-flash`, ≈5,100 calls; verification
  `anthropic/claude-haiku-4.5`, ≈5,400 calls (incl. the 102-text calibration pass); 3 texts
  never yielded parseable JSON and remain unverified (families with ample margin).
- Haiku's `anchors` strings were recorded before the final `3.er → 3.º` normalisation, so a few
  anchor strings show the old form; every `text_es` is normalised.
- Topics are shared across families (same 92-topic pool), so different families' texts about the
  same topic are different texts but similar in subject; uniqueness was enforced within a family.
- Texts naturally contain features of OTHER families (digits, percentages, quotes, rayas) because
  the fixed conventions require them; that is intended and identical across the two polarities
  that will later be tested.

---

# Step 2 — English twins (`pairs/<family>.json`)

For every Spanish text, two English versions that differ **only** in the tested family's style:
`text_nat` (house style) and `text_alt` (the same text with that one family flipped). Built
2026-09-07; user decisions: minimal pairs by translate-once-then-restyle; Gemini translates,
Haiku (one text per call, OpenRouter) checks only the initial translation; pass = style at every
anchor ∧ ≥5 anchors ∧ coherent ∧ faithful.

## House style (= the nat pole of all 17 families; standard American English)

sentence-initial capitals · standard case · US spelling (color, neighbor, center) · -ize
(organize) · -ed pasts (learned) · while/among/amid · one space after periods · serial comma
("bread, cheese, and wine") · straight quotes with the period/comma INSIDE (`"like this,"`) ·
attached em dash (`word—aside—word`) · three-dot ellipsis · digits 2–20 · `15%` · digit
ordinals (1st, 3rd) · contracted forms (don't, it's) · the word "and".

Why: the Spanish already fixes every feature to one convention; if the English did not, the
non-tested features would vary from text to text and confound the later comparison. The house
style is **enforced deterministically**: after Gemini's translation, every family's detector
(`src/sandbox/ext_styleprops/properties.py`) finds its manifestations and the registry renderer
rewrites each to its nat form (`translate_english.normalise_nat`), then an audit confirms no
manifestation of any family remains in alt form (`audit_nat` → must be empty).

## Twin construction

`text_alt = render(text_nat, PROPS[family].find_opps(text_nat), "alt")`. Because both twins are
rendered from the same normalised text, they are identical outside the opportunity spans (checked
per pair). Each record stores `opps = [{k, nat, alt, nat_span, alt_span}]`, the k-th opportunity's
renderings and character spans in each twin — these are the later cue sites. `k_en` = number of
opportunities the detector finds in the English (≥5 required). No item resampling anywhere.

## Verification (Claude Haiku 4.5 via OpenRouter, one translation per call)

Given the Spanish and `text_nat`, the judge returns `coherent`, `fluent`, `faithful` (same
content and sentence order; nothing added, dropped, or merged; numbers, quotations, lists
preserved), `style_consistent` (the tested family appears only in its nat form), `anchors`,
`notes`. Counts and conventions are deterministic (`k_en`, `audit_nat`).

Pass = `k_en ≥ 5` ∧ `audit_nat == []` ∧ coherent ∧ fluent ∧ faithful ∧ style_consistent.
Failures were re-translated with feedback (the Spanish anchors that must survive, family-specific
hints such as "the last two list items must be single words" for oxford_comma, whose detector
requires `A, B, and C` with single-word B and C), up to 6 rounds.

## Files (this step)

| path | contents |
|---|---|
| `pairs/<family>.json` | 200 records per family: everything from `final/` (Spanish + Spanish verification) plus `text_nat`, `text_alt`, `opps`, `k_en`, `verify_en`, `rounds_en` |
| `english/<family>.json` | all translation attempts with raw output, normalised text, audit, verdict, pass |
| `english_audit.csv` | per-family tallies: translated, verified, pass, failure reasons, max rounds, median k_en |

## Example pair (us_uk, `us_uk__t000a`, abridged)

nat: "… which can accumulate a dark **color** from oil and grime. … to prevent any **rumor** of
looseness. If the bicycle is **aluminum**, …"
alt: "… which can accumulate a dark **colour** from oil and grime. … to prevent any **rumour** of
looseness. If the bicycle is **aluminium**, …"
Everything else in the two paragraphs is byte-identical.

## Results
All 17 families reached **200 pairs** (3,400 pairs; 3,664 translation attempts incl. retries and spares;
up to 6 rounds). Integrity sweep over the final pairs (`pairs_check.py`):

```
family         pairs spare k_en med k_en min nat audit bad alt other-fam bad outside-span diff fluent style_ok(judge) k_judge~k_en
sentence_caps    200     2        7        5             0                 0                 0    191             197         0.99
all_caps         200     2        7        5             0                 0                 0    186             197         0.02
double_space     200    10        6        5             0                 0                 0    191             198         0.91
us_uk            200     0        7        5             0                 0                 0    191             190         0.40
ise_ize          200     5        7        5             0                 0                 0    196             200         0.69
brit_t_past      200    10        5        5             0                 0                 0    190              92         0.96
whilst           200     2        7        5             0                 0                 0    194             184         0.73
contractions     200     1        9        5             0                 0                 0    169             165         0.90
ampersand        200     0       13        7             0                 0                 0    186             176         0.51
oxford_comma     200    26        7        5             0                 0                 0    190             162         0.46
curly_quotes     200     0       14       10             0               195                 0    194             200         0.00
quote_punct      200    26        6        5             0                 0                 0    181             143         0.95
em_dash          200     2       13        6             0                 0                 0    186             195         0.01
ellipsis         200     7        7        5             0                 0                 0    145             197         0.99
num_words        200     2        7        5             0                 0                 0    177             157         0.55
percent_sign     200     0        7        5             0                 0                 0    184             200         0.99
ordinal_words    200     1        7        5             0                 0                 0    192             194         0.99
total pairs 3400
```

Columns: `spare` = pairs whose Spanish text came from the spare pass pool (the original text
could not keep ≥5 anchors in English, e.g. multi-word list items for oxford_comma, quotations not
followed by a period/comma for quote_punct); `nat audit bad` = nat twins with any non-house-style
manifestation (0 everywhere); `alt other-fam bad` = alt twins where a *different* family's
detector finds a non-nat form — 0 everywhere except curly_quotes, where the quote_punct detector
sees `.”` instead of `."` (same punctuation-inside placement, curly glyph: benign, by
construction); `outside-span diff` = pairs that differ outside their opportunity spans (0);
`fluent` / `style_ok(judge)` = Haiku's non-gating columns; `k_judge~k_en` = share of pairs where
Haiku's anchor count is within 1 of the detector's (low for all_caps/em_dash/curly_quotes because
the judge counts sentences/pairs differently, not because of disagreement about the text).

Gating: the deterministic checks (`k_en ≥ 5`, `audit_nat == []`, twin identity) plus Haiku's
`coherent` and `faithful`. Haiku's `style_consistent` was NOT used as a gate: it contradicted the
detectors on texts that are demonstrably correct (e.g. "all seven quotations put the period
outside the quote" on a text with every period inside; self-contradicting notes for brit_t_past),
so, as for the Spanish, conventions are judged deterministically. `fluent` is reported but not
gating: most flags are consequences of the fixed conventions ("At 1st," from the Spanish
"1.º,", capitals after ellipses inherited from the Spanish) rather than translation errors.

Two defects the sweep DID catch and that were fixed in the registry (`properties.py`): the
contractions lexicon turned the modal "you have to" into "you've to" (now skipped), and
spelled-out numbers/ordinals opening a sentence were rendered lowercase in the alt twin ("three
days.", "first, …" → now "Three days.", "First, …").

## Caveats
- The translator sometimes paraphrases an anchor away (labor → task, viajero → rider), or merges
  sentences (double_space), or produces list items of more than one word (oxford_comma's detector
  then does not count the list); the retry loop with hints recovers most of these. Families that
  end below 200 pairs are listed in Results.
- `k_en` can exceed the Spanish k (English adds anchors, e.g. extra contractions or quotes) — fine;
  ≥5 is the floor.
- "1st, …" at sentence start (Spanish "Lo 1.º") is house style but reads stiffly; it is consistent.

---

# Step 3a — cue tokens (`pairs/<family>.json` → `cues`)

**Cue token (user definition 2026-09-07):** the token immediately preceding a style choice = the last token that is the same whichever style the model is about to produce. k = 0: the last token shared by the two twins (may be word-internal, e.g. `learn|ed` vs `learn|t`, `it|'s` vs `it| is`). k >= 1: judged on the context as it actually reads (nat or alt twin): the twin up to opportunity k versus the version that differs only in how opportunity k is rendered; cue = last shared token. Sentence families: the period closing the previous sentence; the first sentence of all_caps / sentence_caps has the newline after `English:` as its cue. curly_quotes: one decision per quotation, at the opening mark. Only "cue token" is a fixed term so far.

Computed with the GPT-J tokenizer on the full prompt `Spanish:\n{text_es}\n\nEnglish:\n{twin}` (`cue_tokens.py`). Each record has `cues.nat` and `cues.alt` (per CONTEXT twin), one entry per decision point: `k, opp_index, cue_idx (token index in the prompt), cue_tok, cue_tok_id, cue_char_end, opp_char_start, word_internal, next_nat, next_alt (first tokens after the cue under each rendering), in_header`. Checks: the k=0 cue token id is identical in both contexts for every pair; decision-point counts match. Longest prompt 840 GPT-J tokens. Typical cues: sentence families `.` / `\n`; us_uk / ise_ize / brit_t_past / num_words / ordinal_words the word before the item (` the`, ` to`, ` I`); percent_sign the number (` 50` → `%` vs ` percent`); quote_punct the last quoted word (`." It` vs `". It`); contractions the pronoun (` it` → `'s` vs ` is`); word-internal cues in ~14% of all_caps sentences (` I|t` vs ` IT`) and ~7-10% of the spelling families.
