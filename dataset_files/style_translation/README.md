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
| quotations | angular quotes `« »` only |
| asides | rayas attached to the aside: `palabra —inciso— palabra` |
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
