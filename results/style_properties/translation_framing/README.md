# results/style_properties/translation_framing — SANDBOX variant

> **Exploratory variant, not a headline result.** Same status as `../steering/`: nothing here is
> repo standard unless the user promotes it. (User request 2026-09-06: "is this a good way to
> teach the model things in context?")

## Question

The English-only prescreen (`../behavioral_prescreen/accuracy_by_k.png`) measures how well
GPT-J-6B continues a binary style convention in free-form text as a function of k, the number of
prior manifestations in the prefix. Here the convention is taught through a **translation
framing**: the document is first given in Spanish, then a labelled English section translates it
back *in the target convention*, truncated after the cue token. The model must (a) keep the
convention and (b) keep translating faithfully. Accuracy is therefore the joint event
**style adopted AND translation correct**, compared with the English-only style-only curve.

## Prompt

```
Spanish:
<neutral Spanish translation of the base document>

English:
<English twin (nat or alt convention) up to and including the cue token>
```

The header is tokenised separately and prepended to the unchanged English token ids, so every
cue position / cue-token-id assertion of the plain prescreen holds at the shifted position. Item
counts per property are identical to the plain prescreen (no prompt exceeded 2000 tokens;
longest 1869). Readout: ONE T=1 seeded continuation of the same `max_new` tokens as the plain
prescreen (≤10), classified nat / alt / unscorable by the same property classifier.

## User decisions (2026-09-06)

| choice | decision |
|---|---|
| Spanish form | **neutral and fixed**: one translation per base doc (Gemini 2.5 Flash via OpenRouter, T=0), identical for both English polarities, standard Spanish typography — single spaces, straight quotes, digits, `%`, sentence caps, parenthetical dashes rendered as ` - `. Enforced by a deterministic fix-up; `es_audit.csv` shows zero remaining double spaces / curly quotes / em-dashes / all-caps sentences, length ratio es/en median 1.16 [1.02, 1.32], paragraph structure preserved in all 878 docs. The English prefix is the only source of the convention. |
| translation metric | **Gemini judge primary, normalised exact match secondary** (definitions below) |
| prompt layout | **language labels** `Spanish:` / `English:` |
| scope | translation framing only is sampled (13-property pool on 2026-09-06, the 4 excluded properties added 2026-09-07 at the user's request — see Extension below); the English-only comparison reuses the 2026-09-01/02 prescreen records (same items, different sampling run) |

## Metrics

- **style**: P(continuation classified as the context's own convention | style-scorable) — the
  plain prescreen's accuracy-vs-k definition.
- **judge** (translate framing only): Gemini 2.5 Flash (T=0) sees a ±350-char window of the
  Spanish source around the aligned position, the last 400 chars of the English prefix, the
  document's own next tokens as the gold REFERENCE, and the FRAGMENT; answers CORRECT /
  INCORRECT, explicitly ignoring case, spelling variant, spacing, quotes, dashes, digits-vs-words,
  ampersand, contractions, Oxford comma, quote punctuation, `%`. Empty fragments are INCORRECT
  without a call; judge failures are `null` and excluded (count reported in `summary.csv`).
  Not defined for the English-only run (free continuation has no translation target), hence the
  asymmetry in the figures.
- **exact** (both framings): lowercase, quotes/dashes straightened, whitespace collapsed; True if
  the fragment agrees with EITHER twin's reference on their common prefix (≥ 8 chars).
- **joint** = style adopted AND judge-correct, over style-scorable and judged items — the
  translation framing's accuracy in the user's sense.

## Files

| file | contents |
|---|---|
| `accuracy_by_k.png` | per property: English-only style (grey dashed) vs translation-framed style-only (open) and joint style+translation (filled), by exact k 0..5, both context polarities |
| `translation_by_k.png` | judge-correct and exact-match rates vs k (translate framing), English-only exact-match reference, style-unscorable fractions |
| `summary.csv` | per property × framing × context polarity: style/judge/exact/joint/unscorable by k with denominators, k≥4 aggregates, judge-failure counts |
| `gate.csv` | the Stage-A4 prescreen gate (separation, Spearman, disfavoured-pole adherence, 15 % scorable floor) re-evaluated for all 17 properties under both framings (extension 2026-09-07) |
| `records.npz` | per property × framing arrays `k, pol, label, exact, judge` to regenerate views |
| `es_audit.csv` | per-doc Spanish audit (word ratios, leak counts, paragraph counts) |

Data: `dataset_files/style_properties/base_corpus_es.json` (878 docs). Scripts
(`src/sandbox/ext_styleprops/`): `translate_corpus.py`, `prescreen_adherence.py --framing
translate` (+ `--add_refs` backfill of reference continuations into the plain records),
`judge_translation.py` (+ `--exact_only` for the plain baseline), `plot_translation_framing.py`.
Artifacts: `artifacts/style_properties/prescreen_translate/<prop>.json` (records incl. sampled
tails, refs, judge verdicts); plain records in `artifacts/style_properties/prescreen/` gained
`ref_nat/ref_alt/ctx_tail/trans_exact`. Run: one RTX PRO 4500 Blackwell pod, 2026-09-06.

## Findings (run 2026-09-06; 30,102 translate-framed items, 0 judge failures)

**Short answer to "is this a good way to teach the convention in context?": not better than
plain English examples.** At k ≥ 4 the translation framing continues the convention about as
often as the English-only prefix does (8 of 11 valid properties within ±0.05 on the
alt-convention docs); it is clearly better only for `ampersand` (+0.16) and `double_space`
(+0.07), clearly worse for `us_uk` (−0.16) and `contractions` (−0.09). What it adds is
*content anchoring*: the fragment reproduces the source, so it reaches the feature slot far
more often (style-unscorable fraction drops from 0.2–0.8 to 0.0–0.3), and the model keeps
translating correctly 80–92 % of the time while holding the convention.

Alt-convention docs (the disfavoured pole for most properties), k ≥ 4, from `summary.csv`:

| property | style, English-only | style, translation framing | judge-correct translation | **joint** (style AND correct) | style-unscorable, English-only → framing |
|---|---|---|---|---|---|
| all_caps | 0.942 | 0.918 | 0.684 | 0.609 | 0.01 → 0.00 |
| ampersand | 0.667 | 0.829 | 0.887 | 0.747 | 0.72 → 0.08 |
| contractions | 0.754 | 0.669 | 0.889 | 0.616 | 0.58 → 0.20 |
| curly_quotes | 0.975 | 0.956 | 0.823 | 0.786 | 0.51 → 0.04 |
| double_space | 0.864 | 0.932 | 0.836 | 0.743 | 0.22 → 0.01 |
| em_dash | 1.000 | 1.000 | 0.856 | 0.924 | 0.72 → 0.33 |
| oxford_comma | 0.850 | 0.857 | 0.865 | 0.821 | 0.37 → 0.06 |
| percent_sign | 0.966 | 0.907 | 0.918 | 0.804 | 0.28 → 0.01 |
| quote_punct | 0.901 | 0.931 | 0.822 | 0.744 | 0.29 → 0.02 |
| sentence_caps | 0.798 | 0.845 | 0.804 | 0.675 | 0.06 → 0.00 |
| us_uk | 0.917 | 0.755 | 0.882 | 0.723 | 0.80 → 0.20 |
| num_words † | 0.856 | 0.710 | (0.534) | (0.390) | 0.61 → 0.25 |
| ordinal_words † | 0.994 | 0.917 | (0.354) | (0.202) | 0.79 → 0.17 |

Nat-convention docs sit at 0.93–1.0 style in both framings (see `summary.csv`, `ctx=nat`).

- **Translation fidelity is flat in k** (`translation_by_k.png`): 0.80–0.92 judge-correct for
  the 11 valid properties, independent of how many manifestations precede the site. The one
  convention that costs translation accuracy is ALL CAPS: 0.68 judge-correct on alt docs vs 0.85
  on nat docs of the same property. Exact match is 0.15–0.36 (strict lower bound; the model
  paraphrases freely) vs ≈0.00 for English-only free continuation, confirming the model *is*
  translating rather than continuing.
- **Joint ≈ style × judge**: the two events are close to independent, so the framing's
  accuracy is the style curve scaled by ~0.8–0.9.
- **† `num_words` / `ordinal_words`: translation metric undefined.** Their items are resampled
  at dataset build (`properties.py` `NumWords.resample` / `OrdinalWords.resample`, user decision
  2026-09-02), so in every document the English twin's number differs from the Spanish source
  at exactly the scored site; the judge is asked for a number the source never contained
  (`judge` 0.35–0.53 there is an artefact, and the style curve is also contaminated because the
  Spanish always shows digits). Fixing this needs a user decision: either translate the twin
  text for these two properties (breaks "one Spanish per base doc") or build an unresampled
  variant of the two datasets for this framing.
- **The "neutral" Spanish is not neutral for typographic properties — it instantiates one
  pole.** Visible as the k = 0 gap between framings (`summary.csv` `style_k0`): `em_dash` nat
  docs 1.00 → 0.04 (the Spanish ` - ` rendering is this property's *alt* pole, so at k = 0 the
  model copies the dash form from the source and only the first English manifestation
  overrides it); `oxford_comma` alt docs 0.16 → 0.44 and `quote_punct` alt docs 0.06 → 0.46
  (Spanish has no serial comma and puts punctuation outside quotes = alt pole); `percent_sign`
  alt docs 0.29 → 0.02, `curly_quotes` alt docs 0.21 → 0.07, `num_words` alt 0.31 → 0.10
  (Spanish `%`, straight quotes, digits = nat pole). `us_uk`, `contractions`, `ampersand`,
  `double_space`, `sentence_caps`, `all_caps` show no k = 0 shift. So the k = 0 point of the
  framing is "Spanish typography prior", not the model's own prior; from k ≥ 1 the English
  prefix dominates for every property.
## Extension 2026-09-07: the four properties excluded by the English-only prescreen

User request: the prescreen pruning (`whilst` failed the gate; `ellipsis`, `brit_t_past`,
`ise_ize` pruned for one-sided classifier / < 15 % scorable) was decided under free
continuation, so the four were run through the translation framing too and the **same gate**
(separation at k ≥ 4 ≥ 0.3, Spearman(k, separation) > 0, adherence to the prior-disfavoured pole
at k ≥ 4 ≥ 0.4, plus the 15 % scorable floor) re-evaluated under both framings for all 17
properties → `gate.csv` (`plot_translation_framing.py` reproduces
`behavioral_prescreen/prescreen_summary.csv` exactly on the English-only records).

| property | scorable, English-only → framing | separation k≥4, framing | adherence disfavoured pole k≥4 (n) | judge-correct | gate under framing |
|---|---|---|---|---|---|
| ise_ize (pruned) | 0.107 → 0.690 | 0.805 | 0.805 (118) | 0.889 | **PASS** |
| whilst (failed) | 0.085 → 0.506 | 0.611 | 0.778 (18) | 0.794 | PASS, thin k≥4 cells |
| ellipsis (pruned) | 0.037 → 0.584 | 1.000 | 1.000 (8) | 0.561 * | PASS, very thin (148 sites) |
| brit_t_past (pruned) | 0.050 → 0.608 | 0.266 | 0.316 (19) | 0.850 | **FAIL** (separation) |

- **The scorable-rate objection disappears under the framing**: content anchoring makes the
  model reach the lexical slot (the -ise/-ize word, whilst/while, the past-tense verb, the
  ellipsis), so all four clear the 15 % floor by a wide margin. That was the reason three of
  them were pruned.
- **ise_ize now behaves like a pool property** (0.805 adherence to the British pole at k ≥ 4,
  n = 118) and is the clearest candidate for re-admission — *for this framing*. Whether it
  re-enters the English-only pool is a separate user decision: under free continuation its
  scorable rate is still 10.7 %.
- **whilst and ellipsis pass but on 18 and 8 disfavoured-pole items** at k ≥ 4 — their
  documents rarely contain ≥ 5 opportunities. Treat as provisional.
- **brit_t_past fails on separation, not on scorability**: with ≥ 4 British "-t" pasts in the
  prefix the model still writes the regular form 68 % of the time (alt docs, k ≥ 4: 0.316). The
  convention is not adopted in either framing.
- \* ellipsis judge score is deflated by an artefact, not by bad translation: many ellipsis
  sites end a sentence or the document, so the reference continuation is just "..." and the
  fragment runs on into text the source does not contain, which the judge marks INCORRECT.
- Figures now show all 17 panels; excluded properties are titled in red with their
  original status. `gate.csv` PASS applies the 15 % floor, so the English-only rows for the
  pruned properties read False there although their numeric gate values were True in
  `prescreen_summary.csv`.
- **Provenance (extension)**: pod `obv6xh4cp970td` (RTX PRO 4500 Blackwell, ~30 min incl.
  boot), 3,860 items, judge 3,860 calls, 0 failures.

- **Provenance**: sampling on pod `b0thtqx0mr7h5v` (RTX PRO 4500 Blackwell, ~2 h incl. one OOM
  restart at token budget 16000 → rerun at 8000/batch 16, seeds `crc32("<prop>|prescreen|translate|<batch>")`);
  judge = `google/gemini-2.5-flash` T=0 via OpenRouter, 30,102 calls, 0 failures after retry.
