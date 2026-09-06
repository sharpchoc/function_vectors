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
| scope | translation framing only is sampled; the English-only comparison reuses the 2026-09-01/02 prescreen records (same items, different sampling run) |

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
| `records.npz` | per property × framing arrays `k, pol, label, exact, judge` to regenerate views |
| `es_audit.csv` | per-doc Spanish audit (word ratios, leak counts, paragraph counts) |

Data: `dataset_files/style_properties/base_corpus_es.json` (878 docs). Scripts
(`src/sandbox/ext_styleprops/`): `translate_corpus.py`, `prescreen_adherence.py --framing
translate` (+ `--add_refs` backfill of reference continuations into the plain records),
`judge_translation.py` (+ `--exact_only` for the plain baseline), `plot_translation_framing.py`.
Artifacts: `artifacts/style_properties/prescreen_translate/<prop>.json` (records incl. sampled
tails, refs, judge verdicts); plain records in `artifacts/style_properties/prescreen/` gained
`ref_nat/ref_alt/ctx_tail/trans_exact`. Run: one RTX PRO 4500 Blackwell pod, 2026-09-06.

## Findings

_(filled after the run)_
