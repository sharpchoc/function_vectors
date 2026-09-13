# results/style_translation/multilingual_pilot — can GPT-J write the non-English target languages? (2026-09-13)

Question (user): drop the English-target constraint so that German / Portuguese / French / Spanish (etc.) orthography
reforms can become convention families. Pilot: 20 house-style English source texts (first 3 sentences of the `ampersand`
family's texts, ~60 words), prompt `English:\n{text}\n\n{Lang}:\n` (the study's header layout, k = 0), one seeded T = 1
sample ≤ 160 tokens; second arm with a one-sentence demo pair in front. Gemini 2.5 Flash grades each output: in the
target language, faithful, fluent, coverage. Scripts `multilingual_pilot.py` (GPU, pod ln0z9mibrzgquu, terminated) and
`multilingual_pilot_judge.py`; outputs in `artifacts/style_translation/multilingual_pilot/{rollouts,judged}.json`.

| target | arm | in language | faithful | fluent | faithful ∧ fluent | coverage |
|---|---|---|---|---|---|---|
| German | header / demo | .75 / .95 | .10 / .00 | .10 / .10 | .00 / .00 | .21 / .39 |
| Portuguese | header / demo | .90 / .95 | .10 / .05 | .25 / .10 | .00 / .00 | .48 / .67 |
| French | header / demo | .75 / .95 | .00 / .00 | .15 / .05 | .00 / .00 | .37 / .64 |
| Spanish | header / demo | .95 / 1.00 | .00 / .05 | .25 / .05 | .00 / .00 | .38 / .79 |
| Dutch | header / demo | .90 / .90 | .05 / .05 | .10 / .00 | .00 / .00 | .20 / .51 |
| Romanian | header / demo | .80 / 1.00 | .05 / .00 | .15 / .00 | .05 / .00 | .23 / .35 |

**Verdict: GPT-J cannot serve as the writer for any of these targets.** Spanish is the best (stays in language, covers
the source) but is dense with lexical errors and Spanglish ("el chain", "sintesi", "cómputo"); Portuguese and French
often stay in English or produce non-words ("engarrafadores", "laitage"); German is largely incoherent ("Löffelwischens",
"Dezentriment"). Whole-passage grading is stricter than the study's single-sentence judge, but the error density is
decisive: the accuracy metric (convention ∧ faithful ∧ coherent) would be judge-limited near zero for every family.
Non-English-target families therefore require a different writer (Qwen2.5-7B), which means re-deriving the read/write
geometry on that model — the map question is per model.

## Qwen2.5-7B base (2026-09-13; `multilingual_pilot.py --model qwen25_base`, pods vxmqjpia3ziqr1 / 2kantpu35vn4mg, terminated)

Same 20 texts, prompts and judge. Calibration arm: the study's own direction, Spanish → English, on the same texts
(`--calib_only`, `rollouts_calib.json` / `judged_calib.json`).

| target (demo arm) | in language | faithful | fluent | faithful ∧ fluent | coverage |
|---|---|---|---|---|---|
| **Spanish → English (calibration)** | 1.00 | .85 | .95 | **.80** | .97 |
| English → Portuguese | 1.00 | .60 | .55 | .45 | .96 |
| English → Spanish | 1.00 | .50 | .40 | .35 | .95 |
| English → French | 1.00 | .60 | .25 | .15 | .96 |
| English → German | 1.00 | .25 | .15 | .10 | .85 |
| English → Dutch / Romanian | 1.00 / .95 | .00 | .00 | .00 | .79 / .58 |

Header-only arm within ±.10 of the demo arm except faithfulness (lower without the demo). Qwen stays in language and
covers the source everywhere (GPT-J did neither); the judge's complaints are gender agreement, word choice and, for
German, hallucinated clauses. Since the same judge gives Qwen's English .80, the gap is model competence, not judge
strictness. Per-sentence extrapolation (cube root of the 3-sentence passage rate, independence assumed): English ≈ .93
(matches the judge-OK rates of `../qwen25_base/`), Portuguese ≈ .77, Spanish ≈ .70, French ≈ .55–.65, German ≈ .45.

**Recommendation.** Qwen2.5-7B base can carry Portuguese (BR/PT, Acordo PT, Acordo BR — a 3-family cluster on one
axis) and Spanish 2010 RAE as multilingual convention families at an accuracy ceiling of ≈ .7–.8; French is marginal
(judge-limited like all_caps), German/Dutch/Romanian are out on the base model. English-target additions (Canadian/Oxford
spelling, place names, inclusive language, sub-families of us_uk / contractions / num_words) keep the .93 ceiling.
