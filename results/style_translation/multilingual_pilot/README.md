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
