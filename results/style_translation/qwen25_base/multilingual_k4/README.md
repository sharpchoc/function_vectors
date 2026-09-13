# results/style_translation/qwen25_base/multilingual_k4 — cheap k = 4 check of the non-English convention families

User decision 2026-09-13: before building full corpora, test whether Qwen2.5-7B (base) follows each NON-ENGLISH writing
convention after 4 in-context examples, under the translation prompt `English:\n{source}\n\n{Target}:\n{twin cut at cue k}`.
Only k = 4 (and the free k = 0 baseline) were run.

**Corpus (cheap).** The same 60 house-style English base texts for every family (whilst / oxford_comma / contractions, docs
t000–t019). Gemini 2.5 Flash rewrote each as an anchor-rich paragraph in the target language in the NATURAL style (≥ 7 anchors
requested, up to 3 tries), then back-translated it to English = the source (`ml_k4_build.py`). The natural twin is the text
normalised to the natural pole with the family's registry property; the alternative twin is rendered deterministically
(lexicon / rule ё→е, ß→ss / zhconv script conversion). Texts with ≥ 5 opportunities kept (44–60 per family; Japanese 13 — loanwords
are rare in these topics). No human or model verification of the target text beyond the rollout judge. Families and lexicons:
`ml_families.py`; cue tokens with Qwen's tokeniser; one seeded T = 1 sample ≤ 48 tokens; registry classifier at the cue;
Gemini 2.5 Flash faithful/coherent judge with source/target language filled in and the family's variant listed as ignorable.

## Result (accuracy = convention of the context ∧ faithful; n = 44–60 texts per bar, Japanese 13)

| family | target | natural pole k0 → k4 | alternative pole k0 → k4 | style-only alt k4 | judge OK k4 | unscorable k4 | verdict |
|---|---|---|---|---|---|---|---|
| zh_simp_trad | Chinese (Simplified vs Traditional) | .28 → **.92** | .00 → **.83** | .98 | .90 | .03 | learns strongly |
| pt_acordo_eu | Portuguese (post- vs pre-1990 EP) | .76 → .71 | .07 → **.57** | .62 | .94 | .10 | learns |
| pt_br_eu | Portuguese (BR vs EP) | .78 → .84 | .02 → .31 | .36 | .88 | .09 | partial |
| es_rae2010 | Spanish (2010 RAE accents) | .45 → .65 | .10 → .30 | .33 | .87 | .18 | partial |
| de_1996 | German (1996 reform) | .42 → .64 | .00 → .30 | .38 | .75 | .24 | partial, judge-limited |
| zh_tw_hk | Chinese (Taiwan vs Hong Kong) | .26 → .41 | .17 → .30 | .35 | .86 | .37 | weak, many unscorable |
| ja_long_vowel | Japanese (trailing ー) | .54 → .62 | .08 → .23 | .23 | .88 | .15 | weak (n = 13) |
| ru_yo | Russian (ё vs е) | .04 → .42 | .44 → .50 | .64 | .78 | .33 | model default = е; ё pole learned partially |
| de_swiss | German (ß vs ss) | .49 → .51 | .00 → .11 | .22 | .72 | .39 | weak |
| fr_1990 | French (1990 rectifications) | .55 → .51 | .02 → .08 | .18 | .77 | .32 | not learned |
| pt_acordo_br | Portuguese (post- vs pre-1990 BR) | .56 → .56 | .13 → .09 | .09 | .90 | .36 | not learned |
| uk_2019 | Ukrainian (2019 orthography) | .05 → .05 | .43 → .30 | .55 | .53 | .38 | not learned; judge-limited |

Files: `k4_check.png` (bars = k = 4 accuracy per pole, black ticks = style-only rate, hollow circles = k = 0 baseline),
`k4_check.csv`. Rollouts: `artifacts/style_translation/qwen25_base/rollouts/<family>.json` (k ∈ {0, 4} only); pairs:
`dataset_files/style_translation/pairs/<family>.json` (with `langs`); raw adaptations: `dataset_files/style_translation/multilingual/`.
Pod o9guv3c5v632z6 (~12 min, terminated); 2,260 completions, 2,260 verdicts, 0 failures.

Caveats: small n; cheap corpus (no verification of the generated target text); the k = 0 natural-pole numbers are depressed by
judge failures / unscorable starts rather than by convention choice; ja_long_vowel has 13 texts only.
