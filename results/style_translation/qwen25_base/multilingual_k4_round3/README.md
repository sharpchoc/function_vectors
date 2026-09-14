# results/style_translation/qwen25_base/multilingual_k4_round3 — cheap k = 4 check of the previously pruned ideas

User decision 2026-09-14: test the conventions that had been pruned on the assumption that the model could not learn them.
Same protocol as `../multilingual_k4/` (Qwen2.5-7B base, k ∈ {0, 4}, one seeded sample, registry classifier ∧ Gemini judge; corpus =
English base texts adapted with anchors, back-translated to the source language: Spanish for English-target families, English otherwise).
Texts per family 25–77 (Belarusian 25: the translator rarely produced enough anchors). `ml_families.py` round-3 block; `k4_check.{png,csv}`.

| family | target | natural k0 → k4 | alternative k0 → k4 | judge OK k4 | verdict (cutoff: both poles ≥ .30) |
|---|---|---|---|---|---|
| en_canadian | English (Spanish source) | .86 → .79 | .01 → .42 | .99 | **passes** |
| en_prep_idioms | English | .76 → .63 | .10 → .37 | .97 | **passes** |
| de_inclusive (gender star) | German | .73 → .75 | .00 → .39 (style-only .54) | .75 | **passes**, judge-limited |
| en_inclusive | English | .25 → .32 | .43 → .40 | 1.00 | passes the numbers but no learning (model default = inclusive term); flagged |
| en_collective | English | .74 → .92 | .00 → .24 | .99 | below cutoff |
| en_exonyms | English | .72 → .82 | .27 → .25 | .98 | below cutoff; no learning |
| fr_inclusive | French | .85 → .82 | .00 → .20 | .82 | below cutoff |
| en_archaic | English | .78 → .76 | .00 → .00 | .99 | not learned |
| ja_hist_kana | Japanese | .58 → .42 | .00 → .04 | .65 | not learned |
| ko_north | Korean | .38 → .44 | .02 → .00 | .74 | not learned |
| be_tarask | Belarusian | .04 → .16 | .00 → .00 | .28 | model cannot write Belarusian (66 % unscorable) |
| es_nfd | Spanish | — | — | — | untestable: Qwen's tokenizer normalises NFD → NFC (identical token ids), no rollouts |

Three assumptions were wrong (Canadian/Oxford spelling, US/UK preposition idioms, German gender-star forms); the rare-pole
assumption held for archaic English, historical kana, North Korean spelling and Taraškievica; exonyms and collective nouns
show little or no learning; NFC/NFD is invisible to the model at the tokenizer. Pod 1g0gi7k3lqcoff (~25 min), terminated.
