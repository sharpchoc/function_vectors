# results/style_translation/steering — can a mean-difference vector induce a style at k = 0?

Step 4 of the style-translation study (2026-09-07). Builds on `../` (step 3: accuracy vs k) and the
pairs/cue tokens in `dataset_files/style_translation/`.

## Definitions (user specification)

- **Steering vector.** For family F and layer L (L = 1..28 = output of transformer block L):
  `v_nat(L) = mean h_L(cue) over correct nat-context prompts − mean h_L(cue) over correct alt-context
  prompts`, where "correct" = the step-3 completion used the context's convention AND was judged a
  faithful, coherent translation (`style_ok and judge.ok`; all k = 0..4). `v_alt = −v_nat`. One vector
  per layer per style per family (`capture_cues.py`; `artifacts/style_translation/steering/vectors/`).
- **Injection.** Prompt = `Spanish:\n{es}\n\nEnglish:\n{English up to and including the FIRST cue
  token}` (k = 0; identical for both styles). `α·v` is added to the residual stream at layer L at the
  cue token only (last prompt position, prefill pass; generated tokens are not steered —
  `steer_hooks.CueSteer`, unit-tested: α = 0 is an identity, exactly the last position moves by v).
- **Grading.** Exactly as step 3: the completion (one seeded T=1 sample, ≤48 tokens, cut at the
  sentence end, `capped` flag) must use the TARGET style at the cue (registry classifier via
  `scoring.decide`) AND be judged a faithful, coherent translation (Gemini 2.5 Flash, one completion
  per call). accuracy = P(both). Unscorable (neither style) counts as inaccurate and is reported.
- **Sweep.** Screen: layers {2,4,6,8,10,12,16,20,24} × α ∈ {0, 0.5, 1, 2, 4} on 50 texts per
  (family, target), style-only, 16-token completions (`steer_screen.py`; `screen.csv`,
  `screen_heatmaps.png`). Confirm: the top-2 screen settings per (family, target) on all 200 k = 0
  texts with full grading; the final setting is the one with the higher full accuracy
  (`steer_confirm.py`, `judge_rollouts.py --dir …/confirm`, `steer_analyze.py`).
- **Controls.** `base`: α = 0 on the same 200 texts and seeds (in-run baseline; step-3 k = 0 and k = 4
  accuracies are also reported). `cf`: another family's vector (cyclic within the pod's shard; same
  style role) at the family's own top-1 (L, α) — if accuracy rises there too, the effect is not
  style-specific.

## Files

| file | contents |
|---|---|
| `steering_summary.png` | HEADLINE — per family: unsteered k = 0 · steered at best (L, α); dashed line = step-3 accuracy with 4 in-context examples and no steering (legend on the figure) |
| `steering_summary_controls.png` | DETAILED — the same plus the named other family's vector at the same (L, α) (control) |
| `screen_heatmaps.png` | layer × α target-style rate on the 50-text screen, per family and target |
| `best_config.csv` | final (L, α) per (family, target) with accuracy, CI, style-only, unscorable, judge OK, base, cf, step-3 k0/k4 |
| `steering_summary.csv` | every confirm arm (base, top1, top2, cf) per family/target |
| `screen.csv` | the full screen grid |
| `records.npz` | per-completion arrays of the confirm run |

## Findings
**Status (2026-09-08): 13 of 17 families fully graded.** The OpenRouter workspace hit its $100
lifetime budget mid-judging; `brit_t_past`, `oxford_comma`, `curly_quotes`, `quote_punct` (5,600
completions) and 7 `ellipsis` records have GPU rollouts but no faithfulness verdict yet and are
excluded from `best_config.csv` / `steering_summary.*` (their screens are in `screen.csv` /
`screen_heatmaps.png`). To finish once the budget is restored:
`judge_rollouts.py --dir artifacts/style_translation/steering/confirm --families brit_t_past oxford_comma curly_quotes quote_punct ellipsis --workers 48`
then `steer_analyze.py` (both idempotent). Style-only confirm rates for the four pending families
(n = 200, best of top-2): brit_t_past → -t past 0.33 (base 0.09), oxford_comma → serial comma 0.88
(base 0.27) / → none 0.96 (base 0.70), curly_quotes → straight 0.82 (base 0.42) / → curly 0.38
(base 0.18, 0.56 unscorable), quote_punct → inside 0.87 (base 0.31) / → outside 0.72 (base 0.44).

**Best (L, α) per family and target, full grading, n = 200** (`best_config.csv`; "k4" = step-3
accuracy with 4 in-context examples; "cf" = another family's vector at the same setting):

| family | → alt (rare) convention | L | α | base → steer | cf | k4 | → nat convention | L | α | base → steer | cf | k4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sentence_caps | lowercase sentence start | 24 | 4 | .01 → **.20** | .00 | .68 | Capitalised | 12 | 0.5 | .56 → .52 | .55 | .77 |
| all_caps | ALL CAPS | 10 | 1 | .00 → **.12** | .00 | .35 | standard case | 4 | 0.5 | .54 → .58 | .57 | .76 |
| double_space | two spaces after period | 24 | 4 | .00 → **.64** | .00 | .74 | one space | 8 | 0.5 | .65 → .70 | .69 | .77 |
| us_uk | British spelling | 20 | 4 | .09 → **.56** | .55 | .55 | American | 24 | 2 | .62 → .73 | .65 | .71 |
| ise_ize | -ise | 20 | 4 | .04 → **.56** | .27 | .55 | -ize | 10 | 1 | .67 → .70 | .70 | .76 |
| whilst | whilst/amongst | 24 | 4 | .01 → **.90** | .01 | .54 | while/among | 24 | 4 | .69 → .89 | .67 | .58 |
| contractions | expanded (do not) | 20 | 4 | .41 → **.80** | .23 | .61 | contracted (don't) | 20 | 4 | .23 → .65 | .07 | .69 |
| ampersand | & | 24 | 4 | .00 → **.85** | .00 | .62 | and | 8 | 1 | .81 → .84 | .83 | .83 |
| em_dash | spaced hyphen | 24 | 4 | .02 → **.57** | .00 | .64 | em dash | 24 | 4 | .59 → .76 | .10 | .72 |
| ellipsis | … character | 20 | 4 | .03 → **.74** | .02 | .63 | three periods | 12 | 2 | .45 → .55 | .48 | .71 |
| num_words | spelled cardinals | 24 | 4 | .06 → **.74** | .53 | .28 | digits | 16 | 4 | .66 → .72 | .64 | .79 |
| percent_sign | "percent" | 16 | 2 | .12 → **.86** | .33 | .81 | % | 24 | 0.5 | .76 → .86 | .81 | .88 |
| ordinal_words | spelled ordinals | 20 | 2 | .37 → **.74** | .46 | .65 | 1st/2nd | 16 | 4 | .29 → .47 | .38 | .78 |

1. **One vector at one token induces most conventions with no example.** Toward the rare (alt)
   convention, steering the single cue token lifts full accuracy from ≈0 to 0.56–0.90 in 9 of 13
   families. In 7 of them the steered k = 0 accuracy matches or beats 4 in-context examples
   (whilst .90 vs .54, ampersand .85 vs .62, contractions .80 vs .61, num_words .74 vs .28,
   ellipsis .74 vs .63, ordinal_words .74 vs .65, percent_sign .86 vs .81); us_uk and ise_ize tie
   their k = 4 value (.56 vs .55). num_words is the clearest case: in-context examples never taught
   spelled-out cardinals (≤ .37 in step 3, the Spanish shows digits) but the vector does (.74).
2. **Where steering falls short of ICL it is the translation that breaks, not the style.** The
   style-only rate is 0.79–1.00 in every family except all_caps (.56) and us_uk/ise_ize (.65).
   sentence_caps (style .94, accuracy .20) and all_caps (.56 / .12) lose because the α = 4 push at
   the first sentence produces fragments, restarts and repetition loops: judge OK falls from .56
   (base) to .20 and from .54 to .28. Elsewhere the judge OK rate under steering stays within a
   few points of the unsteered baseline (see `judge_ok` in `best_config.csv`), so the vector
   changes the convention without damaging the translation.
3. **Toward the house (nat) convention the gain is bounded by the baseline**: nat is already the
   k = 0 default in most families, so steering adds 0–.05 there. The exceptions are the families
   whose k = 0 prior is the alt pole — contractions (.23 → .65), ordinal_words (.29 → .47) — plus
   whilst (.69 → .89) and em_dash (.59 → .76), where the vector removes the unscorable/mixed
   completions of the baseline.
4. **Layer and scale.** The best alt setting is late (L20/24 in 11 of 13, L16 for percent_sign,
   L10 for all_caps) and at the top of the α grid (α = 4 in 9 of 13): the heatmaps rise
   monotonically with α at L16–24 for most alt targets. Since the screen ranks on style only, it
   picks the largest α even where α = 4 costs faithfulness; a smaller α could score higher on
   full accuracy for sentence_caps / all_caps (not tested — the confirm run took the screen's
   top-2 as specified). nat targets peak earlier and at small α (L4–12, α = 0.5–1) where the
   baseline is already high.
5. **Controls.** The counterfactual-family vector leaves accuracy at baseline in 7 of 13 alt
   targets and never reaches the own-vector value, except where two families share a convention
   axis — and those transfers are interpretable: the ise_ize (-ise) vector moves us_uk to British
   spelling as well as us_uk's own vector (.55 vs .56); brit_t_past's -t-past vector moves ise_ize
   to -ise (.04 → .27); percent_sign's "percent" vector spells out num_words' cardinals
   (.06 → .53) and ordinal_words' "spelled ordinal" vector spells out percent (.12 → .33); the
   ampersand "and" vector pushes contractions toward expanded forms (.41 → .23 for contracted,
   i.e. a shared spelled-out/expanded direction). Typographic families (double_space,
   sentence_caps, ampersand, em_dash, ellipsis, whilst) show no transfer (cf ≤ .02). Some cf
   vectors disrupt rather than steer: ellipsis' nat vector at L24 α2 makes em_dash emit `--`
   (unscorable .52, accuracy .59 → .10); all_caps' vector at L20 α4 turns sentence_caps
   completions into ALL CAPS.
   The us_uk transfer is explained by the vectors themselves: at L20, cos(v_us_uk, v_ise_ize) = 0.52
   while the median cosine between two families' vectors is 0.01 (90th percentile 0.14; the only other
   pair above 0.4 is curly_quotes/ellipsis at 0.45). The ise_ize vector is also 1.6× longer than
   us_uk's, so its projection onto the us_uk direction (8.6) is 85 % of us_uk's own vector (10.1):
   at the same α it delivers almost the same push along the British-spelling direction. Both
   vectors are extracted from prompts whose English context so far is British (colour/centre vs
   organise/realise) — the cue-token state encodes "this text is British English" rather than the
   specific word class, so either family's vector flips the other's spelling.

## Provenance / caveats
- GPT-J-6B fp16, 3 RunPod RTX PRO 4500 Blackwell pods (l0z8ws4hbilpdw / 6v7f5fgbdvd5q9 /
  z7o86hs5wxw55g, sharded 6/6/5 families, `logs/steer_job.sh <shard>`: capture → screen → confirm,
  ≈ 2 h each, terminated). Vectors from n_nat = 472–837 / n_alt = 146–692 correct step-3 prompts
  per family; ‖v‖/‖mean h‖ at L20 = 0.09–0.37; split-half cosine of v at L20 ≥ 0.87 in 15 families
  (us_uk 0.68, curly_quotes 0.79 — the two weakest steers among the lexical families).
- Hook unit test passed on every pod (α = 0 identity; exactly the last position offset by α·v).
  Screen α = 0 style rates on the 50 texts are within ±0.08 of the step-3 k = 0 rates (n = 200).
- One seeded T=1 sample per (text, arm); seeds `crc32(tag)`; cap 48 tokens (`capped` flag to the
  judge); judge Gemini 2.5 Flash via OpenRouter, one completion per call, same prompt as step 3.
  ellipsis: 7 of 1,400 confirm records unjudged (API failures) → excluded from its rates.
- Caveats: (i) screen on 50 texts → selection optimism (confirm rates on 200 are 0–.07 below the
  screen's); (ii) the screen ranks on style only, so the confirmed α is the largest that still
  produces the style, not the α that maximises accuracy; (iii) injection at the cue token only —
  a vector added at every generated position was not tested; (iv) ise_ize lexicon misses inflected
  forms (step-3 note) — affects the classifier equally in all arms; (v) k = 0 base rates here are
  re-sampled with new seeds, so they differ from step-3 k = 0 by sampling noise (≤ .05).
