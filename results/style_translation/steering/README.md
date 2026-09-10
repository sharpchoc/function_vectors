# results/style_translation/steering — can a mean-difference vector induce a style at k = 0?

Step 4 of the style-translation study (2026-09-07; **paired vectors 2026-09-09, the version reported
here**). Builds on `../` (step 3: accuracy vs k) and the pairs/cue tokens in `dataset_files/style_translation/`.

## Definitions (user specification)

- **Steering vector (paired).** For family F and layer L (L = 1..28 = output of transformer block L):
  `v_nat(L) = mean h_L(cue) over nat-context prompts − mean h_L(cue) over alt-context prompts`, where a
  (text, k) prompt pair enters BOTH means only if the step-3 completion was correct under BOTH contexts —
  used the context's convention AND was judged a faithful, coherent translation (`style_ok and judge.ok`;
  k = 0..4). The two pools therefore contain the same texts at the same k and differ only in the
  convention shown; 87 (brit_t_past) to 589 (percent_sign) pairs per family. `v_alt = −v_nat`. One vector
  per layer per style per family (`capture_cues.py`; `artifacts/style_translation/steering/vectors/`).
  The first version (2026-09-08) selected each pole's correct prompts independently, which left the pools
  unequal in size (nat 2–3× alt where nat is the default) and in k (15–20 % k = 0 on the nat side, ≈0 %
  on the alt side); its results are in git history (commit 17c52c5e; failure-mode notes 146e6a6a,
  aac4cc6f) and its artifacts in `artifacts/style_translation/steering_unpaired/`.
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
  `screen_layer_alpha.png`). Confirm: the top-2 screen settings per (family, target) on all 200 k = 0
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
| `screen_layer_alpha.png` | screen: target-style rate vs injection layer, one line per α (dashed = α = 0), per family and target, all 17 families |
| `best_config.csv` | final (L, α) per (family, target) with accuracy, CI, style-only, unscorable, judge OK, base, cf, step-3 k0/k4 |
| `steering_summary.csv` | every confirm arm (base, top1, top2, cf) per family/target |
| `screen.csv` | the full screen grid |
| `common_layer.png` / `common_layer.csv` | one common injection layer (L24) with a per-family α vs each family's own best (layer, α); all 4 α at L24 per family and target, full grading (`steer_common_layer.py`, `steer_common_analyze.py`) |
| `records.npz` | per-completion arrays of the confirm run |

## Findings

**Best (L, α) per family and target, full grading, n = 200** (`best_config.csv`; "k4" = step-3
accuracy with 4 in-context examples; "cf" = another family's vector at the same setting):

| family | → alt (rare) convention | L | α | base → steer | cf | k4 | → nat convention | L | α | base → steer | cf | k4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sentence_caps | lowercase sentence start | 16 | 4 | .01 → **.17** | .00 | .68 | Capitalised | 12 | 0.5 | .57 → .56 | .56 | .77 |
| all_caps | ALL CAPS | 8 | 2 | .00 → **.10** | .00 | .35 | standard case | 4 | 0.5 | .55 → .58 | .55 | .76 |
| double_space | two spaces after period | 24 | 4 | .00 → **.66** | .01 | .74 | one space | 8 | 0.5 | .66 → .71 | .68 | .77 |
| us_uk | British spelling | 24 | 2 | .09 → **.58** | .56 | .55 | American | 24 | 2 | .61 → .72 | .64 | .71 |
| ise_ize | -ise | 24 | 2 | .04 → **.67** | .41 | .55 | -ize | 16 | 1 | .68 → .74 | .71 | .76 |
| brit_t_past | -t past (learnt, spelt) | 24 | 4 | .07 → **.43** | .10 | .20 | -ed past | 24 | 4 | .55 → .62 | .60 | .51 |
| whilst | whilst/amongst | 24 | 4 | .01 → **.90** | .01 | .54 | while/among | 24 | 4 | .70 → .87 | .68 | .58 |
| contractions | expanded (do not) | 20 | 4 | .41 → **.81** | .23 | .61 | contracted (don't) | 20 | 4 | .23 → .71 | .09 | .69 |
| ampersand | & | 24 | 4 | .00 → **.83** | .00 | .62 | and | 16 | 4 | .81 → .82 | .83 | .83 |
| oxford_comma | no serial comma | 20 | 1 | .64 → **.86** | .73 | .70 | serial comma | 24 | 2 | .23 → .79 | .18 | .84 |
| curly_quotes | curly quotes | 24 | 2 | .09 → **.26** | .06 | .66 | straight quotes | 16 | 4 | .25 → .57 | .12 | .67 |
| quote_punct | punctuation outside quotes | 20 | 1 | .39 → **.62** | .48 | .83 | inside quotes | 24 | 2 | .28 → .74 | .29 | .88 |
| em_dash | spaced hyphen | 24 | 4 | .02 → **.58** | .00 | .64 | em dash | 20 | 4 | .60 → .77 | .09 | .72 |
| ellipsis | … character | 16 | 4 | .03 → **.73** | .02 | .63 | three periods | 24 | 1 | .45 → .54 | .51 | .71 |
| num_words | spelled cardinals | 24 | 4 | .06 → **.73** | .55 | .28 | digits | 24 | 4 | .65 → .76 | .20 | .79 |
| percent_sign | "percent" | 24 | 2 | .12 → **.87** | .34 | .81 | % | 24 | 0.5 | .75 → .85 | .81 | .88 |
| ordinal_words | spelled ordinals | 20 | 2 | .37 → **.74** | .45 | .65 | 1st/2nd | 16 | 4 | .29 → .47 | .40 | .78 |

1. **One vector at one token induces most conventions with no example.** Toward the rare (alt)
   convention, steering the single cue token lifts full accuracy to 0.58–0.90 in 13 of 17 families.
   In 11 of them the steered k = 0 accuracy matches or beats 4 in-context examples (whilst .90 vs .54,
   percent_sign .87 vs .81, oxford_comma .86 vs .70, ampersand .83 vs .62, contractions .81 vs .61,
   ordinal_words .74 vs .65, ellipsis .73 vs .63, num_words .73 vs .28, ise_ize .67 vs .55, us_uk .58
   vs .55, brit_t_past .43 vs .20). num_words and brit_t_past are conventions in-context examples never
   taught (≤ .37 and ≤ .26 in step 3); the vector does. Below the k = 4 reference: double_space (.66 vs
   .74), quote_punct (.62 vs .83), em_dash (.58 vs .64), curly_quotes (.26 vs .66), sentence_caps (.17
   vs .68), all_caps (.10 vs .35).
2. **What the pairing changed.** Against the unpaired vectors (commit 17c52c5e) the paired ones are far
   more reliable — split-half cosine at L20 ≥ .97 in every family (median 1.00; unpaired: us_uk .68,
   curly_quotes .79, brit_t_past .83) — and the accuracy gains land exactly on the families whose
   unpaired vectors were noisy: brit_t_past → -t past .19 → .43, ise_ize → -ise .56 → .67, brit_t_past
   → -ed .56 → .62, contractions → contracted .65 → .71; every other (family, target) moved by ≤ .03.
   The layer/α picture is unchanged.
3. **Where steering falls short of ICL it is the translation that breaks, not the style.** The
   style-only rate is 0.71–0.99 in every family except all_caps (.61), brit_t_past (.52) and
   curly_quotes (.38). sentence_caps (style .90, accuracy .17) and all_caps (.61 / .10) lose because
   the push at the first sentence produces fragments, restarts and repetition loops: judge OK falls
   from .57 (base) to .17 and from .55 to .17. Elsewhere the judge OK rate under steering stays within a
   few points of the unsteered baseline (`judge_ok` in `best_config.csv`), so the vector changes the
   convention without damaging the translation.
4. **Toward the house (nat) convention the gain is bounded by the baseline**: nat is already the k = 0
   default in most families, so steering adds 0–.06 there. The exceptions are the families whose k = 0
   prior is the alt pole — oxford_comma (.23 → .79), quote_punct (.28 → .74), contractions (.23 → .71),
   curly_quotes (.25 → .57), ordinal_words (.29 → .47) — plus whilst (.70 → .87) and em_dash (.60 → .77),
   where the vector removes the unscorable/mixed completions of the baseline.
5. **Layer and scale.** The best alt setting is late (L20/24 in 14 of 17; L16 for sentence_caps and
   ellipsis, L8 for all_caps) and high on the α grid (α = 4 in 9, α = 2 in 6, α = 1 in 2): the screen
   curves rise monotonically with α at L16–24 for most alt targets. Since the screen ranks on style
   only, it picks the largest α even where α = 4 costs faithfulness; a smaller α could score higher on
   full accuracy for sentence_caps / all_caps (not tested — the confirm run took the screen's top-2 as
   specified). nat targets peak earlier and at small α (L4–16, α = 0.5–1) where the baseline is high.
6. **Controls.** The counterfactual-family vector leaves accuracy near baseline in 9 of 17 alt targets
   and never reaches the own-vector value except where two families share a convention axis, and those
   transfers are the geometry of the vectors themselves (cosines at L20, median over all pairs .01,
   90th percentile .21): a **British-English cluster** — ise_ize/us_uk .82, brit_t_past/ise_ize .69,
   brit_t_past/us_uk .69, brit_t_past/whilst .62, ise_ize/whilst .52 — so the ise_ize vector moves us_uk
   to British spelling as well as its own vector (.56 vs .58) and the brit_t_past vector moves ise_ize
   to -ise (.04 → .41); a **spelled-out-numbers axis** — num_words/ordinal_words .62 — so percent_sign's
   "percent" vector spells out num_words' cardinals (.06 → .55) and ordinal_words' vector spells out
   percent (.12 → .34); and the ampersand "and" vector pushes contractions toward expanded forms
   (contracted .41 → .23). curly_quotes/ellipsis (.50) are both Unicode typographic characters.
   Typographic families show no transfer (cf ≤ .02). Some cf vectors disrupt rather than steer: the
   ellipsis nat vector at L20 α4 makes em_dash emit `--` (unscorable .52, accuracy .60 → .09); the
   all_caps vector at L16 α4 turns sentence_caps completions into ALL CAPS.

## One common layer for every family? (2026-09-10, user question)

Setting: inject at **layer 24 for all 17 families**, α still chosen per family and target, and ask whether
each stays within −0.03 of the accuracy at its own best (layer, α) from the table above. Layer 24 is
the only candidate the screen allows (style-only: 13/17 alt and 14/17 nat targets within .03 at L24;
7/17 and 9/17 at L20; nothing earlier). Every (family, target) was then sampled at L24 with all four
α on the 200 texts and fully graded (27 of 136 arms reused from the confirm run; 21,800 new rollouts,
judged with 0 failures). α per family is picked by full accuracy on the same 200 texts — the same
kind of selection optimism as the per-family best, which picked the better of two confirmed settings.

**Result: 30 of 34 targets are within 0.03 of their own best at layer 24** (alt 14/17, nat 16/17;
`common_layer.png`). Mean accuracy over the 34 targets is 0.67 at the common layer vs 0.66 at the
per-family best. The four misses are all inside their 95 % intervals (±.06–.07): all_caps → ALL CAPS
.05 vs .10 (L8), em_dash → em dash .72 vs .77 (L20), ellipsis → … .67 vs .73 (L16), ordinal_words →
spelled ordinals .70 vs .74 (L20). Nine targets already had L24 as their own best; six others improve
at L24 with a different α (sentence_caps → lowercase .24 vs .17 at α = 2 instead of 4 — the gentler
push keeps more translations intact; ise_ize → -ize .81 vs .74; brit_t_past → -ed .68 vs .62; quote_punct
→ outside .67 vs .62; oxford_comma → none .91 vs .86; contractions → expanded .85 vs .81). With the α
picked by the style-only screen instead (protocol parity with step 4), 28 of 34 stay within .03.

So a single late layer suffices for this family of conventions; the per-family layer choice in the
main table buys ≤ .06 anywhere, and per-family α is what matters. The α at L24 is 4 for the strong
symbol/digit conventions (double_space, whilst, ampersand, num_words, em_dash, brit_t_past → -t) and
0.5–2 where a strong push damages the translation (sentence_caps, all_caps, quote_punct, ellipsis).

## Failure modes of the weak families (confirm records, n = 200 per arm)

| family → target | style | judge OK | accuracy | what fails |
|---|---|---|---|---|
| sentence_caps → lowercase (L16 α4) | .90 | .17 (base .57) | .17 | translation quality, plus a judge artefact (below) |
| all_caps → ALL CAPS (L8 α2 / L20 α4) | .61 / .61 | .17 / .09 (base .55) | .10 / .03 | translation quality; style only half-forced |
| curly_quotes → curly (L24 α2 / α1) | .38 / .33 | .67 / .66 (base .64) | .26 / .22 | the decision itself: single curly quotes and dropped quotations |
| brit_t_past → -t past (L24 α4) | .52 | .77 | .43 | the form is rarely produced at all (unscorable .24) |
| ordinal_words → digits (L16 α4) | .60 | .69 | .47 | malformed digit ordinals (`1.st`, bare `1.`, `1.º`) |

- **sentence_caps and all_caps are real generation damage, measured without the judge.** Content-word
  F1 between completion and reference falls from .51 (unsteered) to .19 / .24 (lowercase, L12/L16 α4)
  and from .49 to .28 / .17 (ALL CAPS α2 / α4); the share of completions sharing < 20 % of content
  words with the reference rises .17 → .68 and .14 → .47 / .69. ALL CAPS output is shorter (18 → 11
  words) and 27 % of it never becomes ALL CAPS. Rejected steered completions are fragments, restarts,
  list openers, repetition loops, unrelated text or letter-salad. Both families' cue is the newline
  after `English:`, i.e. the vector is injected before any English exists, and a strong push there
  derails the whole first sentence rather than only its case.
- **Judge case-bias is small** (measured on the unpaired run: re-judging the rejected style-correct
  completions with the case normalised flipped 12 % for sentence_caps and 5 % for all_caps).
- **Judge artefact at the first-sentence cue.** For k = 0 sentence families the completion IS the first
  sentence, and Gemini sometimes rejects a faithful rendering as "repeats the beginning of the passage"
  (these overlap the reference as well as accepted completions do). It fires on 22 / 200 unsteered and
  30–34 / 200 lowercase-steered completions; counting every rejected completion with F1 ≥ .5 as possibly
  faithful adds ≤ .12 — sentence_caps' lowercase accuracy stays ≤ .3, far below .68 with 4 examples.
  Step-3 baselines carry the same artefact, so within-family comparisons are fair.
- **curly_quotes is not a coherence failure** (judge OK and F1 unchanged under steering). The vector
  moves the quotation mark away from straight `"` (.40 → .06) but only .38 land on the scored curly
  pair “ ”; .17 use single curly quotes ‘ ’ (unscored — the alt convention is defined as “ ”), .24 drop
  the quotation altogether (up from .15) and .10 copy « » from the Spanish. Tokenisation explains the
  ‘ ’ spill: GPT-J has no token for “ or ”; each curly mark is two byte tokens and the FIRST token is
  shared by “ ” ‘ ’, so the vector selects the curly branch at the cue while the distinguishing second
  byte is a within-branch choice it barely constrains. Straight `"` is a single token, hence the .81
  style rate toward it. Counting ‘ ’ as curly would raise the style rate to .55 — a definitional
  choice for the user, not changed here.
- **ordinal_words → digits stalls at .47 (style .60, unscorable .33) against .65–.78 with 1–4 examples
  because the vector carries "digit here" but not the English surface form "1st".** Steering makes
  88 % of completions digit-initial (unsteered 42 %), but 19 % come out as the hybrid `1.st`/`1.er`,
  4 % as a bare `1.` that stalls and 2 % as the Spanish `1.º` — 25 % unscorable digit forms (unsteered
  8.5 %). One in-context example fixes this (style .33 → .91 at k = 1) because it shows the form; the
  word direction has no such problem ("first" is one token): .94 style, .74 accuracy.
- **brit_t_past → -t past** is now the family where the vector beats examples by the widest relative
  margin (.43 vs .20) yet remains weak in absolute terms: .52 style-only, .24 unscorable (regular
  "-ed" avoided but no "-t" form either), judge OK .77 (translation intact).

## Provenance / caveats

- GPT-J-6B fp16, 3 RunPod RTX PRO 4500 Blackwell pods (avjdbbxxghcbaa / g6d0gx85kq6t5s /
  aoo1a76hsg5qxb, sharded 6/6/5 families, `logs/steer_job.sh <shard>`: capture → screen → confirm,
  ≈ 2 h each, terminated). Paired vectors from 87–589 (text, k) pairs per family; ‖v‖/‖mean h‖ at L20 =
  0.07–0.36; split-half cosine of v at L20 ≥ 0.97 in all 17 families (halves split by (text, k)).
- Hook unit test passed on every pod (α = 0 identity; exactly the last position offset by α·v).
  Screen α = 0 style rates on the 50 texts are within ±0.08 of the step-3 k = 0 rates (n = 200).
- One seeded T=1 sample per (text, arm); seeds `crc32(tag)`; cap 48 tokens (`capped` flag to the
  judge); judge Gemini 2.5 Flash via OpenRouter, one completion per call, same prompt as step 3. Judge
  robustness: a `notes` string that overruns `max_tokens` is parsed by regex for the `ok` field; a
  completion identical to its reference is marked OK by rule ONLY when Gemini never returns a verdict
  (5 of 23,800 records; the model echoes the prompt on bare `...` completions).
- Caveats: (i) screen on 50 texts → selection optimism (confirm rates on 200 are 0–.07 below the
  screen's); (ii) the screen ranks on style only, so the confirmed α is the largest that still
  produces the style, not the α that maximises accuracy; (iii) injection at the cue token only —
  a vector added at every generated position was not tested; (iv) ise_ize lexicon misses inflected
  forms (step-3 note) — affects the classifier equally in all arms; (v) k = 0 base rates here are
  re-sampled with new seeds, so they differ from step-3 k = 0 by sampling noise (≤ .05); (vi) the paired
  pools contain almost no k = 0 prompts (0–22 per family), so the vectors describe the convention as it
  appears after ≥ 1 prior decision in the context.

## Figure layout (2026-09-10)
Per-family panels are grouped into two blocks: **left, blue tint = lexically diverse conventions** (us_uk, ise_ize, brit_t_past, contractions, num_words, ordinal_words — the rule applies across many different words) and **right, warm tint = lexically identical conventions** (whilst, ampersand, percent_sign, em_dash, ellipsis, curly_quotes, quote_punct, double_space, oxford_comma, sentence_caps, all_caps — one fixed marker or formatting choice). Definition in DECISIONS 2026-09-10; lists in `src/sandbox/style_translation/family_groups.py`.
