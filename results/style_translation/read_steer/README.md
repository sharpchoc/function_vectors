# results/style_translation/read_steer — steering at the evidence tokens with the read-feature difference

Step 7 of the style-translation study (2026-09-10). Question (user): can mean-difference steering at the
**evidence tokens only** change how the model perceives the style of the text? Take a 3-shot prompt (three
rendered instances of a convention in context, model about to make the 4th choice), add the read-feature
difference at every evidence token of those three instances, and see whether the 4th choice flips.

## Definitions (user decisions 2026-09-10)

- **Read steering vector.** `u_nat(L) = r_nat(L) − r_alt(L)` — the step-6 evidence-token means
  (`../read_features/`, `artifacts/style_translation/read_features/<family>.npz`); `u_alt = −u_nat`.
  No normalisation. ‖u‖ is 0.2–0.5 of the residual norm and 10–35× the cue-token write vector at early
  layers; α = 1 is roughly "swap the evidence-token state for the other pole's".
- **Prompts.** The step-3 **k = 3 translation prompts** (Spanish + English twin cut at the 4th cue),
  200 texts × 2 context poles per family. Every k = 3 prompt is an exact prefix of the k = 4 prompt, so
  the evidence positions of instances 0..2 from the step-6 extraction are reused unchanged
  (`read_steer_screen.k3_items`; natural-tokenisation, style-bearing tokens only).
- **Injection.** `α·u` added to the residual stream at layer L at **every evidence token of the 3
  instances** (multi-token instances: all their tokens; all_caps ≈ 77 tokens per prompt), prefill pass
  only (`steer_hooks.PositionSteer`, unit-tested). The cue token is untouched unless it is itself an
  evidence token (adjacent opportunities: 5.5 % of prompts, 182/200 of them all_caps, where the 3rd
  sentence's final period is the 4th decision's cue).
- **Directions.** `nat2alt`: nat-context prompt + `α·u_alt`, target = alt at the 4th decision;
  `alt2nat`: alt-context prompt + `α·u_nat`, target = nat.
- **Sweep and grading.** Screen layers {2,4,6,8,10,12,16,20,24} × α ∈ {0.5,1,2,4} on 50 texts
  (style only, 16 tokens) → confirm the top-2 settings per (family, direction) on all 200 texts with
  full grading (one seeded T=1 sample, ≤ 48 tokens, sentence cut, registry classifier at the 4th
  decision AND Gemini faithful/coherent); final = higher full accuracy.
- **Baseline / reference / control.** Unsteered = the same 3-shot prompt scored toward the target (how
  often the model already deviates from its context). Reference (dashed) = step-3 accuracy at k = 3 when
  the context genuinely was the target pole — the ceiling for "perceiving the context as the other
  style". Control (user choice) = another family's read vector at the same positions and (L, α).

## Files

| file | contents |
|---|---|
| `read_steer_summary.png` | HEADLINE — per family and direction: unsteered · steered at the best (L, α); dashed = genuine-context reference (grouped: lexically diverse left, lexically identical right) |
| `read_steer_summary_controls.png` | the same plus the other-family control bar |
| `read_steer_layer_alpha.png` | screen: target rate vs injection layer, one line per α, per family and direction |
| `best_config.csv` | final (L, α) per (family, direction) with accuracy, CI, style-only, unscorable, judge OK, unsteered, control, genuine-context reference |
| `read_steer_summary.csv` | every confirm arm |
| `screen.csv`, `records.npz` | the screen grid; per-completion arrays of the confirm run |
| `read_steer_layer_alpha_full.png`, `read_steer_layer_mean.png`, `screen_full.csv`, `best_layer_full.csv` | full-layer sweep (every layer 0–27, 0 = embedding output): per-cell curves, mean over cells per group, best layer per cell vs the 9-layer sweep |

## Findings (n = 200 per bar; accuracy = target convention at the 4th decision ∧ faithful, coherent)

| family | nat ctx → alt: unsteered → steered (L, α) | genuine alt ctx | alt ctx → nat: unsteered → steered (L, α) | genuine nat ctx |
|---|---|---|---|---|
| us_uk | .05 → **.68** (L10, 2) | .58 | .14 → **.67** (L12, 2) | .65 |
| ise_ize | .01 → **.65** (L20, 2) | .60 | .16 → **.71** (L12, 4) | .68 |
| brit_t_past | .07 → .17 (L8, 2) | .17 | .38 → .49 (L10, 4) | .43 |
| contractions | .10 → **.75** (L6, 4) | .61 | .12 → **.62** (L2, 2) | .65 |
| num_words | .01 → **.55** (L2, 4) | .33 | .45 → **.74** (L16, 4) | .73 |
| ordinal_words | .01 → **.72** (L2, 4) | .64 | .20 → **.70** (L10, 2) | .72 |
| whilst | .05 → **.66** (L4, 2) | .53 | .12 → .57 (L10, 4) | .58 |
| ampersand | .00 → **.81** (L6, 4) | .66 | .15 → **.81** (L8, 1) | .77 |
| percent_sign | .01 → **.89** (L2, 2) | .84 | .07 → **.87** (L4, 2) | .86 |
| em_dash | .00 → **.68** (L4, 2) | .70 | .00 → **.81** (L6, 1) | .79 |
| ellipsis | .00 → **.82** (L20, 4) | .63 | .01 → .04† (L6, 1) | .60 |
| curly_quotes | .00 → .53 (L4, 2) | .63 | .01 → **.69** (L6, 2) | .65 |
| quote_punct | .03 → .74 (L10, 4) | .80 | .02 → **.85** (L6, 2) | .79 |
| double_space | .00 → .41 (L10, 4) | .59 | .04 → **.74** (L4, 2) | .69 |
| oxford_comma | .07 → **.88** (L8, 4) | .70 | .10 → .66 (L6, 2) | .77 |
| sentence_caps | .00 → .59 (L4, 1) | .67 | .04 → **.76** (L4, 2) | .70 |
| all_caps | .00 → .21 (L6, 2) | .36 | .00 → .34 (L4, 1) | .71 |
| uk_vocab ¹ | .09 → **.47** (L6, 4) | .28 | .30 → **.52** (L6, 4) | .51 |
| register ¹ | .14 → **.28** (L8, 4) | .21 | .52 → **.65** (L6, 2) | .55 |
| unit_abbr ¹ | .00 → **.56** (L2, 4) | .28 | .59 → **.82** (L6, 1) | .80 |
| diacritics ¹ | .24 → **.41** (L2, 4) | .28 | .30 → **.38** (L6, .5) | .38 |
| latin_abbr ¹ | .10 → **.21** (L6, 4) | .17 | .30 → **.42** (L6, 4) | .35 |
| hyphen_compound ¹ | .10 → **.12** (L8, 4) | .14 | .46 → **.57** (L10, 2) | .60 |
| flat_adverb ¹ | .07 → **.14** (L10, 4) | .09 | .52 → **.61** (L8, 4) | .58 |
| irreg_past ¹ | .02 → **.07** (L2, .5) | .04 | .29 → **.34** (L10, 4) | .31 |
| latin_plural ¹ | .26 → **.29** (L10, 2) | .28 | .36 → **.47** (L10, .5) | .42 |
| title_abbr ¹ | .14 → .21 (L2, 2) | .27 | .22 → **.47** (L4, 4) | .40 |

¹ the ten lexically diverse families added 2026-09-11 (paragraph 7 below); bold = within .03 of the genuine-context reference or above it.

† ellipsis alt → nat: style .98 (the model writes `...`) but a bare `...` completion against a reference
starting with `…` is judged "empty" (1 % OK) — the same verdict the judge gave the identical completions in
step 3 (0 of 32 OK), so this cell is a judge artefact, not a steering failure.

1. **Yes — steering the evidence tokens alone makes the model re-read the context as the other
   convention.** Mean accuracy toward the target over the 34 (family, direction) cells of the 17 original families: unsteered .07,
   steered .64, genuine-context reference .64. 29 of 34 cells come within .10 of the reference and 21
   match or exceed it: after the intervention the model behaves as if the three examples had been
   written in the other convention. The two clear misses are all_caps (style flips — 1.00 / .92 — but
   the judge rejects the capitalised or de-capitalised continuation as in every earlier step) and the
   ellipsis artefact above; brit_t_past → -t reaches its own ceiling (.17 = genuine .17).
2. **Reading happens early.** The best injection layer is ≤ 12 in 31 of 34 cells (L2–6 in 20); the
   screen curves drop back to the unsteered rate at L20–24 for most families (`read_steer_layer_alpha.png`).
   This is the mirror image of cue-token steering, where the best layers were 20–24. The evidence
   representation that later layers consult is set in the first third of the network.
3. **The translation survives.** Judge OK under the best steering averages .76 vs .78 unsteered; in 31 of
   34 cells the steered judge rate is within .1 of the baseline. Where cue-token steering of the first
   sentence broke the text (sentence_caps k = 0: accuracy .17–.24), evidence-token steering of a 3-shot
   context gives .59 / .76 with judge OK .78 / .76 — changing what the model *reads* is far gentler than
   forcing what it *writes*. all_caps is the exception (judge .25 / .34).
4. **α.** Best α = 2 in 17 cells, 4 in 12, 1 in 5; never 0.5. Since α = 1 ≈ a full state swap, the
   model needs roughly 2× the natural difference at the evidence tokens for the re-reading to dominate.
5. **Controls.** Another family's read vector leaves 22 of 34 cells within .10 of the unsteered rate.
   The transfers follow the shared axes seen on the write side: ise_ize's vector re-reads us_uk as
   British (.65 / .63 vs own .67 / .68) and vice versa (.55 / .47 vs .71 / .65); ordinal_words' digits-
   minus-words vector re-reads percent_sign's " percent" as "%" (.84 vs own .87) and percent_sign's
   vector moves num_words (.27); whilst's vector moves quote_punct → outside (.47); all_caps' case vector
   partly re-capitalises sentence_caps (.26). Typographic families otherwise show none (≤ .13).
6. **Lexically diverse vs identical.** Both groups flip; the diverse group is where evidence steering
   matches the genuine context most closely (us_uk, ise_ize, contractions, num_words, ordinal_words all
   ≥ reference − .03), and where the write-side cue vector was weakest — the read side carries the
   lexical conventions better than the cue side does.
7. **The ten lexically diverse families added 2026-09-11** (uk_vocab … title_abbr; rows marked ¹).
   Evidence-token steering reaches the genuine-context reference here too — 17 of 20 cells match or
   exceed it (mean steered .40 vs reference .35, unsteered .25; best layers L2–10, α mostly 4) — and
   for unit_abbr (spelled-out units .00 → .56 vs genuine .28) and uk_vocab (British words .09 → .47 vs
   .28) the steered model adopts the flipped convention about twice as often as a real 3-shot context
   makes it. But the references themselves are low (step 3: these families barely learn in context),
   so the flips toward alt stay at .07–.56 in absolute terms, and title_abbr → abbreviated (.21 vs .27)
   is the one miss. **Controls differ by direction:** toward alt, another family's read vector does
   nothing (mean control .13 vs steered .28, unsteered .12 — specific); toward nat, the control is
   almost as effective as the family's own vector (mean control .40 vs steered .53, unsteered .38;
   register .59 vs .65, hyphen_compound .54 vs .57, flat_adverb .52 vs .61, latin_plural .46 vs .47).
   For these families any large perturbation at the evidence tokens erases the alt evidence and the
   model falls back to its default — the house-style word — so the alt → nat "re-reading" is largely
   evidence destruction, whereas the nat → alt flip requires the family-specific direction. The judge
   OK rate stays within .1 of the unsteered rate in all 20 cells (translation intact).

## Full-layer sweep (layers 0–27, 2026-09-11)

The 9-layer screen was extended to every layer, including **layer 0 = the embedding output** (GPT-J adds
no positional vector to the residual stream, so the layer-0 read difference is exactly the mean embedding
difference of the evidence tokens; steering there is a token swap in embedding space). Same protocol:
50 texts, style only, α ∈ {0.5,1,2,4}, both directions (`read_steer_screen.py --layers … --tag screen_extra`,
`read_steer_layers_analyze.py`).

- **Denser layers change little (17 original families).** Mean best rate over the 34 cells: .89 with the 9 screened layers, .91
  with all 28; only 4 cells gain more than .05 (num_words → words .74 → .86 at L25, contractions → contracted
  .74 → .84 at L1, curly_quotes → curly .76 → .86 at L3, us_uk → American .86 → .94 at L3). 22 of 34 cells
  now have their best layer at 0–3.
- **Layer 0 alone gives .82 on average (17 original families)**, i.e. adding the mean *embedding* difference at the evidence tokens
  already flips the reading in most cells (26 of 34 within .10 of the best layer). So for most conventions the
  intervention is well described as a token swap in the model's input space, which the network then reads
  normally — a shared direction across all the family's word pairs (one vector serves 712 distinct us_uk pairs),
  but an input-level one.
- **Where the embedding difference is not enough**, the residual stream at layers 3–9 carries something the
  embeddings do not: em_dash → spaced hyphen .18 at L0 vs .92 at L5; ise_ize → -ise .58 vs .80 (L4);
  oxford_comma → serial comma .66 vs .86 (L3); double_space → two spaces .72 at L0 but .10 at L2 and .70 only
  from L10; num_words → words .64 at L0 vs .86 at L25; brit_t_past → -t .12 vs .28. These are the cells where
  activation geometry beyond token identity does the work; they are a minority.
- **The mean curves** (`read_steer_layer_mean.png`) are flat and high from layer 0 to ~10 for both groups at
  α ≥ 2, then fall toward the unsteered rate by layers 20–27; the lexically identical group falls earlier and
  further than the lexically diverse one. The read-side window is the first third of the network.
- **The ten lexically diverse families added 2026-09-11 break the "token swap" picture.** For the 17
  original families layer 0 accounts for .87 of the best-layer gain on average (L0 .82 vs best .91, unsteered
  .10; L0 is itself the best layer in 8 of 34 cells). For the ten new families it accounts for .14 (L0 .34 vs
  best .55, unsteered .30; L0 is never the best layer): uk_vocab → British .12 at L0 vs .68 at L7, unit_abbr
  → spelled-out .02 vs .62 (L3), title_abbr → abbreviated .32 vs .56 (L5) and → full .24 vs .68 (L4),
  register → formal .20 vs .50 (L9), latin_plural → classical .34 vs .42 (L10), hyphen_compound → closed
  .66 vs .80 (L3), latin_abbr → Latin .12 vs .30 (L10). The best layers are 1–10 in 18 of 20 cells. So the
  mean embedding difference of the evidence tokens is NOT what re-reads a lexically diverse context; the
  residual-stream direction that the first blocks build from it is. In the mean curves
  (`read_steer_layer_mean.png`) the lexically diverse block now rises from L0 to a plateau at L2–12 before
  decaying, while the fixed-marker block starts high at L0. (Screen metric, 50 texts; the judged 200-text
  confirm of the 9-layer settings for these families is in the table above.)
- Caveat: this is the screen metric (style only, 50 texts); the judged 200-text confirm was run for the
  9-layer settings only. cos(read difference at layer 1, embedding difference) ≈ .1 in every family — the
  first block already rotates the token representation away from the embedding, yet steering at either
  layer works, so the flip does not depend on the exact direction being the embedding difference.

## Provenance / caveats

- GPT-J-6B fp16; 3 RunPod RTX PRO 4500 Blackwell pods (hvoj51p99eb6g0 / xjk0sj1lk9m2fc / 80ppa2z6ty93vw,
  6/6/5 families, `logs/read_steer_job.sh <shard>`: screen → confirm, ≈ 2.5 h each, terminated).
  Screen 61,200 16-token rollouts; confirm 27,200 48-token rollouts; 27,200 Gemini verdicts, 0 failures.
  `PositionSteer` unit test passed on every pod (α = 0 identity; exactly the listed positions move by v).
  Screen α = 0 rates match the step-3 k = 3 style rates (both poles) within ±.08.
- Full-layer sweep: 3 pods 5dfax5jhq3bdvl / yqqiaonph7ryxa / 9sezvyj8ea1oap (~2 h each, terminated); 19 extra layers × 4 α × 50 texts × 34 cells = 129,200 16-token rollouts; hook unit test incl. layer 0 passed on each pod.
- Caveats: (i) the same vector is added at every evidence token of an instance, including sub-word
  pieces and the ~77 tokens of an all-caps sentence; (ii) selection optimism from the 50-text screen and
  from picking the better of two confirmed settings (same protocol as step 4); (iii) the genuine-context
  reference comes from step 3's own sampling (different seeds), so ±.05 gaps are noise; (iv) for all_caps
  the cue token is steered too in 91 % of prompts (evidence sentence ends at the cue); (v) the ellipsis
  alt → nat judge artefact above; (vi) k = 3 only — how the flip scales with the number of steered
  instances (1, 2, 3) was not run.
