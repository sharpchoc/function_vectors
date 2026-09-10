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

† ellipsis alt → nat: style .98 (the model writes `...`) but a bare `...` completion against a reference
starting with `…` is judged "empty" (1 % OK) — the same verdict the judge gave the identical completions in
step 3 (0 of 32 OK), so this cell is a judge artefact, not a steering failure.

1. **Yes — steering the evidence tokens alone makes the model re-read the context as the other
   convention.** Mean accuracy toward the target over the 34 (family, direction) cells: unsteered .07,
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

## Provenance / caveats

- GPT-J-6B fp16; 3 RunPod RTX PRO 4500 Blackwell pods (hvoj51p99eb6g0 / xjk0sj1lk9m2fc / 80ppa2z6ty93vw,
  6/6/5 families, `logs/read_steer_job.sh <shard>`: screen → confirm, ≈ 2.5 h each, terminated).
  Screen 61,200 16-token rollouts; confirm 27,200 48-token rollouts; 27,200 Gemini verdicts, 0 failures.
  `PositionSteer` unit test passed on every pod (α = 0 identity; exactly the listed positions move by v).
  Screen α = 0 rates match the step-3 k = 3 style rates (both poles) within ±.08.
- Caveats: (i) the same vector is added at every evidence token of an instance, including sub-word
  pieces and the ~77 tokens of an all-caps sentence; (ii) selection optimism from the 50-text screen and
  from picking the better of two confirmed settings (same protocol as step 4); (iii) the genuine-context
  reference comes from step 3's own sampling (different seeds), so ±.05 gaps are noise; (iv) for all_caps
  the cue token is steered too in 91 % of prompts (evidence sentence ends at the cue); (v) the ellipsis
  alt → nat judge artefact above; (vi) k = 3 only — how the flip scales with the number of steered
  instances (1, 2, 3) was not run.
