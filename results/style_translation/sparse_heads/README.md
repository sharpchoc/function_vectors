# results/style_translation/sparse_heads — a sparse set of attention heads as a style function vector

Step 5 of the style-translation study (2026-09-08). Builds on step 4 (`../steering/`: one residual
mean-difference vector at the cue token) and asks the function-vector question: can a **sparse,
shared set of attention heads**, whose mean outputs are summed and injected once at the cue token,
induce the rare convention with no in-context example?

## Definitions (user decisions 2026-09-08)

- **Head vectors.** For family F and head (l, h): the mean out_proj-projected output of that head at
  the cue token over step-3 prompts whose completion used the ALT convention and was judged faithful
  (`style_ok and judge.ok`, all k pooled), restricted to the 120 fit texts. `C_alt[F]` ∈ ℝ^{448×4096}
  (`capture_heads.py`; linearity gate Σ_h W_O^h a_h = attention output on every family).
- **Sparse optimisation** (Hu et al. 2505.05145 §3.1, reusing the repo's sandbox implementation):
  one coefficient vector c ∈ [0,1]^448 **shared across the 17 families**; the family's steering vector
  is v_F = Σ_h c_h C_alt[F, h], added to the residual stream at the output of block L at the cue token
  (last prompt position) of the k = 0 prompt (Spanish + English up to the first cue; identical for both
  styles). Loss per prompt = mean over label tokens of −log p(token), teacher-forced, + λ‖c‖₁; AdamW
  lr 0.01, batch 64, c clamped to [0,1] after each step, early stopping on the validation slice.
- **Golden completion / label.** The `text_alt` twin's tokens from cue + 1 through the end of the alt
  style span (e.g. " colour", " first", "ise", " &", "…"; for all_caps the whole capitalised first
  sentence, for sentence_caps its lowercase first token). Per-token MEAN so the long all_caps labels
  do not dominate the pooled objective.
- **Split** (doc_id order, per family): 0..99 train, 100..119 validation (early stop, λ and L
  selection), 120..199 test (80 texts, all graded numbers below).
- **Sweep.** L ∈ {6, 9, 12, 16, 20} × λ ∈ {0.001, 0.003, 0.01, 0.03, 0.1}. λ* = largest λ within 0.02
  nats of the best validation NLL at that layer; L* = lowest validation NLL at λ*. Selected heads =
  c > 0.2 (c > 0.8 also reported).
- **Grading** exactly as steps 3–4: one seeded T = 1 sample, ≤ 48 tokens, cut at the sentence end,
  registry classifier for the convention, Gemini 2.5 Flash for faithfulness/coherence;
  accuracy = both. Arms: unsteered · sparse (learned c) · sparse unweighted (indicator sum, the
  canonical FV construction) · other family's sparse vector (control) · step-4 residual vector at its
  best (L, α) on the same 80 texts · 4 in-context examples (step 3, same 80 texts).

## Files

| file | contents |
|---|---|
| `sparse_summary.png` | per family: unsteered · sparse heads · unweighted · other family · step-4 residual vector; dashed = 4 in-context examples |
| `heads_vs_accuracy.png` | validation NLL vs number of heads per layer (left); graded accuracy vs number of heads at L* (right) |
| `selected_heads_map.png` | learned coefficients over (layer, head); the 37 ICL function-vector heads outlined |
| `sparse_summary.csv` | every arm per family (accuracy, CI, style-only, unscorable, judge OK) |
| `layer_lambda.csv` | the 25 fitted cells: validation/test NLL, c = 0 baseline, heads at c > 0.2 / 0.8 |
| `records.npz` | per-completion arrays and the coefficient vector |

## Findings
**Selected setting:** layer 20, λ = 0.001 → **114 heads** with c > 0.2 (94 with c > 0.8), spread over
blocks 3–27 with clusters at blocks 4–12 and 15–20; 20 of the 114 are among the 37 ICL function-vector
heads. Validation NLL per label token 2.62 (unsteered) → 0.68; test 2.61 → 0.65. Every layer fits
(L6 0.91, L9 0.95, L12 0.79, L16 0.72, L20 0.68); the λ = 0.1 cells are degenerate (early stopping
kept the epoch-0 coefficients, all ≈ 0.5) and are excluded from the curves.

**Accuracy on the 80 held-out texts per family, alt direction** (`sparse_summary.csv`; k4 = 4 in-context
examples, resid = step-4 residual mean-difference vector — PAIRED version, 2026-09-10 — at its per-family best (L, α), both on the same 80 texts):

| family | unsteered | sparse 114 heads | unweighted | other family | resid (step 4) | k4 |
|---|---|---|---|---|---|---|
| sentence_caps | .00 | .19 | .21 | .00 | .17 | .65 |
| all_caps | .00 | .06 | .05 | .00 | .03 | .44 |
| double_space | .00 | .45 | .38 | .00 | .66 | .75 |
| us_uk | .11 | .64 | .68 | .74 | .61 | .56 |
| ise_ize | .06 | **.79** | .60 | .24 | .72 | .57 |
| brit_t_past | .09 | .25 | .19 | .07 | .45 | .24 |
| whilst | .00 | .75 | .71 | .00 | .90 | .54 |
| contractions | .36 | .62 | .53 | .29 | .80 | .72 |
| ampersand | .01 | .55 | .51 | .00 | .81 | .61 |
| oxford_comma | .61 | .75 | .85 | .71 | .85 | .72 |
| curly_quotes | .14 | **.36** | .39 | .05 | .28 | .68 |
| quote_punct | .40 | .57 | .56 | .50 | .61 | .81 |
| em_dash | .04 | .45 | .45 | .00 | .53 | .65 |
| ellipsis | .01 | .55 | .68 | .10 | .79 | .68 |
| num_words | .07 | .25 | .24 | .09 | .70 | .31 |
| percent_sign | .07 | .75 | .81 | .11 | .88 | .79 |
| ordinal_words | .38 | .51 | .45 | .29 | .74 | .64 |
| **mean** | .14 | **.50** | .49 | .19 | .62 | .61 |

1. **One shared set of 114 heads, summed and injected once at the cue token, induces the rare
   convention in most families with no example**: mean accuracy .14 → .50 (unweighted indicator sum
   .49, so the learned weights matter little once the set is chosen). It matches or beats 4 in-context
   examples in 5 families (ise_ize .79 vs .57, whilst .75 vs .54, us_uk .64 vs .56, oxford_comma .75
   vs .72, brit_t_past .25 vs .24) and is within .05 of the per-family-tuned residual vector in 4
   (sentence_caps, all_caps, us_uk, quote_punct).
2. **Versus the residual mean-difference vector (step 4, paired vectors): lower style forcing, better
   translations, lower accuracy.** Mean accuracy .50 vs .62 (residual) on the same 80 texts. The sparse
   heads use the target convention less often (style-only .64 vs .82 averaged over families) but keep
   the judge rate at the unsteered level (.76 vs .72; unsteered .78). They win on ise_ize (.79 vs .72)
   and curly_quotes (.36 vs .28; unscorable .25 vs .56 — the heads produce the two-token “ more often
   than the residual vector), tie within .05 on sentence_caps / all_caps / us_uk / quote_punct, and lose
   by > .10 on double_space (.45 vs .66), brit_t_past (.25 vs .45), whilst (.75 vs .90), contractions
   (.62 vs .80), ampersand (.55 vs .81), ellipsis (.55 vs .79), num_words (.25 vs .70), percent_sign
   (.75 vs .88) and ordinal_words (.51 vs .74) — the symbol / digit conventions and the two families
   whose paired residual vector improved most. Caveat: the residual comparison is per-family tuned
   (best of 45 (L, α) cells per family) whereas the sparse set is one shared vector at one layer with
   α = 1 and was chosen by validation NLL, not by accuracy. (Against the earlier unpaired residual
   vectors the residual mean was .59 and brit_t_past was a sparse-heads win, .25 vs .19; see git
   history before 2026-09-10.)
3. **Sparsity trades accuracy smoothly.** Graded accuracy at layer 20 (mean over families,
   `heads_vs_accuracy.png`): 114 heads .47, 90 heads .46, 39 heads .40, 16 heads .32 (unsteered .14;
   the 114-head point re-sampled with other seeds gives .50 in the main run, so ±.03 is seed noise).
   The judge rate is flat (.75–.79) across sparsity: fewer heads means less style, not worse text.
   Validation NLL prefers late injection at every sparsity level (L20 < L16 < L12 < L6 ≈ L9).
4. **Controls.** Another family's head vector (same 114 heads, that family's means) leaves 11 of 17
   families at or below baseline; it transfers where step 4 also transferred: us_uk ← ise_ize (.74,
   above us_uk's own .64 — the shared British-spelling direction), oxford_comma ← curly_quotes (.71 vs
   base .61), quote_punct ← whilst (.50 vs .40), ise_ize ← brit_t_past (.24 vs .06).
5. **The style heads overlap the ICL function-vector heads only partly** (20 of 114 vs 37): the
   optimiser recruits many late-layer heads (blocks 15–27) that the task-imitation FV set does not use.

## Provenance / caveats
- GPT-J-6B; 3 RunPod RTX PRO 4500 Blackwell pods (b1r93c3e80lq8l / 1dn87vinj3zggo / x67vz2lxhwyktb,
  ≈ 5.5 h each, terminated). Pipeline `logs/sparse_job.sh` (capture → train shards by layer → eval);
  points and label spans verified on CPU (`sparse_heads_train.py points`: every prompt equals the
  stored k = 0 prompt, every label starts with the alt rendering; label lengths 1–3 tokens except
  all_caps, median 38).
- Per-head capture from the fit texts only (n_alt 76 brit_t_past … 421 oxford_comma); linearity gate
  rel dev ≤ 6e-4 and indicator check ≤ 5e-5 in every family. Training bf16, micro-batch 4 (L6/9), 6
  (L12), 8 (L16), 10 (L20) — larger micro-batches OOM on 32 GB with ~500-token all_caps sequences;
  batch 64, AdamW lr .01, patience 3, cap 10 epochs (15 for the first λ = 0.001 cell of each layer;
  those ran to the cap with < 0.003 nats/epoch change). Hook unit test passed at L20 (tolerance made
  fp16-aware: hidden entries at late layers reach the hundreds, where the fp16 spacing is 0.25).
- Evaluation: one seeded T = 1 sample per (text, arm), ≤ 48 tokens, sentence cut, `scoring.decide`,
  Gemini 2.5 Flash judge (0 failed calls, 13,600 verdicts incl. the λ curve). `base` arm agrees with
  step-4 `base` on the same 80 texts within CI in every family. The λ-curve arm at λ = 0.001 is the
  same vector as `sparse_w` with different sampling seeds (.47 vs .50 mean).
- Caveats: (i) one shared head set for all 17 families — per-family fits were not run; (ii) α fixed
  at 1 (the coefficients set the scale) — no α sweep; (iii) selection by validation NLL with a 0.02-nat
  tolerance picks the densest fit; the accuracy curve shows 90 heads within .02 of it; (iv) the
  residual-vector reference inherits step 4's selection optimism (screen on 50 texts incl. these);
  (v) the residual reference is the PAIRED step-4 vector (2026-09-10; `results/style_translation/steering/`),
  re-analysed with `sparse_heads_analyze.py` on 2026-09-10 — the sparse head means themselves are
  alt-correct-only means on the 120 fit texts (no pairing applies: they are not a nat−alt difference).

## Figure layout (2026-09-10)
Per-family panels are grouped into two blocks: **left, blue tint = lexically diverse conventions** (us_uk, ise_ize, brit_t_past, contractions, num_words, ordinal_words — the rule applies across many different words) and **right, warm tint = lexically identical conventions** (whilst, ampersand, percent_sign, em_dash, ellipsis, curly_quotes, quote_punct, double_space, oxford_comma, sentence_caps, all_caps — one fixed marker or formatting choice). Definition in DECISIONS 2026-09-10; lists in `src/sandbox/style_translation/family_groups.py`.
