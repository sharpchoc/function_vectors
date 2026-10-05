# Subspace-patch steering at the cue token, towards the ALTERNATIVE convention (53 code families, Qwen2.5-7B base)

2026-10-05. `write_subspace_patch.py` (+ `capture_cue_layers.py`, job `logs/code_styles_subspace_all_job.sh`), analysis `write_subspace_patch_analyze.py`.
Records: `artifacts/style_translation/qwen25_code/steering/subspace_patch_k4/{s1,sp1..sp4}/<family>.json`; subspaces `cue_pairs_layers/<family>_k4_subspace.npz`.

## Definitions
- Pair differences d_i = h_nat,i − h_alt,i at the cue token (last prompt position) of the k = 4 twin prompts of the 150 TRAINING documents, one matrix per
  layer 20..28 (layer 28 = un-normed output of the last block). Twins whose two prompts end on different cue tokens are skipped (py_type_hints 16,
  py_fstring 1, py_with_open 1).
- V = top-2 right singular vectors of the stacked differences, UNCENTERED: PC1 is the mean-difference direction (cos ≥ .54, mean .97–.98).
- Target coordinates = mean over the training documents' k = 4 ALTERNATIVE prompts of h V^T.
- Steering of the first 40 held-out documents' 0-shot prompts, ONE layer at a time, prefill pass only, cue token only:
  `patch2` replaces the cue token's coordinates in span(PC1, PC2) by the target; `patch1` the same with PC1 only; `add2` adds 2 × the mean
  difference (towards the alternative) as reference; `base` = unsteered. Success = completion uses the alternative convention AND Gemini judge OK.

## Results (`pooled_by_layer.csv`, `per_family_best.csv`, `full.csv`, `summary.json`, `pooled_by_layer.png`)
Pooled over 53 families; unsteered .14, k = 4 in context .74.

| layer | patch2 | patch1 | add2 |
|---|---|---|---|
| 20 | .17 | .15 | .24 |
| 22 | .20 | .19 | .30 |
| 24 | .31 | .31 | .51 |
| 25 | .38 | .36 | .51 |
| 26 | .39 | .39 | .51 |
| 27 | .39 | .39 | .47 |
| 28 | .43 | .42 | .57 |

- 2-D vs 1-D patch: +.006 pooled at layer 28 (own-best layers +.009); 11 families better by ≥ .10, 6 worse. PC2 holds 8–11% of the difference
  energy and, being ⟂ to the mean difference, its target coordinate is nearly pole-independent (|Δc2| / |Δc1| median .05), so patching it mostly
  moves the cue token towards "k = 4-ness", not towards the convention.
- Patch vs add 2 × mean difference: the patch is −.14 pooled at every layer ≥ 23 (30 families worse by ≥ .10, 4 better); first-token margin 2.9
  vs 11.5. The patch sets the natural k = 4 magnitude along PC1; the additive arm overshoots it (≈ 2× the pole gap) and that is what pays.
- All three improve monotonically with layer up to 28; unlike the earlier layer sweep (additive, where L24/L26 were chosen), the un-normed
  last-block output is the best single patch site.
- Judge-OK at layer 28: .78 (patch) / .73 (add); unscorable .25 / .19.
