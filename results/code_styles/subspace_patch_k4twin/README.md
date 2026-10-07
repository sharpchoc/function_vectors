# Twin patch at the k = 4 cue: natural-context prompt patched with its alternative twin (53 code families)

2026-10-07. `write_subspace_patch_k4.py` (job `logs/code_styles_subspace_k4_job.sh`, pods sk1–sk4), analysis `write_subspace_patch_k4_analyze.py`.
Records `artifacts/style_translation/qwen25_code/steering/subspace_patch_k4twin/sk*/<family>.json` (judged, 0 fails).

## Design (user, 2026-10-07)
- Source prompt: the held-out document's k = 4 NATURAL-context prompt (demonstrations and target in the natural convention, ending at the cue).
- Counterpart: the same document's k = 4 ALTERNATIVE-context twin — same demonstrations and cue token, only the convention differs — kept only when
  the twin's own k = 4 completion was correct (convention AND judge OK). Up to 40 such held-out documents per family (min 17, median 39).
- Per layer 20..28 (28 = un-normed last-block output) the natural prompt's cue-token coordinates in the training subspace (top-r uncentered right
  singular vectors of the training documents' k = 4 pair differences) are replaced by the twin's: `own1` / `own2` / `own10`; `full` replaces the whole
  cue activation by the twin's; `mean1` / `mean2` use the training-mean alternative coordinates instead. `base` = unpatched natural prompt.
- Metric: fraction of completions that use the ALTERNATIVE convention and pass the judge (the flip rate). First-token margin recorded.

## Results (`pooled_by_layer.csv`, `per_family_L28.csv`, `full.csv`, `summary.json`, `pooled_by_layer.png`, `success_by_layer_slide.png`)
Pooled over 53 families, layer 28 (every twin arm peaks at 27 or 28):

| unpatched | mean1 | mean2 | own1 | own2 | own10 | full |
|---|---|---|---|---|---|---|
| .03 | .43 | .45 | .46 | .52 | .56 | .64 |

- With exact twins the rank ladder is real: twin PC1 .46 → PC1 + PC2 .52 → top 10 .56 → full .64 (first-token margin 1.8 → 3.0 → 4.2 → 5.1;
  judge-OK .85–.86 throughout). The second direction adds +.06 with the twin's coordinates (23 families better by ≥ .10, 4 worse), whereas with the
  training mean it adds only +.02 — the twin's PC2 coordinate carries document-specific information the mean cannot.
- Own vs mean target: +.03 at rank 1, +.07 at rank 2.
- Ceiling check: the full swap at layer 28 reproduces the twin's next-token distribution exactly (first-token margins match the twin's within bf16
  noise). Its convention rate .74 equals the twin's own probability of sampling the alternative rendering at T = 1 (.77 averaged over these
  documents), and .64 after the judge — so "100%" is not reachable with T = 1 sampling; the twin itself would flip ~.77 of the time on re-sampling.
- Layers 20–22 do nothing (≤ .14); the jump is between 22 and 25.
