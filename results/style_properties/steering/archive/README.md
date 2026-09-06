# Excluded earlier steering runs

These predate the current protocol and are **not** part of the variant grid, because their
measurements are not comparable:

- `artifacts/style_properties/steering/_archive_allk/{sweep,full}` — evidence-token
  injection, **all** cue sites (not just the first), 5–16-token rollouts, **no coherence
  judge**, and adherence conditional on scorable rollouts only.
- `artifacts/style_properties/steering/_archive_allk/{sweep_cue,full_cue}` — cue-token
  injection but still all cue sites and unjudged short rollouts.
- `artifacts/style_properties/steering/_archive_allk/_old_classifier` — num_words results
  under the pre-2026-09-01 classifier (counted out-of-range digits as violations).

Also excluded from the grid, though they live inside the current runs' JSONs: arms whose
vector was derived from **evidence** tokens rather than cue tokens
(`meandiff_cue_*`, `rawalt_cue_nat2alt_*`). The grid is cue-derived vectors injected at the
cue; derivation site is not one of its axes.

Nothing here is deleted — only unassigned.
