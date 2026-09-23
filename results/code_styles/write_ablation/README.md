# Write-feature ablation at the cue token (code-convention families, 2026-09-23) — INTERIM, 25 of 54 families

USER DECISIONS: k = 4 prompts of the first 40 held-out documents per pole, both contexts; ablation at the cue token at EVERY layer 1..27,
each layer with the family's own unit write direction ŵ_A^l = unit(mean_nat − mean_alt) (vectors_k3_train) — zero-projection
(h_l −= (h_l·ŵ) ŵ) or mean ablation (h_l += (m_A^l − h_l·ŵ) ŵ, m_A^l = corpus-wide mean projection over all pool families' cue means, both poles,
prompt-count weighted; at L24 it matches the activation-store mean within 0.2); counterfactual control = the same two edits with one
seeded random other pool family's directions and means (`artifacts/.../ablation/means_all.json` cf_family). Success = the completion keeps
the DEMONSTRATED convention and the judge accepts it. Scripts: `ablation_means.py`, `write_ablate.py` (+ `steer_hooks.MultiCueAblate`),
`write_ablate_analyze.py`; records `artifacts/.../ablation/full_k4/<shard>/<family>.json`. Files: full.csv, success_by_arm.csv, margin.csv,
summary.json, headline.png, per_family_grid.png.
