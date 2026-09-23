# Read-feature ablation at the evidence tokens (code-convention families, 2026-09-23)

Mirror of `../write_ablation/`. USER DECISIONS: k = 4 prompts of the first 40 held-out documents, ALTERNATIVE-convention demonstrations only;
ablation at EVERY evidence token (code_evidence.py positions, all four demonstrations) at EVERY layer 0..27 (0 = embedding output), each layer
with the family's own unit READ direction ŵ_A^l = unit(mean_nat − mean_alt) of `read_features/vectors_k3_train` — zero-projection
(h_l −= (h_l·ŵ) ŵ) or mean ablation (h_l += (m_A^l − h_l·ŵ) ŵ, m_A^l = corpus-wide mean projection of the evidence-token activations: prompt-count
weighted mean over all pool families' natural- and alternative-pole evidence means, `ablation_means.py --feature read` → `means_read_all.json`;
at L8 it lies between the family's own two pole means in 53/53 families); counterfactual control = the same edits with the write ablation's seeded
partner family's read directions and means (js_var's partner re-drawn js_hungarian → js_quotes after js_hungarian left the pool; its write
ablation was re-run with the new partner). Success = keeps the demonstrated (alternative) convention AND judge OK.
Scripts: `read_ablate.py` (+ `steer_hooks.PositionAblate/MultiPositionAblate`), `read_ablate_analyze.py`; records
`artifacts/.../ablation/read_full_k4/<shard>/<family>.json`. Files: full.csv, success_by_arm.csv, margin.csv, summary.json, headline.png, per_family_grid.png.

## Results (53 families, alternative-convention demonstrations, k = 4)

| arm | read ablation (evidence tokens, L0–27) | write ablation (cue token, L1–27) |
|---|---|---|
| unablated | .741 | .718 |
| own direction, zero-projection | .310 | .405 |
| own direction, mean ablation | .339 | .408 |
| counterfactual direction, zero-projection | .738 | .718 |
| counterfactual direction, mean ablation | .736 | .741 |

Own-direction read ablation removes .43 (zero) / .40 (mean) of success (95 % CI over families ≈ ±.07); the counterfactual removes nothing (.003 / .005).
First-token margin towards the demonstrated convention: 4.6 → −0.8 (own zero) / −0.9 (own mean) vs 4.6 (counterfactual) — the model flips to the
natural convention. 31 of 53 families are individually specific (own drop ≥ .20 with cf drop < .10), 15 families fall to ≤ .05. No effect (own drop
< .10): comment_language, c_comment_style, py_private, py_literal_ctor, py_optional, py_tabs, py_not_in. Judge-OK rate and unscorable rate move
little (.89 → .85, .10 → .13): the loss is a convention flip, not broken code.
