# Write-feature ablation at the cue token (code-convention families, 2026-09-23) — all pool families

USER DECISIONS: k = 4 prompts of the first 40 held-out documents per pole, both contexts; ablation at the cue token at EVERY layer 1..27,
each layer with the family's own unit write direction ŵ_A^l = unit(mean_nat − mean_alt) (vectors_k3_train) — zero-projection
(h_l −= (h_l·ŵ) ŵ) or mean ablation (h_l += (m_A^l − h_l·ŵ) ŵ, m_A^l = corpus-wide mean projection over all pool families' cue means, both poles,
prompt-count weighted; at L24 it matches the activation-store mean within 0.2); counterfactual control = the same two edits with one
seeded random other pool family's directions and means (`artifacts/.../ablation/means_all.json` cf_family). Success = the completion keeps
the DEMONSTRATED convention and the judge accepts it. Scripts: `ablation_means.py`, `write_ablate.py` (+ `steer_hooks.MultiCueAblate`),
`write_ablate_analyze.py`; records `artifacts/.../ablation/full_k4/<shard>/<family>.json`. Files: full.csv, success_by_arm.csv, margin.csv,
summary.json, headline.png, per_family_grid.png.

## Results (alternative-convention demonstrations = headline; USER DECISION 2026-09-23: the natural convention is the model's default, so only the
alternative context tests whether the demonstrations' effect is carried by the write feature; natural-context numbers in `supp_natural_context.png` / full.csv)

| arm | alternative demos | natural demos |
|---|---|---|
| unablated | .718 | .807 |
| own direction, zero-projection | .405 | .647 |
| own direction, mean ablation | .408 | .606 |
| counterfactual direction, zero-projection | .718 | .799 |
| counterfactual direction, mean ablation | .741 | .805 |

Own-direction ablation removes .31 of the alternative-context success (95 % CI over families ≈ ±.05); the counterfactual direction changes nothing.
First-token margin towards the demonstrated convention: 6.0 → 1.2 (own) vs 5.9 (counterfactual). 30 of 53 families show an own drop ≥ .20 with a
counterfactual drop < .10 (zero-projection; 30 for mean ablation). Largest: num_separators .90 → .15, py_literal_ctor .88 → .15, py_class_naming
.72 → .02, bash_subst .78 → .08, comment_case .72 → .05. No effect (own drop < .10): sql_keyword_case, bash_test, comment_language, py_join_concat,
py_fstring, js_template, py_enumerate, operator_spaces, sql_join_style — mostly structural families whose k = 4 baseline is already low or whose
convention is carried at positions other than the cue.

Note 2026-09-23: js_var's counterfactual partner was re-drawn (js_hungarian → js_quotes) after js_hungarian left the pool; js_var re-run, pooled numbers above refreshed (changes ≤ .003). Read-feature counterpart: `../read_ablation/`.
