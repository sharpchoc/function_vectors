# Read-feature steering (qwen25_code)

`sweep10/` — layer (2..26 step 2) × strength (0.5, 1, 2, 4) sweep on the 10 write-sweep families (`read_sweep.py`,
`read_sweep_analyze.py`): k = 3 prompts of the held-out documents whose demonstrations are in the WRONG style are steered with
±α·(μ_nat − μ_alt) of the read feature at every evidence token (prefill only); success = target convention AND judge OK.
Selected (shared max): layer 8, α = 4 → .65 vs .05 unsteered (right-style k = 3 ceiling .77). Read steering works at layers 2–22 and
fails at 24/26 (the write feature's layers); success still rises at α = 4.

`full56/` — all 56 pool families at the selected setting (L8, α = 4), first 40 held-out documents per context style, with the
unsteered baseline (`read_sweep.py --layers 8 --alphas 4 --with_base --limit 40`, `read_full_analyze.py`): → natural .09 → .63
(k = 3 right-style ceiling .74), → alternative .03 → .48 (ceiling .67); ≥ +20 points in 53/56 families; docstring_style has no effect.

## Update 2026-09-22 — corpus-wide padding clean-up refreshed
All numbers here were recomputed after the corpus-wide padding clean-up (2,438 of 2,464 flagged documents regenerated with Opus 5 + Opus 5/GPT-5 review; 49 families changed; 26 documents still flagged: js_hungarian 9, py_join_concat 6, py_private 11). Pool is now 55 / 60: py_ternary dropped (k = 4 natural pole .375 → .27, below the cutoff). Before/after per family: `results/code_styles/padding_regen_before_after.csv`, pooled: `padding_regen_before_after_pooled.csv` (55 pool families: k = 4 .726 → .719; write L24 α2 .508 → .543; read L8 α4 .569 → .581). The 10-family hyperparameter sweeps (sweep10) were NOT re-run and predate the clean-up. The read→write map sandbox was NOT refit (user will instruct).

## Update 2026-09-22 (later) — py_private redefined
py_private now = underscore-prefixed vs plain private members (alternative derived by rule, see WORKLOG); its numbers are not comparable with the earlier double-underscore definition. Residual flagged documents in the pool: 1 (py_join_concat); js_hungarian is out of the pool (decision 2026-09-23).

## Update 2026-09-23 — regen8 (corpus-wide regeneration) refreshed
All numbers recomputed after regen8: 7,271 of 7,567 strict-review failures regenerated (Opus 5 generator, GPT-5 sole judge = USER DECISION 2026-09-23, exact rule-based alternative twins for 37 families, designed tasks, static pre-checks); 127 docs failed every attempt and keep their old text (sql_keyword_case 56, py_join_concat 32, sql_join_style 14, line_wrap 9, …). Pool is now 53 / 60: py_abbrev dropped (k = 4 .30) and js_hungarian removed by decision (USER DECISION 2026-09-23: Hungarian prefixes need type inference, no clean alternative twin was achievable). Pooled means over the 53 pool families: k = 4 .723 → .763, write L24 α2 .551 → .564, read L8 α4 .591 → .600, 3-shot reference .716 → .760. Tables: `results/code_styles/regen8/{pooled_before_after,per_family_before_after}.csv`. Pre-regen8 data and results are archived under `dataset_files/style_translation/code/archive/2026-09-22_pre_regen8/`. The 10-family hyperparameter sweeps (sweep10) were not re-run. The read→write map sandbox and the read→write causal test were NOT re-run (user will instruct).
