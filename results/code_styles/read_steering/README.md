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
