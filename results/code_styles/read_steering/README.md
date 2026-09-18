# Read-feature steering (qwen25_code)

`sweep10/` — layer (2..26 step 2) × strength (0.5, 1, 2, 4) sweep on the 10 write-sweep families (`read_sweep.py`,
`read_sweep_analyze.py`): k = 3 prompts of the held-out documents whose demonstrations are in the WRONG style are steered with
±α·(μ_nat − μ_alt) of the read feature at every evidence token (prefill only); success = target convention AND judge OK.
Selected (shared max): layer 8, α = 4 → .65 vs .05 unsteered (right-style k = 3 ceiling .77). Read steering works at layers 2–22 and
fails at 24/26 (the write feature's layers); success still rises at α = 4.

`full56/` — all 56 pool families at the selected setting (L8, α = 4), first 40 held-out documents per context style, with the
unsteered baseline (`read_sweep.py --layers 8 --alphas 4 --with_base --limit 40`, `read_full_analyze.py`): → natural .09 → .63
(k = 3 right-style ceiling .74), → alternative .03 → .48 (ceiling .67); ≥ +20 points in 53/56 families; docstring_style has no effect.
