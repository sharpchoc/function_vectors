# debug/ — FV sets that are NOT canonical

`train_varicl_max4_top40` was a DEBUG test set (variable-ICL capped at 4 shots). It is not the
project's function-vector definition. "Function vectors" means `../train_varicl_top40`
(variable 1-10-shot CIE, top-40 multitask heads) unless explicitly stated otherwise.
See DECISIONS.md 2026-07-10. Historical artifacts fit against the max4 FVs
(preimage_pairdiff/, preimage_pairdiff_tsvdk16/, twoshot_pairdiff results, Stream W v1 run)
record the old path in their run_config.json.
