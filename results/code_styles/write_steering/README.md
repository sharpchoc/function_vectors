# Write-feature steering on code conventions (Qwen2.5-7B base, model key `qwen25_code`)

Cue-token steering of 0-shot prompts with `±α · (μ_nat − μ_alt)`; μ = mean cue-token activation over k ∈ {3, 4} prompts whose
sampled completion was correct, from the 150 training documents of each family (`write_split.heldout`: crc32(doc_id) % 4 == 0 are
held out). Only held-out 0-shot prompts are steered. Success = completion uses the target convention AND the Gemini judge accepts it.
Grading identical to step-3 rollouts. See DECISIONS.md 2026-09-17 / 2026-09-18.

- `sweep10/` — layer (2…26) × strength (0.5, 1, 2, 4) sweep on 10 random pool families, ~50 held-out prompts each
  (`write_sweep_analyze.py`). Only layers 24/26 work; L26 α2 and L24 α4 tie at ~.55 mean success. At those layers the vector
  mostly writes the first token, not the convention.
**Canonical write feature (USER DECISION 2026-09-18): layer 24, α = 2.** Headline over 56 families: → natural .56 (unsteered .40),
→ alternative .43 (unsteered .13), judge OK .84; k = 4 in context .74 / .69. The other cells are context only.

- NOTE 2026-09-18 (bugs 14/15): py_comprehension's rows in `sweep10/` come from the OLD opportunity lists (its cues moved); the
  sweep was not re-run because it only served the layer/strength choice. `full56/` is current: the 12 changed families were re-sampled.

- `full56/` — all 56 pool families at (L24, L26) × (α 2, 4), first 40 held-out documents per family, both directions
  (`write_full_analyze.py`). No cell selected (user decision): `by_cell.csv` gives the four means; `full.csv` per family/cell;
  `success_grid.png` (families × cells with unsteered and k = 4 references), `gain_by_family.png`, `steer_vs_k4.png`.
  Means over 56 families, both directions: unsteered .27, L24α2 .50, L24α4 .50, L26α2 .49, L26α4 .43, k = 4 in context .72.

## Update 2026-09-22 — corpus-wide padding clean-up refreshed
All numbers here were recomputed after the corpus-wide padding clean-up (2,438 of 2,464 flagged documents regenerated with Opus 5 + Opus 5/GPT-5 review; 49 families changed; 26 documents still flagged: js_hungarian 9, py_join_concat 6, py_private 11). Pool is now 55 / 60: py_ternary dropped (k = 4 natural pole .375 → .27, below the cutoff). Before/after per family: `results/code_styles/padding_regen_before_after.csv`, pooled: `padding_regen_before_after_pooled.csv` (55 pool families: k = 4 .726 → .719; write L24 α2 .508 → .543; read L8 α4 .569 → .581). The 10-family hyperparameter sweeps (sweep10) were NOT re-run and predate the clean-up. The read→write map sandbox was NOT refit (user will instruct).

## Update 2026-09-22 (later) — py_private redefined
py_private now = underscore-prefixed vs plain private members (alternative derived by rule, see WORKLOG); its numbers are not comparable with the earlier double-underscore definition. Residual flagged documents in the pool: 1 (py_join_concat); js_hungarian is out of the pool (decision 2026-09-23).

## Update 2026-09-23 — regen8 (corpus-wide regeneration) refreshed
All numbers recomputed after regen8: 7,271 of 7,567 strict-review failures regenerated (Opus 5 generator, GPT-5 sole judge = USER DECISION 2026-09-23, exact rule-based alternative twins for 37 families, designed tasks, static pre-checks); 127 docs failed every attempt and keep their old text (sql_keyword_case 56, py_join_concat 32, sql_join_style 14, line_wrap 9, …). Pool is now 53 / 60: py_abbrev dropped (k = 4 .30) and js_hungarian removed by decision (USER DECISION 2026-09-23: Hungarian prefixes need type inference, no clean alternative twin was achievable). Pooled means over the 53 pool families: k = 4 .723 → .763, write L24 α2 .551 → .564, read L8 α4 .591 → .600, 3-shot reference .716 → .760. Tables: `results/code_styles/regen8/{pooled_before_after,per_family_before_after}.csv`. Pre-regen8 data and results are archived under `dataset_files/style_translation/code/archive/2026-09-22_pre_regen8/`. The 10-family hyperparameter sweeps (sweep10) were not re-run. The read→write map sandbox and the read→write causal test were NOT re-run (user will instruct).
