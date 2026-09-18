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

- `full56/` — all 56 pool families at (L24, L26) × (α 2, 4), first 40 held-out documents per family, both directions
  (`write_full_analyze.py`). No cell selected (user decision): `by_cell.csv` gives the four means; `full.csv` per family/cell;
  `success_grid.png` (families × cells with unsteered and k = 4 references), `gain_by_family.png`, `steer_vs_k4.png`.
  Means over 56 families, both directions: unsteered .27, L24α2 .50, L24α4 .50, L26α2 .49, L26α4 .43, k = 4 in context .72.
