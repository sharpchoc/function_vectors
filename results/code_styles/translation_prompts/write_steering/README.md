# Write-feature steering on the source-program prompts (35 Python convention families, Qwen2.5-7B base, model key `qwen25_code_tr`)

2026-10-14/15. Same procedure as `results/code_styles/write_steering/`: cue-token mean activation per family, convention and layer over the k = 3 and 4
prompts of the TRAINING documents whose completion was correct (`capture_cues.py --ks 3 4 --unpaired --split train`, vectors in
`artifacts/style_translation/qwen25_code_tr/steering/vectors_k3_train/`); held-out 0-shot prompts (first 40 per family) steered at the cue token with
±α · (μ_nat − μ_alt) at ONE layer; success = target convention AND Gemini judge OK (`write_sweep.py`, `write_sweep_analyze.py --tag sweep35`).
Grid: layers 4, 8, 12, 16, 20, 22, 24, 26, 27 × α 1, 2, 4, ALL 35 families (the original sweep used 10 families).

`sweep35/`: sweep.csv (family × target × layer × α), baseline.csv, selected.json, per_family_L24a2_vs_original.csv, sweep_heatmap.png,
sweep_by_family.png, sweep_margin.png, sweep_slide.png.

Pooled success (35 families; unsteered 0-shot: natural .64, alternative .17):

| layer | alt α1 | alt α2 | alt α4 | nat α1 | nat α2 | nat α4 |
|---|---|---|---|---|---|---|
| 4–16 | .17–.18 | .17–.20 | .18–.27 | .65–.67 | .66–.67 | .65–.67 |
| 20 | .26 | .39 | .53 | .70 | .73 | .72 |
| 22 | .35 | .49 | .59 | .72 | .77 | .74 |
| 24 | .54 | **.69** | .66 | .79 | **.80** | .72 |
| 26 | .59 | .67 | .59 | .78 | .77 | .58 |
| 27 | .60 | .66 | .56 | .77 | .70 | .53 |

- Best shared cell = layer 24, α 2 (as in the original setting): mean .75 (alt .69, nat .80), judge-OK .96. On the same 35 families the original
  prompts gave alt .49 / nat .57 at L24 α2.
- Per family at L24 α2: 10 families ≥ .90 towards the alternative (1 before); 23 families improve by ≥ .10 towards the alternative (2 worse), 28
  towards the natural (1 worse). Still hard: docstring_quotes .08, py_builtin_generics .12, comment_language .15, py_const_naming .20,
  py_type_hints .28, float_literals .30.
- Layer profile is unchanged in shape: nothing before layer 20, a sharp rise at 20–24, and α 4 at layers 26–27 breaks the natural-side prompts
  (first-token margins 14–17 at those cells).
- Note: with the source program the prompts are 1,000–2,300 tokens; `write_sweep.py` now computes the first-token logits at the last position only
  and uses an adaptive batch (token budget), which was needed to fit 32 GB.

Selection check (2026-10-15): the shared cell is chosen on the same held-out prompts that are reported. Re-selecting on the first 20 held-out docs per
family and scoring the last 20 picks L24 α2 again: alt .67 / nat .79; family-split folds (18/17) also pick L24 α2: alt .71 / .67, nat .79 / .81
(`sweep35/bars_summary.json`). Bar chart `sweep35/bars_slide.png`: original prompts vs source program, 0-shot / steered 0-shot / 4-shot.
