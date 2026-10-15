# Read-feature steering on the source-program prompts (35 Python convention families, Qwen2.5-7B base, model key `qwen25_code_tr`)

2026-10-15. Same procedure as `results/code_styles/read_steering/`: evidence tokens of the translation prompts = the original evidence positions
shifted by the header length (`build_translation_evidence.py`; the Python part tokenises identically after the header, verified for all 55,288
records, 0 mismatches); read vectors = mean evidence-token activation per family, convention and layer over the k = 3 and 4 TRAINING prompts
(`capture_read.py --ks 3 4 --split train`); held-out k = 3 prompts whose demonstrations are in the WRONG convention (40 per context) are steered
at EVERY evidence token with ±α · r_L towards the other convention; success = target convention AND judge OK (`read_sweep.py`,
`read_sweep_analyze.py --tag sweep35`). Grid: layers 2–16 step 2, 20, 24 × α 1, 2, 4, 8, all 35 families.

`sweep35/`: sweep.csv, baseline.csv (unsteered wrong-convention prompts and the k = 3 right-convention ceiling), selected.json,
per_family_L8a4_vs_original.csv, compare_summary.json, sweep_heatmap.png, sweep_by_family.png, sweep_margin.png, sweep_slide.png.

Pooled success (35 families):

| layer | alt α2 | alt α4 | alt α8 | nat α2 | nat α4 | nat α8 |
|---|---|---|---|---|---|---|
| 2 | .42 | .34 | .27 | .55 | .48 | .43 |
| 6 | .50 | .49 | .44 | .61 | .64 | .59 |
| 8 | .49 | .50 | .43 | .60 | .64 | .58 |
| 10 | .49 | .53 | .45 | .61 | .62 | .58 |
| 12 | .48 | .53 | .44 | .62 | .62 | .53 |
| 16 | .45 | .49 | .36 | .59 | .59 | .51 |
| 20 | .40 | .41 | .34 | .49 | .51 | .44 |
| 24 | .18 | .23 | .23 | .24 | .27 | .26 |

- Best shared cell L12 α4 (.576 mean; nat .62, alt .53), with L6–L14 at α4 all within .02; the previous canonical L8 α4 gives nat .64, alt .50.
- Same 35 families, L8 α4, original prompts: nat .64, alt .49. So the read feature steers to the SAME absolute level as before, while the k = 3
  right-convention ceiling rose from .79 / .73 to .94 / .85 and the unsteered wrong-convention rate stayed ≈ .07 / .01. Fraction of the ceiling
  recovered: 67% / 59% now vs 82% / 67% before. Judge-OK of steered completions .76 / .75 (was .83 / .81); unscorable .08 / .12 (was .15 / .21).
- Per family at L8 α4 vs original: towards the alternative 13 families better by ≥ .10, 10 worse; towards the natural 15 better, 10 worse.
  Strong: py_self_name 1.00 / 1.00, py2_print .98 / .95, py2_except .95 / .92, blank_lines .95 / .98. Zero towards the alternative:
  docstring_style, py_class_naming, py_comprehension, py_not_in.
- Contrast with the write feature on the same prompts (L24 α2: alt .69 / nat .80 vs .49 / .57 before): the source program helps the cue-token
  write feature a lot and the evidence-token read feature not at all in absolute terms.
