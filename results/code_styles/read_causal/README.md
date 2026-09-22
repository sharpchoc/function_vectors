# Are the read features causal for the write features? (code-convention families, 2026-09-22)

Analogue of the function-vector read→write experiment. Script `src/sandbox/style_translation/read_causal.py`, job `logs/code_styles_read_causal_job.sh`;
raw per-prompt outputs in `artifacts/style_translation/qwen25_code/read_causal/<family>.npz`; analysis tables and figures in this folder.

**Design (user-approved 2026-09-22).** For every pool family (55), the first 40 held-out documents' k = 3 prompts with NATURAL-convention demonstrations
(the read_sweep prompt set). The read feature r = mean_nat − mean_alt (L8, evidence-token mean, training documents) is injected at layer 8 at every
evidence token as −α·r (towards the ALTERNATIVE convention), α ∈ {0.5, 1, 2, 4, 8}; prefill only. Recorded at the query cue token for layers 20..28:
h_α (steered), h_0 (unsteered, same prompt) and h_cf = the same document's k = 3 prompt with ALTERNATIVE demonstrations (identical query and cue).

**Metrics per prompt, layer and α** (the FV metric, user decision 2026-07-14 = dircos):
- dircos = cos(h_α − h_0, h_cf − h_0)
- proj_frac = ((h_α − h_0)·ŵ) / ((h_cf − h_0)·ŵ), ŵ = unit write feature (mean_alt − mean_nat at the cue, that layer)
- cos_w = cos(h_α − h_0, ŵ); norm_ratio = |h_α − h_0| / |h_cf − h_0|; cf_cos_w = cos(h_cf − h_0, ŵ) as reference
- next-token log-prob margin towards the alternative first token, every arm

No controls in this run (user decision); `--controls` adds an other-family read feature and a random direction of the same norm.

## Results (2026-09-22, 55 pool families, 40 prompts each; `summary.json`, `by_alpha.csv`, `per_family_L24.csv`, `margin.csv`, figures)

L24 cue token, mean over families (95 % CI on dircos ≈ ±.05):

| α | dircos | fraction of counterfactual shift along ŵ (median) | cos(Δ steered, ŵ) | ‖Δ steered‖ / ‖Δ cf‖ | alt first token top-1 |
|---|---|---|---|---|---|
| unsteered | – | 0 | – | 0 | .04 (real alt context: .10) |
| 0.5 | .55 | .16 | .46 | .21 | .08 |
| 1 | .66 | .48 | .55 | .51 | .43 |
| 2 | .72 | .82 | .59 | .87 | .66 |
| 4 | .69 | 1.02 | .56 | 1.18 | .75 |
| 8 | .58 | .92 | .47 | 1.29 | .70 |

Reference: the real context effect itself aligns with ŵ at cos .68 (cf_cos_w). Layer curve (α 2): dircos .61 (L20) → .72 (L24) → .78 (L28).
48 / 55 families reach dircos ≥ .5 at α 2; lowest rust_question .12, docstring_style .23, py_not_in .26; highest blank_lines .99, py2_except .96.
Files for 24 families (first launch) also hold layers 1..19 and the two control arms; the analysis uses layers 20..28 and the read arm only.
