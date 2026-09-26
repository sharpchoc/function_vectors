# Identification -> execution readout for coding styles, v2 (2026-09-23): 53-family pool WITH controls

Rerun of `../read_causal/` (55 families, older corpus, no controls) on the current 53-family pool with `read_causal.py --controls`
(artifacts `artifacts/style_translation/qwen25_code/read_causal_v2/`). Natural-convention k = 3 prompts, 40 held-out documents per
family; −α·r (identification style-contrast vector, evidence-token mean at hidden state 8 = block 7 output) added at every evidence
token, prefill only. Readout at the query cue, hidden state 24 = block 23 output (the paper's execution block). Controls: another
pool family's identification vector (seeded, same language when possible) and a random direction with |r|.
Summary: `summarize_code_read_causal_v2.py` → `summary.csv` (family bootstrap 10k, seed 20260923), `per_family.csv`.

At α = 2 (mean over 53 families; 95 % CI):
| arm | cos(steered shift, real shift) | share of real shift along execution direction (median) | next token = alternative convention |
|---|---|---|---|
| own identification vector | .746 | .827 | .674 |
| other-family vector | .173 | .092 | .026 |
| random direction | .039 | −.004 | .015 |
References: real shift's cosine with the execution direction .721; alternative first token top-1 unsteered .016, with real alternative
demonstrations .084. Own > both controls (cosine) in 52/53 families; own cosine ≥ .5 in 46/53.

## Cue shift as cosine with the execution feature, cos(Δz_α, −d_f^exec) (2026-09-26, stored activations, CPU)
Paper §5.3 metric. Δz_α = h_α − h_0 at the query cue, hidden state 24; d_f^exec = `v_nat` = mean_nat − mean_alt of the cue-token
activation at hidden state 24 over paired successful training-split prompts (`steering/vectors_k3_train`). This is exactly the stored
`cos_w` (w = unit(−v_nat)), so no new numbers were computed from scratch; a recomputation from the fp16 `h_L24`/`h0_L24` agrees to 4e-7.
Same aggregation as above (mean over 40 prompts per family, mean over 53 families, family bootstrap 10k, seed 20260923; the script first
re-validates the dircos numbers above). Script `src/eval_scripts/code_cue_shift_exec_cos.py` → `exec_feature_cos_summary.csv`,
`exec_feature_cos_per_family.csv`, candidate Figure 21 `exec_feature_cos_by_alpha.{png,pdf}` (left panel swapped, right unchanged).

| arm (α = 2) | cos(Δz_α, −d_f^exec) [95 % CI] |
|---|---|
| own identification vector | .628 [.572, .682] |
| other-family vector | .100 [.068, .133] |
| random direction | −.006 [−.024, .012] |
| real shift Δz_alt (reference) | .721 [.684, .755] |
Own > both controls in 52/53 families; own ≥ .5 in 42/53 (real shift ≥ .5 in 51/53); no family negative at α ≤ 4. Own peaks at α = 2
(.492/.587/.628/.588/.500 for α = .5/1/2/4/8). Lowest own families at α 2: rust_question .06, py_not_in .14, docstring_style .21,
py_comprehension .22, py_with_open .23, py_enumerate .24.
