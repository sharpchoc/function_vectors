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
