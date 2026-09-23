# results/code_styles — coding-convention families, clean restart (2026-09-17)

Model key `qwen25_code` (Qwen2.5-7B base; artifacts under `artifacts/style_translation/qwen25_code/`). Single source of truth for the data:
`dataset_files/style_translation/pairs/<family>.json` — 60 families, exactly 200 documents each (12,000 pairs). The corpus was rebuilt and
cleaned one bug at a time with user approval (bugs 1–10, see DECISIONS.md 2026-09-17 entries): whitespace and bash_subst alternatives are
derived by rule, every twin passes a real parser (Python / JavaScript / Rust / PHP / Bash), renames never collide, and only genuine
convention decisions are counted as opportunities (`opps`; the full diff is kept in `opps_all`). The earlier code-family results
(2026-09-14..17, with the forced-decision confound and the data defects) are archived under `results/style_translation/qwen25_base/code_archive/`.

## Step 3 — in-context accuracy vs k (this bucket)

Prompts: `build_prompts.py --model qwen25_code --K 5` → 200 docs × 2 poles × k = 0..4 per family (120,000 items). One seeded T = 1 sample of
48 tokens per prompt (`rollout.py`), cut at the code boundary; convention decided by the family's regex pair on the first 160 characters
(deterministic); correctness by Gemini 2.5 Flash told to ignore all style (`judge_rollouts.py`); accuracy = convention shown in context AND
judge OK (`analyze.py`, Wilson 95 % CI).

| file | what |
|---|---|
| `accuracy_by_k.png`, `summary.csv` | accuracy vs k per family, one line per pole (nat = natural convention in context, alt = alternative) |
| `unscorable_by_k.png`, `unscorable.csv` | share of completions that use neither convention (regex finds nothing), per family and k |
| `cutoff_k4.png`, `cutoff_k4.csv`, `code_pool.json` | the k = 4 cutoff: a family survives if accuracy at k = 4 ≥ .30 on BOTH poles (`code_cutoff.py`) |
| `logprob/logprob_margin_by_k.png`, `logprob/logprob_top1_by_k.png`, `logprob/logprob_summary.png`, `logprob/logprob_margin.csv` | classifier- and judge-free failsafe: first-token log-prob margin at the cue, log p(first token of the context's rendering) − log p(first token of the other rendering), and the share of prompts whose top-1 next token is the context's rendering (`logprob_margin.py` on GPU, `logprob_analyze.py`) |
| `records.npz` | per-record decisions / judge flags for re-analysis |

Run: `logs/code_styles_step3_job.sh <tag> <families>` per pod (rollout + log-prob margin), `judge_rollouts.py --dir artifacts/style_translation/qwen25_code/rollouts`,
then `analyze.py --model qwen25_code`, `code_cutoff.py --model qwen25_code`, `logprob_analyze.py --model qwen25_code`.

- 2026-09-19: py_not_in, rust_question, py_is_none, float_literals regenerated without padding (Opus 5 + dual review); all their
  downstream artefacts and the results here were rebuilt; `explainer/` lists every family; `read_write_map/sandbox` is not yet refit.

## Update 2026-09-22 — corpus-wide padding clean-up refreshed
All numbers here were recomputed after the corpus-wide padding clean-up (2,438 of 2,464 flagged documents regenerated with Opus 5 + Opus 5/GPT-5 review; 49 families changed; 26 documents still flagged: js_hungarian 9, py_join_concat 6, py_private 11). Pool is now 55 / 60: py_ternary dropped (k = 4 natural pole .375 → .27, below the cutoff). Before/after per family: `results/code_styles/padding_regen_before_after.csv`, pooled: `padding_regen_before_after_pooled.csv` (55 pool families: k = 4 .726 → .719; write L24 α2 .508 → .543; read L8 α4 .569 → .581). The 10-family hyperparameter sweeps (sweep10) were NOT re-run and predate the clean-up. The read→write map sandbox was NOT refit (user will instruct).

## Update 2026-09-22 (later) — py_private redefined
py_private now = underscore-prefixed vs plain private members (alternative derived by rule, see WORKLOG); its numbers are not comparable with the earlier double-underscore definition. Residual flagged documents in the pool: 1 (py_join_concat); js_hungarian is out of the pool (decision 2026-09-23).

## Update 2026-09-23 — regen8 (corpus-wide regeneration) refreshed
All numbers recomputed after regen8: 7,271 of 7,567 strict-review failures regenerated (Opus 5 generator, GPT-5 sole judge = USER DECISION 2026-09-23, exact rule-based alternative twins for 37 families, designed tasks, static pre-checks); 127 docs failed every attempt and keep their old text (sql_keyword_case 56, py_join_concat 32, sql_join_style 14, line_wrap 9, …). Pool is now 53 / 60: py_abbrev dropped (k = 4 .30) and js_hungarian removed by decision (USER DECISION 2026-09-23: Hungarian prefixes need type inference, no clean alternative twin was achievable). Pooled means over the 53 pool families: k = 4 .723 → .763, write L24 α2 .551 → .564, read L8 α4 .591 → .600, 3-shot reference .716 → .760. Tables: `results/code_styles/regen8/{pooled_before_after,per_family_before_after}.csv`. Pre-regen8 data and results are archived under `dataset_files/style_translation/code/archive/2026-09-22_pre_regen8/`. The 10-family hyperparameter sweeps (sweep10) were not re-run. The read→write map sandbox and the read→write causal test were NOT re-run (user will instruct).
