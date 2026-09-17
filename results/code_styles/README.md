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
