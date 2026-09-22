# Archive: code-style data and results BEFORE the 2026-09-22 corpus-wide regeneration (regen8)

State of git commit 09ef0e01 (main, 2026-09-22). Everything is gzip-compressed JSON.
- `pairs/<family>.json.gz` — the 60 code families' pair files (`dataset_files/style_translation/pairs/`).
- `code/<family>.json.gz` — the raw builder records (`dataset_files/style_translation/code/`), plus every task pool (`tasks_<lang>.json`, `tasks_designed_<family>.json`).
- `cues/<family>.json.gz` — the qwen25_code cue files (`artifacts/style_translation/qwen25_code/cues/`).
- `results/` — a copy of `results/code_styles/` at that commit (all tables, figures and READMEs).
The token-exact prompt files (342 MB) and evidence positions are NOT in git: a full copy is on the shared volume at
`/workspace/archive_code_styles_2026-09-22/{prompts,evidence}/`; they are also reproducible from the archived pairs + cues with
`build_prompts.py --model qwen25_code --K 5` and `code_evidence.py`.
Restore: `gunzip -c pairs/<f>.json.gz > dataset_files/style_translation/pairs/<f>.json` (same for code/ and cues/), then rebuild prompts.
