# Project Instructions

Canonical definitions:
- "Function vectors" (GPT-J) means `artifacts/function_vectors/gpt-j/train_varicl_top40`
  unless explicitly stated otherwise. FV sets under `artifacts/function_vectors/gpt-j/debug/`
  (e.g. `train_varicl_max4_top40`) were debug tests — do not build new results on them.
  See DECISIONS.md 2026-07-10.
- Extended-task experiments use ONLY the 69-task working pool / split
  `task_splits/extended_steerable_69_prunedfail.json` (55 train / 14 heldout) unless
  explicitly stated otherwise. The 21 tasks in
  `dataset_files/extended_tasks/head_intensive_pruned_tasks/` are excluded — never mix them
  back in implicitly. See DECISIONS.md 2026-08-16.

Before starting work:
- Read WORKLOG.md and DECISIONS.md.
- Identify your tmux window/session and current stream.
- Rename the tmux window to reflect the task before starting it (if the current name doesn't already fit the work) — this makes it easier to switch between tmux windows running Claude. Use `tmux rename-window <name>`.
- Add or update your stream entry in WORKLOG.md.

Results & artifacts layout:
- Intermediates (activation captures, function vectors, head selections, paired-task captures) live in
  git-ignored `artifacts/`. Study deliverables (figures, summary tables) live in tracked `results/`,
  bucketed by research direction: `direction1_ambiguous`, `direction2_label_geometry`,
  `direction3_fv_formation`, `steering_vector_comparison`, `general`. Run logs go in git-ignored `logs/`.
- NEVER hardcode `results/...` or `figures/...` paths. Import `ARTIFACTS_ROOT`, `RESULTS_ROOT`,
  `LOGS_ROOT` and the bucket constants from `src/utils/paths.py`. See README "Repository layout".

While working:
- Keep changes narrowly scoped.
- Prefer existing repo patterns over new abstractions.
- For big experiments, save the reusable intermediates (within reason on storage) so variations can be rerun without recomputing the expensive stage — see "Save intermediates" in DECISIONS.md Conventions.
- Record important commands, outputs, files changed, and findings in WORKLOG.md.
- Put reusable lessons, conventions, and project decisions in DECISIONS.md.
- Do not let multiple agents edit the same source files at the same time unless explicitly coordinated.

Before stopping:
- Update WORKLOG.md with:
  - Status
  - Commands run
  - Files changed
  - Findings
  - Next
  - Blockers