# Project Instructions

Before starting work:
- Read WORKLOG.md and DECISIONS.md.
- Identify your tmux window/session and current stream.
- Rename the tmux window to reflect the task before starting it (if the current name doesn't already fit the work) — this makes it easier to switch between tmux windows running Claude. Use `tmux rename-window <name>`.
- Add or update your stream entry in WORKLOG.md.

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