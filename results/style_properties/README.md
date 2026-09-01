# results/style_properties — free-form style-property read/write study

Extension of the 69-task read/write-feature line (results/69_task_run) to a new
substrate: **free-form text with a binary stylistic property** (all-lowercase sentence
starts, UK vs US spelling, double space after periods, ...). Property manifestations play
the role of demo *labels* (evidence / read sites); the last token before a
manifestation's nat/alt divergence — the *point of no return*, identity-matched across
the minimal-pair twins — plays the role of the *cue* (decision / write site).
Full staged design + user decisions of 2026-09-01: see `adjudication_memo.md` here and
WORKLOG stream "style-properties".

Model: GPT-J-6B. Readout: **sampled adherence only** (T=1 seeded generation classified
nat / alt / unscorable at decision points) — user decision 2026-09-01; no logit-diff
readouts anywhere in reported results.

| Folder / file | Contents |
|---|---|
| `dataset_audit.csv` | Stage A2 tokenization audit: eligible/used docs, sites, decision-token mismatch drops, nat/alt token-count deltas, k coverage. |
| `behavioral_prescreen/` | Stage A4: sampled adherence by (context polarity, k prior manifestations) and by token distance since last manifestation; `prescreen_summary.csv` with the proposed pass/fail gate. |

Data: `dataset_files/style_properties/` (base corpus + per-property minimal-pair
datasets); scripts: `src/sandbox/ext_styleprops/`; run artifacts:
`artifacts/style_properties/` (gitignored). Pool artifact after the gate:
`task_splits/style_properties_pool.json`.
