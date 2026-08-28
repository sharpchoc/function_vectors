# Exploratory results (NOT mainstream)

Quarantined on **2026-08-28** (see DECISIONS.md entry of that date). These research directions
were explored before the project settled on its mainstream line — the 69-task read/write-feature
study in `results/69_task_run/` (write feature = 37-head function vector; read feature =
early-layer label-token mean) plus the steering-method comparisons in
`results/steering_vector_comparison/`.

The directions below **did not pan out** and are kept only for possible later revisits.
Do not build new mainstream results on them without an explicit user decision promoting them
(same rule as `results/sandbox/`). Their plotting/eval scripts moved to
`src/eval_scripts/exploratory/` at the same time.

| Folder | Old path (pre 2026-08-28) | What it was |
|---|---|---|
| `direction1_ambiguous/` | `results/direction1_ambiguous/` | FV development on ambiguous ICL tasks (magnitude/identity disambiguation, held-out ambiguous steering). Dormant since 2026-06. |
| `direction2_label_geometry/` | `results/direction2_label_geometry/` | Label- vs pre-label-token geometry, one/two-shot paired captures, task-switch steering, GPT-judged evals. Dormant since 2026-07. |
| `direction3_fv_formation/` | `results/direction3_fv_formation/` | FV formation across layers/positions: activation→FV ridge/OLS decoding, preimage analysis and ablation, attention-head mechanisms. Has its own README with a fixed 6-folder taxonomy (DECISIONS 2026-07-30). |
| `general/` | `results/general/` | Uncategorized exploration: fig-8 recreation, embedding geometry, task-accuracy catalogs, extended-task n-shot sweep, mixed-ICL and rhyme-judge trials. |

Notes:
- Paths in the subfolders' own READMEs, WORKLOG.md, and DECISIONS.md entries that predate
  2026-08-28 refer to the old `results/<bucket>` locations; prepend `exploratory/`.
- Runtime code never hardcoded these paths — the bucket constants (`AMBIGUOUS_DIR`,
  `LABEL_GEOMETRY_DIR`, `FV_FORMATION_DIR`, `GENERAL_DIR`) in `src/utils/paths.py` were
  reparented under `EXPLORATORY_ROOT`, so old scripts (including on unmerged branches, once
  they merge main) automatically read/write the new locations.
