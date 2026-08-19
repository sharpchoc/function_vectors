# results/69_task_run — structure

All studies on the canonical 69-task pool (split `task_splits/extended_steerable_69_prunedfail.json`,
55 train / 14 heldout; pooled 37-head FV from `prunedfail_seed43/pooled_sparse/selection.json`).
Reorganised 2026-08-19; WORKLOG entries before that date use the old paths (mapping below).

| Folder | Contents |
|---|---|
| `FV_train_test_generalisation/` | 37-head pooled sparse FV steers held-out tasks (zs .09→.73); poster visuals. Was `train_test_generalisation/`. |
| `FV_dimensionality_analysis/` | PCA / stable rank of task-mean + per-prompt FVs (spectra, n90=24 task-level). |
| `FV_dimensionality_reduction/train_test_split/` | 46-PC sparse selection on the TRAIN-fit 512-PC basis (+ `debugging/`: span coverage, all-512 oracle probe — heldout drop is basis-fit artifact). Files were in `FV_dimensionality_analysis/` + `FV_dimensionality_reduction/debugging/`. |
| `FV_dimensionality_reduction/train_test_together_50d/` | Sparse PC selection on the all-69 basis (50 PCs @L9, 48 @L13) — heldout fully recovered. Was `sparse_all69{,_L13}/`. |
| `FV_dimensionality_reduction/low_dim_22d/` | Fixed 22-PC task-mean subspace, alpha-rescaled: .68 vs .75 zs, no train/heldout gap. Was `debugging/taskmean_k90_*` + `poster_lowdim.png`. |
| `FV_linear_decodability/token_layer_regressions/` | Ridge activation→FV over the (token, layer) grid; heldout R² peaks at pre-label cue tokens, mid layers. |
| `FV_linear_decodability/labeltoken_fv_ridge/` | Label-token L6 ridge (+ layer sweep, seed splits, error decomposition: task-identity readout, not per-prompt). |
| `write_feature_and_model_accuracy/` | FV presence (cos at query cue, L9–20) vs task accuracy; within-task positive, between-task negative (Simpson). Was `FV_location/presence_vs_accuracy/`. |
| `feature_locations/` | Token×layer presence maps: write feature (`direct_FV_presence/`, `low_dim_FV_presence/`), top-down read feature (`top_down_read_dir_presence/`), bottom-up read feature (`bottom_up_label_mean_L6_presence/`); `poster_visuals/` keeps only `read_vs_write_presence_label_mean_dual.{png,csv}` (rest deleted on user request 2026-08-19). Was `FV_location/`. |
| `bottom_up_read_features/layer_selection/` | Raw label-token-mean steering layer×alpha sweep (peak L6–7; shared-mean control ≈0). Was `raw_mean_steering/` top level. |
| `bottom_up_read_features/head_selection/` | Sparse head selection at the label slot vs mean-diff vs raw mean (raw mean wins). Was `read_vector_head_selection/`. |
| `bottom_up_read_features/steering_results/` | L6-fixed steering: `sixshot_dummy/` (six dummy slots → 71% of real 6-shot), `narrow_patch/` (41-PC remove-and-replace). |
| `bottom_up_read_features/dimensionality_analysis/` | PCA of L6 label-token task means (`dimensionality/`), sparse PC-retention (`sparse_pc40/`). |
| `bottom_up_read_features/ablation/` | (empty — to be done) |
| `top_down_read_features/definition_sweep/` | Read-direction definition levers: cross-bracket overlap, unit-vs-natural containment, write-up assets. Was in `Read_direction_geometry/`. |
| `top_down_read_features/steering_results/` | Read-direction steering (`steering/`: dot/cos perhead @L7, patch-vs-direct; `steering_methods/`: 4 brackets × layers × alpha). |
| `top_down_read_features/dimensionality_analysis/` | Per-definition PC spectra: `{cosine,dot}_{M,perhead}__{unit,natural}/`. |
| `top_down_read_features/ablation/` | `pc50_ablation/` (4 definitions' 50-PC subspaces ablated at label tokens), `dot_perhead_unit_sparse_optimisation/` (24-dir learned ablation halves 10-shot acc). |
| `read_write_relationship/bottom_up/` | Injecting the L6 label-mean at dummy slots makes the task's own FV form at L13. Was `mean_read_steering_effect_on_write/`. |
| `read_write_relationship/top_down/fv_presence_heatmaps/` | Read-dir@L3 injection → downstream FV presence + attention-to-slot maps (2 tasks). |

Note: plotting scripts on main and on the unmerged experiment branches still write to the OLD paths;
path fixes for the three scripts in this tree live on branch `worktree-tlr-line-poster` (merge later).

Renames 2026-08-19 (later same day): `feature_locations/read_dir_presence` -> `top_down_read_dir_presence`,
`feature_locations/label_mean_L6_presence` -> `bottom_up_label_mean_L6_presence` (make read-feature
provenance explicit). After merging any old experiment branch, run `bin/fix_69_task_run_paths.sh`
(on branch worktree-tlr-line-poster) to rewrite its scripts to this layout; it prints the few
context-dependent cases needing manual review.
