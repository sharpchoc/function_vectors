# Task-level linear mapping: read feature -> task FV (all 28 layers)

Task-level ridge regression treating each TASK as one sample (2026-08-20):

- Features: task-mean label-token residual activation at layer L
  (`artifacts/69_task_run/label_resid_means/<task>.pt`, `resid_means[L]`, 4096-d;
  same capture used for read-feature steering).
- Targets: task FV = mean of the 150 per-prompt FVs
  (`artifacts/69_task_run/perprompt_fvs/<task>.pt`).
- Fit: 55 train tasks (split `task_splits/extended_steerable_69_prunedfail.json`),
  dual/kernel ridge with intercept (features and targets centered on train stats),
  lambda chosen by leave-one-task-out CV on the train tasks (grid 1e-2..1e6).
- Scored on the 14 held-out tasks. R^2 conventions:
  - `heldout_r2_trainmean`: denominator = variance around the train-mean FV
    (split-average-FV predictor; comparable to the per-prompt layer sweep in
    `FV_linear_decodability/labeltoken_fv_ridge/layer_sweep/`).
  - `heldout_r2_testmean`: standard R^2, denominator = variance around the
    14 test tasks' own mean FV.
  - `loo_train_r2`: honest train-side number (in-sample R^2 is 1.000 for
    n=55 << d=4096 at small lambda — the fit interpolates).

## Findings

- Held-out R^2 (train-mean baseline) rises steeply L0 0.464 -> L6 0.669, broad
  plateau L6-L20 peaking 0.683 at L12-L13, gentle decay to 0.652 at L27 —
  task-identity information at label tokens never leaves the residual stream.
- Harsh regularisation is NOT needed despite n=55 << d=4096: LOO CV picks the
  smallest lambda from L6 onward (min-norm interpolation regime); explicit
  shrinkage only hurts (L6: lambda 1e3 -> R^2 0.55, 1e4 -> 0.27). Only L0-L5
  prefer moderate lambda (10-32).
- Matches the per-prompt ridge where they overlap (per-prompt test-centroid peak
  0.692 at L13 vs 0.683 here): the 55 task centroids already carry all the
  transferable signal — consistent with the centroid-memorization finding.
- L6 per-task pattern (see `tasklevel_ridge.py` output): morphology/translation
  near-perfect (ends_with_ing cos 0.995), classification-like worst (ag_news 0.63).
- Mean cos(predicted, true FV) on held-out: 0.82 (L0) -> 0.90 (plateau), vs 0.64
  for the train-mean baseline.

## Files

- `tasklevel_ridge_all_layers.csv` — per-layer: best lambda, LOO train R^2,
  held-out R^2 (both conventions), held-out mean cos.
- `tasklevel_ridge_r2_by_layer.png` — held-out R^2 vs layer (+ LOO train curve).
- `tasklevel_ridge.py` — single-layer (L6) version with per-task breakdown and
  lambda-sensitivity printout.
- `tasklevel_ridge_all_layers.py` — the 28-layer sweep (writes the csv).
- `plot_tasklevel_ridge_layers.py` — renders the png from the csv.

Scripts are self-contained (read artifacts directly, write to their own dir when
paths are adjusted); they were run from a scratch dir, so the hardcoded TMP output
paths need pointing here to regenerate.
