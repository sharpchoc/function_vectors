# Poster visuals — train/held-out generalisation

Condensed, label-light versions of the per-task bars in the parent folder, for poster use.
Same underlying numbers (`../train_heldout_summary.csv`); no new compute.
Regenerate with `python src/eval_scripts/plot_69_task_run_poster.py`.

| File | What it shows |
|---|---|
| `headline_bars.png` | The hero figure: mean accuracy for train (55) and held-out (14) tasks plus the held-out unsteered baseline, in the zero-shot and mixed-task mixed-label 10-shot settings. |
| `per_task_lift.png` | All 69 tasks ranked by steered accuracy, each a line from its unsteered baseline to its steered accuracy; held-out tasks in orange. Shows the lift is universal and held-out tasks interleave with train tasks. |
| `poster_summary.png` | Single-slot combination of the two above (bars + zero-shot per-task strip). |
| `selected_heads.png` | Which 37 of the 448 (28 layers × 16) heads the pooled sparse optimisation selected, plus a heads-per-layer marginal. |
| `poster_numbers.csv` | The plotted aggregates. |
| `selected_heads.csv` | The selected (layer, head, c) triples. |

Setup: GPT-J-6B; one shared 37-head set from pooled sparse optimisation (λ=0.005, zero-shot
train metric) fitted on the 55 train tasks of the seed-43 split of the 69-task pool; per-task
FV = unweighted sum of those heads' mean activations; α=1; best-layer accuracy, full-label
teacher-forced readout, 50 queries/task.

Headline numbers (steered vs unsteered):

| Setting | Train (55) | Held-out (14) | Held-out baseline |
|---|---|---|---|
| Zero-shot | 0.75 | 0.73 | 0.09 |
| Mixed-task mixed-label 10-shot | 0.68 | 0.78 | 0.18 |

Every one of the 69 tasks improves under steering in both settings (minimum lift +0.30
zero-shot, +0.14 mixed-task); no task falls below 0.4 zero-shot.

Head set: 37 heads spanning layers 3–27, densest at layers 12–15 (layer 13 contributes 5);
selection coefficients run 0.82–1.00 (threshold c > 0.8), source
`artifacts/sandbox/ext_steerability/prunedfail_seed43/pooled_sparse/selection.json`.
