# Read → write map (qwen25_code, code-convention families)

`read_write_map_code.py`: per-prompt read (L8 evidence mean) / write (L24 cue) features from `prompt_pairs/`; 80/20 family split
(seed 43: 45 train / 11 test, `split.json`); ridge with intercept, λ by leave-one-family-out CV (pooled predictions, `cv.csv`);
map A = prompt read → family-pole write centroid, map B = read centroid → write centroid. Held-out (variance-weighted) R²:
A .121 (λ 1e5), B .111 (λ 1e2); controls: constant −.131, shuffled targets −.309. Mean cos of the predicted vs true nat−alt write
direction on test families: A .26, B .27. `results.json`, `test_predictions.npz`, `cv_curve.png`, `test_scatter.png`.
