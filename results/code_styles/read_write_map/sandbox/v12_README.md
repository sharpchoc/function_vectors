# v12 — coding map without double normalisation (backup, NOT in the paper; 2026-09-25)

v11 (paper) unit-normalises every (document, k) pair contrast before fitting, and scores on family centroids built by
averaging those unit contrasts and normalising again. v12 normalises once: c^q_f = normalize(mean_j raw delta^q_fj).
Same ten split files as v11 (`../split_v10_{66,80}_s{1..5}.json`, 53-family pool), read L8 (layer-7 output) -> write L24
(layer-23 output). Script: `src/eval_scripts/code_map_single_norm.py`; variant-A fits: `read_write_map_code.py` without
`--unit_norm` (tags `v12a_rawpairs_*`). All rows: `v12_single_norm_summary.csv`.

| variant | 66/34 R² (train-mean ref) | 80/20 R² | cos (66/34) | notes |
|---|---|---|---|---|
| v11_ref — paper protocol | **.237 ± .030** | .222 ± .075 | .426 | reproduces v11 exactly |
| v11_on_single — v11 map, single-norm centroids | .238 ± .031 | .224 ± .074 | .422 | scoring change alone is neutral |
| B_centroids — ridge on single-norm family centroids (1 example / family) | .198 ± .028 | .189 ± .078 | .373 | lambda = 0.1 every split (grid 1e-4..1e3) |
| A_rawpairs — ridge on RAW pair contrasts, single-norm scoring | −.380 ± .088 | −.380 ± .094 | .296 | lambda 1e4–1e5; raw-scale R² .087 (66/34) / .092 |

A's R² is computed after normalising its prediction to unit length (targets are unit); v11/B predictions are not
renormalised, so for A the scale-free cosine (.30 vs .43) and raw-scale R² (.09) are the fair comparisons.
Reading: normalising each pair before fitting is what makes the map work — raw pair contrasts vary widely in norm, so an
unnormalised fit is dominated by high-norm pairs. Normalising the family average once vs twice makes no difference.
