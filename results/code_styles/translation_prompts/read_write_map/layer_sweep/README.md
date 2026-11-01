# Read → write map: where to take the identification feature from (2026-11-01)

Question: can a different identification (read) site raise the held-out-family R² of the shared linear map above .3–.4, trading some steering
strength? Same recipe and scoring as the paper map (v11): unit (document, k) pair contrasts, ridge, held-out family centroids, R² vs the
training-family mean. Write/execution feature fixed at the cue token, L24.

Data: `artifacts/style_translation/{qwen25_code,qwen25_code_tr}/prompt_pairs_sites/` (capture_pair_sites_layers.py: evidence-token mean,
last evidence token and cue token at every layer 0–28, same rows as prompt_pairs; alignment with the stored L8 read / L24 write checked;
5 of 780 num_separators translation rows drift at L24 (min cos .81, bf16 batch-layout sensitivity), not re-checked further).
Scripts: map_layer_sweep.py (prep / screen at fixed λ / confirm with LOFO λ), map_methods.py, map_diagnostics.py.
Protocol: SCREEN (exploration) on the five paper 66/34 splits → `screen.jsonl`; CONFIRM with the full LOFO protocol on ten FRESH random
66/34 splits never used for screening (`../split_fresh66_s201..210.json`) → `confirm.jsonl`. This folder = original prompts (53 families);
`../../translation_prompts/read_write_map/layer_sweep/` = translation prompts (35 families, `split_v10_*_tr35` / `split_fresh66_*`).

Confirmed (fresh splits, mean ± SD over 10):

| identification feature X (→ cue L24) | original 53 | translation 35 | steering at that site (from earlier sweeps) |
|---|---|---|---|
| evidence mean L8 (paper) | .192 ± .045 | .129 ± .036 | read steering best region (L6–14) |
| evidence mean L19 | .202 ± .050 | .148 ± .032 | read steering still works (tr L20 .46 vs L8 .57) |
| cue L20 | .215 ± .045 | .260 ± .024 | cue steering weak on original (.35 vs ~.30 unsteered), .63 on translation (L24 .75) |
| **evidence mean L8 ⊕ cue L20** | **.311 ± .046** | **.285 ± .026** | both sites usable; concatenation of unit features |
| evidence mean L19 ⊕ cue L20 | .316 ± .049 | .287 ± .026 | |
| cue L22 | .379 ± .040 | .379 ± .022 | no map needed for .16 / .24 of it (identity R²); near-execution |

Screen findings (paper splits): evidence-token layer is flat from L6 to L23 (.23–.25 original, .15–.17 translation); bands, concatenated
evidence layers, last evidence token, PCA-truncated X, other write layers, reduced-rank ridge, family-analogy kernels and shrinkage do not
beat plain ridge; cue L23 → L24 (.75) is identity (identity R² .73).

Diagnostics (`diagnostics.jsonl`): the limit is NEW families, not the map. Held-out DOCUMENTS of seen families: R² .92 (paper map,
both prompt sets; own-centroid reference .98). Oracle projection of held-out write centroids onto the span of the training families'
write centroids: .30 original / .19 translation. Learning curve (8 fixed test families): .13 at 5 training families → .26 at 45, still
rising. Pooling both prompt sets' rows for the same training families: no gain (.231 vs .237).
