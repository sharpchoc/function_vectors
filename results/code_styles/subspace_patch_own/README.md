# Cue-token patch with the document's OWN k = 4 counterpart as target (53 code families, alternative direction)

2026-10-07. `write_subspace_patch_own.py` (job `logs/code_styles_subspace_own_job.sh`, 4 pods so1–so4), analysis `write_subspace_patch_own_analyze.py`.
Records `artifacts/style_translation/qwen25_code/steering/subspace_patch_own_k4/so*/<family>.json` (judged, 0 fails).

Same subspace as `../subspace_patch/` (top-r uncentered right singular vectors of the k = 4 pair differences, training documents, per layer 20..28),
but the 0-shot cue token's coordinates are replaced by the coordinates of the SAME held-out document's k = 4 ALTERNATIVE prompt (its stored cue-token
activation), not by the training mean. Arms: own1 / own2 / own10 (rank 1, 2, 10) and `full` (the whole cue activation replaced by the document's
k = 4 alternative activation). `mean1` / `mean2` are the previous run's training-mean patches on the same documents. Success = alternative convention
AND judge OK; 40 held-out 0-shot prompts per family (documents without a stored twin dropped).

Pooled success at layer 28 (best layer for every arm except full, L27 .43): unsteered .14, k = 4 in context .74.

| mean1 | mean2 | own1 | own2 | own10 | full |
|---|---|---|---|---|---|
| .42 | .43 | .41 | .42 | .44 | .42 |

- Own counterpart vs training mean: −.012 (rank 1), −.007 (rank 2); 7 families better by ≥ .10, 8 worse. The per-document target adds nothing.
- Rank ladder with own targets: 1 → 2 → 10 → full = .41 → .42 → .44 → .42. The full replacement raises the first-token margin (3.9 vs 2.9) but
  lowers judge-OK (.62 vs .77), so success does not improve.
- Reading (interpretation): a single-layer cue-token activation from the k = 4 prompt, even the exact one, reproduces only ~.42 of the .74 that the
  demonstrations give in context, so the rest of the in-context effect comes from later layers reading the demonstrations directly, not from the
  cue-token residual at any one layer.
