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
- **CAVEAT (2026-10-07, after user challenge): the "own counterpart" is NOT the same cue position.** In this corpus the k-shot prompt of a document
  is the document itself continued: the k = 4 prompt ends at the FIFTH occurrence of the construct (the first four occurrences are the
  demonstrations), while the 0-shot prompt ends at the FIRST occurrence. So the patched activation comes from a different construct instance with
  different local content (0 / 200 documents share the cue position between k = 0 and k = 4). Projected on 1–10 convention directions this is
  harmless and explains why own ≈ mean; the full swap imports the wrong local content, which is why judge-OK drops to .62 and its first-token
  log-probs match the real k = 4 prompt in only 5 of 53 families (uniform constructs such as py2_except, bash_test, sql_join_style). The
  earlier reading that "a single-layer cue activation reproduces only .42 of the in-context effect" is therefore NOT supported by this run.
  A true counterpart would need new prompts: demonstrations from OTHER documents prepended to the 0-shot prompt (same cue position).
