# Subspace-patch steering at the cue token (py_is_none, Qwen2.5-7B base)

2026-10-02. `write_subspace_patch.py` (job `logs/code_styles_subspace_patch_job.sh`), records in
`artifacts/style_translation/qwen25_code/steering/subspace_patch_k4/s1/py_is_none.json`, subspace in `cue_pairs_layers/py_is_none_k4_subspace.npz`.

- Pair differences d_i = h_nat − h_alt at the cue token of the k = 4 twin prompts, 150 TRAINING documents, per layer 20..28 (layer 28 = un-normed
  output of the last block). V = top-2 right singular vectors, uncentered (PC1 = mean difference, cos ≥ .994).
- Target coordinates: mean over the training documents' k = 4 prompts of each pole, projected on V.
- Steering: 0-shot prompts of the first 40 held-out documents; at ONE layer the cue token's coordinates in the subspace are replaced by the target
  pole's mean coordinates (prefill pass only). Arms: `patch2` (PC1 + PC2), `patch1` (PC1 only), `add2` (reference: add 2 × mean difference), `base`.
- Metric here = convention rate (completion uses the target convention) and first-token margin. **The Gemini judge has NOT run** (OpenRouter key
  rejected, 401); `judge` is null in the records, so these are not yet the usual success numbers (convention AND judge OK).

Results (`py_is_none_by_layer.csv`): unsteered natural .70 / alternative .075. Best patch layer = 25 (patch2 → natural .80, → alternative .475;
patch1 .80 / .325); layers 20–24 the patch does not move the alternative (≤ .15). PC2 adds nothing consistent (its target coordinate is the same for
both poles because PC2 ⟂ mean difference). Adding 2 × mean difference is stronger from layer 23 on (L25 .85 / .50, margin 12 vs 2.5 for the patch).
