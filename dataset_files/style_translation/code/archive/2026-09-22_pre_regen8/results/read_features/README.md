# Read features for the code-convention families (qwen25_code)

`read_vectors.csv`, `read_summary.png`, `read_vs_write.png` from `read_features_summary.py`. Vectors in
`artifacts/style_translation/qwen25_code/read_features/vectors_k3_train/<family>.npz` (`capture_read.py`): mean activation over the
EVIDENCE tokens (`code_evidence.py`, rule of DECISIONS 2026-09-18), per prompt then over prompts, for the same prompts as the write
feature (k = 3, 4 correct completions, training documents, unpaired pools); `mean_nat/mean_alt [28, D]`, `diff`, layer 0, split-half
cosine. Evidence positions per prompt: `read_features/evidence/<family>.json` (k = 1..4, all 56 families).
