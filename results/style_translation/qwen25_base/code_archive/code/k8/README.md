# results/style_translation/qwen25_base/code/k8 — do 8 in-context examples beat 4? (55 code families, Qwen2.5-7B base)

User question 2026-09-16: is k = 8 a game changer? Protocol: only k = 8 was sampled (no k = 5–7). A k = 8 prompt = the document's first 8
opportunities rendered in the context pole, decision at the 9th; only documents with ≥ 9 opportunities qualify (8,049 of 10,059; per-family
n below, 5 families under 50). Cue tokens for every opportunity were already on disk, so the k = 8 items were appended to the existing
prompt files (`build_prompts.py --K 9 --ks 8 --append`) and rolled out with the step-3 protocol (`rollout.py --ks 8 --append`: one seeded
T = 1 sample, 48 tokens, code cut, regex classifier ∧ Gemini judge = correct solution), then appended to the same rollouts files so
`capture_cues.py --ks 8 --out_tag k8` can build a k = 8 write vector directly. The k = 0..4 records are unchanged (verified against
`rollouts_backup_k4/`). Because the k = 8 subset is smaller, the comparison is **paired**: k = 4 accuracy on the same documents vs k = 8.

## Verdict: a small, consistent gain — not a game changer

| context pole | families (n ≥ 50) | k = 4 on the same documents | k = 8 | mean paired Δ | significant gains / losses | capped at k = 8 | unscorable k = 4 → k = 8 |
|---|---|---|---|---|---|---|---|
| nat | 50 | .73 | .76 | +.03 | 9 / 6 | .25 | .19 → .17 |
| alt | 50 | .68 | .72 | +.04 | 7 / 1 | .28 | .20 → .17 |

- Doubling the examples adds .03–.04 on average. The larger gains (Δ ≥ .10, significant): py_const_naming (nat, +0.17, n=195), py_class_naming (nat, +0.44, n=186), py_class_naming (alt, +0.25, n=186), py_private (nat, +0.40, n=200), py_private (alt, +0.47, n=200), py_bool_prefix (nat, +0.17, n=200), py_loop_vars (alt, +0.16, n=108), py_quotes (alt, +0.14, n=173), py2_iter (nat, +0.15, n=58), py2_iter (alt, +0.21, n=58), py_type_hints (nat, +0.20, n=102), py_optional (nat, +0.18, n=79), py_optional (alt, +0.18, n=79), line_wrap (nat, +0.10, n=191), rust_question (alt, +0.16, n=147).
- The k = 4 accuracy on the k = 8 document subset equals the k = 4 accuracy on all documents (.726 vs .725; .679 vs .679), so the subset is not
  biased. Prompts are longer at k = 8 (max 2,024 tokens) and the `capped` rate (completion not finished within 48 tokens) rises to .25–.28.
- Per-family curves with the k = 8 point: `../accuracy_by_k.png`; paired view: `k8_vs_k4.png`; table: `k8_vs_k4.csv`.

## Per family (k = 4 on the same documents → k = 8, paired Δ; * = |Δ| > 1.96 SE)

| family | n (k = 8 docs) | nat context | alt context | capped k = 8 |
|---|---|---|---|---|
| py_snake_camel | 199 | 0.96 → 0.96 (-0.01) | 0.92 → 0.93 (+0.01) | 0.26 |
| js_camel_snake | 200 | 0.89 → 0.94 (+0.06*) | 0.90 → 0.93 (+0.03) | 0.24 |
| py_const_naming | 195 | 0.66 → 0.83 (+0.17*) | 0.60 → 0.61 (+0.01) | 0.46 |
| py_class_naming | 186 | 0.44 → 0.88 (+0.44*) | 0.62 → 0.87 (+0.25*) | 0.24 |
| py_private | 200 | 0.46 → 0.85 (+0.40*) | 0.39 → 0.85 (+0.47*) | 0.27 |
| py_bool_prefix | 200 | 0.57 → 0.75 (+0.17*) | 0.32 → 0.37 (+0.06) | 0.23 |
| py_loop_vars | 108 | 0.56 → 0.58 (+0.03) | 0.50 → 0.66 (+0.16*) | 0.23 |
| js_hungarian | 200 | 0.66 → 0.69 (+0.04) | 0.85 → 0.88 (+0.03) | 0.28 |
| py_abbrev | 188 | 0.72 → 0.79 (+0.07) | 0.74 → 0.82 (+0.07) | 0.25 |
| py_quotes | 173 | 0.82 → 0.81 (-0.01) | 0.72 → 0.86 (+0.14*) | 0.41 |
| js_quotes | 164 | 0.78 → 0.85 (+0.07) | 0.80 → 0.84 (+0.04) | 0.29 |
| py_fstring | 200 | 0.70 → 0.79 (+0.08) | 0.84 → 0.82 (-0.02) | 0.19 |
| js_template | 190 | 0.91 → 0.89 (-0.02) | 0.82 → 0.80 (-0.02) | 0.17 |
| num_separators | 51 | 0.82 → 0.84 (+0.02) | 0.51 → 0.61 (+0.10) | 0.29 |
| hex_constants | 111 | 0.49 → 0.56 (+0.07) | 0.69 → 0.74 (+0.05) | 0.17 |
| float_literals | 87 | 0.51 → 0.54 (+0.03) | 0.53 → 0.64 (+0.12) | 0.34 |
| sql_bool_case | 98 | 0.94 → 0.97 (+0.03) | 0.95 → 0.95 (+0.00) | 0.31 |
| py2_print | 200 | 0.73 → 0.74 (+0.01) | 0.67 → 0.74 (+0.07) | 0.41 |
| py2_iter | 58 | 0.72 → 0.88 (+0.15*) | 0.53 → 0.74 (+0.21*) | 0.12 |
| py2_except | 21 | 0.91 → 0.95 (+0.05) | 0.91 → 0.91 (+0.00) | 0.24 |
| js_var | 107 | 0.63 → 0.46 (-0.17*) | 0.58 → 0.46 (-0.12) | 0.28 |
| js_arrow | 175 | 0.61 → 0.68 (+0.07) | 0.57 → 0.65 (+0.08) | 0.21 |
| js_semicolons | 97 | 0.87 → 0.68 (-0.19*) | 0.55 → 0.55 (+0.00) | 0.21 |
| js_strict_eq | 26 | 0.81 → 0.81 (+0.00) | 0.69 → 0.85 (+0.15) | 0.15 |
| trailing_commas | 68 | 0.84 → 0.71 (-0.13) | 0.63 → 0.53 (-0.10) | 0.28 |
| py_paren_if | 174 | 0.63 → 0.60 (-0.03) | 0.56 → 0.53 (-0.02) | 0.30 |
| py_type_hints | 102 | 0.72 → 0.92 (+0.20*) | 0.79 → 0.83 (+0.04) | 0.27 |
| py_optional | 79 | 0.58 → 0.76 (+0.18*) | 0.59 → 0.77 (+0.18*) | 0.27 |
| py_builtin_generics | 47 | 0.85 → 0.94 (+0.09) | 0.96 → 0.89 (-0.06) | 0.47 |
| py_comprehension | 158 | 0.60 → 0.63 (+0.03) | 0.63 → 0.68 (+0.04) | 0.12 |
| py_not_in | 104 | 0.50 → 0.61 (+0.11) | 0.36 → 0.40 (+0.05) | 0.29 |
| py_is_none | 19 | 0.68 → 0.84 (+0.16) | 0.58 → 0.95 (+0.37*) | 0.21 |
| py_ternary | 147 | 0.45 → 0.51 (+0.06) | 0.54 → 0.55 (+0.01) | 0.11 |
| early_return | 199 | 0.49 → 0.36 (-0.14*) | 0.82 → 0.79 (-0.04) | 0.17 |
| py_join_concat | 148 | 0.65 → 0.66 (+0.01) | 0.52 → 0.59 (+0.07) | 0.15 |
| py_with_open | 199 | 0.57 → 0.60 (+0.03) | 0.86 → 0.83 (-0.03) | 0.16 |
| py_enumerate | 175 | 0.75 → 0.76 (+0.01) | 0.78 → 0.81 (+0.03) | 0.18 |
| py_self_name | 200 | 0.91 → 0.94 (+0.03) | 0.90 → 0.92 (+0.02) | 0.09 |
| py_indent | 183 | 0.93 → 0.93 (-0.01) | 0.55 → 0.60 (+0.05) | 0.27 |
| py_tabs | 189 | 0.92 → 0.95 (+0.04) | 0.91 → 0.94 (+0.03) | 0.26 |
| operator_spaces | 145 | 0.86 → 0.87 (+0.01) | 0.68 → 0.72 (+0.05) | 0.25 |
| comma_space | 169 | 0.86 → 0.89 (+0.03) | 0.72 → 0.69 (-0.03) | 0.41 |
| line_wrap | 191 | 0.67 → 0.78 (+0.10*) | 0.40 → 0.42 (+0.02) | 0.39 |
| blank_lines | 1 | 0.00 → 1.00 (+1.00) | 1.00 → 0.00 (-1.00) | 1.00 |
| docstring_style | 167 | 0.98 → 0.96 (-0.01) | 0.97 → 0.96 (-0.01) | 0.19 |
| docstring_quotes | 191 | 0.99 → 0.83 (-0.16*) | 0.92 → 0.83 (-0.09*) | 0.18 |
| comment_case | 146 | 0.66 → 0.60 (-0.06) | 0.57 → 0.56 (-0.01) | 0.47 |
| comment_language | 186 | 0.62 → 0.48 (-0.14*) | 0.56 → 0.48 (-0.09) | 0.40 |
| c_comment_style | 200 | 0.89 → 0.78 (-0.10*) | 0.78 → 0.76 (-0.01) | 0.28 |
| sql_keyword_case | 200 | 0.89 → 0.93 (+0.04) | 0.87 → 0.89 (+0.01) | 0.21 |
| sql_join_style | 112 | 0.88 → 0.80 (-0.08) | 0.72 → 0.70 (-0.02) | 0.29 |
| r_assignment | 195 | 0.90 → 0.93 (+0.03) | 0.81 → 0.85 (+0.04) | 0.23 |
| rust_question | 147 | 0.80 → 0.82 (+0.02) | 0.58 → 0.74 (+0.16*) | 0.15 |
| bash_test | 196 | 0.87 → 0.85 (-0.02) | 0.77 → 0.74 (-0.03) | 0.33 |
| php_array | 178 | 0.43 → 0.46 (+0.03) | 0.60 → 0.58 (-0.02) | 0.23 |
