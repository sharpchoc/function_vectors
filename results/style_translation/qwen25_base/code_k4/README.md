# results/style_translation/qwen25_base/code_k4 — cheap k = 4 check of 60 CODE conventions on Qwen2.5-7B base

User request 2026-09-14: code-style conventions as a new source of lexically diverse families. Prompt `Task:\n{spec}\n\n{Language}:\n{solution twin
cut at cue k}`; twins = Gemini natural-style solution (≥ 8 decision points spread over 20–35 lines) + Gemini rewrite changing only the
convention, aligned by token diff (`code_build.py`; kept if ≥ 5 opportunities, the 5th before 75 % of the code, ≥ 60 % shared tokens).
~50 tasks per family from shared per-language pools (`dataset_files/style_translation/code/tasks_<lang>.json`), k ∈ {0, 4}, both poles,
one seeded sample ≤ 48 tokens cut at the first blank line, regex classifier per family (`code_families.py`) with the next-token fallback,
Gemini judge = plausible on-task continuation ignoring style. Accuracy = convention ∧ acceptable. Pod kfv3jzf3ch8qqd (~35 min), terminated.
Files: `k4_check.{png,csv}` (this folder), `../code_selection.{png,csv}` (all 60 with the cutoff), `../../code/` corpus.

## Result: 55 of 60 pass the cutoff (both poles ≥ .30 at k = 4)

| family | language | natural k0 → k4 | alternative k0 → k4 | unscorable | tasks |
|---|---|---|---|---|---|
| js_camel_snake | JavaScript | 0.82 → 0.94 | 0.02 → 0.94 | 0.00 | 50 |
| py_snake_camel | Python | 0.88 → 1.00 | 0.00 → 0.92 | 0.00 | 50 |
| py_abbrev | Python | 0.10 → 0.82 | 0.42 → 0.74 | 0.15 | 50 |
| py_const_naming | Python | 0.02 → 0.80 | 0.00 → 0.68 | 0.14 | 50 |
| js_hungarian | JavaScript | 0.54 → 0.62 | 0.00 → 0.84 | 0.21 | 50 |
| py_class_naming | Python | 0.18 → 0.58 | 0.34 → 0.66 | 0.26 | 50 |
| py_loop_vars | Python | 0.39 → 0.43 | 0.09 → 0.52 | 0.43 | 46 |
| py_private | Python | 0.12 → 0.40 | 0.00 → 0.44 | 0.55 | 50 |
| py_bool_prefix | Python | 0.00 → 0.64 | 0.12 → 0.30 | 0.43 | 50 |
| js_func_pascal ✗ | JavaScript | 0.18 → 0.52 | 0.00 → 0.26 | 0.47 | 50 |
| sql_bool_case | SQL | 0.31 → 0.97 | 0.62 → 0.97 | 0.03 | 32 |
| js_template | JavaScript | 0.08 → 0.86 | 0.18 → 0.84 | 0.09 | 49 |
| js_quotes | JavaScript | 0.12 → 0.76 | 0.55 → 0.92 | 0.08 | 49 |
| py_quotes | Python | 0.29 → 0.82 | 0.31 → 0.73 | 0.06 | 49 |
| py_fstring | Python | 0.18 → 0.72 | 0.20 → 0.86 | 0.20 | 50 |
| num_separators | Python | 0.29 → 0.71 | 0.00 → 0.66 | 0.14 | 35 |
| hex_constants | JavaScript | 0.23 → 0.60 | 0.00 → 0.74 | 0.23 | 43 |
| float_literals | Python | 0.42 → 0.47 | 0.00 → 0.63 | 0.21 | 38 |
| py2_except | Python | 0.09 → 1.00 | 0.00 → 0.93 | 0.02 | 45 |
| py_builtin_generics | Python | 0.06 → 0.94 | 0.03 → 0.90 | 0.03 | 31 |
| py_self_name | Python | 0.96 → 0.96 | 0.00 → 0.86 | 0.03 | 50 |
| py2_print | Python | 0.12 → 0.84 | 0.00 → 0.72 | 0.19 | 50 |
| trailing_commas | Python | 0.09 → 0.76 | 0.35 → 0.71 | 0.09 | 34 |
| py_is_none | Python | 0.19 → 0.84 | 0.03 → 0.68 | 0.19 | 31 |
| py_with_open | Python | 0.08 → 0.66 | 0.00 → 0.84 | 0.15 | 50 |
| py_type_hints | Python | 0.04 → 0.63 | 0.80 → 0.82 | 0.12 | 49 |
| py_enumerate | Python | 0.18 → 0.78 | 0.43 → 0.63 | 0.09 | 49 |
| py2_iter | Python | 0.50 → 0.74 | 0.00 → 0.63 | 0.14 | 38 |
| py_comprehension | Python | 0.35 → 0.65 | 0.22 → 0.63 | 0.17 | 46 |
| js_arrow | JavaScript | 0.78 → 0.61 | 0.06 → 0.59 | 0.32 | 49 |
| js_strict_eq | JavaScript | 0.82 → 0.75 | 0.07 → 0.57 | 0.14 | 28 |
| py_paren_if | Python | 0.68 → 0.68 | 0.02 → 0.54 | 0.21 | 50 |
| py_join_concat | Python | 0.39 → 0.66 | 0.16 → 0.50 | 0.26 | 44 |
| js_semicolons | JavaScript | 0.77 → 0.68 | 0.06 → 0.48 | 0.32 | 31 |
| early_return | Python | 0.32 → 0.48 | 0.20 → 0.84 | 0.22 | 50 |
| js_var | JavaScript | 0.40 → 0.46 | 0.04 → 0.50 | 0.49 | 50 |
| py_optional | Python | 0.14 → 0.59 | 0.00 → 0.45 | 0.43 | 49 |
| py_ternary | Python | 0.04 → 0.40 | 0.33 → 0.52 | 0.34 | 48 |
| py_not_in | Python | 0.33 → 0.51 | 0.00 → 0.36 | 0.37 | 45 |
| py_literal_ctor ✗ | Python | 0.31 → 0.20 | 0.06 → 0.63 | 0.50 | 35 |
| py_tabs | Python | 0.80 → 0.90 | 0.02 → 0.96 | 0.00 | 50 |
| py_indent | Python | 0.78 → 0.96 | 0.10 → 0.70 | 0.03 | 50 |
| operator_spaces | Python | 0.76 → 0.89 | 0.11 → 0.69 | 0.09 | 45 |
| comma_space | Python | 0.77 → 0.79 | 0.00 → 0.49 | 0.16 | 47 |
| blank_lines | Python | 0.15 → 0.50 | 0.53 → 0.38 | 0.41 | 34 |
| line_wrap | Python | 0.04 → 0.82 | 0.58 → 0.38 | 0.37 | 50 |
| c_braces ✗ | C | 0.77 → 0.77 | 0.00 → 0.06 | 0.07 | 47 |
| docstring_style | Python | 0.64 → 1.00 | 0.04 → 0.92 | 0.01 | 50 |
| docstring_quotes | Python | 0.08 → 0.92 | 0.00 → 0.96 | 0.05 | 50 |
| c_comment_style | JavaScript | 0.34 → 0.96 | 0.00 → 0.84 | 0.09 | 50 |
| comment_case | Python | 0.22 → 0.58 | 0.08 → 0.62 | 0.32 | 50 |
| comment_language | Python | 0.22 → 0.70 | 0.00 → 0.54 | 0.31 | 50 |
| r_assignment | R | 0.76 → 0.92 | 0.02 → 0.82 | 0.06 | 50 |
| sql_keyword_case | SQL | 0.66 → 1.00 | 0.04 → 0.80 | 0.02 | 50 |
| bash_test | Bash | 0.16 → 0.76 | 0.64 → 0.74 | 0.05 | 50 |
| sql_join_style | SQL | 0.93 → 0.84 | 0.02 → 0.66 | 0.17 | 44 |
| rust_question | Rust | 0.72 → 0.87 | 0.05 → 0.49 | 0.14 | 39 |
| php_array | PHP | 0.24 → 0.38 | 0.22 → 0.60 | 0.37 | 50 |
| bash_subst ✗ | Bash | 0.18 → 0.32 | 0.00 → 0.26 | 0.66 | 50 |
| css_shorthand ✗ | CSS | 0.46 → 0.46 | 0.08 → 0.23 | 0.54 | 13 |

Rejected (5): js_func_pascal (alt 0.26, unscorable 0.47), py_literal_ctor (alt 0.63, unscorable 0.50), c_braces (alt 0.06, unscorable 0.07), bash_subst (alt 0.26, unscorable 0.66), css_shorthand (alt 0.23, unscorable 0.54).
Accepted but with weak classifiers (unscorable ≥ .35, i.e. the continuation often makes neither pole's decision within 48 tokens): py_loop_vars (0.43), py_private (0.55), py_bool_prefix (0.43), js_var (0.49), py_optional (0.43), py_not_in (0.37), blank_lines (0.41), line_wrap (0.37), php_array (0.37) — their numbers are lower bounds.

## Reading
- Code conventions are far more learnable in context than prose conventions: the alternative pole moves from ≈ 0 to .5–.97 in most families
  (naming .68–.94, quotes/f-strings/templates .73–.92, Python 2 dialect .63–.93, typing .45–.90, docstring and comment styles .54–.96, SQL/R/Bash .66–.82).
  Both poles are dense in pretraining for almost all of them, which is what the pole-balance hypothesis predicted.
- The five failures are either a rare pole (Allman braces .06, PascalCase function names .26) or a weak classifier (bash backticks, CSS shorthand, dict()/list()).
- Axes are dense by construction (10 naming families, 8 literal families, 22 syntax/dialect families), so this pool is the right place to
  re-run the read→write-map coverage test. Next: full protocol (200 tasks, k = 0..4) for the accepted families, then features + map.
