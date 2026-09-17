# results/style_translation/qwen25_base/code/read_steer — evidence-token (read-feature) steering of 3-shot code prompts, 55 coding-convention families

Qwen2.5-7B (base). Step 7 of the style-translation study applied to code with the 2026-09-15 decisions (DECISIONS.md): read vector
u = r_alt − r_nat from `../read_features/` (paired evidence-token means of the k = 4 prompts), added with ±α at **every evidence token of
the 3 earlier in-context opportunities** of a 3-shot prompt (never at the cue). Directions: nat context → steered toward alt
(`nat2alt`), alt context → toward nat (`alt2nat`). Screen: all layers 0–27 × α ∈ {0.5, 1, 2, 4} on 50 prompts, style only
(`read_steer_layer_alpha.png`, `read_steer_layer_alpha_full.png`, `read_steer_layer_mean.png`, `screen.csv`, `screen_full.csv`,
`best_layer_full.csv`); confirm: top-2 cells per direction on all prompts of the family (≤ 200; 48 tokens, code cut, regex classifier ∧
Gemini 2.5 Flash judge = the completion is still a correct solution). **No control arm** (user decision).

Accuracy = target convention at the next decision AND correct solution. Unsteered = the same 3-shot prompt, scored toward the target
(how often the model already deviates from its context). Reference ("flipped reference") = step-3 accuracy at k = 3 of prompts whose
context genuinely shows the target (`../summary.csv`). Gate columns in `best_config.csv`: `reach` = steered / reference, `pass_50` =
reach ≥ .5 (default rule — the user judges the final results), `sig_vs_unsteered` = Wilson CI of the steered accuracy excludes the unsteered rate.
Figures: `summary_bars.png` (three-bar summary), `steered_vs_unsteered.png` (one row per family), `read_steer_summary.png` (bar grid).

## Result (55 families, k = 3)

| direction | unsteered | steered (best L, α) | reference | mean reach | pass_50 | significant lift | steered ≥ .5 | ≥ .7 |
|---|---|---|---|---|---|---|---|---|
| nat context → alt | 0.03 | 0.53 | 0.68 | 0.77 | 44 / 55 | 55 / 55 | 34 | 16 |
| alt context → nat | 0.07 | 0.61 | 0.73 | 0.81 | 50 / 55 | 55 / 55 | 36 | 21 |

- Both directions pass the 50 % reach rule in **43 / 55** families; significant lift in both directions in 55 / 55.
- Best injection layer is early: 76% of the confirmed best cells are at L0–L9 (screen mean over families: plateau L4–L12, collapse after Shared read layer fixed at **L8** by user decision (2026-09-15).
  L22 — `read_steer_layer_mean.png`). The write feature of the same families sits at L24 (`../steering/`): the read site precedes the write site.
- Families failing the both-direction rule: comment_language, early_return, js_hungarian, py_abbrev, py_bool_prefix, py_comprehension, py_join_concat, py_loop_vars, py_not_in, py_ternary, py_with_open, rust_question.

## Per family (unsteered → steered / reference, best layer and α; ✓ = reach ≥ .5, ✗ = below, "ns" = CI does not exclude unsteered)

| family | nat / alt (language) | n | nat context → alt | alt context → nat |
|---|---|---|---|---|
| py_snake_camel | snake_case identifiers / camelCase identifiers (Python) | 200 | 0.00 → **0.86** / ref 0.92 (L4, 1) ✓ | 0.03 → **0.89** / ref 0.97 (L3, 2) ✓ |
| js_camel_snake | camelCase identifiers / snake_case identifiers (JavaScript) | 200 | 0.00 → **0.87** / ref 0.89 (L2, 2) ✓ | 0.01 → **0.87** / ref 0.92 (L2, 1) ✓ |
| py_const_naming | UPPER_SNAKE constants / camelCase constants (Python) | 200 | 0.00 → **0.41** / ref 0.47 (L11, 4) ✓ | 0.01 → **0.60** / ref 0.67 (L4, 2) ✓ |
| py_class_naming | PascalCase class names / snake_case class names (Python) | 200 | 0.18 → **0.37** / ref 0.56 (L12, 1) ✓ | 0.01 → **0.44** / ref 0.50 (L7, 1) ✓ |
| py_private | single-underscore private attributes / double-underscore private attributes (Python) | 200 | 0.00 → **0.41** / ref 0.35 (L1, 4) ✓ | 0.04 → **0.39** / ref 0.43 (L0, 1) ✓ |
| py_bool_prefix | is_/has_ boolean names / bare boolean names (Python) | 200 | 0.01 → **0.20** / ref 0.42 (L5, 4) ✗ | 0.00 → **0.34** / ref 0.82 (L8, 4) ✗ |
| py_loop_vars | single-letter loop variables / descriptive loop variables (Python) | 188 | 0.01 → **0.12** / ref 0.51 (L22, 4) ✗ | 0.04 → **0.40** / ref 0.49 (L6, 2) ✓ |
| js_hungarian | plain identifiers / Hungarian notation (JavaScript) | 200 | 0.00 → **0.18** / ref 0.84 (L5, 4) ✗ | 0.03 → **0.47** / ref 0.68 (L16, 4) ✓ |
| py_abbrev | abbreviated identifiers / spelled-out identifiers (Python) | 200 | 0.05 → **0.41** / ref 0.71 (L1, 2) ✓ | 0.03 → **0.32** / ref 0.67 (L7, 4) ✗ |
| py_quotes | single-quoted strings / double-quoted strings (Python) | 192 | 0.06 → **0.84** / ref 0.85 (L0, 2) ✓ | 0.12 → **0.83** / ref 0.86 (L8, 2) ✓ |
| js_quotes | double-quoted strings / single-quoted strings (JavaScript) | 189 | 0.06 → **0.85** / ref 0.81 (L0, 1) ✓ | 0.02 → **0.80** / ref 0.83 (L6, 2) ✓ |
| py_fstring | f-strings / str.format (Python) | 200 | 0.04 → **0.56** / ref 0.77 (L6, 4) ✓ | 0.06 → **0.56** / ref 0.69 (L7, 4) ✓ |
| js_template | template literals / string concatenation (JavaScript) | 198 | 0.01 → **0.52** / ref 0.70 (L2, 4) ✓ | 0.04 → **0.59** / ref 0.84 (L6, 4) ✓ |
| num_separators | plain numeric literals / underscore digit separators (Python) | 131 | 0.01 → **0.68** / ref 0.55 (L22, 4) ✓ | 0.22 → **0.79** / ref 0.77 (L0, 0.5) ✓ |
| hex_constants | decimal constants / hexadecimal constants (JavaScript) | 177 | 0.00 → **0.63** / ref 0.70 (L7, 4) ✓ | 0.06 → **0.32** / ref 0.53 (L15, 2) ✓ |
| float_literals | full float literals / abbreviated float literals (Python) | 162 | 0.22 → **0.50** / ref 0.50 (L17, 2) ✓ | 0.14 → **0.45** / ref 0.54 (L3, 4) ✓ |
| sql_bool_case | lowercase true/false / uppercase TRUE/FALSE (SQL) | 133 | 0.01 → **0.92** / ref 0.95 (L10, 1) ✓ | 0.00 → **0.93** / ref 0.90 (L0, 1) ✓ |
| py2_print | print() function / print statement (Python) | 200 | 0.00 → **0.58** / ref 0.68 (L8, 4) ✓ | 0.00 → **0.57** / ref 0.77 (L8, 4) ✓ |
| py2_iter | range/items/keys / xrange/iteritems/iterkeys (Python) | 144 | 0.00 → **0.50** / ref 0.59 (L8, 4) ✓ | 0.22 → **0.79** / ref 0.79 (L14, 2) ✓ |
| py2_except | except E as e / except E, e (Python) | 177 | 0.00 → **0.93** / ref 0.92 (L2, 1) ✓ | 0.01 → **0.95** / ref 0.97 (L4, 1) ✓ |
| js_var | const/let / var (JavaScript) | 200 | 0.01 → **0.66** / ref 0.61 (L1, 4) ✓ | 0.04 → **0.61** / ref 0.69 (L10, 2) ✓ |
| js_arrow | arrow functions / function expressions (JavaScript) | 196 | 0.01 → **0.26** / ref 0.48 (L7, 4) ✓ | 0.05 → **0.50** / ref 0.66 (L5, 4) ✓ |
| js_semicolons | semicolons / no semicolons (JavaScript) | 146 | 0.03 → **0.68** / ref 0.66 (L5, 4) ✓ | 0.04 → **0.77** / ref 0.76 (L0, 1) ✓ |
| js_strict_eq | === / !== / == / != (JavaScript) | 107 | 0.00 → **0.70** / ref 0.68 (L23, 4) ✓ | 0.17 → **0.75** / ref 0.79 (L4, 1) ✓ |
| trailing_commas | trailing commas / no trailing commas (Python) | 130 | 0.17 → **0.71** / ref 0.66 (L2, 4) ✓ | 0.23 → **0.65** / ref 0.77 (L4, 4) ✓ |
| py_paren_if | if x: / if (x): (Python) | 199 | 0.01 → **0.50** / ref 0.75 (L17, 4) ✓ | 0.06 → **0.60** / ref 0.80 (L2, 2) ✓ |
| py_type_hints | type hints / no type hints (Python) | 192 | 0.24 → **0.77** / ref 0.77 (L12, 2) ✓ | 0.01 → **0.40** / ref 0.65 (L4, 2) ✓ |
| py_optional | X | None / Optional[X] (Python) | 191 | 0.00 → **0.29** / ref 0.48 (L5, 4) ✓ | 0.08 → **0.46** / ref 0.55 (L17, 4) ✓ |
| py_builtin_generics | list[int] builtins / typing.List (Python) | 113 | 0.00 → **0.81** / ref 0.89 (L5, 1) ✓ | 0.00 → **0.82** / ref 0.81 (L0, 2) ✓ |
| py_comprehension | list comprehensions / explicit loops (Python) | 194 | 0.02 → **0.08** / ref 0.60 (L1, 4) ✗ | 0.10 → **0.20** / ref 0.63 (L18, 4) ✗ |
| py_not_in | x not in y / not x in y (Python) | 164 | 0.06 → **0.13** / ref 0.41 (L18, 4) ✗ | 0.10 → **0.26** / ref 0.49 (L14, 4) ✓ |
| py_is_none | is None / == None (Python) | 114 | 0.00 → **0.51** / ref 0.63 (L0, 2) ✓ | 0.09 → **0.67** / ref 0.65 (L2, 2) ✓ |
| py_ternary | conditional expressions / if/else blocks (Python) | 187 | 0.12 → **0.17** / ref 0.76 (L1, 4) ✗ | 0.02 → **0.05** / ref 0.63 (L2, 4) ✗ |
| early_return | guard clauses / nested if/else (Python) | 200 | 0.23 → **0.33** / ref 0.78 (L1, 4) ✗ | 0.08 → **0.23** / ref 0.38 (L10, 2) ✓ |
| py_join_concat | str.join / + concatenation (Python) | 176 | 0.03 → **0.25** / ref 0.58 (L3, 4) ✗ | 0.09 → **0.31** / ref 0.62 (L13, 4) ✗ |
| py_with_open | with open(...) / open() / close() (Python) | 200 | 0.03 → **0.29** / ref 0.76 (L5, 4) ✗ | 0.04 → **0.54** / ref 0.71 (L5, 4) ✓ |
| py_enumerate | enumerate / range(len()) (Python) | 198 | 0.09 → **0.32** / ref 0.62 (L1, 4) ✓ | 0.16 → **0.42** / ref 0.65 (L17, 4) ✓ |
| py_self_name | self / this (Python) | 200 | 0.00 → **0.86** / ref 0.90 (L0, 2) ✓ | 0.00 → **0.86** / ref 0.89 (L4, 1) ✓ |
| py_indent | 4-space indentation / 2-space indentation (Python) | 200 | 0.00 → **0.69** / ref 0.53 (L1, 4) ✓ | 0.41 → **0.93** / ref 0.95 (L1, 1) ✓ |
| py_tabs | spaces / tabs (Python) | 200 | 0.00 → **0.75** / ref 0.93 (L9, 4) ✓ | 0.01 → **0.69** / ref 0.94 (L2, 2) ✓ |
| operator_spaces | spaces around operators / no spaces around operators (Python) | 187 | 0.01 → **0.60** / ref 0.60 (L9, 4) ✓ | 0.26 → **0.74** / ref 0.81 (L2, 2) ✓ |
| comma_space | space after comma / no space after comma (Python) | 192 | 0.01 → **0.57** / ref 0.59 (L9, 4) ✓ | 0.21 → **0.72** / ref 0.84 (L9, 2) ✓ |
| line_wrap | wrapped long calls / single long lines (Python) | 200 | 0.11 → **0.20** / ref 0.37 (L7, 2) ✓ | 0.00 → **0.35** / ref 0.61 (L8, 4) ✓ |
| blank_lines | two blank lines between defs / one blank line (Python) | 142 | 0.02 → **0.69** / ref 0.69 (L1, 1) ✓ | 0.00 → **0.68** / ref 0.65 (L0, 1) ✓ |
| docstring_style | Google docstrings / NumPy docstrings (Python) | 199 | 0.00 → **0.56** / ref 0.94 (L4, 4) ✓ | 0.01 → **0.87** / ref 0.97 (L13, 4) ✓ |
| docstring_quotes | triple double quotes / triple single quotes (Python) | 200 | 0.00 → **0.96** / ref 0.96 (L0, 1) ✓ | 0.00 → **0.95** / ref 0.98 (L1, 1) ✓ |
| comment_case | capitalised comments / lowercase comments (Python) | 200 | 0.01 → **0.59** / ref 0.62 (L4, 2) ✓ | 0.04 → **0.65** / ref 0.71 (L4, 1) ✓ |
| comment_language | English comments / Spanish comments (Python) | 200 | 0.00 → **0.23** / ref 0.60 (L2, 2) ✗ | 0.04 → **0.56** / ref 0.67 (L17, 2) ✓ |
| c_comment_style | // comments / /* */ comments (JavaScript) | 200 | 0.00 → **0.69** / ref 0.74 (L17, 2) ✓ | 0.01 → **0.79** / ref 0.73 (L0, 2) ✓ |
| sql_keyword_case | uppercase keywords / lowercase keywords (SQL) | 200 | 0.00 → **0.80** / ref 0.76 (L8, 2) ✓ | 0.01 → **0.90** / ref 0.94 (L10, 1) ✓ |
| sql_join_style | explicit JOIN ... ON / implicit comma join (SQL) | 183 | 0.00 → **0.37** / ref 0.60 (L9, 4) ✓ | 0.12 → **0.60** / ref 0.86 (L3, 4) ✓ |
| r_assignment | <- assignment / = assignment (R) | 200 | 0.00 → **0.80** / ref 0.79 (L8, 2) ✓ | 0.03 → **0.93** / ref 0.89 (L0, 1) ✓ |
| rust_question | ? operator / explicit match (Rust) | 165 | 0.01 → **0.03** / ref 0.82 (L13, 4) ✗ | 0.05 → **0.47** / ref 0.79 (L7, 4) ✓ |
| bash_test | [[ ]] tests / [ ] tests (Bash) | 200 | 0.01 → **0.90** / ref 0.92 (L2, 2) ✓ | 0.01 → **0.83** / ref 0.88 (L13, 1) ✓ |
| php_array | [] arrays / array() arrays (PHP) | 193 | 0.03 → **0.31** / ref 0.41 (L3, 4) ✓ | 0.04 → **0.36** / ref 0.40 (L2, 2) ✓ |
