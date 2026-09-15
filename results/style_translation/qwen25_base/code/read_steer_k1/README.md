# results/style_translation/qwen25_base/code/read_steer_k1 — evidence-token (read-feature) steering of 1-shot code prompts, 55 coding-convention families

Qwen2.5-7B (base). Step 7 of the style-translation study applied to code with the 2026-09-15 decisions (DECISIONS.md): read vector
u = r_alt − r_nat from `../read_features/` (paired evidence-token means of the k = 4 prompts), added with ±α at **every evidence token of
the 1 earlier in-context opportunity** of a 1-shot prompt (never at the cue). Directions: nat context → steered toward alt
(`nat2alt`), alt context → toward nat (`alt2nat`). Screen: all layers 0–27 × α ∈ {0.5, 1, 2, 4} on 50 prompts, style only
(`read_steer_layer_alpha.png`, `read_steer_layer_alpha_full.png`, `read_steer_layer_mean.png`, `screen.csv`, `screen_full.csv`,
`best_layer_full.csv`); confirm: top-2 cells per direction on all prompts of the family (≤ 200; 48 tokens, code cut, regex classifier ∧
Gemini 2.5 Flash judge = the completion is still a correct solution). **No control arm** (user decision).

Accuracy = target convention at the next decision AND correct solution. Unsteered = the same 1-shot prompt, scored toward the target
(how often the model already deviates from its context). Reference ("flipped reference") = step-3 accuracy at k = 1 of prompts whose
context genuinely shows the target (`../summary.csv`). Gate columns in `best_config.csv`: `reach` = steered / reference, `pass_50` =
reach ≥ .5 (default rule — the user judges the final results), `sig_vs_unsteered` = Wilson CI of the steered accuracy excludes the unsteered rate.
Figures: `summary_bars.png` (three-bar summary), `steered_vs_unsteered.png` (one row per family), `read_steer_summary.png` (bar grid).

## Result (55 families, k = 1)

| direction | unsteered | steered (best L, α) | reference | mean reach | pass_50 | significant lift | steered ≥ .5 | ≥ .7 |
|---|---|---|---|---|---|---|---|---|
| nat context → alt | 0.05 | 0.49 | 0.59 | 0.84 | 46 / 55 | 51 / 55 | 25 | 13 |
| alt context → nat | 0.10 | 0.57 | 0.66 | 0.87 | 50 / 55 | 53 / 55 | 34 | 21 |

- Both directions pass the 50 % reach rule in **43 / 55** families; significant lift in both directions in 50 / 55.
- Best injection layer is early: 79% of the confirmed best cells are at L0–L9 (screen mean over families: plateau L4–L12, collapse after
  L22 — `read_steer_layer_mean.png`). The write feature of the same families sits at L24 (`../steering/`): the read site precedes the write site.
- Families failing the both-direction rule: comment_language, js_hungarian, py_abbrev, py_comprehension, py_enumerate, py_join_concat, py_loop_vars, py_not_in, py_optional, py_ternary, py_with_open, rust_question.

## Per family (unsteered → steered / reference, best layer and α; ✓ = reach ≥ .5, ✗ = below, "ns" = CI does not exclude unsteered)

| family | nat / alt (language) | n | nat context → alt | alt context → nat |
|---|---|---|---|---|
| py_snake_camel | snake_case identifiers / camelCase identifiers (Python) | 200 | 0.00 → **0.69** / ref 0.73 (L3, 1) ✓ | 0.15 → **0.94** / ref 0.96 (L3, 2) ✓ |
| js_camel_snake | camelCase identifiers / snake_case identifiers (JavaScript) | 200 | 0.00 → **0.57** / ref 0.55 (L2, 2) ✓ | 0.17 → **0.73** / ref 0.69 (L8, 4) ✓ |
| py_const_naming | UPPER_SNAKE constants / camelCase constants (Python) | 200 | 0.00 → **0.34** / ref 0.37 (L1, 2) ✓ | 0.01 → **0.51** / ref 0.65 (L6, 2) ✓ |
| py_class_naming | PascalCase class names / snake_case class names (Python) | 200 | 0.15 → **0.34** / ref 0.44 (L6, 2) ✓ | 0.04 → **0.35** / ref 0.35 (L5, 2) ✓ |
| py_private | single-underscore private attributes / double-underscore private attributes (Python) | 200 | 0.00 → **0.43** / ref 0.42 (L7, 2) ✓ | 0.01 → **0.43** / ref 0.47 (L1, 1) ✓ |
| py_bool_prefix | is_/has_ boolean names / bare boolean names (Python) | 200 | 0.01 → **0.38** / ref 0.55 (L2, 4) ✓ | 0.01 → **0.47** / ref 0.84 (L0, 4) ✓ |
| py_loop_vars | single-letter loop variables / descriptive loop variables (Python) | 188 | 0.02 → **0.29** / ref 0.60 (L11, 2) ✗ | 0.04 → **0.54** / ref 0.61 (L1, 1) ✓ |
| js_hungarian | plain identifiers / Hungarian notation (JavaScript) | 200 | 0.01 → **0.13** / ref 0.68 (L3, 4) ✗ | 0.17 → **0.51** / ref 0.57 (L1, 4) ✓ |
| py_abbrev | abbreviated identifiers / spelled-out identifiers (Python) | 200 | 0.14 → **0.42** / ref 0.58 (L2, 2) ✓ | 0.06 → **0.16** / ref 0.44 (L1, 2) ✗ |
| py_quotes | single-quoted strings / double-quoted strings (Python) | 192 | 0.02 → **0.84** / ref 0.82 (L1, 1) ✓ | 0.05 → **0.82** / ref 0.87 (L0, 2) ✓ |
| js_quotes | double-quoted strings / single-quoted strings (JavaScript) | 187 | 0.05 → **0.84** / ref 0.87 (L0, 2) ✓ | 0.03 → **0.77** / ref 0.83 (L15, 2) ✓ |
| py_fstring | f-strings / str.format (Python) | 200 | 0.01 → **0.61** / ref 0.34 (L21, 4) ✓ | 0.05 → **0.44** / ref 0.67 (L0, 4) ✓ |
| js_template | template literals / string concatenation (JavaScript) | 198 | 0.01 → **0.50** / ref 0.53 (L21, 4) ✓ | 0.03 → **0.77** / ref 0.89 (L13, 4) ✓ |
| num_separators | plain numeric literals / underscore digit separators (Python) | 131 | 0.01 → **0.61** / ref 0.46 (L22, 4) ✓ | 0.23 → **0.63** / ref 0.68 (L6, 2) ✓ |
| hex_constants | decimal constants / hexadecimal constants (JavaScript) | 177 | 0.01 → **0.32** / ref 0.58 (L9, 4) ✓ | 0.16 → **0.40** / ref 0.55 (L15, 2) ✓ |
| float_literals | full float literals / abbreviated float literals (Python) | 162 | 0.15 → **0.34** / ref 0.32 (L14, 4) ✓ | 0.24 → **0.37** / ref 0.46 (L7, 1) ✓ |
| sql_bool_case | lowercase true/false / uppercase TRUE/FALSE (SQL) | 133 | 0.00 → **0.93** / ref 0.94 (L5, 1) ✓ | 0.01 → **0.93** / ref 0.91 (L5, 1) ✓ |
| py2_print | print() function / print statement (Python) | 200 | 0.01 → **0.35** / ref 0.34 (L2, 4) ✓ | 0.00 → **0.35** / ref 0.37 (L6, 4) ✓ |
| py2_iter | range/items/keys / xrange/iteritems/iterkeys (Python) | 144 | 0.00 → **0.40** / ref 0.33 (L1, 4) ✓ | 0.27 → **0.68** / ref 0.67 (L2, 2) ✓ |
| py2_except | except E as e / except E, e (Python) | 177 | 0.00 → **0.80** / ref 0.76 (L9, 2) ✓ | 0.05 → **0.86** / ref 0.82 (L5, 4) ✓ |
| js_var | const/let / var (JavaScript) | 200 | 0.01 → **0.52** / ref 0.41 (L23, 4) ✓ | 0.18 → **0.61** / ref 0.64 (L2, 1) ✓ |
| js_arrow | arrow functions / function expressions (JavaScript) | 193 | 0.01 → **0.14** / ref 0.22 (L1, 4) ✓ | 0.20 → **0.65** / ref 0.72 (L2, 4) ✓ |
| js_semicolons | semicolons / no semicolons (JavaScript) | 146 | 0.03 → **0.64** / ref 0.65 (L8, 4) ✓ | 0.09 → **0.73** / ref 0.71 (L1, 2) ✓ |
| js_strict_eq | === / !== / == / != (JavaScript) | 107 | 0.00 → **0.85** / ref 0.57 (L24, 4) ✓ | 0.25 → **0.87** / ref 0.90 (L13, 4) ✓ |
| trailing_commas | trailing commas / no trailing commas (Python) | 130 | 0.35 → **0.66** / ref 0.61 (L0, 2) ✓ | 0.17 → **0.41** / ref 0.54 (L4, 4) ✓ |
| py_paren_if | if x: / if (x): (Python) | 199 | 0.00 → **0.26** / ref 0.45 (L8, 4) ✓ | 0.19 → **0.58** / ref 0.65 (L2, 4) ✓ |
| py_type_hints | type hints / no type hints (Python) | 192 | 0.32 → **0.68** / ref 0.73 (L19, 2) ✓ | 0.01 → **0.36** / ref 0.41 (L4, 2) ✓ |
| py_optional | X | None / Optional[X] (Python) | 191 | 0.01 → **0.01** / ref 0.57 (L8, 1) ✗ ns | 0.07 → **0.35** / ref 0.19 (L13, 4) ✓ |
| py_builtin_generics | list[int] builtins / typing.List (Python) | 113 | 0.22 → **0.44** / ref 0.79 (L0, 4) ✓ | 0.01 → **0.45** / ref 0.26 (L2, 2) ✓ |
| py_comprehension | list comprehensions / explicit loops (Python) | 191 | 0.02 → **0.13** / ref 0.78 (L1, 4) ✗ | 0.01 → **0.16** / ref 0.60 (L0, 4) ✗ |
| py_not_in | x not in y / not x in y (Python) | 155 | 0.10 → **0.09** / ref 0.33 (L12, 4) ✗ ns | 0.16 → **0.24** / ref 0.41 (L13, 4) ✓ |
| py_is_none | is None / == None (Python) | 114 | 0.00 → **0.46** / ref 0.44 (L0, 1) ✓ | 0.06 → **0.56** / ref 0.56 (L6, 2) ✓ |
| py_ternary | conditional expressions / if/else blocks (Python) | 186 | 0.10 → **0.28** / ref 0.68 (L1, 4) ✗ | 0.02 → **0.06** / ref 0.60 (L10, 4) ✗ |
| early_return | guard clauses / nested if/else (Python) | 200 | 0.26 → **0.29** / ref 0.39 (L3, 0.5) ✓ ns | 0.36 → **0.34** / ref 0.44 (L9, 4) ✓ ns |
| py_join_concat | str.join / + concatenation (Python) | 131 | 0.09 → **0.26** / ref 0.52 (L6, 4) ✓ | 0.08 → **0.22** / ref 0.58 (L4, 4) ✗ |
| py_with_open | with open(...) / open() / close() (Python) | 200 | 0.03 → **0.24** / ref 0.86 (L1, 4) ✗ | 0.00 → **0.81** / ref 0.84 (L3, 4) ✓ |
| py_enumerate | enumerate / range(len()) (Python) | 197 | 0.00 → **0.31** / ref 0.62 (L4, 4) ✓ | 0.04 → **0.08** / ref 0.80 (L11, 4) ✗ ns |
| py_self_name | self / this (Python) | 200 | 0.01 → **0.79** / ref 0.80 (L0, 2) ✓ | 0.01 → **0.85** / ref 0.81 (L2, 1) ✓ |
| py_indent | 4-space indentation / 2-space indentation (Python) | 200 | 0.00 → **0.60** / ref 0.59 (L0, 1) ✓ | 0.33 → **0.91** / ref 0.93 (L0, 1) ✓ |
| py_tabs | spaces / tabs (Python) | 200 | 0.00 → **0.81** / ref 0.88 (L4, 2) ✓ | 0.01 → **0.73** / ref 0.96 (L0, 4) ✓ |
| operator_spaces | spaces around operators / no spaces around operators (Python) | 187 | 0.05 → **0.54** / ref 0.50 (L9, 4) ✓ | 0.34 → **0.74** / ref 0.83 (L3, 2) ✓ |
| comma_space | space after comma / no space after comma (Python) | 192 | 0.02 → **0.35** / ref 0.24 (L5, 4) ✓ | 0.49 → **0.74** / ref 0.79 (L2, 1) ✓ |
| line_wrap | wrapped long calls / single long lines (Python) | 200 | 0.37 → **0.36** / ref 0.45 (L2, 4) ✓ ns | 0.00 → **0.11** / ref 0.14 (L5, 4) ✓ |
| blank_lines | two blank lines between defs / one blank line (Python) | 142 | 0.09 → **0.75** / ref 0.79 (L4, 1) ✓ | 0.01 → **0.68** / ref 0.67 (L22, 2) ✓ |
| docstring_style | Google docstrings / NumPy docstrings (Python) | 199 | 0.00 → **0.70** / ref 0.95 (L5, 4) ✓ | 0.04 → **0.82** / ref 0.93 (L5, 4) ✓ |
| docstring_quotes | triple double quotes / triple single quotes (Python) | 200 | 0.00 → **0.91** / ref 0.91 (L1, 1) ✓ | 0.00 → **0.87** / ref 0.93 (L0, 1) ✓ |
| comment_case | capitalised comments / lowercase comments (Python) | 200 | 0.01 → **0.49** / ref 0.53 (L4, 2) ✓ | 0.09 → **0.56** / ref 0.59 (L2, 1) ✓ |
| comment_language | English comments / Spanish comments (Python) | 200 | 0.00 → **0.21** / ref 0.56 (L2, 2) ✗ | 0.04 → **0.51** / ref 0.69 (L17, 2) ✓ |
| c_comment_style | // comments / /* */ comments (JavaScript) | 200 | 0.00 → **0.54** / ref 0.50 (L19, 2) ✓ | 0.05 → **0.46** / ref 0.65 (L5, 2) ✓ |
| sql_keyword_case | uppercase keywords / lowercase keywords (SQL) | 200 | 0.01 → **0.86** / ref 0.79 (L3, 2) ✓ | 0.03 → **0.92** / ref 0.94 (L2, 1) ✓ |
| sql_join_style | explicit JOIN ... ON / implicit comma join (SQL) | 183 | 0.00 → **0.47** / ref 0.68 (L8, 4) ✓ | 0.02 → **0.79** / ref 0.77 (L3, 4) ✓ |
| r_assignment | <- assignment / = assignment (R) | 200 | 0.06 → **0.70** / ref 0.63 (L15, 2) ✓ | 0.09 → **0.84** / ref 0.81 (L1, 2) ✓ |
| rust_question | ? operator / explicit match (Rust) | 163 | 0.01 → **0.03** / ref 0.70 (L0, 0.5) ✗ | 0.04 → **0.53** / ref 0.70 (L13, 4) ✓ |
| bash_test | [[ ]] tests / [ ] tests (Bash) | 200 | 0.00 → **0.88** / ref 0.88 (L4, 2) ✓ | 0.03 → **0.84** / ref 0.86 (L17, 2) ✓ |
| php_array | [] arrays / array() arrays (PHP) | 193 | 0.05 → **0.32** / ref 0.42 (L3, 4) ✓ | 0.08 → **0.36** / ref 0.37 (L2, 2) ✓ |
