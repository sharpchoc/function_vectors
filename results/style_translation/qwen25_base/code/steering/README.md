# results/style_translation/qwen25_base/code/steering — cue-token (write-feature) steering for the 55 coding-convention families, full 200-task corpora

Qwen2.5-7B (base). Same protocol as the text buckets (`../../steering/`, GPT-J `../../../steering/`): PAIRED cue-token mean-difference vectors
pooled over k = 0..4 from the judged step-3 rollouts (`capture_cues.py --model qwen25_base`; paired pools 202–596 prompts per pole, split-half
cos ≥ .88 at L20), screen layers {2,4,…,24} × α ∈ {0.5,1,2,4} on 50 k = 0 prompts (style only, 16 tokens), confirm the top-2 settings per
target on all k = 0 prompts of the family (≤ 200; 48 tokens, code cut, registry classifier ∧ Gemini 2.5 Flash judge: the completion must still be
a correct solution of the task). **No control arm** (another family's vector at the same layer/α) in this run — user decision 2026-09-15.

Prompt seen by the model at k = 0: `Task:\n<task description>\n\n<Language>:\n<code up to the cue token>`; the vector is added at the cue token only.
Accuracy = target convention used (regex classifier over the completion) AND judge finds the code a correct solution; "unsteered" = the same k = 0
prompt scored toward the same target; dashed line = k = 4 in-context accuracy from `../summary.csv`. Run: `steer_analyze.py --model qwen25_base --tag code --families <code_pool_full.json>`.

Files: `steered_vs_unsteered.png` (one-row-per-family overview, both directions, with the 4-shot reference; `code_steering_overview.py`), `steering_summary.png` (per-family bar grid), `screen_layer_alpha.png` (screen grid), `best_config.csv`, `steering_summary.csv`, `screen.csv`, `records.npz`.

## Result (55 families, full corpus)

| direction | unsteered mean | steered mean (best L, α) | significant lift (Wilson CI excludes unsteered) | unsteered ≥ .70 (no headroom) | steered ≥ .5 | steered ≥ .7 |
|---|---|---|---|---|---|---|
| → alt (the pole the model does not default to) | .14 | .42 | 44 / 55 | 0 | 21 | 11 |
| → nat | .39 | .56 | 31 / 55 | 13 | 30 | 23 |

- Strict gate (significant lift in BOTH directions): 24 / 55. Headroom-aware gate (a direction with unsteered ≥ .70 is exempt): 32 / 55.
- Best layer: L24 in 39/55 (→ alt) and 33/55 (→ nat) families; α = 4 most often (33 and 24). Same late-layer preference as the text families on Qwen.
- Full vs cheap (50-task) run (`../../code_k4/README.md`): steered accuracies correlate .87 (→ alt) / .91 (→ nat), mean |difference| .08.
  Families with no significant lift in either direction: 4 on the cheap corpus, 4 here (js_hungarian, py_abbrev, py_comprehension, py_ternary).
  Cheap-flat families that lift now: js_template, py_optional. Newly flat: js_hungarian, py_comprehension.
- Steering reaches the 4-example reference in 11 (→ alt) / 12 (→ nat) families; for the rest it lands below it, as for the text families.

## Per family (accuracy, unsteered → steered at the best layer/α; ✓ = CI excludes unsteered, ceil = unsteered ≥ .70, ✗ = no significant lift)

| family | nat / alt | n | → alt | → nat |
|---|---|---|---|---|
| py_snake_camel | snake_case identifiers / camelCase identifiers | 200 | 0.00 → **0.80** (L24, 4) ✓ | 0.91 → **0.90** (L2, 0.5) ceil |
| js_camel_snake | camelCase identifiers / snake_case identifiers | 200 | 0.00 → **0.63** (L24, 2) ✓ | 0.86 → **0.83** (L2, 0.5) ceil |
| py_const_naming | UPPER_SNAKE constants / camelCase constants | 200 | 0.00 → **0.01** (L24, 4) ✓ | 0.02 → **0.03** (L12, 0.5) ✗ |
| py_class_naming | PascalCase class names / snake_case class names | 200 | 0.39 → **0.43** (L20, 2) ✗ | 0.12 → **0.17** (L6, 4) ✓ |
| py_private | single-underscore private attributes / double-underscore private attributes | 200 | 0.03 → **0.40** (L24, 4) ✓ | 0.05 → **0.28** (L24, 4) ✓ |
| py_bool_prefix | is_/has_ boolean names / bare boolean names | 200 | 0.07 → **0.07** (L24, 1) ✗ | 0.01 → **0.36** (L24, 4) ✓ |
| py_loop_vars | single-letter loop variables / descriptive loop variables | 188 | 0.08 → **0.21** (L24, 4) ✓ | 0.34 → **0.69** (L24, 4) ✓ |
| js_hungarian | plain identifiers / Hungarian notation | 200 | 0.01 → **0.01** (L12, 0.5) ✗ | 0.54 → **0.57** (L20, 1) ✗ |
| py_abbrev | abbreviated identifiers / spelled-out identifiers | 200 | 0.39 → **0.41** (L6, 0.5) ✗ | 0.13 → **0.14** (L20, 4) ✗ |
| py_quotes | single-quoted strings / double-quoted strings | 192 | 0.33 → **0.48** (L24, 2) ✓ | 0.28 → **0.41** (L24, 1) ✓ |
| js_quotes | double-quoted strings / single-quoted strings | 189 | 0.53 → **0.66** (L24, 1) ✓ | 0.20 → **0.70** (L24, 2) ✓ |
| py_fstring | f-strings / str.format | 200 | 0.12 → **0.33** (L24, 4) ✓ | 0.18 → **0.85** (L24, 4) ✓ |
| js_template | template literals / string concatenation | 198 | 0.14 → **0.34** (L24, 4) ✓ | 0.24 → **0.25** (L24, 1) ✗ |
| num_separators | plain numeric literals / underscore digit separators | 131 | 0.00 → **0.64** (L24, 4) ✓ | 0.31 → **0.47** (L24, 4) ✓ |
| hex_constants | decimal constants / hexadecimal constants | 177 | 0.01 → **0.26** (L24, 4) ✓ | 0.37 → **0.37** (L16, 0.5) ✗ |
| float_literals | full float literals / abbreviated float literals | 162 | 0.03 → **0.34** (L24, 4) ✓ | 0.47 → **0.65** (L24, 4) ✓ |
| sql_bool_case | lowercase true/false / uppercase TRUE/FALSE | 133 | 0.61 → **0.86** (L24, 2) ✓ | 0.22 → **0.90** (L24, 2) ✓ |
| py2_print | print() function / print statement | 200 | 0.00 → **0.08** (L8, 4) ✓ | 0.22 → **0.23** (L24, 2) ✗ |
| py2_iter | range/items/keys / xrange/iteritems/iterkeys | 144 | 0.04 → **0.75** (L24, 4) ✓ | 0.69 → **0.72** (L24, 4) ✗ |
| py2_except | except E as e / except E, e | 177 | 0.00 → **0.14** (L24, 2) ✓ | 0.09 → **0.16** (L10, 0.5) ✓ |
| js_var | const/let / var | 200 | 0.04 → **0.31** (L24, 4) ✓ | 0.41 → **0.47** (L24, 0.5) ✗ |
| js_arrow | arrow functions / function expressions | 196 | 0.05 → **0.85** (L24, 4) ✓ | 0.77 → **0.85** (L2, 1) ✓ |
| js_semicolons | semicolons / no semicolons | 146 | 0.07 → **0.60** (L24, 4) ✓ | 0.81 → **0.80** (L8, 2) ceil |
| js_strict_eq | === / !== / == / != | 107 | 0.05 → **0.78** (L24, 4) ✓ | 0.76 → **0.84** (L24, 2) ✓ |
| trailing_commas | trailing commas / no trailing commas | 130 | 0.52 → **0.54** (L10, 0.5) ✗ | 0.13 → **0.41** (L24, 2) ✓ |
| py_paren_if | if x: / if (x): | 199 | 0.02 → **0.46** (L24, 4) ✓ | 0.64 → **0.72** (L20, 4) ✓ |
| py_type_hints | type hints / no type hints | 192 | 0.68 → **0.76** (L4, 2) ✓ | 0.10 → **0.91** (L24, 4) ✓ |
| py_optional | X | None / Optional[X] | 191 | 0.00 → **0.01** (L6, 4) ✓ | 0.16 → **0.29** (L20, 4) ✓ |
| py_builtin_generics | list[int] builtins / typing.List | 113 | 0.00 → **0.00** (L10, 1) ✗ | 0.12 → **0.19** (L20, 4) ✓ |
| py_comprehension | list comprehensions / explicit loops | 194 | 0.13 → **0.11** (L4, 0.5) ✗ | 0.36 → **0.37** (L24, 1) ✗ |
| py_not_in | x not in y / not x in y | 164 | 0.02 → **0.20** (L24, 4) ✓ | 0.37 → **0.41** (L24, 1) ✗ |
| py_is_none | is None / == None | 114 | 0.03 → **0.18** (L24, 4) ✓ | 0.22 → **0.91** (L24, 4) ✓ |
| py_ternary | conditional expressions / if/else blocks | 187 | 0.25 → **0.30** (L20, 1) ✗ | 0.14 → **0.13** (L8, 0.5) ✗ |
| early_return | guard clauses / nested if/else | 200 | 0.20 → **0.35** (L20, 4) ✓ | 0.31 → **0.34** (L24, 4) ✗ |
| py_join_concat | str.join / + concatenation | 176 | 0.07 → **0.22** (L24, 4) ✓ | 0.46 → **0.60** (L24, 4) ✓ |
| py_with_open | with open(...) / open() / close() | 200 | 0.01 → **0.02** (L20, 1) ✓ | 0.07 → **0.08** (L24, 4) ✗ |
| py_enumerate | enumerate / range(len()) | 198 | 0.27 → **0.30** (L24, 2) ✗ | 0.18 → **0.75** (L24, 4) ✓ |
| py_self_name | self / this | 200 | 0.00 → **0.68** (L24, 2) ✓ | 0.96 → **0.99** (L6, 0.5) ✓ |
| py_indent | 4-space indentation / 2-space indentation | 200 | 0.01 → **0.76** (L24, 4) ✓ | 0.74 → **0.90** (L24, 1) ✓ |
| py_tabs | spaces / tabs | 200 | 0.01 → **0.77** (L10, 4) ✓ | 0.80 → **0.86** (L20, 2) ✓ |
| operator_spaces | spaces around operators / no spaces around operators | 187 | 0.03 → **0.48** (L20, 2) ✓ | 0.78 → **0.80** (L12, 2) ceil |
| comma_space | space after comma / no space after comma | 192 | 0.01 → **0.44** (L24, 4) ✓ | 0.77 → **0.78** (L10, 2) ceil |
| line_wrap | wrapped long calls / single long lines | 200 | 0.66 → **0.67** (L8, 0.5) ✗ | 0.01 → **0.81** (L24, 4) ✓ |
| blank_lines | two blank lines between defs / one blank line | 142 | 0.63 → **0.68** (L24, 2) ✗ | 0.05 → **0.57** (L24, 4) ✓ |
| docstring_style | Google docstrings / NumPy docstrings | 199 | 0.05 → **0.86** (L24, 4) ✓ | 0.68 → **0.95** (L24, 4) ✓ |
| docstring_quotes | triple double quotes / triple single quotes | 200 | 0.03 → **0.23** (L24, 4) ✓ | 0.15 → **0.18** (L10, 1) ✗ |
| comment_case | capitalised comments / lowercase comments | 200 | 0.05 → **0.28** (L24, 4) ✓ | 0.22 → **0.22** (L8, 1) ✗ |
| comment_language | English comments / Spanish comments | 200 | 0.00 → **0.06** (L20, 2) ✓ | 0.35 → **0.37** (L24, 1) ✗ |
| c_comment_style | // comments / /* */ comments | 200 | 0.01 → **0.11** (L24, 4) ✓ | 0.30 → **0.38** (L24, 4) ✓ |
| sql_keyword_case | uppercase keywords / lowercase keywords | 200 | 0.02 → **0.29** (L24, 4) ✓ | 0.69 → **0.76** (L24, 4) ✓ |
| sql_join_style | explicit JOIN ... ON / implicit comma join | 183 | 0.01 → **0.58** (L24, 4) ✓ | 0.82 → **0.85** (L6, 0.5) ceil |
| r_assignment | <- assignment / = assignment | 200 | 0.05 → **0.68** (L24, 4) ✓ | 0.79 → **0.79** (L24, 2) ceil |
| rust_question | ? operator / explicit match | 165 | 0.04 → **0.12** (L24, 2) ✓ | 0.70 → **0.73** (L2, 1) ceil |
| bash_test | [[ ]] tests / [ ] tests | 200 | 0.68 → **0.92** (L24, 4) ✓ | 0.19 → **0.51** (L24, 2) ✓ |
| php_array | [] arrays / array() arrays | 193 | 0.12 → **0.71** (L24, 4) ✓ | 0.31 → **0.42** (L24, 4) ✓ |
