# Scorer fixes (audit 9, G7) — before / after, all 55 pool families (2026-09-22)

Self-test = each twin's own continuation after its cue scored with decide_any + cut_code (before = audit 9 partA A9). Real-rollout check = stored
rollouts whose raw tail starts exactly like ONE pole's expected next tokens (the scorer's own fallback prefix): stored decision (before) vs
decide_any recomputed with the fixed code (after). Only families that changed are listed; the other 47 are identical before and after.

| family | self-test fail before → after | unambiguous rollouts | mislabelled before → after | unscorable before → after |
|---|---|---|---|---|
| float_literals | 20.2% → 0.0% | 1187 | 16.3% → 0.8% | 1.8% → 0.0% |
| js_semicolons | 19.6% → 0.1% | 1360 | 0.2% → 0.0% | 11.8% → 0.0% |
| operator_spaces | 3.5% → 1.3% | 1355 | 5.9% → 1.0% | 2.2% → 1.0% |
| py_abbrev | 3.4% → 0.4% | 287 | 8.7% → 3.5% | 0.0% → 0.0% |
| py_builtin_generics | 0.6% → 0.1% | 1347 | 0.0% → 0.0% | 0.2% → 0.0% |
| py_optional | 3.1% → 0.7% | 955 | 1.8% → 2.1% | 1.7% → 0.0% |
| py_type_hints | 1.1% → 1.0% | 1558 | 1.4% → 1.4% | 1.1% → 1.0% |
| trailing_commas | 16.7% → 16.0% | 1577 | 11.4% → 10.4% | 3.7% → 1.9% |

Unchanged (47): bash_subst, bash_test, blank_lines, c_comment_style, comma_space, comment_case, comment_language, docstring_quotes, docstring_style, hex_constants, js_arrow, js_camel_snake, js_func_pascal, js_hungarian, js_quotes, js_strict_eq, js_template, js_var, line_wrap, num_separators, php_array, py2_except, py2_iter, py2_print, py_class_naming, py_comprehension, py_const_naming, py_enumerate, py_fstring, py_indent, py_is_none, py_join_concat, py_literal_ctor, py_loop_vars, py_not_in, py_paren_if, py_private, py_quotes, py_self_name, py_snake_camel, py_tabs, py_with_open, r_assignment, rust_question, sql_bool_case, sql_join_style, sql_keyword_case

## Notes
- trailing_commas: the remaining self-test failures are DATA, not the scorer: 251 of the 320 failing items have a twin that contradicts its own
  pole at the first closer after the cue (alternative twins keeping a trailing comma; single-line `x = f(),` commas whose "closer" never comes),
  26 have no closer inside the cut tail; the family is scheduled for rule-based regeneration (plan Part 0.1 / audit G2).
- py_optional: the rollout "mislabelled" count rises 17 → 20 only because the 6-character expectation is wrong for those records (natural and
  alternative next tokens share the prefix `\n\ndef `); the tails contain `Optional[` and are now correctly labelled alternative. The k = 0
  unscorable cases (absent import, tail starting with blank lines) are fixed by the cut_code change: 1.7 % → 0 %.
- py_builtin_generics / py_type_hints: tiny improvements from the cut_code change (tails starting with blank lines are no longer cut to '').

## Edits
1. `src/sandbox/style_translation/scoring.py` cut_code: a blank line only cuts the tail AFTER the first non-empty line (previously a tail starting
   with blank lines was cut to '' and became unscorable); the `Task:` cut and the blank_lines special case are unchanged.
2. `src/sandbox/style_translation/code_scoring.py` (decide_code, CTX_FAMILIES += float_literals, trailing_commas, operator_spaces, js_semicolons):
   - float_literals: cue-anchored — the digits (and a point) that end the prompt line belong to the literal: `25.` + `5` → nat, `25.` + `:` → alt,
     `1` + `.0` → nat, `1` + `.` → alt, ` ` + `.5` → alt; otherwise the old regex pair on string-masked code with the prompt digits prepended.
   - trailing_commas: a comma right after the cue → nat; else the FIRST closing bracket on its own line after the cue decides (comma before it or
     not), with the prompt's last character prepended so a closer at the start of the tail is scorable; no closer → None.
   - operator_spaces: the prompt's last character is prepended and string literals are masked; a match at the cue (`a = b` / `a=b`) decides,
     otherwise the earliest match as before.
   - js_semicolons: `;` right after the cue → nat; a newline right after a statement-ending cue → alt; else the first statement line of the tail
     (comment stripped) ends with `;` → nat / without → alt; `}` / `{` / `,` lines are no decision; then the old regex pair as fallback.
   - py_abbrev: only the FIRST new identifier after the cue decides — an abbreviation part (cfg, idx, …) → nat, a full-word part → alt, else its
     lexicon label; later identifiers no longer override it.
   The regex strings in code_families.py were left unchanged (the new branches supersede them for these families).