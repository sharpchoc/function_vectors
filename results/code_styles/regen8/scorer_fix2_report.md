# Scorer fix 2 (2026-09-23): c_comment_style, comment_case, py_abbrev after the regen8 wave-1 refresh

## Diagnosis (fresh rollouts, k = 0..4, both poles)
- **comment_case — scorer miss.** The cue is the comment opener (` #`), so the tail starts with the comment text (` Deduct the allowed …`).
  The regex classifier (`#\s+[A-Z]` vs `#\s+[a-z]`) never sees a `#` in seg_prefix + tail and the next-token fallback rarely matches
  (next_nat = ` The patient`). Of 1,224 unscorable rollouts, 1,181 have the `#` cue and a letter as first character (685 upper, 496 lower);
  the rest start with a digit / quote / symbol (genuine no-decision).
- **py_abbrev — scorer miss by design of the 2026-09-22 patch.** Only the very first NEW identifier decided; tails such as
  ` i in range(cfg[...])` (first new name `i`, unmapped) were None although a mapped new name followed. Of 1,230 unscorable rollouts,
  ~300 had no mapped identifier at all; the remainder had one later, a plural form (`values`), or a lexicon-labelled name later.
  Names already present in the prompt (`ctx`, `num`, `parameter`) stay excluded as forced reuses (the family's first-mention rule).
- **c_comment_style — genuine.** Of 955 unscorable rollouts, 783 contain no `//` or `/*` anywhere in the completion (the model continued
  with code where the twin had a comment), 169 have a comment only after a blank line (a later block, not this decision), 3 beyond 160
  characters. The cue placement is right (next_nat ` // Re` / next_alt ` /* Re`). No scorer change; the k = 4 drop (.80 → .51) reflects the
  regenerated documents: comments now sit where the model does not spontaneously comment.

## Edits (`src/sandbox/style_translation/code_scoring.py`)
1. `comment_case` added to CTX_FAMILIES with a cue-anchored branch: if the prompt's last line ends with `#`, the first letter of the tail
   decides (upper → nat, lower → alt; non-letter → None); otherwise the first `#<space>letter` in seg[:160] decides.
2. `py_abbrev`: the first NEW identifier (prompt names excluded, strings masked, 400-char window) whose parts (split on `_`, plural `s`
   stripped) intersect the abbreviation set → nat, the full-word set → alt, or that has a lexicon label — instead of only the first new identifier.
No other family's branch or regex was touched.

## New script `src/sandbox/style_translation/rescore_records.py`
Recomputes `decision` / `style_ok` of stored rollout, write-steering and read-steering records from the stored cut `tail` with the current
scorer (prompt text decoded from the prompts file: rollout (doc, style, k); write-steering the k = 0 prompt; read-steering the k = 3 prompt of
the context style); `judge` untouched; dry run by default, `--apply` writes. Applied to the three families (all records were already judged):

| family | file | records | decisions changed | unscorable before → after | style_ok before → after |
|---|---|---|---|---|---|
| comment_case | rollouts | 2000 | 1361 | 1224 → 58 | 568 → 1636 |
| comment_case | steering/full_k3/w2 | 360 | 266 | 272 → 30 | 48 → 213 |
| comment_case | read_steer/full_k3/w2 | 160 | 103 | 90 → 1 | 31 → 70 |
| py_abbrev | rollouts | 2000 | 305 | 1230 → 942 | 475 → 662 |
| py_abbrev | steering/full_k3/w2 | 360 | 93 | 260 → 167 | 71 → 118 |
| py_abbrev | read_steer/full_k3/w2 | 160 | 39 | 95 → 56 | 32 → 62 |
| c_comment_style | all three | 2520 | 0 | unchanged | unchanged |

## k = 4 accuracy (style_ok AND judge ok) and steering, wave-1 records

| family | k4 before | k4 after | unscorable k4 before → after | write L24 α2 after | read L8 α4 after |
|---|---|---|---|---|---|
| c_comment_style | .510 | .510 | .40 → .40 | .24 | .47 |
| comment_case | .260 | .733 | .62 → .01 | .49 | .60 |
| py_abbrev | .205 | .297 | .64 → .48 | .25 | .26 |

## Regression check, all 55 pool families (self-test = each twin's own continuation; rollout cross-check on unambiguous stored rollouts)
Families with any change: []. Everything else identical before and after.

| family | self-test fail before | after | of | unambiguous rollouts | mislabel before | after | unscorable before | after |
|---|---|---|---|---|---|---|---|---|
| bash_subst | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| bash_test | 0 | 0 | 2000 | 1294 | 0 | 0 | 0 | 0 |
| blank_lines | 0 | 0 | 2000 | 1646 | 4 | 4 | 0 | 0 |
| c_comment_style | 0 | 0 | 2000 | 268 | 0 | 0 | 0 | 0 |
| comma_space | 70 | 70 | 2000 | 1553 | 41 | 41 | 21 | 21 |
| comment_case | 0 | 0 | 2000 | 182 | 0 | 0 | 0 | 0 |
| comment_language | 57 | 57 | 1992 | 393 | 3 | 3 | 0 | 0 |
| docstring_quotes | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| docstring_style | 5 | 5 | 2000 | 1528 | 0 | 0 | 4 | 4 |
| float_literals | 0 | 0 | 2000 | 1187 | 9 | 9 | 0 | 0 |
| hex_constants | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| js_arrow | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| js_camel_snake | 0 | 0 | 2000 | 694 | 0 | 0 | 0 | 0 |
| js_func_pascal | 29 | 29 | 2000 | 0 | 0 | 0 | 0 | 0 |
| js_hungarian | 28 | 28 | 2000 | 391 | 5 | 5 | 0 | 0 |
| js_quotes | 0 | 0 | 2000 | 1301 | 0 | 0 | 0 | 0 |
| js_semicolons | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| js_strict_eq | 0 | 0 | 2000 | 1395 | 0 | 0 | 0 | 0 |
| js_template | 0 | 0 | 2000 | 847 | 16 | 16 | 0 | 0 |
| js_var | 0 | 0 | 2000 | 668 | 0 | 0 | 0 | 0 |
| line_wrap | 9 | 9 | 2000 | 1369 | 20 | 20 | 6 | 6 |
| num_separators | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| operator_spaces | 26 | 26 | 2000 | 1355 | 14 | 14 | 13 | 13 |
| php_array | 11 | 11 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py2_except | 0 | 0 | 2000 | 1509 | 1 | 1 | 0 | 0 |
| py2_iter | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py2_print | 2 | 2 | 2000 | 1005 | 0 | 0 | 1 | 1 |
| py_abbrev | 5 | 5 | 2000 | 173 | 5 | 5 | 0 | 0 |
| py_builtin_generics | 1 | 1 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_class_naming | 0 | 0 | 2000 | 972 | 0 | 0 | 0 | 0 |
| py_comprehension | 6 | 6 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_const_naming | 1 | 1 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_enumerate | 101 | 101 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_fstring | 0 | 0 | 2000 | 738 | 4 | 4 | 0 | 0 |
| py_indent | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_is_none | 0 | 0 | 2000 | 1387 | 0 | 0 | 0 | 0 |
| py_join_concat | 23 | 23 | 2000 | 1043 | 26 | 26 | 4 | 4 |
| py_literal_ctor | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_loop_vars | 17 | 17 | 2000 | 722 | 3 | 3 | 0 | 0 |
| py_not_in | 0 | 0 | 2000 | 1360 | 7 | 7 | 0 | 0 |
| py_optional | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_paren_if | 5 | 5 | 2000 | 1152 | 26 | 26 | 0 | 0 |
| py_private | 49 | 49 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_quotes | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_self_name | 0 | 0 | 2000 | 1522 | 0 | 0 | 0 | 0 |
| py_snake_camel | 0 | 0 | 2000 | 631 | 0 | 0 | 0 | 0 |
| py_tabs | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| py_type_hints | 8 | 8 | 2000 | 1398 | 12 | 12 | 9 | 9 |
| py_with_open | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| r_assignment | 0 | 0 | 2000 | 0 | 0 | 0 | 0 | 0 |
| rust_question | 1 | 1 | 2000 | 1671 | 30 | 30 | 0 | 0 |
| sql_bool_case | 0 | 0 | 2000 | 1560 | 0 | 0 | 0 | 0 |
| sql_join_style | 1 | 1 | 2000 | 0 | 0 | 0 | 0 | 0 |
| sql_keyword_case | 0 | 0 | 2000 | 1219 | 1 | 1 | 0 | 0 |
| trailing_commas | 31 | 31 | 2000 | 1797 | 37 | 37 | 3 | 3 |
