# Rules group B — validation report (2026-09-22)

Module: `src/sandbox/style_translation/code_rules_b.py` (`RULES` dict, 12 families). Validation: `tmp/rules8/validate_b.py`
(`validate_b.json`, `validate_b.log`): rule output vs the STORED alternative twin on reviewer-approved pairs (`reviewed_by`) and on all 200 pairs;
"differs" cases inspected; align + counted / filter_free check on 20 random docs per family (`counted>=5`).

| family | verdict | reviewed identical | all 200 (identical / differs / declined) | counted ≥ 5 (20 docs) | notes |
|---|---|---|---|---|---|
| py_const_naming | ENABLE | 20 / 20 | 179 / 21 / 0 | 20 / 20 | differs = stored twins left constants UNCONVERTED (t014 SPLIT_PATTERN, t023 PI_VALUE) or converted names inside comments; single-word UPPER constants are lower-cased like the approved pairs (PROMPT → prompt) |
| py_class_naming | ENABLE | 19 / 19 | 196 / 1 / 3 | 19 / 19 | differs = class names inside comments (rule leaves comments); declines = genuine collisions (`string_cleaner = StringCleaner()` already exists) — the stored twins of those docs are buggy |
| py_loop_vars | ENABLE (scope-precise) | 121 / 131 (+4 declined, 6 differ) | 123 / 70 / 7 | 19 / 20 | renames single-letter FOR / comprehension targets in the scope where they loop; stored twins also renamed the same letter in `while` loops or other functions (d009, d063) and non-letters (`val`→`value`, d090); declines = target word already used (`index`); LOOP_WORDS map documented in the module |
| py_abbrev | ENABLE-with-caveat | 30 / 34 (4 declined) | 64 / 132 / 4 | 11 / 18 | rule expands only the 16 abbreviations of the family hint (case-preserving, per `_` segment, names + parameters + self attributes); stored twins expanded extras (char→character, inp→input, txt→text) and text inside strings/docstrings. Old docs that relied on extra abbreviations lose counted occurrences (7 / 18 < 5) — new docs generated with the hint are unaffected; declines = collision (`index` exists next to `idx`) |
| py_quotes | ENABLE | 30 / 31 | 157 / 41 / 2 | 20 / 20 | converts every single-quoted literal (plain and triple, prefixes kept); literals containing a double quote: plain non-raw strings are escaped (`\"`), f-strings / raw / backslash-containing ones are left (stored twins did the same); `\'` unescaped in converted plain strings; differs = stored twins left `'__main__'` unconverted or converted quotes inside COMMENTS; declines = nothing to convert (t158 has no string literal) |
| docstring_quotes | ENABLE | 98 / 98 | 200 / 0 / 0 | 20 / 20 | every `"""` string token (docstrings and standalone block strings), declines if the content holds `'''` or ends with `'` |
| float_literals | ENABLE | 137 / 200 | 137 / 63 / 0 | 20 / 20 | all 63 differs are `0.0`: stored twins used `.0` (64×) or `0.` (98×) inconsistently; rule = `0.` always; 2.50 → 2.5, exponents kept |
| trailing_commas | ENABLE | 62 / 62 | 84 / 115 / 1 | 18 / 20 | rule removes EVERY trailing comma before a closer except 1-tuples / subscript tuples (AST-verified): stored twins left 165 multi-line trailing commas unconverted (audit G2) and in 12 docs removed a 1-tuple comma (meaning change, audit t118/t151); the 2 counted failures are old docs whose 5th occurrence sits after 75 % of the file (builder rule), not rule errors. The single-line `x = [],` bug of the audit lives in the NATURAL twins and needs regeneration (the rule does not touch it: the comma is not before a closer) |
| comment_case | ENABLE | 65 / 65 | 173 / 25 / 2 | 20 / 20 | first word of every `#` comment: ALLCAPS word → lowercase, else first letter lowered (shebang / coding lines skipped); differs = stored twins also lowered a capital after a colon or kept inline comments; declines = the 2 known `< 5` docs (no capitalised comment) |
| comma_space | ENABLE (delegates to code_whitespace_alt) | 58 / 58 | 200 / 0 / 0 | 20 / 20 | already the production rule |
| operator_spaces | ENABLE (delegates) | 31 / 31 | 200 / 0 / 0 | 20 / 20 | already the production rule |
| blank_lines | ENABLE (delegates) | 16 / 16 | 200 / 0 / 0 | 20 / 20 | already the production rule |

All rules return None instead of guessing (unparsable source, rename collision, f-string with a double quote for py_quotes, nothing to convert).
Naming rules verify by AST-transforming the natural twin the same way and comparing `ast.dump`; the others require `same_ast` (values unchanged).
Not touched by any rule: comments and string contents (by definition of the rewrites); reviewers occasionally cite a class / constant name left in
a comment — acceptable per "change nothing else", flag if it becomes a rejection reason.
