# Group C rules — validation report (2026-09-22)

Module: `src/sandbox/style_translation/code_rules_c.py` (`RULES` dict, 11 families + `hex_constants_pow2` variant). Validator:
`tmp/rules8/validate_c.py` (results `validate_c.json`). "identical" = the rule reproduces the STORED alternative twin byte for byte;
"differs" = differs from the stored (LLM-made) twin; "declined" = rule returned None. Reviewed = pairs with `reviewed_by` (strict reviewers
accepted the stored twin). counted = of 20 random docs, how many keep >= 5 counted opportunities (nat != alt, 5th inside the threshold) with
the rule's twin (low values reflect the OLD documents' padding-era opportunities, e.g. quotes in comments; regenerated docs get the hint).

| family | reviewed identical / differs / declined | all 200 identical / differs / declined | counted >= 5 | verdict |
|---|---|---|---|---|
| js_quotes | 30 / 0 / 0 | 126 / 69 / 5 | 14/20 | ENABLE. Differs = stored twins also changed quotes INSIDE COMMENTS (rule leaves comments); declines = no double-quoted literal in code (templates only). Escapes handled (`\"`→`"`, `'`→`\'`). |
| js_semicolons | 8 / 0 / 0 | 184 / 16 / 0 | 20/20 | ENABLE. Differs = stored twins removed `;` inside comments or moved one INTO a string (t054, stored is wrong). Removes end-of-line and before-`}` terminators outside ( ) [ ]; `;;` runs; declines on ASI hazards (next line starts with ( [ ` + - /) or two statements on one line. |
| js_var | 20 / 0 / 0 | 198 / 2 / 0 | 20/20 | ENABLE. Differs = commented-out code converted by the stored twin. |
| js_strict_eq | 81 / 0 / 0 | 199 / 1 / 0 | 20/20 | ENABLE. Also rewrites `${...}` template expressions (reviewed t145 needs it). The 1 differ: stored twin changed `item` to `item.price` (stored wrong). |
| js_camel_snake | 8 / 0 / 0 | 185 / 10 / 5 | 20/20 | ENABLE-with-caveat. Renames names BOUND in the file (functions, params, const/let/var incl. array destructuring, catch) everywhere incl. template expressions and comments; object-literal keys and property names stay (data); task-pinned names stay. Declines (5) when an unbound camelCase identifier that is neither a key of the file, a JS builtin member (whitelist of 190) nor pinned remains (foreign object properties). Differs = stored twins left a bound name unconverted (t057) or renamed keys. Caveat: key/variable coupling via object destructuring `const {a} = o` is treated as variables (renamed) — the key is not; such docs are rare (0 in reviewed). |
| php_array | 22 / 0 / 0 | 180 / 20 / 0 | 20/20 | ENABLE. `[ ... ]` literals → `array( ... )` (nested, multi-line; index access `$x[..]`, `f()[..]` untouched; `[$a,$b] = ` / `as [$k,$v]` destructuring untouched, as in the reviewed pairs). Differs = stored twins used `array_merge` for spread literals (meaning change) or appended `?>`. Heredoc → decline. |
| r_assignment | 70 / 0 / 0 | 198 / 2 / 0 | 20/20 | ENABLE. `<-` → `=` in code (not strings/comments); declines on `<<-`/`->` in code or an assignment inside call arguments. Differs = commented-out code converted by the stored twin. |
| c_comment_style | 19 / 0 / 0 | 138 / 62 / 0 | 20/20 | ENABLE. Per-line `// text` → `/* text */`; a second `// ` on a comment line becomes its own block (matches stored). Differs = stored twins merged consecutive comment lines into one block (t018/t022) — style choice; reviewed pairs are per-line. |
| bash_test | 46 / 2 / 1 | 89 / 110 / 1 | 11/20 | ENABLE-with-caveat. Converts `[[ e ]]` → `[ e ]` only when `[ ]` can express it: NOT `=~`, `&&`, `||`, unquoted `<`/`>`, `( )`, glob patterns, `-nt/-ot/-ef`; bare `$var` operands get quoted (except in arithmetic tests). Non-convertible tests stay `[[ ]]` in both twins (non-opportunities → counted only 11/20 on OLD docs, whose tests are often regex/glob). Differs vs stored = stored twins converted the unsafe tests (audit: 23 invalid alt twins) or split `&&` tests. For regeneration, add to the family hint: "tests must be expressible in POSIX [ ]: no =~, no glob patterns, no && / || inside the brackets, quote every variable". |
| sql_bool_case | 40 / 0 / 0 | 192 / 7 / 1 | 20/20 | ENABLE. `true`/`false` → upper outside strings/comments; differs = stored twins changed comments or fixed a typo. |
| hex_constants (set) | 71 / 20 / 0 | 83 / 106 / 11 | 13/18 | ENABLE the SET rule (the family definition: exactly the 9 constants). Reviewed differs = stored twins ALSO hex-ified out-of-set literals (65533, 127, 67, 0, 1, 2, 3 …), which GPT-5/Opus also flagged as scope creep in other docs — the reviewers were inconsistent; the set rule is what the rewrite instruction says. `hex_constants_pow2` variant (powers of two ≥ 16, 2^n−1 ≥ 15): reviewed 64 / 27 — worse; not recommended. Declines = no set constant in the doc. Regeneration hint must require ≥ 5 set constants (each on its own line). |

Common properties: every rule masks strings / templates / regex / comments (JS lexer `js_segments`; generic lexer for PHP/R/Bash/SQL), rewrites only
code (+ `${}` template expressions, + comments for the identifier rule), re-parses the result (node --check / php -l / bash -n; R and SQL have no
parser here), and returns None instead of guessing. String/comment contents are never changed except the comment rule (comments are the construct)
and the identifier rule (a renamed identifier mentioned in a comment follows, as the reviewed pairs do).

Out of scope (noted): js_hungarian needs type inference — not attempted; js_arrow / js_template / js_func_pascal are structural.
