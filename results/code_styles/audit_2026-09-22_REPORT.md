# Audit 9 — sweeping check of the code-style data (2026-09-22, read-only)

Scope: 60 code families in `dataset_files/style_translation/pairs/`, the qwen25_code prompt files (k = 0..4), the evidence files, the scorer,
the held-out split, and a strict two-reviewer sample of documents that no reviewer had ever seen. Scratch: `tmp/audit9/` (partA.json,
partB.json, sample_review.jsonl, targeted.json, pyflakes.json). Nothing was modified.

## Part A — mechanical checks (all 60 families; numbers for the 55 pool families unless stated)

1. **Twin integrity** (rebuild text_alt from text_nat + opps_all, spans sorted/inside/non-overlapping, k_en, nat != alt): **0 failures in all 60 families.**
2. **k = 0 prefix** identical in both twins and both k = 0 prompt_ids identical: 0 failures except the known comment_language 11 docs.
3. **Parse / validate** (valid_source / bash_ok as in the builder): 0 failures. (Python-2 alt twins exempt, as in the builder — see G3.)
4. **Mixed conventions by regex** (foreign-pole regex outside the opportunity spans): large counts in 50 families, but inspection shows they
   are overwhelmingly regex artefacts (literals inside strings/docstrings/comments, JS builtins, pinned names). Real leftovers were then
   measured with targeted checks — see G2, G6.
5. **Duplicates**: no duplicate task specs, titles or natural twins within any family; 5 natural twins shared by py_indent and py_tabs
   (same task ids t158, t199, t252, …, both derived from the same generation) — harmless.
6. **Degeneracy**: no fences, `<ctrl>` or U+FFFD; padding_lines / leak_lines hit only the known residuals (js_hungarian 8+1, py_join_concat 1,
   js_arrow 1) and the four out-of-pool families; 4 docs with non-ASCII characters in code (an em dash and a degree sign inside JS regex
   literals: js_camel_snake t095, js_hungarian t095, js_template t005; Korean identifiers in py_optional t103) — harmless; docs > 120 lines:
   hex_constants 12, sql_join_style 11, docstring_quotes 7, docstring_style 6, sql_bool_case 5, py_not_in 4, others ≤ 3.
7. **Prompt files**: every (doc, style, k) present except the known < 5-occurrence residuals (comment_case 16 items, comment_language 8);
   every prompt decodes to an exact prefix of header + twin; cue_tok is the last token everywhere; next_nat != next_alt everywhere; header
   format identical in all families. Prompts over 1,100 tokens: num_separators 4 (max 2,096), rust_question 1 (1,141), float_literals max 1,078.
8. **Evidence files**: all positions inside the prompt, after the header and before the cue; docstring_style averages 133 evidence tokens
   (whole-docstring SECTION rule, by design). **py_type_hints: 26 alt k = 1 prompts have an EMPTY evidence list** (first opportunity
   `: int` → `` at k = 1, absence pole; n_A = n_B = 0). k = 3/4 are unaffected (prompt_pairs: 0 zero-evidence rows, no NaN read vectors).
9. **Scorer self-test** (each twin's own continuation after its cue, scored with the rollout scorer decide_any + cut_code). Families with
   ≥ 3 % failures: {"docstring_style": "60/2000", "float_literals": "404/2000", "js_semicolons": "392/2000", "operator_spaces": "71/2000", "py_abbrev": "68/2000", "py_enumerate": "66/2000", "py_optional": "63/2000", "trailing_commas": "333/2000"}.
   Cross-checked on the REAL rollouts (records whose first 6 characters equal exactly one pole's expected continuation):
   float_literals 16.3 % MISLABELLED (185 natural continuations `.0` scored as alternative: the natural regex `\d+\.\d+` needs the digits
   that sit before the cue, the alternative regex `\.\d+` then matches), trailing_commas 11.4 % mislabelled (the 160-char look-ahead hits a
   later trailing comma), py_abbrev 8.7 %, operator_spaces 5.9 % (`\w-\w` inside strings such as "non-negative"), js_semicolons 11.8 %
   unscorable (closing-brace lines), comment_language 0.8 %. See G7.
10. **Held-out split**: every pool family has ≥ 45 held-out docs (min 45); no family below 40.

## Part B — strict two-reviewer sample of never-reviewed documents (Opus 5 + GPT-5, AND rule; the regeneration's exact checklist)

204 documents (4 per pool family, 51 families had ≥ 4 unreviewed docs; some families are fully reviewed), cost $17.
**Accepted by both reviewers: 11 / 204 (5.4%).** Both reject 149, one rejects 44. Items cited (Opus / GPT-5): padding 497 / 615,
fake decision 37 / 149, family rule 12 / 88, inconsistent convention 51 / 65, scope creep 43 / 65, extra change 14 / 65, wrong task 30 / 70,
invalid 16 / 35, too few / bunched 30 / 15, meaning change 8 / 4.
Issues cited by BOTH reviewers on the same document: padding 81 docs (40 %), wrong task 17 (8 %), inconsistent convention 16, invalid 11 (5 %),
extra change 7, meaning change 2. Per family acceptance: c_comment_style 2/4, comment_language 2/4, docstring_quotes 1/4, js_semicolons 1/4, js_var 1/4, py_indent 2/4, py_self_name 1/4, py_tabs 1/4 — every other family 0/4.
Reading: the never-reviewed 76 % of the corpus (Gemini 2.5 Flash, ≥ 8-occurrence hint) does not meet the standard the regenerated 24 % was
held to. Most of the rejections are "padding" in the reviewers' sense — dead branches, never-firing checks, unused helpers, `$(echo "$1")`,
example blocks — which the regex sweep cannot see. This is systematic, not a handful of glaring documents.

15 most damning quoted issues (both reviewers agree on the item):
- bash_test bash_test__t004 · item 3 invalid · `if ! [ "$N" =~ ^[0-9]+$ ]; then` — The POSIX `[` builtin does not support the `=~` regex operator, so this line fails at runtime with "too many arguments" and the script exits with an error for every input.
- php_array php_array__t080 · item 3 invalid · `    if (count(explode('.', $filteredHost)) > 2 && in_array(end(explode` — `end()` takes its argument by reference, so passing the result of `explode()` directly is invalid and emits a notice/error rather than running cleanly.
- py2_except py2_except__t202 · item 3 invalid · `        return f"Error: {e}"` — The alternative twin mixes Python-2-only `except E, e:` syntax with f-strings, so the file cannot run under either Python 2 or Python 3.
- py2_except py2_except__t183 · item 3 invalid · `        print(f"Error: {err}")` — The alternative twin combines Python 2-only `except E, err:` syntax with Python 3-only f-strings, so it cannot run under either interpreter.
- py2_iter py2_iter__t162 · item 3 invalid · `            print(f"Invalid unit: {unit_list[i]}")` — The alternative twin mixes Python-2-only constructs (xrange, iteritems, iterkeys) with Python-3-only f-strings, so it cannot run under either interpreter.
- py_builtin_generics py_builtin_generics__t251 · item 3 invalid · `    return Dict(word_counts)` — Calling `typing.Dict(...)` raises TypeError, so the alternative twin does not run and fails the task examples.
- py_private py_private__t041 · item 3 invalid · `        self.n_max = _n_max` — The alternative renamed the __init__ parameter to `n_max` but still reads `_n_max`, so the code raises NameError and cannot run.
- py_with_open py_with_open__t031 · item 3 invalid · `    with open("temp_log_4.txt", "r") as f:` — Opening a non-existent file for reading raises FileNotFoundError, so `distance_2d(0, 0, 3, 4)` would not return 5.0 in either twin.
- sql_bool_case sql_bool_case__t181 · item 3 invalid · `    last_moved DATETIME,` — The script mixes incompatible dialects — DATETIME is not a PostgreSQL type while the later `ON CONFLICT ... DO UPDATE` clause is PostgreSQL-only (MySQL would reject it), so the code cannot run as writ
- sql_bool_case sql_bool_case__t039 · item 3 invalid · `VALUES (2, 103, 100, '2023-10-27', FALSE);` — Warehouse 103 was never inserted into Warehouses, so this row violates the declared FOREIGN KEY (warehouse_id) REFERENCES Warehouses(warehouse_id) and the script would error at runtime (same flaw in t
- trailing_commas trailing_commas__t041 · item 3 invalid · `        row = [],` — While syntactically valid, `row` becomes a tuple so `row.append` raises AttributeError and the natural twin cannot run.
- trailing_commas trailing_commas__t118 · item 1 meaning_change · `            (f"Input: '{input_text}', Expected: '{expected_output}', G` — In the natural twin this was a one-element tuple `(f"...", )`; removing the comma turns it into a plain parenthesized string, changing the assert message's type/meaning.
- trailing_commas trailing_commas__t151 · item 1 meaning_change · `            f"Input: {input_strings}, Expected: {expected_output}, Got` — In the natural twin the trailing comma makes the assert message a one-element tuple, whereas the alternative makes it a plain string, so the two twins differ in meaning (value/type of the assertion me
- bash_subst bash_subst__t180 · item 11 wrong_task · `current_dir_files=$(find . -maxdepth 1 -type f -print0)` — Command substitution discards NUL bytes, so the -print0 output loses its separators and the subsequent NUL count is always 0, making the script exit early and never print any matching file names, so i
- bash_subst bash_subst__t099 · item 11 wrong_task · `for FILE in "$(find "$(readlink -f "$TARGET_DIR")" -maxdepth 1 -type f` — Quoting the whole find output makes the loop iterate once over a multi-line blob, so `stat` fails and the script does not correctly list the matching files.

## New glaring issues (not previously known)

G1. **py_private: 8 alternative twins raise NameError** (t041, t046, t087, t103, t118, t162, t163, t169): a constructor parameter that shares
    the private attribute's name (`def __init__(self, _n_max)`) was renamed in the signature but not in its use (`self.n_max = _n_max`).
    Cause: `code_free.propagate_renames` (run inside filter_free for identifier families) rebuilds the alternative twin from the opportunity
    mapping with its own regex and overrides the AST-verified rule output; its 2026-09-22 patch covers `def _x(` / `_x =` / `_x:` but not a
    plain use of the name. Impact: 8 / 200 alt twins invalid at run time; the cue-level style signal is unaffected. Fix = let the rule's
    twin stand for ALWAYS_RULE families (or extend the regex), re-derive, rebuild prompts, refresh py_private.
G2. **trailing_commas: counted "decisions" that are single-line tuple-making commas.** 43 natural twins (117 of 1,600 counted opportunities,
    7 %) contain lines such as `processed_chars = [],`, `merged_dict = {},`, `return False,` — a trailing comma on a single-line assignment
    or return turns the value into a tuple, so the natural twin's code is wrong and the twins differ in meaning (both reviewers: t041, t118,
    t151). The family definition only allows the comma after the last element of a MULTI-LINE literal. In addition **92 alternative twins keep
    a trailing comma before a closing bracket somewhere** (unconverted occurrences outside the spans, e.g. t003 `(10, 3628800),⏎ ]`), so the
    alternative pole shows the natural convention in the context. Impact: both poles of this family are contaminated (k = 4 alt .70 with
    wrong_style .20; write/read steering for the family unreliable). Fix = regenerate the family with a rule-based twin and the single-line
    guard, or drop it.
G3. **py2_except: 190 / 200 alternative twins run under no Python version** — Python-2-only `except E, e:` combined with f-strings (same for
    py2_iter 15 and py2_print 6). The builder exempts these alt twins from parsing, so nothing caught it. The cue-level signal (the except
    clause) is real, but the alternative pole is incoherent code; a base model has never seen this mixture. Impact on steering numbers:
    unknown; worth noting in the write-up; fix = generate Python-2-only-compatible natural twins (no f-strings) for these three families.
G4. **bash_test: 16 alternative twins use `[ "$x" =~ regex ]`**, which POSIX `[` does not support (runtime error "too many arguments"); 3 more
    use glob patterns with `==` inside `[ ]`, 4 use unquoted `<` / `>`. The `[[ ]] → [ ]` rewrite is unsound for regex / glob tests. Impact:
    ~10 % of alt twins invalid; fix = keep regex tests as `[[ ]]` (treat as non-opportunities) or regenerate those docs with tasks that need
    no regex tests.
G5. **py_builtin_generics: 5 alternative twins call typing aliases as constructors** (`Dict(word_counts)` → TypeError; t193, t198, t205, t251, …):
    the rewrite converted `dict(...)` calls as if they were annotations. Small; fix by hand or rule.
G6. **Unconverted occurrences in alternative twins** (mixed conventions in the alt pole, outside the counted spans): py_join_concat 32 docs
    with a non-path `.join(` left; py_loop_vars 9 docs with a single-letter loop left; sql_join_style 41 docs keep `LEFT JOIN` (no implicit
    form exists — a family-design limit rather than an error, but reviewers reject it); operator_spaces 7 docs keep spaced operators inside
    f-string expressions. Impact: alt-pole contexts carry natural-pole evidence for those docs; moderate for py_join_concat (16 %).
G7. **Scorer errors in the stored results** (measurement, not data): float_literals 16 % of unambiguous natural continuations are recorded as
    alternative; trailing_commas 11 %, py_abbrev 9 %, operator_spaces 6 % mislabelled; js_semicolons 12 % unscorable at closing braces;
    py_optional k = 0 natural continuations unscorable (26 %, the decision is an absent import). The k = 4 accuracies and the steering
    success rates of these families are biased by these amounts. Fix = scorer patches (float_literals: include the cue digits in seg_prefix
    or match `\.\d+` vs `\.(?!\d)` at the cue; trailing_commas: score only the first closer; js_semicolons: allow `}\n` lines).
G8. **py_type_hints: 26 alt k = 1 prompts with an empty evidence list** (absence pole at the first opportunity). Only k = 1 is affected; the
    k = 3/4 read features and prompt_pairs are complete.
G9. **Undefined names in both twins** (pre-existing Gemini bugs, code would not run): trailing_commas t082, py_type_hints (none), py_optional
    t072 + 1, py_comprehension t156, py_ternary t086 + 1, line_wrap t199, py_join_concat t065 (alt only: `no_odds_msg`) — 8 docs across
    36 Python families (pyflakes, undefined-name only).
G10. **Systematic (Part B): 95 % of never-reviewed documents fail the strict reviewers; 40 % are cited for padding by both**, 8 % do not
    solve their stated task (wrong signature, wrong output for the task's own example), 5 % contain invalid code by the reviewers' reading
    (dialect mixing in SQL, `end(explode())` in PHP, FK violations). Individually these are not glaring at the cue level, but the corpus
    that the 2,464-document clean-up left untouched is of the same kind the clean-up removed.

## Not new (already known, confirmed here)
- 10 residual padded docs (js_hungarian 9, py_join_concat 1) — the only padding_lines / leak_lines hits in pool families besides js_arrow t041.
- comment_language: 11 docs with differing k = 0 prompts; comment_case 2 docs and comment_language 1 doc with < 5 counted occurrences (missing prompt items).
- js_hungarian task-pinned parameters left unprefixed (21 docs); reviewers cite it as inconsistent.
- The `5th > 75 %` residuals; py_bool_prefix / early_return / c_braces / css_shorthand / py_ternary outside the pool (they still carry padding / leaks and have no evidence files).
- Prompt lengths up to 2,096 tokens (num_separators) — within the model's context.
