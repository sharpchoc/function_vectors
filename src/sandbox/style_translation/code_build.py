#!/usr/bin/env python
"""Cheap k = 4 check corpus for the CODE conventions (code_families.py). Per language a shared pool of small tasks (Gemini); per family
and task: Gemini writes the NATURAL-style solution (>= 8 decision points spread over 20-35 lines), then rewrites it changing ONLY the
convention; the twins are aligned by a token diff (non-equal blocks = opportunities, adjacent blocks merged). Kept if >= 5 opportunities,
the 5th starts before 75 % of the code, and >= 60 % of the tokens are shared. Records -> dataset_files/style_translation/pairs/<family>.json
(text_es = task spec, langs = {src: Task, tgt: <Language>}); raw cache dataset_files/style_translation/code/<family>.json;
task pools dataset_files/style_translation/code/tasks_<lang>.json. Then cue tokens + prompts (Qwen) filtered to k in {0, 4}."""
import argparse
import argparse, ast, difflib, json, re, subprocess, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.gen_spanish import load_key, URL
from src.sandbox.style_translation.code_families import CODE_FAMILIES, CODE_FAMILY
from src.sandbox.style_translation.models import paths as model_paths

PAIRS = STYLE_TRANSLATION_DATA / "pairs"; RAW = STYLE_TRANSLATION_DATA / "code"; PY = sys.executable
TASK_PROMPT = {
 "default": """Produce {n} short, self-contained programming tasks for {lang}, each solvable in 20-35 lines of code. Vary the domains widely
(text processing, arithmetic, records/dictionaries, dates, parsing, simple simulations, geometry, inventory, scheduling, statistics, games).
Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}} where spec is 2-4 sentences stating what to implement,
the inputs and outputs, the exact function/script name and signature to use, and two example calls with their expected results.""",
 "SQL": """Produce {n} short SQL tasks. Each spec describes a small schema (2-3 tables with columns) and asks for 3-5 queries (selects with
filters, joins, aggregates, inserts/updates) that a 20-35 line SQL script can satisfy. Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}}.""",
 "CSS": """Produce {n} short CSS tasks. Each spec describes a small page (5-8 elements with class/id names) and the visual styling required
(colors given as hex, spacing, borders, fonts) that a 20-35 line stylesheet can satisfy. Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}}.""",
 "Bash": """Produce {n} short Bash scripting tasks (file handling, string processing, argument checking, simple loops), each solvable in a 20-35 line
script, with the inputs/outputs and 2 example invocations. Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}}.""",
}
GEN = """Write a complete {lang} solution to the task below.
STYLE REQUIREMENT (this is the point of the exercise): {hint}. Apply it consistently, and make sure the places where it applies are spread
over the WHOLE solution, including the second half — not clustered at the top.
NO PADDING (2026-09-18): every occurrence of the required construct must arise from the task's own logic; an independent strict reviewer
rejects the solution if ANY line exists only to raise the occurrence count. Forbidden: filler, placeholder, dummy or illustrative lines;
checks that can never fire or whose body is only `pass`; unused variables; helper functions that are never called or not needed for the
task; self-test / example-usage blocks; discarded results (`_ = ...`, `let _ = ...`); comments that announce examples, count occurrences or
name the style; stacked negations (`not x not in y`, `!!x`); `elif True`, `for _ in range(1)`, repeated boilerplate. If the task alone cannot
produce enough occurrences, extend the solution with genuinely useful, actually-used logic (validation the task's inputs really need, a
parsing / formatting / reporting step, a caller that the task's examples exercise) — the result must still solve the stated task with the
stated name and signature.
Length {length} lines. Plain code only: no markdown fences, no prose before or after.

TASK:
{spec}"""
REWRITE = """Below is a {lang} program. Rewrite it so that you {rewrite}.
EVERYTHING ELSE must stay byte-for-byte identical: same logic, names, spacing, comments, blank lines, line order. Do not fix or improve anything.
Return only the code, no markdown fences, no prose.

{code}"""
TOK = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)
from src.sandbox.style_translation.code_rule_alt import RULE_ALT, ALWAYS_RULE
from src.sandbox.style_translation.code_whitespace_alt import WS_FAMILIES, same_ast, transform as ws_transform
from src.sandbox.style_translation.code_subst_alt import convert as subst_convert, same_tree as subst_same_tree
from src.sandbox.style_translation.code_sql_alt import transform as sql_transform, sql_ok
from src.sandbox.style_translation.code_free import filter_free, AFFECTED, LINE_EXEMPT, fifth_frac
# extra generation hints for the families whose decisions can be forced by earlier code (free-opportunity rebuild, 2026-09-16):
# the NATURAL solution must contain many FRESH choice points (new names / new loops / new blocks), not re-mentions
LENGTH = {"rust_question": "40-80", "c_comment_style": "30-45", "py_ternary": "30-45", "py_enumerate": "35-50", "sql_join_style": "35-50", "docstring_style": "40-60",
          "js_strict_eq": "30-45", "py_comprehension": "30-45", "py_join_concat": "30-45", "js_template": "30-45", "py_not_in": "30-45", "py_optional": "30-45",
          "py_fstring": "30-45", "py_builtin_generics": "30-45", "py_is_none": "40-80", "js_arrow": "30-45",
          "py_not_in": "40-80", "float_literals": "40-80"}     # solution length guidance per family (default 20-35); 40-80 = FREE_OCC_HINT families (2026-09-18)
# occurrence count asked of the generator in the bug-13 line hint (default 8); per-family overrides (none at present)
OCC_HINT = {}
# user decision 2026-09-18 (OPTION A, strengthened): these families get NO numeric occurrence requirement — the construct is described and the
# generator writes a realistic, longer solution in which occurrences arise naturally (LENGTH 40-80); EXTRA_HINT / OCC_HINT are not applied
FREE_OCC_HINT = {
    "py_not_in": "negated membership tests written as `x not in y`",
    "rust_question": "error propagation with the `?` operator in functions that return Result / Option",
    "py_is_none": "comparisons with None written as `is None` / `is not None`",
    "float_literals": "float literals with digits on both sides of the point (0.5, 1.0, 2.25, 0.75)",
}
FREE_OCC_TEXT = ("use the construct wherever the task's logic genuinely calls for it; write a realistic, complete solution — it may be a small "
                 "module with several related functions and a short driver/main that actually uses them — so that occurrences arise naturally; "
                 "never add code, branches, checks, variables or comments whose only purpose is to create an occurrence")
EXTRA_HINT = {
    "py_snake_camel": "introduce at least 8 DIFFERENT multi-word variable, parameter and function names, each a new name (do not just re-use two or three names)",
    "js_camel_snake": "introduce at least 8 DIFFERENT multi-word variable, parameter and function names, each a new name (do not just re-use two or three names)",
    "py_const_naming": "define at least 6 DIFFERENT module-level constants, each with a distinct multi-word name",
    "py_class_naming": "define at least 7 DIFFERENT small classes (each with a distinct multi-word name), spread through the whole solution",
    "py_private": "assign at least 7 DIFFERENT private attributes in __init__ and other methods, each with a distinct multi-word name",
    "py_bool_prefix": "introduce at least 7 DIFFERENT boolean variables / parameters, each with a distinct multi-word name",
    "py_loop_vars": "write at least 7 DIFFERENT for-loops (nested or sequential) spread through the whole solution, each with its own loop variable",
    "js_hungarian": "introduce at least 8 DIFFERENT variables and parameters, each with a distinct multi-word name",
    "py_abbrev": "introduce at least 8 DIFFERENT variables and parameters, each a distinct name (do not re-use the same few names)",
    "py_self_name": "define a class with at least 7 methods (each `def method(self, ...)`), spread through the whole solution",
    "py_indent": "use at least 8 DIFFERENT indented blocks (if / for / while / def / try / with), several of them nested, spread through the whole solution",
    "py_tabs": "use at least 8 DIFFERENT indented blocks (if / for / while / def / try / with), several of them nested, spread through the whole solution",
    "early_return": "write at least 7 SEPARATE early-exit checks (if <bad case>: return ... / continue) in several functions and loops, spread through the whole solution",
    "trailing_commas": "the trailing comma goes ONLY after the last element of a multi-line list / dict / tuple / call literal, never after a statement; the code must be valid Python",
    "comment_case": "ONLY the comment text starts with a capital letter; Python keywords and identifiers keep their normal spelling",
    "bash_subst": "use at least 9 SEPARATE $(...) command substitutions spread through the whole script, several of them in the first half (only the opening of each substitution counts as an opportunity)",
    "py_with_open": "use at least 8 SEPARATE `with open(...) as f:` blocks, each opening one file, spread through the whole solution with several in the first half (only the `with` line of each block counts as an opportunity)",
    "py_class_naming": "no variable, parameter or attribute may be the snake_case form of a class name (e.g. a class ResultContainer must not coexist with a name result_container)",
    "py_bool_prefix": "the boolean names must stay distinct from every other name once the is_/has_/should_ prefix is dropped (has_numbers next to a parameter numbers is NOT allowed)",
    # bug 10 (2026-09-17): only the OPENING symbol of a pair is a decision, so paired-symbol families need more, and earlier, occurrences
    "py_paren_if": "ONLY the if / elif / while conditions are written without parentheses; function calls, definitions, subscripts and every other parenthesis stay as normal valid Python; write at least 10 such conditions, several of them in the first half of the solution",
    "docstring_quotes": "write at least 8 docstrings (module, classes, methods, functions), several of them in the first half of the file",
    "py_quotes": "write at least 12 string literals spread through the whole solution, several of them in the first half",
    "js_quotes": "write at least 12 string literals spread through the whole solution, several of them in the first half",
    "js_template": "write at least 10 template literals spread through the whole solution, several of them in the first half",
    "php_array": "write at least 10 array literals spread through the whole solution, several of them in the first half",
    "bash_test": "write at least 10 test expressions spread through the whole script, several of them in the first half",
    "py_join_concat": "write at least 10 such string constructions spread through the whole solution, several of them in the first half",
    "py_not_in": "write at least 8 such membership tests spread through the whole solution, several of them in the first half",
    # bug 12: one decision per natural line -> these families need many constructs, each on its own line
    "rust_question": "use the ? operator at least 9 times, each use on its OWN statement line, spread through the whole solution with several in the first half",
    "c_comment_style": "write at least 10 SEPARATE single-line // comments (not blocks of consecutive comment lines), each on its own line above or beside code, spread through the whole solution",
    "py_ternary": "write at least 9 SEPARATE one-line conditional-expression assignments (x = a if cond else b), each on its own line, spread through the whole solution with several in the first half",
    "py_enumerate": "write at least 8 SEPARATE `for i, x in enumerate(...)` loops, spread through the whole solution with several in the first half",
    # bug 13: one construct = one decision, at most one counted per line -> spread the occurrences over separate lines / statements
    "sql_join_style": "write at least 8 SEPARATE queries, each its own statement ending in a semicolon and each containing exactly ONE join between two tables",
    "docstring_style": "define at least 8 functions or methods, EACH with its own docstring that has an argument section, spread through the whole file",
    "js_strict_eq": "write at least 9 equality comparisons, each on its OWN line (never two comparisons on one line), spread through the whole solution",
    "py_comprehension": "write at least 8 SEPARATE one-line list comprehensions, each on its own line, spread through the whole solution with several in the first half",
    "py_optional": "write at least 8 SEPARATE function signatures or annotated variables that use Optional[...], one per line, spread through the whole solution",
    # bug 15 / fix 4 (2026-09-18): line-level counting = one opportunity per arrow function, so the solution needs many of them
    "js_arrow": "define at least 10 arrow functions (const f = (a, b) => {...} assignments AND inline callbacks such as arr.map(x => x * 2)), each on its own line, spread through the whole solution with several in the first half",
    "css_shorthand": "EVERY hex colour must be of the shortenable kind, made of three repeated digit pairs (#ffffff, #336699, #aabbcc, #000000), never #2a7f62-like; write at least 8 such colours AND at least 6 zero lengths with units (0px, 0em), spread through the whole stylesheet with several in the first half",
}


GEN_MODEL = "google/gemini-2.5-flash"          # default generator / rewriter; override per run with --gen_model or build_one(gen_model=...)
REFUSAL_FALLBACK = None                         # model used when the provider of `model` refuses a call (set by drivers; None = raise)
REJECTS = Counter()                             # guard rejections inside build_one, by reason
USAGE = Counter(); _USAGE_LOCK = threading.Lock()  # per-model prompt / completion tokens and OpenRouter cost (USD) of every _chat call in this process


def _chat(key, content, temperature, max_tokens=2000, model=None, reasoning=None, timeout=150):
    """One OpenRouter chat call; returns the stripped text (markdown fence removed). `model` defaults to GEN_MODEL; `reasoning` (e.g.
    {"enabled": False} or {"effort": "low"}) is passed through for reasoning models. Usage is accumulated in USAGE (per model)."""
    model = model or GEN_MODEL
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}], "usage": {"include": True}}
    if temperature is not None:                                    # None = omit (OpenAI reasoning models reject the parameter)
        body["temperature"] = temperature
    if reasoning is not None:
        body["reasoning"] = reasoning
    r = requests.post(URL, json=body, timeout=timeout, headers={"Authorization": f"Bearer {key}"}); r.raise_for_status()
    js = r.json()
    if "choices" not in js:
        raise RuntimeError(f"openrouter: {js.get('error')}")
    ch0 = js["choices"][0]
    if ch0.get("finish_reason") == "content_filter" or ch0.get("native_finish_reason") == "refusal":   # provider safety filter (false positives on
        with _USAGE_LOCK:                                          # network / bit-level code, found 2026-09-21): never treat the empty reply as code
            REJECTS["provider refusal " + model.split("/")[0]] += 1
        if REFUSAL_FALLBACK and model != REFUSAL_FALLBACK:
            return _chat(key, content, temperature, max_tokens, REFUSAL_FALLBACK, {"effort": "low"}, max(timeout, 300))
        raise RuntimeError("provider refusal (content filter)")
    u = js.get("usage") or {}
    with _USAGE_LOCK:
        USAGE[(model, "prompt_tokens")] += u.get("prompt_tokens", 0); USAGE[(model, "completion_tokens")] += u.get("completion_tokens", 0)
        USAGE[(model, "cost")] += u.get("cost", 0.0); USAGE[(model, "calls")] += 1
    out = js["choices"][0]["message"]["content"] or ""
    out = re.sub(r"^\s*```[a-zA-Z0-9+#-]*\s*\n", "", out.strip()); out = re.sub(r"\n\s*```\s*$", "", out)
    return out.rstrip() + "\n"


def tasks_for(key, lang, n=60, batch=60):
    """Task pool of at least n tasks, grown in batches of `batch` (each batch told to avoid the existing titles); ids t001.. are stable."""
    f = RAW / f"tasks_{lang}.json"
    pool = json.load(open(f)) if f.exists() else []
    while len(pool) < n:
        avoid = "; ".join(t["title"] for t in pool[-120:])
        extra = f"\nDo NOT repeat or closely paraphrase any of these existing task titles: {avoid}" if pool else ""
        for attempt in range(4):
            try:
                raw = _chat(key, TASK_PROMPT.get(lang, TASK_PROMPT["default"]).format(n=batch, lang=lang) + extra, 0.9, 12000)
                js = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
                js = [t for t in js if isinstance(t, dict) and t.get("spec")]
                assert len(js) >= 30, len(js)
                seen = {t["title"].strip().lower() for t in pool}
                for t in js:
                    if t["title"].strip().lower() in seen:
                        continue
                    seen.add(t["title"].strip().lower()); t["id"] = f"t{len(pool) + 1:03d}"; pool.append(t)
                RAW.mkdir(parents=True, exist_ok=True); json.dump(pool, open(f, "w"), indent=1); break
            except Exception as e:
                print(f"tasks {lang} batch attempt {attempt} failed: {e}", flush=True); time.sleep(3)
        else:
            raise RuntimeError(f"could not grow the task pool for {lang}")
        print(f"task pool {lang}: {len(pool)}", flush=True)
    return pool[:max(n, len(pool))]


def align(nat, alt, merge_gap=2):
    """Opportunities from a token diff. Returns (opps, shared_fraction) or (None, 0) if the rewrite is not a local edit."""
    ta, tb = TOK.findall(nat), TOK.findall(alt)
    if "".join(ta) != nat or "".join(tb) != alt:
        return None, 0.0
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    blocks = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    shared = sum(i2 - i1 for tag, i1, i2, _, _ in sm.get_opcodes() if tag == "equal") / max(len(ta), 1)
    if not blocks:
        return [], shared
    merged = [list(blocks[0])]
    for b in blocks[1:]:
        if b[0] - merged[-1][1] <= merge_gap and b[2] - merged[-1][3] <= merge_gap:
            merged[-1][1], merged[-1][3] = b[1], b[3]
        else:
            merged.append(list(b))
    offa = [0]; [offa.append(offa[-1] + len(t)) for t in ta]
    offb = [0]; [offb.append(offb[-1] + len(t)) for t in tb]
    opps = []
    for k, (i1, i2, j1, j2) in enumerate(merged):
        opps.append({"k": k, "nat": nat[offa[i1]:offa[i2]], "alt": alt[offb[j1]:offb[j2]], "nat_span": [offa[i1], offa[i2]], "alt_span": [offb[j1], offb[j2]]})
    return opps, shared


def bash_ok(text):
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text); p = fh.name
    r = subprocess.run(["bash", "-n", p], capture_output=True, text=True); os.unlink(p)
    return r.returncode == 0


PARSERS = Path("/workspace/micromamba/envs/parsers/bin")                     # node / rustfmt / php (bug 9, 2026-09-17)


def valid_source(lang, text):
    """Syntax check with a real parser: Python (ast), JavaScript (node --check), Rust (rustfmt, parse only), PHP (php -l), Bash (bash -n).
    Languages without a parser here (SQL, CSS, R) return True."""
    import os, tempfile
    if lang == "Python":
        return valid_python(text)
    if lang == "Bash":
        return bash_ok(text)
    suf, cmd = {"JavaScript": (".js", [str(PARSERS / "node"), "--check"]), "Rust": (".rs", [str(PARSERS / "rustfmt"), "--check", "--edition", "2021"]),
                "PHP": (".php", [str(PARSERS / "php"), "-l"])}.get(lang, (None, None))
    if suf is None or not (PARSERS / cmd[0].split("/")[-1]).exists():
        return True
    with tempfile.NamedTemporaryFile("w", suffix=suf, delete=False) as fh:
        fh.write(text); p = fh.name
    r = subprocess.run(cmd + [p], capture_output=True, text=True, timeout=120); os.unlink(p)
    return ("error" not in r.stderr) if lang == "Rust" else r.returncode == 0


def valid_python(text):
    try:
        ast.parse(text); return True
    except SyntaxError:
        return False


# padding guard (2026-09-18, user finding): the >= 8-occurrence hint made the generator manufacture occurrences. A NATURAL twin is rejected
# if any line matches one of these. "code" patterns are matched on the code part of the line (trailing # / // comment removed), "line"
# patterns on the whole line. The sweep (code_pairs_check.padding_line) reuses the same constant.
PADDING_PATTERNS = [
    # genuine double negation of ONE operand only (`not x not in y`, `not (not x)`, `!!x`, `!(!x)`); `not a or not b` is ordinary code (user decision 2026-09-18)
    ("stacked_negation", re.compile(r"\bnot\b(?:(?!\b(?:and|or|if|else)\b)[^#\n])*?\bnot\s+in\b|\bnot\s*\(\s*not\b|!\s*!|!\s*\(\s*!"), "code"),
    ("filler_wording", re.compile(r"dummy|placeholder|demonstrat|showcase|to show|for illustration|just to|just for|another example|yet another|meet .* count", re.I), "line"),
    ("example_comment", re.compile(r"#\s*(Example(?!\s+usage)|Test case|Demo)\b"), "line"),        # `# Example usage` = a normal trailing usage block (user decision)
    ("discarded_result", re.compile(r"^\s*(let\s+_\s*=|_\s*=|const\s+_\w*\s*=|var\s+_\w*\s*=)"), "line"),
    ("trivial_branch", re.compile(r"\belif\s+True\b|\bif\s+True\s*:|\bfor\s+\w+\s+in\s+range\(1\)\s*:"), "code"),
    ("style_meta_comment", re.compile(r"(?:#|//).*(?:style requirement|(?:the|this) (?:style|convention)\b|for (?:the )?style\b|opportunit|as (?:required|requested) by the (?:style|exercise))", re.I), "line"),
]
_COMMENT_MARK = {"Python": "#", "Bash": "#", "R": "#", "SQL": "--"}


# style-leak guard (2026-09-21, padding audit category 1): a COMMENT or docstring that talks about the style / the exercise, or names the
# family's own convention words (nat_label / alt_label minus generic words). Matched on comment text only (`comment_texts`).
LEAK_PATTERNS = [
    # "style" only in a code-style sense: stylistic, or style next to a signal word (code / naming / this / the same / required ... style,
    # style requirement / guide / rule / convention); "output styles", "'x' style flags", "clock-style" are task content and do not match
    ("style_vocab", re.compile(r"\bstylistic\b|\b(?:code|coding|naming|this|that|the|same|consistent|required|requested|python|pep\s*8|our|preferred|"
                               r"correct|proper|new|old|alternative|chosen|desired|target|intended|specified|mandated)\s+styl(?:e|es|ing)\b|"
                               r"\bstyl(?:e|es|ing)\s+(?:requirement|guide|guideline|rule|constraint|preference|consistency|convention|of the exercise|"
                               r"as required|as requested|demand|check|compliance|point|purpose|reason)s?\b|"
                               r"\bconventions?\b|\brequirements?\b|\bconsisten(?:cy|t|tly)\b|\bidiom(?:s|atic)?\b|\bPEP\s*-?\s*8\b|\bfor the exercise\b|"
                               r"\bas (?:required|requested)\b|\bto (?:satisfy|meet|fulfil)", re.I)),
]
_GENERIC_LABEL_WORDS = {"identifiers", "identifier", "names", "name", "function", "functions", "constants", "constant", "class", "private", "attributes",
                        "boolean", "loop", "variables", "variable", "strings", "string", "literals", "literal", "numeric", "plain", "full", "this",
                        "self", "if", "x", "none", "statement", "blocks", "block", "explicit", "implicit", "comments", "comment", "tests", "test",
                        "arrays", "array", "substitution", "assignment", "operator", "join", "on", "with", "open", "and", "no", "long", "lines",
                        "calls", "between", "defs", "one", "two", "blank", "digit", "single-letter", "descriptive", "wrapped", "int", "e", "as",
                        "print", "print()", "docstrings", "spelled-out", "abbreviated", "bare", "keywords", "match", "arrow", "expressions", "clauses",
                        "guard", "nested", "hex", "list", "builtins", "typing.list", "list[int]", "dict()", "list()", "{}", "[]", "/", "in", "y",
                        "not", "is", "==", "===", "!==", "!=", "<-", "=", "?", "[[", "]]", "[", "]", "$(...)", "backtick", "//", "/*", "*/",
                        "except", "range/items/keys", "xrange/iteritems/iterkeys", "0px", "0", "//", "range(len())", "enumerate", "str.join", "+",
                        "concatenation", "str.format", "f-strings", "conditional", "if/else", "spaces", "space", "around", "after", "comma",
                        "operators", "uppercase", "lowercase", "true/false", "capitalised", "english", "spanish", "4-space", "2-space", "indentation",
                        "tabs", "k&r", "allman", "braces", "google", "numpy", "triple", "double", "single", "quotes", "single-quoted", "double-quoted",
                        "semicolons", "trailing", "commas", "type", "hints", "optional[x]", "x | none", "|", "template", "hungarian", "notation",
                        "underscore", "separators", "decimal", "hexadecimal", "float", "abbreviated", "x:", "(x):", "is_/has_", "pascalcase",
                        "camelcase", "snake_case", "upper_snake", "const/let", "var", "?"}


def family_words(fam):
    """Words of the family's two labels that name its convention (used by the leak check). Generic words are dropped; words with an
    underscore / mixed case (snake_case, camelCase, PascalCase, UPPER_SNAKE, is_/has_, Hungarian) are the informative ones."""
    F = CODE_FAMILY[fam]; words = set()
    for lab in (F.nat_label, F.alt_label):
        for w in re.split(r"[\s/]+", lab):
            w = w.strip("()").strip()
            if len(w) >= 3 and w.lower() not in _GENERIC_LABEL_WORDS:
                words.add(w)
    return words | set(FAMILY_LEAK_WORDS.get(fam, ()))


# explicit convention words per family for the leak check (matched case-insensitively as whole words in comments / docstrings)
FAMILY_LEAK_WORDS = {
    "py_snake_camel": ("snake_case", "camelCase", "snake case", "camel case"), "js_camel_snake": ("snake_case", "camelCase", "snake case", "camel case"),
    "js_func_pascal": ("camelCase", "PascalCase", "camel case", "pascal case"), "py_const_naming": ("UPPER_SNAKE", "camelCase", "upper snake", "screaming"),
    "py_class_naming": ("PascalCase", "snake_case", "pascal case", "snake case", "CapWords"), "py_private": ("single underscore", "double underscore", "name mangling"),
    "py_bool_prefix": ("is_/has_", "boolean prefix", "is_ prefix", "has_ prefix"), "py_loop_vars": ("single-letter", "single letter", "descriptive loop", "loop variable"),
    "js_hungarian": ("Hungarian",), "py_abbrev": ("abbreviated", "abbreviation", "spelled-out", "spelled out"),
    "py_quotes": ("single-quoted", "double-quoted", "single quotes", "double quotes"), "js_quotes": ("single-quoted", "double-quoted", "single quotes", "double quotes"),
    "py_fstring": ("f-string", "f-strings", "str.format", ".format("), "js_template": ("template literal", "template literals", "concatenation"),
    "num_separators": ("digit separator", "underscore separator", "numeric separator", "separators"), "hex_constants": ("hexadecimal", "hex constant", "hex literal", "decimal literal"),
    "float_literals": ("float literal",), "sql_bool_case": ("uppercase", "lowercase"), "py2_print": ("print statement", "print function", "Python 2", "Python 3"),
    "py2_iter": ("xrange", "iteritems", "iterkeys", "Python 2", "Python 3"), "py2_except": ("Python 2", "Python 3", "except syntax"),
    "js_var": ("const/let", "var keyword", "let/const", "block-scoped", "block scoped"), "js_arrow": ("arrow function", "arrow functions", "function expression"),
    "js_semicolons": ("semicolon", "semicolons"), "js_strict_eq": ("strict equality", "strict comparison", "loose equality", "===", "!=="),
    "trailing_commas": ("trailing comma", "trailing commas"), "py_paren_if": ("parenthes",), "py_type_hints": ("type hint", "type hints", "annotation", "annotations"),
    "py_optional": ("Optional[", "X | None", "union syntax"), "py_builtin_generics": ("typing.List", "builtin generic", "built-in generic", "PEP 585"),
    "py_literal_ctor": ("literal syntax", "dict()", "list()", "constructor"), "py_comprehension": ("comprehension", "comprehensions"),
    "py_not_in": ("membership test",), "py_is_none": ("None comparison", "None check"), "py_ternary": ("ternary", "conditional expression"),
    "early_return": ("early return", "early-return", "guard clause", "guard clauses"), "py_join_concat": ("str.join", ".join(", "concatenation"),
    "py_with_open": ("context manager", "with statement", "with open"), "py_enumerate": ("range(len",), "py_self_name": ("self keyword", "this keyword"),
    "py_indent": ("indentation", "4-space", "four-space", "2-space"), "py_tabs": ("tabs", "spaces for indentation", "indentation"),
    "c_braces": ("K&R", "Allman", "brace style", "braces"), "operator_spaces": ("spaces around", "operator spacing", "whitespace"),
    "comma_space": ("space after comma", "comma spacing", "whitespace"), "line_wrap": ("line wrap", "wrapped", "line length", "long line"),
    "blank_lines": ("blank line", "blank lines"), "docstring_style": ("Google style", "NumPy style", "Google-style", "NumPy-style", "docstring format"),
    "docstring_quotes": ("triple double", "triple single", "triple quotes", "triple-quoted"), "comment_case": ("capitalised", "capitalized", "lowercase comment"),
    "comment_language": ("English", "Spanish"), "c_comment_style": ("line comment", "block comment", "// comment", "/* comment"),
    "sql_keyword_case": ("uppercase", "lowercase", "keyword case"), "sql_join_style": ("explicit join", "implicit join", "comma join", "JOIN ... ON"),
    "r_assignment": ("<- assignment", "assignment operator", "arrow assignment"), "rust_question": ("? operator", "question mark", "error propagation"),
    "bash_test": ("double bracket", "single bracket", "[[ ]]", "POSIX test"), "bash_subst": ("command substitution", "backtick", "backticks", "$(...)"),
    "css_shorthand": ("shorthand", "short hex", "long hex"), "php_array": ("short array", "array()", "array syntax", "short syntax"),
}
_COMMENT_SYNTAX = {"Python": ("#", None, True), "Bash": ("#", None, False), "R": ("#", None, False), "SQL": ("--", ("/*", "*/"), False),
                   "JavaScript": ("//", ("/*", "*/"), False), "C": ("//", ("/*", "*/"), False), "Rust": ("//", ("/*", "*/"), False),
                   "PHP": ("//", ("/*", "*/"), False), "CSS": (None, ("/*", "*/"), False)}


def comment_texts(text, lang="Python"):
    """{line_no (1-based): comment text on that line} — the part after a line-comment marker outside string literals, the content of a
    block comment (/* */) or, for Python, of a triple-quoted string (docstring); lines without comment text are absent."""
    line_mark, block, triple = _COMMENT_SYNTAX.get(lang, ("#", None, False)); out = {}; i = 0; n = len(text); ln = 1
    cur = []                                                       # comment chars of the current line
    def flush():
        nonlocal cur
        t = "".join(cur).strip()
        if t:
            out[ln] = out.get(ln, "") + t
        cur = []
    while i < n:
        c = text[i]
        if c == "\n":
            flush(); ln += 1; i += 1; continue
        if triple and text.startswith(('"""', "\'\'\'"), i):
            q = text[i:i + 3]; j = text.find(q, i + 3); j = n if j < 0 else j + 3
            seg = text[i:j]
            for ch in seg:
                if ch == "\n":
                    flush(); ln += 1
                else:
                    cur.append(ch)
            i = j; continue
        if block and text.startswith(block[0], i):
            j = text.find(block[1], i + 2); j = n if j < 0 else j + 2
            for ch in text[i:j]:
                if ch == "\n":
                    flush(); ln += 1
                else:
                    cur.append(ch)
            i = j; continue
        if line_mark and text.startswith(line_mark, i):
            j = text.find("\n", i); j = n if j < 0 else j
            cur.extend(text[i:j]); i = j; continue
        if c in "\"'`":                                            # skip a string literal (same-line; unterminated -> rest of line)
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            i = min(j + 1, n); continue
        i += 1
    flush()
    return out


def leak_lines(text, lang="Python", fam=None):
    """[(line_no, pattern_name, line)] for every line whose COMMENT / docstring text matches LEAK_PATTERNS or (with `fam`) names the
    family's convention words (FAMILY_LEAK_WORDS); empty list = clean."""
    lines = text.split("\n"); out = []; fw = FAMILY_LEAK_WORDS.get(fam, ()) if fam else ()
    fw_rx = re.compile("|".join(r"(?<![\w])" + re.escape(w) + r"(?![\w])" for w in fw), re.I) if fw else None
    for ln, ctext in sorted(comment_texts(text, lang).items()):
        hit = next((name for name, rx in LEAK_PATTERNS if rx.search(ctext)), None)
        if hit is None and fw_rx is not None and fw_rx.search(ctext):
            hit = "family_word"
        if hit:
            out.append((ln, hit, lines[ln - 1]))
    return out


_PAD_STR_RX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\\n])*`')


def padding_lines(text, lang="Python"):
    """[(line_no, pattern_name, line)] for every (line, pattern) of `text` that matches a PADDING_PATTERNS entry (empty list = clean).
    String-literal contents are masked before matching (test data such as "Another example!!" is not padding); "code" patterns are matched on
    the code part of the line (trailing comment removed), "line" patterns on the whole masked line. A line can appear with several patterns."""
    mark = _COMMENT_MARK.get(lang, "//"); out = []
    for i, line in enumerate(text.split("\n"), 1):
        masked = _PAD_STR_RX.sub(lambda m: m.group()[0] + "s" * (len(m.group()) - 2) + m.group()[-1] if len(m.group()) >= 2 else m.group(), line)
        code = masked.split(mark, 1)[0]
        for name, rx, where in PADDING_PATTERNS:
            if rx.search(code if where == "code" else masked):
                out.append((i, name, line))
    return out


def degenerate(text):
    """Generator output to reject (2026-09-17, bugs 3a/3b): a markdown fence inside the code, a repeated top-level definition (the same
    function / class defined more than once = padded re-implementations), or a control-token artefact."""
    if "```" in text or re.search(r"<ctrl\d+>", text):
        return "fence/artefact"
    heads = re.findall(r"^(?:def|class|function|fn|const|async function)\s+([A-Za-z_]\w*)", text, re.M)
    if any(n > 1 for n in Counter(heads).values()):
        return "repeated definition"
    return None


FREE_OCC_LENGTH = "40-80"


def free_occ_description(fam):
    """The construct description used with the no-count hint: the explicit FREE_OCC_HINT text or, for any other family, its own gen_hint
    with the numeric requirement removed ("use at least 6 float literals" -> "use float literals")."""
    if fam.name in FREE_OCC_HINT:
        return FREE_OCC_HINT[fam.name]
    h = re.sub(r"\s*\((?:with )?at least \d+[^)]*\)", "", fam.gen_hint)          # "(at least 12 statements)" -> dropped
    h = re.sub(r"\bat least \d+ (?:times|such [a-z]+)\b", "several times", h)
    h = re.sub(r"\bat least \d+\b", "several", h)
    return re.sub(r"\s{2,}", " ", h).strip(" ;,")


def gen_hint(fam, free_occ=False):
    """The STYLE REQUIREMENT text of GEN for a family: gen_hint + EXTRA_HINT + the bug-13 one-per-line sentence (>= OCC_HINT.get(name, 8)
    lines), or — for FREE_OCC_HINT families, or for any family when `free_occ` is set (regenerated documents, 2026-09-21) — the construct
    description + FREE_OCC_TEXT + one-per-line / several-early without a number."""
    if fam.name in FREE_OCC_HINT or free_occ:
        h = free_occ_description(fam) + ": " + FREE_OCC_TEXT
        if fam.name not in LINE_EXEMPT:
            h += "; IMPORTANT: at most one occurrence of the construct per line, and let several occurrences appear early in the file rather than all at the end"
        return h
    n_occ = OCC_HINT.get(fam.name, 8)
    extra = re.sub(r"at least \d+", f"at least {n_occ}", EXTRA_HINT[fam.name]) if fam.name in OCC_HINT and fam.name in EXTRA_HINT else EXTRA_HINT.get(fam.name)
    h = re.sub(r"at least \d+", f"at least {n_occ}", fam.gen_hint) if fam.name in OCC_HINT else fam.gen_hint
    h += ("; ALSO: " + extra) if extra else ""
    if fam.name not in LINE_EXEMPT:                                # bug 13: only the first occurrence on a line is counted
        h += f"; IMPORTANT: put each occurrence of this style on its OWN line (at most one per line), with at least {n_occ} such lines spread through the whole solution, several of them in the first half"
    return h


def finish_pair(key, fam, task, nat, gen_model=None, free_occ=False, rewrite_notes=None, rule_alt=False):
    """Everything after the natural twin exists: guards (length, degenerate, padding, parser), the alternative twin (rule or LLM rewrite with
    `gen_model`), alignment, the free filter. Raises ValueError on a guard failure; returns the record (pass may be False when the
    alignment / counted rules fail). Used by build_one and by the feedback-repair loop (revised natural twins)."""
    rew_r = {"enabled": False} if (gen_model or GEN_MODEL).startswith("anthropic/") else None
    if nat.count("\n") < 8:
        raise ValueError("too short")
    if degenerate(nat):
        raise ValueError(degenerate(nat))
    pad = padding_lines(nat, fam.tgt_lang)
    if pad:                                                        # padding guard (2026-09-18): manufactured occurrences
        raise ValueError("padding: " + "; ".join(f"L{i} {n}: {l.strip()[:60]}" for i, n, l in pad[:3]))
    leak = leak_lines(nat, fam.tgt_lang, fam.name)
    if leak:                                                       # style-leak guard (2026-09-21): comments naming the style / convention
        raise ValueError("style leak: " + "; ".join(f"L{i} {n}: {l.strip()[:60]}" for i, n, l in leak[:3]))
    if not valid_source(fam.tgt_lang, nat):
        raise ValueError("natural twin does not parse")
    if fam.name in WS_FAMILIES:                                    # bug 4 (2026-09-17): whitespace-only families derive alt by rule
        alt = ws_transform(fam.name, nat)
        if alt is None or not same_ast(nat, alt):
            raise ValueError("whitespace rule failed")
    elif fam.name == "bash_subst":                                 # bug 7 (2026-09-17): $(...) -> backticks by rule, verified by parse-back
        alt = subst_convert(nat)
        if not subst_same_tree(nat, alt) or not bash_ok(nat) or not bash_ok(alt):
            raise ValueError("substitution rule failed")
    elif fam.name == "sql_keyword_case":                           # bug 15 (2026-09-18): only SQL keywords are lower-cased, by rule (round-trip verified)
        alt = sql_transform(nat)
        if not sql_ok(nat, alt):
            raise ValueError("sql keyword rule failed (mixed keyword case or no keyword)")
    elif (rule_alt or fam.name in ALWAYS_RULE) and fam.name in RULE_ALT:                        # 2026-09-21: exact rename / literal rules (code_rule_alt.py), self-verified by AST
        alt = RULE_ALT[fam.name](nat, task.get("spec", ""))
        if alt is None and fam.name in ALWAYS_RULE:
            raise ValueError("rule rewrite failed (rename collision, nothing to convert, or AST check)")
        if alt is None:                                            # regen8: a declining rule falls back to the LLM rewrite (never silently: counted + flagged)
            with _USAGE_LOCK:
                REJECTS["rule declined -> llm rewrite"] += 1
            alt = _chat(key, REWRITE.format(lang=fam.tgt_lang, rewrite=fam.rewrite, code=nat), 0.2, max_tokens=16000, model=gen_model, reasoning=rew_r); rule_alt = False
            if degenerate(alt):
                raise ValueError("alt " + degenerate(alt))
    else:
        content = REWRITE.format(lang=fam.tgt_lang, rewrite=fam.rewrite, code=nat)
        if rewrite_notes:                                          # rewrite-repair (2026-09-21): reviewer feedback on an earlier rewrite of this same program
            content = content.replace("\nReturn only the code", "\nAn earlier rewrite of this exact program was REJECTED for the problems below; do not repeat them. Convert EVERY "
                                      "occurrence the rule covers (no exceptions), and touch nothing the rule does not cover (strings, comments, output text):\n" + rewrite_notes + "\nReturn only the code", 1)
        alt = _chat(key, content, 0.2, max_tokens=16000, model=gen_model, reasoning=rew_r)
        if degenerate(alt):
            raise ValueError("alt " + degenerate(alt))
    if fam.name not in ("py2_print", "py2_except") and not valid_source(fam.tgt_lang, alt):
        raise ValueError("alt twin does not parse")
    opps, shared = align(nat, alt)
    rec = {"doc_id": f"{fam.name}__{task['id']}", "family": fam.name, "topic": task["title"], "angle": "code", "text_es": task["spec"].strip(),
           "langs": {"src": "Task", "tgt": fam.tgt_lang}, "text_nat": nat, "text_alt": alt, "opps": opps or [], "k_en": len(opps or []),
           "shared_fraction": round(shared, 3), "pass": False, "verify": None}
    if free_occ:
        rec["free_occ"] = True                                     # no-count hint: 5th opportunity threshold 85 % (code_free.fifth_frac)
    ok = opps is not None and len(opps) >= 5 and shared >= 0.6 and opps[4]["nat_span"][0] < fifth_frac(fam.name, rec) * len(nat) and all(o["nat"] != o["alt"] for o in opps)
    rec["pass"] = bool(ok)
    if fam.name in WS_FAMILIES or fam.name in ("bash_subst", "sql_keyword_case") or ((rule_alt or fam.name in ALWAYS_RULE) and fam.name in RULE_ALT):
        rec["alt_rule"] = True
    if ok and fam.name in AFFECTED:                               # free-opportunity rule: the pair must keep >= 5 genuine choice points
        fr = filter_free(rec, fam.name); ok = fr is not None; rec["pass"] = bool(ok)
        if fr is not None:
            rec.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
    return rec


def gen_length(fam, free_occ=False):
    return FREE_OCC_LENGTH if (free_occ or fam.name in FREE_OCC_HINT) else LENGTH.get(fam.name, "20-35")


def generate_nat(key, fam, task, gen_model=None, free_occ=False):
    """A fresh natural twin from the task (GEN prompt; Anthropic models: reasoning effort medium). `free_occ`: no-count hint + 40-80 lines."""
    gen_r = {"effort": "medium"} if (gen_model or GEN_MODEL).startswith("anthropic/") else None
    return _chat(key, GEN.format(lang=fam.tgt_lang, hint=gen_hint(fam, free_occ), spec=task["spec"], length=gen_length(fam, free_occ)), 0.9, max_tokens=16000, model=gen_model, reasoning=gen_r)


def build_one(key, fam, task, gen_model=None, free_occ=False):
    """Natural twin from the task (GEN, all guards), alternative twin by rule or by LLM rewrite (same model), align + free filter;
    up to 3 attempts. `gen_model` overrides GEN_MODEL for both calls; `free_occ` = the per-document no-count hint (regenerated docs)."""
    for attempt in range(3):
        try:
            rec = finish_pair(key, fam, task, generate_nat(key, fam, task, gen_model, free_occ), gen_model, free_occ)
            if rec["pass"] or attempt == 2:
                return rec
        except Exception as e:
            with _USAGE_LOCK:                                      # guard-rejection tally (reason up to the first colon), for run reports
                REJECTS[str(e).split(":")[0][:60]] += 1
            if attempt == 2:
                print(f"{fam.name}/{task['id']} FAILED: {e}", flush=True); return None
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in CODE_FAMILIES])
    ap.add_argument("--n_tasks", type=int, default=50); ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--ks", default="0,4"); ap.add_argument("--skip_prompts", action="store_true")
    ap.add_argument("--target", type=int, default=None, help="stop generating once this many passing pairs exist (chunks of 60 tasks); pairs file capped at --target")
    ap.add_argument("--gen_model", default=GEN_MODEL, help="OpenRouter model for the natural twin and the LLM rewrite (default GEN_MODEL)")
    args = ap.parse_args()
    key = load_key(); RAW.mkdir(parents=True, exist_ok=True)
    langs = sorted({CODE_FAMILY[f].tgt_lang for f in args.families})
    with ThreadPoolExecutor(len(langs)) as ex:                   # pools grow independently per language (each batch depends on the previous one)
        pools = dict(zip(langs, ex.map(lambda lang: tasks_for(key, lang, n=args.n_tasks + 10), langs)))
    for lang, ts in pools.items():
        print(f"task pool {lang}: {len(ts)}", flush=True)
    for name in args.families:
        fam = CODE_FAMILY[name]; raw_path = RAW / f"{name}.json"
        raw = {r["doc_id"]: r for r in json.load(open(raw_path))} if raw_path.exists() else {}
        def apply_free():                                        # the free-opportunity rule, applied to every stored raw doc (old and new)
            if name in AFFECTED:
                for r in raw.values():
                    fr = filter_free(r, name) if (r.get("opps_all") or r.get("opps")) else None
                    r["pass"] = bool(r.get("pass")) and fr is not None      # the free rule can only REMOVE a pass, never grant one
                    if fr is not None:
                        r.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
        apply_free()
        todo = [t for t in pools[fam.tgt_lang][: args.n_tasks] if f"{name}__{t['id']}" not in raw]
        chunks = [todo[i:i + 60] for i in range(0, len(todo), 60)] if args.target else [todo]
        for chunk in chunks:
            n_pass = sum(r["pass"] for r in raw.values())
            if args.target and n_pass >= args.target:
                break
            if not chunk:
                continue
            with ThreadPoolExecutor(args.workers) as ex:
                for fu in as_completed([ex.submit(build_one, key, fam, t, args.gen_model) for t in chunk]):
                    r = fu.result()
                    if r: raw[r["doc_id"]] = r
            json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_path, "w"), ensure_ascii=False, indent=0)
            apply_free()
            print(f"{name}: {sum(r['pass'] for r in raw.values())} passing after {len(raw)} docs", flush=True)
        apply_free(); json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_path, "w"), ensure_ascii=False, indent=0)
        pairs = [r for r in sorted(raw.values(), key=lambda r: r["doc_id"]) if r["pass"]][: args.target or None]
        json.dump(pairs, open(PAIRS / f"{name}.json", "w"), ensure_ascii=False, indent=0)
        ks = sorted(r["k_en"] for r in raw.values())
        print(f"{name:20s} built={len(raw):3d} opps median={ks[len(ks)//2] if ks else 0:3d} pass={len(pairs):3d} shared={sum(r['shared_fraction'] for r in raw.values())/max(len(raw),1):.2f}", flush=True)
    print("openrouter usage:", {f"{m}/{k}": (round(v, 4) if k == "cost" else v) for (m, k), v in sorted(USAGE.items())}, flush=True)
    if args.skip_prompts:
        return
    fams = [f for f in args.families if (PAIRS / f"{f}.json").exists() and json.load(open(PAIRS / f"{f}.json"))]
    subprocess.run([PY, "src/sandbox/style_translation/cue_tokens.py", "--model", "qwen25_base", "--families", *fams], check=True, cwd=_BOOT)
    subprocess.run([PY, "src/sandbox/style_translation/build_prompts.py", "--model", "qwen25_base", "--K", "5", "--families", *fams], check=True, cwd=_BOOT)
    if args.ks != "all":
        keep = {int(k) for k in args.ks.split(",")}; MP = model_paths("qwen25_base")
        for f in fams:
            p = MP["prompts"] / f"{f}.json"; items = [it for it in json.load(open(p)) if it["k"] in keep]
            json.dump(items, open(p, "w")); print(f"{f}: {len(items)} prompts kept")


if __name__ == "__main__":
    main()
