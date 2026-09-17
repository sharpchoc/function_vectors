#!/usr/bin/env python
"""Cheap k = 4 check corpus for the CODE conventions (code_families.py). Per language a shared pool of small tasks (Gemini); per family
and task: Gemini writes the NATURAL-style solution (>= 8 decision points spread over 20-35 lines), then rewrites it changing ONLY the
convention; the twins are aligned by a token diff (non-equal blocks = opportunities, adjacent blocks merged). Kept if >= 5 opportunities,
the 5th starts before 75 % of the code, and >= 60 % of the tokens are shared. Records -> dataset_files/style_translation/pairs/<family>.json
(text_es = task spec, langs = {src: Task, tgt: <Language>}); raw cache dataset_files/style_translation/code/<family>.json;
task pools dataset_files/style_translation/code/tasks_<lang>.json. Then cue tokens + prompts (Qwen) filtered to k in {0, 4}."""
import argparse
import argparse, ast, difflib, json, re, subprocess, sys, time
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
Length {length} lines. Plain code only: no markdown fences, no prose before or after.

TASK:
{spec}"""
REWRITE = """Below is a {lang} program. Rewrite it so that you {rewrite}.
EVERYTHING ELSE must stay byte-for-byte identical: same logic, names, spacing, comments, blank lines, line order. Do not fix or improve anything.
Return only the code, no markdown fences, no prose.

{code}"""
TOK = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)
from src.sandbox.style_translation.code_whitespace_alt import WS_FAMILIES, same_ast, transform as ws_transform
from src.sandbox.style_translation.code_subst_alt import convert as subst_convert, same_tree as subst_same_tree
from src.sandbox.style_translation.code_free import filter_free, AFFECTED, LINE_EXEMPT
# extra generation hints for the families whose decisions can be forced by earlier code (free-opportunity rebuild, 2026-09-16):
# the NATURAL solution must contain many FRESH choice points (new names / new loops / new blocks), not re-mentions
LENGTH = {"rust_question": "35-50", "c_comment_style": "30-45", "py_ternary": "30-45", "py_enumerate": "35-50", "sql_join_style": "35-50", "docstring_style": "40-60",
          "js_strict_eq": "30-45", "py_comprehension": "30-45", "py_join_concat": "30-45", "js_template": "30-45", "py_not_in": "30-45", "py_optional": "30-45",
          "py_fstring": "30-45", "py_builtin_generics": "30-45", "py_is_none": "30-45"}     # solution length guidance per family (default 20-35)
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
    "css_shorthand": "EVERY hex colour must be of the shortenable kind, made of three repeated digit pairs (#ffffff, #336699, #aabbcc, #000000), never #2a7f62-like; write at least 8 such colours AND at least 6 zero lengths with units (0px, 0em), spread through the whole stylesheet with several in the first half",
}


def _chat(key, content, temperature, max_tokens=2000):
    body = {"model": "google/gemini-2.5-flash", "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    r = requests.post(URL, json=body, timeout=150, headers={"Authorization": f"Bearer {key}"}); r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
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


def degenerate(text):
    """Generator output to reject (2026-09-17, bugs 3a/3b): a markdown fence inside the code, a repeated top-level definition (the same
    function / class defined more than once = padded re-implementations), or a control-token artefact."""
    if "```" in text or re.search(r"<ctrl\d+>", text):
        return "fence/artefact"
    heads = re.findall(r"^(?:def|class|function|fn|const|async function)\s+([A-Za-z_]\w*)", text, re.M)
    if any(n > 1 for n in Counter(heads).values()):
        return "repeated definition"
    return None


def build_one(key, fam, task):
    for attempt in range(3):
        try:
            hint = fam.gen_hint + ("; ALSO: " + EXTRA_HINT[fam.name] if fam.name in EXTRA_HINT else "")
            if fam.name not in LINE_EXEMPT:                        # bug 13: only the first occurrence on a line is counted
                hint += "; IMPORTANT: put each occurrence of this style on its OWN line (at most one per line), with at least 8 such lines spread through the whole solution, several of them in the first half"
            nat = _chat(key, GEN.format(lang=fam.tgt_lang, hint=hint, spec=task["spec"], length=LENGTH.get(fam.name, "20-35")), 0.9)
            if nat.count("\n") < 8:
                raise ValueError("too short")
            if degenerate(nat):
                raise ValueError(degenerate(nat))
            if not valid_source(fam.tgt_lang, nat):
                raise ValueError("natural twin does not parse")
            if fam.name in WS_FAMILIES:                            # bug 4 (2026-09-17): whitespace-only families derive alt by rule
                alt = ws_transform(fam.name, nat)
                if alt is None or not same_ast(nat, alt):
                    raise ValueError("whitespace rule failed")
            elif fam.name == "bash_subst":                         # bug 7 (2026-09-17): $(...) -> backticks by rule, verified by parse-back
                alt = subst_convert(nat)
                if not subst_same_tree(nat, alt) or not bash_ok(nat) or not bash_ok(alt):
                    raise ValueError("substitution rule failed")
            else:
                alt = _chat(key, REWRITE.format(lang=fam.tgt_lang, rewrite=fam.rewrite, code=nat), 0.2)
                if degenerate(alt):
                    raise ValueError("alt " + degenerate(alt))
            if fam.name not in ("py2_print", "py2_except") and not valid_source(fam.tgt_lang, alt):
                raise ValueError("alt twin does not parse")
            opps, shared = align(nat, alt)
            ok = opps is not None and len(opps) >= 5 and shared >= 0.6 and opps[4]["nat_span"][0] < 0.75 * len(nat) and all(o["nat"] != o["alt"] for o in opps)
            rec = {"doc_id": f"{fam.name}__{task['id']}", "family": fam.name, "topic": task["title"], "angle": "code", "text_es": task["spec"].strip(),
                   "langs": {"src": "Task", "tgt": fam.tgt_lang}, "text_nat": nat, "text_alt": alt, "opps": opps or [], "k_en": len(opps or []),
                   "shared_fraction": round(shared, 3), "pass": bool(ok), "verify": None}
            if fam.name in WS_FAMILIES or fam.name == "bash_subst":
                rec["alt_rule"] = True
            if ok and fam.name in AFFECTED:                       # free-opportunity rule: the pair must keep >= 5 genuine choice points
                fr = filter_free(rec, fam.name); ok = fr is not None; rec["pass"] = bool(ok)
                if fr is not None:
                    rec.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
            if ok or attempt == 2:
                return rec
        except Exception as e:
            if attempt == 2:
                print(f"{fam.name}/{task['id']} FAILED: {e}", flush=True); return None
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in CODE_FAMILIES])
    ap.add_argument("--n_tasks", type=int, default=50); ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--ks", default="0,4"); ap.add_argument("--skip_prompts", action="store_true")
    ap.add_argument("--target", type=int, default=None, help="stop generating once this many passing pairs exist (chunks of 60 tasks); pairs file capped at --target")
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
                for fu in as_completed([ex.submit(build_one, key, fam, t) for t in chunk]):
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
