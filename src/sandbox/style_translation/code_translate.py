#!/usr/bin/env python
"""Code-translation corpus for the Python convention families (user design 2026-10-13): every document's NATURAL-convention Python program is
translated into ONE other language (JavaScript / Go / Java / Ruby, 50 documents each per family, seeded split), structure kept line by line, and
written in a FIXED style guide for that language so the source program's own conventions are identical across all prompts. Judges (Gemini 2.5 Pro
and Claude Sonnet 4.5, both must accept) check faithfulness, style-guide adherence and runnability; failures are resampled up to --max_attempts
with the judges' notes appended. Fidelity pre-check: the translator lists the Python definitions it translated (must equal the source's, in order).
Output <out>/<family>.json: one record per document {doc_id, lang, code, attempts, judges, passed, history}.
"""
import argparse
import json
import random
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.gen_spanish import load_key, URL
from src.sandbox.style_translation.logprob_margin import PAIRS

LANGS = ["JavaScript", "Go", "Java", "Ruby"]
STYLE = {
    "JavaScript": """JavaScript (Node.js 18, CommonJS). Style guide, to be followed exactly:
- `const` by default, `let` when reassigned, never `var`; a semicolon after every statement (not after function or class declarations); 2-space indentation.
- camelCase for variables and functions, PascalCase for classes, UPPER_SNAKE_CASE for module-level constants.
- Double-quoted string literals; template literals only when interpolating.
- Strict equality (`===`, `!==`); `null` for Python's None; arrays and plain objects (or Map) for lists and dicts.
- `//` for line comments, `/** ... */` doc comments above functions (Python docstrings become doc comments).
- Named `function` declarations for top-level functions, arrow functions only for callbacks. Python exceptions become `throw new Error(...)` (TypeError for type errors) with try/catch.""",
    "Go": """Go (1.20), a single `package main` file. Style guide, to be followed exactly:
- gofmt formatting: tabs for indentation, opening braces on the same line.
- camelCase for unexported identifiers, PascalCase only for exported ones; constants in camelCase unless exported.
- `//` comments; a doc comment above every function (Python docstrings become doc comments).
- Double-quoted string literals; slices and maps for lists and dicts; errors returned as a trailing `error` value (no panics for expected failures);
  Python's None becomes nil pointers or a bool `ok` flag as appropriate; `fmt` for output; Python exceptions become returned errors created with `fmt.Errorf`.""",
    "Java": """Java (17), a single file with one public class `Main` holding static methods (nested static classes for Python classes). Style guide, to be
followed exactly:
- 4-space indentation, K&R braces; camelCase methods and variables, PascalCase classes, UPPER_SNAKE_CASE for static final constants.
- Double-quoted strings; `List`/`Map` from java.util for lists and dicts; `null` for Python's None.
- `//` for line comments, `/** ... */` Javadoc above every method (Python docstrings become Javadoc).
- Python ValueError becomes IllegalArgumentException, TypeError becomes IllegalArgumentException, other exceptions the closest java.lang type; try/catch.""",
    "Ruby": """Ruby (3.x). Style guide, to be followed exactly:
- 2-space indentation; snake_case methods and variables, PascalCase classes, SCREAMING_SNAKE_CASE constants.
- Double-quoted string literals; `#{}` interpolation; `nil` for Python's None; arrays and hashes for lists and dicts.
- `#` comments; a comment block above every method (Python docstrings become comment blocks).
- `do ... end` for multi-line blocks, `{ }` for single-line blocks; no semicolons; Python ValueError and TypeError become `raise ArgumentError, "..."`, KeyError becomes `raise KeyError`; `def`/`end` methods.""",
}

PROMPT = """Translate the following Python program into {lang_head}.

Rules:
- Keep the structure line by line as far as the target language allows: the same functions in the same order, the same control flow, the same
  variable roles, the same string literals and the same behaviour. Do not add or remove functions. Do not reorder statements. Do not optimise.
- Follow the style guide below EXACTLY; the same style must be used for every program, so do not improvise. Never copy Python naming,
  quoting, comment or layout conventions into the translation.
- In statically typed targets use proper static types (int, []int, string, ...) in signatures and drop Python runtime type checks that the
  type system makes impossible; keep None checks only where the native type can be nil/null.
- Map Python exceptions to the closest native mechanism of the target language (named in the style guide); map repr()/%r formatting to the
  closest native equivalent; these mappings are expected and not deviations.
- Declare every variable; do not leave unused helpers or constants. Keep the Python function names (converted to the guide's case); if a name
  collides with a builtin of the target language, append a short suffix. If the language needs its own entry point: in Go, Python's `main()` becomes `func main()`;
  in Java, Python's `main()` becomes `public static void main(String[] args)` (if Python's main takes parameters, keep it as `run(...)` and call it
  from `main`). Translate the `if __name__ == "__main__"` guard into that entry point; do not add any other wrapper.
- The program must be self-contained and runnable.

Style guide:
{style}
{feedback}
Return ONLY a JSON object with two keys: "code" (the full translated program as one string) and "defs" (the ordered list of the Python
function and class names you translated, as they appear in the Python source).

Task description of the program:
{task}

Python program:
```python
{code}
```"""

JUDGE = """You are reviewing a translation of a Python program into {lang_head}. The translation exists to fix the CONTENT of the program (which
functions, in which order, doing what) in ONE fixed style, so that a later Python rendering has nothing left to decide except Python's own
conventions. Answer the questionnaire below with a JSON object only, and nothing before it.

Differences that the target language forces are NOT defects and must be answered as if absent: static types in signatures replacing Python's
runtime type checks (isinstance, bool rejection, None checks the type system makes impossible); optional or keyword parameters rendered as
pointers, variadics, overloads or options; errors returned instead of raised; exception types mapped to native ones; message wording, quoting or
repr() rendering; whitespace or line-reading details of the standard library; integer sizes; container types; an added entry point required by
the language; a Python name adjusted to the guide's case or suffixed to avoid a builtin; tiny helpers the language needs (e.g. a repr helper).

Questions:
q1_definitions: does every Python function and class have exactly one counterpart, in the same order, with no Python definition dropped,
  merged or split? (true/false)
q2_logic: does each counterpart perform the same steps as the Python (same branches, loops, early returns and computations on ordinary
  inputs), ignoring the forced differences listed above? Do NOT report: rounding or float-formatting semantics, truthiness or emptiness checks
  rendered the native way, iteration or zip length details, string-formatting mechanics, added bounds or nil checks that the native container
  needs, parsing or date handling re-implemented because the target library differs. Error handling is FREE in mechanism: as long as the same
  conditions lead to an error and the same message text is produced, it does not matter whether it is raised, thrown, returned, caught by a
  different native type or checked before the operation. Report only a missing or extra step, a different branch structure, or a different
  computation on ordinary inputs. (true/false; if false, name the function
  and the differing step)
q3_runnable: would the program run or compile? Report only definite errors: syntax errors, undeclared or out-of-scope variables, missing
  imports, type errors that fail compilation. (true/false; if false, name them)
q4_style: does the translation follow the style guide on these four points only: identifier case (camelCase / snake_case / PascalCase as the
  guide says); quote character of plain string literals (template literals, interpolation and escaped quotes are fine); comment marker style
  (`//`, `#`, `/** */` as the guide says; the internal format of doc comments is free); indentation unit (spaces/tabs and width)? Anything
  else about style is NOT a violation. (true/false; if false, name the point)

Style guide:
{style}

Return {{"q1_definitions": true/false, "q2_logic": true/false, "q3_runnable": true/false, "q4_style": true/false, "notes": "<for every false
answer, at most 25 words naming the concrete problem; otherwise 'none'>"}}.

Python program:
```python
{code}
```

Translation:
```
{translation}
```"""


REPAIR = """Below is a Python program, its translation into {lang_head}, and reviewer notes explaining why the translation was rejected.
Fix the translation so that every note is addressed, changing as little as possible otherwise. The translation must keep the structure of the
Python line by line (same functions in the same order, same control flow, same behaviour) and must follow the style guide EXACTLY.

Style guide:
{style}

Reviewer notes:
{feedback}

Return ONLY a JSON object with two keys: "code" (the full corrected program as one string) and "defs" (the ordered list of the Python function
and class names, as they appear in the Python source).

Python program:
```python
{code}
```

Current translation:
```
{translation}
```"""


def defs_of(src):
    return re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", src, flags=re.M)


def top_defs_of(src):
    return re.findall(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", src, flags=re.M)


def defs_match(got, src):
    """The translator's list may name the top-level definitions only, or include methods; order must be preserved either way."""
    return got == defs_of(src) or got == top_defs_of(src) or (set(got) >= set(top_defs_of(src)) and [g for g in got if g in defs_of(src)] == [d for d in defs_of(src) if d in set(got)])


def _chat(key, model, content, max_tokens, timeout=900, attempts=4, temperature=1.0, reasoning=None):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    err = None
    for _ in range(attempts):
        try:
            resp = requests.post(URL, json=body, timeout=timeout, headers={"Authorization": f"Bearer {key}"}); resp.raise_for_status()
            raw = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            v, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])
            return v, None
        except Exception as e:
            err = str(e)[:200]
    return None, err


def translate(key, model, rec, lang, feedback="", previous=None):
    head = STYLE[lang].split("\n")[0].rstrip(":")
    if previous:                                    # repair the previous translation instead of starting over
        content = REPAIR.format(lang_head=head, style=STYLE[lang], feedback=feedback, code=rec["text_nat"], translation=previous)
    else:
        content = PROMPT.format(lang_head=head, style=STYLE[lang], feedback="", task=rec["text_es"], code=rec["text_nat"])
    v, err = _chat(key, model, content, 20000, reasoning="high")
    if v is None or not isinstance(v.get("code"), str):
        return None, [], err or "no code"
    return v["code"], [str(x) for x in v.get("defs", [])], None


def syntax_check(lang, code):
    """Local syntax check where an interpreter is installed (node --check, ruby -c); None when unavailable."""
    import shutil, subprocess, tempfile
    cmd = {"JavaScript": (["node", "--check"], ".js"), "Ruby": (["ruby", "-c"], ".rb")}.get(lang)
    if cmd is None or shutil.which(cmd[0][0]) is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=cmd[1], delete=False) as f:
        f.write(code); path = f.name
    r = subprocess.run(cmd[0] + [path], capture_output=True, text=True, timeout=60)
    return None if r.returncode == 0 else (r.stderr or r.stdout).strip()[:300]


def judge(key, judges, rec, lang, code):
    out = {}
    syn = syntax_check(lang, code)
    if syn:
        return {"syntax": {"ok": False, "notes": f"syntax check failed: {syn}"}}
    checked = syntax_check.__doc__ and lang in ("JavaScript", "Ruby") and __import__("shutil").which({"JavaScript": "node", "Ruby": "ruby"}[lang]) is not None
    note = "\n(The translation PASSED a local syntax check with the language's own parser; do not report syntax errors.)\n" if checked else ""
    for name, model in judges.items():
        v, err = _chat(key, model, JUDGE.format(lang_head=STYLE[lang].split("\n")[0].rstrip(":"), style=STYLE[lang], code=rec["text_nat"], translation=code) + note, 6000, attempts=4, temperature=0.0)
        if v:
            qs = {q: bool(v.get(q)) for q in ("q1_definitions", "q2_logic", "q3_runnable", "q4_style")}
            out[name] = {"ok": all(qs.values()), **qs, "notes": str(v.get("notes", ""))[:400] if not all(qs.values()) else "none"}
        else:
            out[name] = {"ok": False, "notes": f"judge error: {err}"}
    return out


def process(key, args, judges, rec, lang, prior):
    """Translate + judge with resampling; `prior` = existing record (resumed) or None."""
    r = prior or dict(doc_id=rec["doc_id"], family=rec["family"], lang=lang, code=None, attempts=0, judges=None, passed=False, history=[])
    feedback = ""
    if r["history"]:
        feedback = r["history"][-1].get("feedback", "")
    if prior and not r["passed"] and r["code"] and r["history"] and r["history"][-1].get("judges"):   # resumed: re-judge the last code under the current rules
        J = judge(key, judges, rec, lang, r["code"]); ok = all(j["ok"] for j in J.values())
        feedback = "; ".join(f"{n}: {j['notes']}" for n, j in J.items() if not j["ok"])
        r["history"].append({"attempt": r["attempts"], "rejudge": True, "judges": J, "feedback": feedback}); r["judges"], r["passed"] = J, ok
        if not ok:
            r["attempts"] = min(r["attempts"], args.max_attempts - 2)
    while not r["passed"] and r["attempts"] < args.max_attempts:
        r["attempts"] += 1
        code, defs, err = translate(key, args.model, rec, lang, feedback, previous=r["code"] if feedback else None)
        src = defs_of(rec["text_nat"])
        if code is None:
            r["history"].append({"attempt": r["attempts"], "error": err}); continue
        if not defs_match(defs, rec["text_nat"]):
            feedback = f"The list of translated definitions {defs} does not match the Python source definitions {src}; translate every definition in order."
            r["history"].append({"attempt": r["attempts"], "fidelity_ok": False, "defs_model": defs, "feedback": feedback}); r["code"] = code; continue
        J = judge(key, judges, rec, lang, code)
        ok = all(j["ok"] for j in J.values())
        feedback = "; ".join(f"{n}: {j['notes']}" for n, j in J.items() if not j["ok"])
        r["history"].append({"attempt": r["attempts"], "fidelity_ok": True, "judges": J, "feedback": feedback})
        r["code"], r["judges"], r["passed"] = code, J, ok
    return r


def split_langs(doc_ids, seed):
    ids = sorted(doc_ids); rng = random.Random(zlib.crc32(f"{seed}|{ids[0].split('__')[0]}".encode())); rng.shuffle(ids)
    return {d: LANGS[i % len(LANGS)] for i, d in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="openai/gpt-5"); ap.add_argument("--judge_gemini", default="google/gemini-2.5-pro"); ap.add_argument("--judge_claude", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--max_attempts", type=int, default=4, help="1 initial + 3 resamples"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", default="translation_v1"); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); key = load_key()
    judges = {"gemini": args.judge_gemini, "claude": args.judge_claude}
    for fam in args.families:
        recs = json.load(open(PAIRS / f"{fam}.json")); lang_of = split_langs([r["doc_id"] for r in recs], args.seed)
        recs = recs[: args.limit]; out = args.out / f"{fam}.json"
        done = {r["doc_id"]: r for r in json.load(open(out))} if out.exists() else {}
        todo = [r for r in recs if not done.get(r["doc_id"], {}).get("passed")]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(lambda rec: process(key, args, judges, rec, lang_of[rec["doc_id"]], done.get(rec["doc_id"])), todo):
                done[r["doc_id"]] = r
                if len(done) % 25 == 0:
                    json.dump(list(done.values()), open(out, "w"), ensure_ascii=False, indent=1)
        json.dump(list(done.values()), open(out, "w"), ensure_ascii=False, indent=1)
        rs = list(done.values()); by = {l: [r for r in rs if r["lang"] == l] for l in LANGS}
        print(f"{fam}: passed {sum(r['passed'] for r in rs)}/{len(rs)} | " + " ".join(f"{l} {sum(r['passed'] for r in v)}/{len(v)}" for l, v in by.items())
              + f" | mean attempts {sum(r['attempts'] for r in rs) / max(len(rs), 1):.2f}", flush=True)
    print("translation done", flush=True)


if __name__ == "__main__":
    main()
