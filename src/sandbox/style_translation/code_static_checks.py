"""Free automatic pre-checks for regenerated code-style pairs (2026-09-22, regen8): everything that can fail a strict review without
paying for a reviewer call. Python: pyflakes (unused variables / imports, undefined names, redefinitions) + a sandboxed run of the natural
twin (NameError, AttributeError, never-run code); JavaScript: `node --check` via valid_source + unused declarations + a sandboxed `node` run;
all languages: string literals identical between the twins. Returns (codes, feedback lines)."""
import io, os, re, subprocess, sys, tempfile
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
NODE = "/workspace/micromamba/envs/parsers/bin/node"
_STR = re.compile(r'([A-Za-z]*)("(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\])*`)')
_JS_MASK = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|/(?![/*])(?:\\.|\[(?:\\.|[^\]\\])*\]|[^/\\\n])+/[gimsuy]*", re.S)


def python_flakes(text):
    """pyflakes messages that a strict reviewer would call padding or invalid code."""
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:
        return []
    out, err = io.StringIO(), io.StringIO(); check(text, "<doc>", Reporter(out, err))
    msgs = []
    for line in out.getvalue().splitlines():
        m = re.match(r"<doc>:(\d+):\d+:? (.*)", line)
        if not m:
            continue
        msg = m.group(2)
        if any(k in msg for k in ("assigned to but never used", "imported but unused", "undefined name", "redefinition of unused", "is assigned to but never used")):
            msgs.append(f"line {m.group(1)}: {msg}")
    return msgs


def js_unused(text):
    """const / let / var / function declarations never referenced elsewhere (strings, comments, regex literals masked)."""
    code = _JS_MASK.sub(lambda m: m.group(0) if m.group(0).startswith("`") else " " * len(m.group(0)), text); msgs = []   # template literals kept: ${name} is a use
    for m in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", code):          # declarations only: a top-level function may be the task's entry point
        name = m.group(1)
        if len(re.findall(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])", code)) < 2:
            msgs.append(f"line {code[:m.start()].count(chr(10)) + 1}: `{name}` is declared but never used")
    return msgs


def run_sandboxed(lang, text, timeout=6):
    """Run the natural twin (Python / JavaScript) in a temp dir; returns None if it exits 0, else the last stderr line."""
    if lang not in ("Python", "JavaScript"):
        return None
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "main.py" if lang == "Python" else "main.js"); open(p, "w").write(text)
        cmd = [sys.executable, "-I", p] if lang == "Python" else [NODE, p]
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return None                                            # waits for input / a timer: not a programming error
        if r.returncode != 0:
            tail = [l for l in r.stderr.strip().splitlines() if l.strip()]; last = tail[-1] if tail else ""
            if re.match(r"^\s*(NameError|AttributeError|TypeError|SyntaxError|IndentationError|UnboundLocalError|ZeroDivisionError|IndexError|KeyError|RecursionError|ReferenceError|RangeError)\b", last) \
                    or re.search(r"\b(ReferenceError|TypeError|SyntaxError|RangeError): ", "\n".join(tail[-3:])):
                return last[:200]                                  # a programming error; CLI usage / missing-input exits are fine
    return None


def string_leak(nat, alt):
    """Plain string literals must be identical between the twins; f-strings / template literals are skipped (an identifier rename inside
    an interpolation is part of the convention)."""
    lit = lambda t: sorted(s for pre, s in _STR.findall(t) if "f" not in pre.lower() and not s.startswith("`"))
    a, b = lit(nat), lit(alt)
    return None if a == b else [x for x in b if x not in a][:5]


STRING_FAMILIES = {"py_quotes", "js_quotes", "docstring_quotes", "docstring_style", "comment_language", "comment_case", "c_comment_style", "py_fstring", "js_template", "py_join_concat", "py2_print", "bash_subst", "bash_test", "sql_keyword_case", "line_wrap"}


def precheck(F, rec):
    """Static checks on a builder record; returns (codes, feedback). Empty codes = pass."""
    codes, fb = [], []; lang = F.tgt_lang; nat = rec["text_nat"]; alt = rec["text_alt"]
    if lang == "Python":
        for m in python_flakes(nat):
            codes.append("flakes"); fb.append(f"[automatic static check] {m} (a strict reviewer calls an unused name padding)")
        if F.name not in ("py2_print", "py2_iter", "py2_except", "py_type_hints", "py_optional", "py_builtin_generics"):
            for m in python_flakes(alt)[:2]:
                codes.append("flakes_alt"); fb.append(f"[automatic static check, alternative twin] {m}")
    elif lang == "JavaScript":
        for m in js_unused(nat):
            codes.append("unused_js"); fb.append(f"[automatic static check] {m}")
    err = run_sandboxed(lang, nat)
    if err:
        codes.append("runtime"); fb.append(f"[automatic run of the natural twin] it fails: {err}")
    if F.name not in STRING_FAMILIES:
        leak = string_leak(nat, alt)
        if leak:
            codes.append("string_leak"); fb.append(f"[automatic check] the rewrite changed string literals: {leak} — strings and printed text must stay identical")
    return sorted(set(codes)), fb
