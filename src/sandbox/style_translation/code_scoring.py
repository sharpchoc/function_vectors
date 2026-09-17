#!/usr/bin/env python
"""Context-aware convention scoring for code families whose regex pair mis-scores (2026-09-17, user-approved scoring fixes).

Identifier families: the opportunity is the FIRST MENTION of a new name, so only names the completion INTRODUCES may decide — a name
already present in the prompt (code so far or task text) is a forced reuse and carries no decision (the v1 regexes scanned 160 characters
and were triggered by such reuses in 10–70 % of scored completions). `decide_code(fam, prompt_text, seg_prefix, tail, next_nat, next_alt,
lexicon)` returns 'nat' / 'alt' / None, or NotImplemented for families that keep the regex classifier.

  py_snake_camel / js_camel_snake  first new multi-word name: snake_case vs camelCase
  py_const_naming                  first new constant-like name: UPPER_SNAKE vs camelCase
  py_class_naming                  first new name after `class`: PascalCase vs snake_case
  js_hungarian                     first new name: type-prefixed (nCount, strName, ...) vs plain
  py_loop_vars                     first new `for` target: single letter vs descriptive
  py_abbrev / py_bool_prefix       first new name with a clear label in the LLM lexicon (artifacts/.../scoring_lexicon.json)
  hex_constants                    first integer literal: decimal vs 0x... (v1 only knew nine specific decimal numbers)
"""
import builtins
import keyword
import re

IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
PY_SKIP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls", "this", "args", "kwargs"}
JS_SKIP = set("""break case catch class const continue debugger default delete do else export extends finally for function if import in instanceof let new
return super switch this throw try typeof var void while with yield async await of static get set null undefined true false NaN Infinity console Math JSON Object
Array String Number Boolean Date RegExp Error Map Set Promise parseInt parseFloat isNaN require module exports length push pop log""".split())
LEX_FAMILIES = {"py_abbrev", "py_bool_prefix"}
CTX_FAMILIES = {"py_snake_camel", "js_camel_snake", "py_const_naming", "py_class_naming", "js_hungarian", "py_loop_vars", "hex_constants"} | LEX_FAMILIES


def strip_literals(code, js=False):
    code = re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S) if js else re.sub(r"#[^\n]*", " ", code)
    return re.sub(r'"""(?:\\.|[^\\])*?"""|\'\'\'(?:\\.|[^\\])*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\])*`', '""', code, flags=re.S)


def split_prompt(prompt_text):
    """(task text, code so far): the prompt is 'Task:\n<spec>\n\n<Language>:\n<code>'."""
    m = list(re.finditer(r"\n\n[A-Za-z+#]+:\n", prompt_text))
    return (prompt_text[:m[-1].start()], prompt_text[m[-1].end():]) if m else ("", prompt_text)


def known_names(prompt_text):
    """Names the completion cannot 'introduce': every identifier of the code so far, plus the identifiers the task text pins down as code
    (inside backticks or written as a call) — NOT the ordinary English words of the task description."""
    task, code = split_prompt(prompt_text)
    pinned = set()
    for m in re.finditer(r"`([^`]+)`", task):
        pinned.update(IDENT.findall(m.group(1)))
    pinned.update(re.findall(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", task))
    return set(IDENT.findall(code)) | pinned


def new_identifiers(seg, seg_prefix, known, js=False, limit=160):
    """Names the completion introduces, in order: not in the prompt, not a keyword / builtin, not an attribute access (`.name`), not inside
    a string or comment. A word-internal cue (seg_prefix) means the name at position 0 is the one being introduced, whatever its prefix."""
    code = strip_literals(seg[:limit], js); skip = JS_SKIP if js else PY_SKIP; out = []
    for m in IDENT.finditer(code):
        name = m.group()
        if m.start() == 0 and seg_prefix:
            out.append(name); continue
        if name in skip or name in known or (m.start() > 0 and code[m.start() - 1] == "."):
            continue
        if name not in out:
            out.append(name)
    return out


def _snake(n):
    return re.fullmatch(r"_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+", n) is not None


def _camel(n):
    return re.fullmatch(r"[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+", n) is not None


def decide_code(fam, prompt_text, seg_prefix, tail, next_nat, next_alt, lexicon=None):
    if fam not in CTX_FAMILIES:
        return NotImplemented
    seg = seg_prefix + tail
    if fam == "hex_constants":
        code = strip_literals(seg[:160], js=True)
        md, mh = re.search(r"(?<![\w.$])\d+(?![\w.$])", code), re.search(r"(?<![\w$])0[xX][0-9a-fA-F]+", code)
        return "nat" if md and (mh is None or md.start() < mh.start()) else ("alt" if mh else None)
    js = fam.startswith("js_"); known = known_names(prompt_text); last_line = prompt_text.rsplit("\n", 1)[-1]
    if fam == "py_class_naming":
        for m in re.finditer(r"\bclass\s+([A-Za-z_]\w*)", last_line + strip_literals(tail[:200])):   # last_line already ends with seg_prefix
            n = m.group(1); at_cue = m.start(1) <= len(last_line)          # the name that straddles / follows the cue is THE decision
            if n in known and not at_cue:
                continue
            return "nat" if re.fullmatch(r"[A-Z][A-Za-z0-9]*", n) else ("alt" if re.fullmatch(r"[a-z_][a-z0-9_]*", n) else None)
        return None
    if fam == "py_loop_vars":
        for m in re.finditer(r"\bfor\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s+in\b", last_line + strip_literals(tail[:200])):
            names = [x for x in re.split(r"\s*,\s*", m.group(1)) if x != "_"]; at_cue = m.start() < len(last_line) and m.end(1) >= len(last_line)
            if not names or (all(x in known for x in names) and not at_cue):
                continue
            return "alt" if any(len(x) > 1 for x in names) else "nat"
        return None
    for n in new_identifiers(seg, seg_prefix, known, js):
        if fam in ("py_snake_camel", "js_camel_snake"):
            lab = "snake" if _snake(n) else ("camel" if _camel(n) else None)
            if lab:
                return ("nat" if lab == "snake" else "alt") if fam == "py_snake_camel" else ("nat" if lab == "camel" else "alt")
        elif fam == "py_const_naming":
            if re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*", n) and len(n) > 1:
                return "nat"
            if _camel(n):
                return "alt"
        elif fam == "js_hungarian":
            if re.fullmatch(r"(?:[a-z]{0,2}(?:str|arr|obj|bool|num|int|flt|dbl|fn|func|var|map|set|lst|arg)|n|i|f|d|s|sz|a|b|o|c|ch|l|p|rg|dw)[A-Z][A-Za-z0-9]*", n):
                return "alt"
            if re.fullmatch(r"[a-z][A-Za-z0-9]*", n):
                return "nat"
        elif fam in LEX_FAMILIES:
            lab = (lexicon or {}).get(fam, {}).get(n)
            if lab in ("nat", "alt"):
                return lab
    return None


def decide_any(fam, prompt_text, seg_prefix, tail, next_nat, next_alt, lexicon=None):
    """Single entry point for rollout scoring: the context-aware rule for CTX_FAMILIES (with the exact next-token fallback), the regex
    classifier of scoring.decide for every other family."""
    from src.sandbox.style_translation.scoring import decide
    d = decide_code(fam, prompt_text, seg_prefix, tail, next_nat, next_alt, lexicon)
    if d is NotImplemented:
        return decide(fam, seg_prefix, tail, next_nat, next_alt)
    if d is None:
        for label, nxt in sorted((("nat", next_nat), ("alt", next_alt)), key=lambda kv: -len(kv[1])):
            if nxt and tail.startswith(nxt[: max(1, min(len(nxt), 6))]):
                return label
    return d
