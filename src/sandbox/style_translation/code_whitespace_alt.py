#!/usr/bin/env python
"""Deterministic alternative twins for the six whitespace-only Python families (bug 4, 2026-09-17).

The alternative rendering of py_indent, py_tabs, blank_lines, operator_spaces, comma_space and line_wrap is a pure
formatting change, so it is derived from the natural twin by rule instead of by an LLM rewrite (which slipped in
content edits in 11 of ~1,100 documents). Every rule works on the `tokenize` stream, so string literals and comments
are never touched, and the result must have the same AST as the natural twin (`same_ast`).

  py_indent        leading spaces of every code line halved (4-space levels -> 2-space levels)
  py_tabs          every 4 leading spaces -> one tab (remainder kept as spaces)
  blank_lines      a run of >= 2 blank lines before a top-level def / class / decorator -> one blank line
  operator_spaces  whitespace around every BINARY / assignment / augmented operator removed (unary -, *args, **kw kept)
  comma_space      whitespace after every comma removed (comma before a comment or a line break kept)
  line_wrap        every bracket group that spans several lines and holds no comment is joined into one line; the
                   trailing comma of a wrapped call / definition / literal is dropped (kept for a 1-tuple / subscript)

`transform(family, text)` returns the alternative text or None when the natural twin does not tokenize.
"""
import ast
import io
import keyword
import re
import tokenize

WS_FAMILIES = ("py_indent", "py_tabs", "blank_lines", "operator_spaces", "comma_space", "line_wrap")
BINARY_OPS = {"=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>=", ":=",
              "==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "<<", ">>", "@", "@="}
OPEN, CLOSE = "([{", ")]}"
SKIP = (tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.NEWLINE)


def _tokens(src):
    """(tokens, offset(pos)) - offset maps a tokenize (row, col) to an absolute character index; (None, None) if src does not tokenize."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None, None
    starts = [0]
    for ln in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(ln))

    def off(pos):
        r, c = pos
        return starts[r - 1] + c if r - 1 < len(starts) else len(src)
    return toks, off


def _apply(src, edits):
    """edits: (start, end, replacement) in absolute offsets, non-overlapping."""
    out = src
    for s, e, rep in sorted(set(edits), reverse=True):
        out = out[:s] + rep + out[e:]
    return out


def _string_rows(toks):
    """rows (1-based) that lie INSIDE a multi-line string token (its first row is ordinary code, its later rows are not)."""
    rows = set()
    for t in toks:
        if t.type == tokenize.STRING and t.end[0] > t.start[0]:
            rows.update(range(t.start[0] + 1, t.end[0] + 1))
    return rows


def _reindent(src, fn):
    toks, _ = _tokens(src)
    if toks is None:
        return None
    skip = _string_rows(toks)
    out = []
    for i, ln in enumerate(src.split("\n"), 1):
        m = re.match(r" +", ln)
        if m and i not in skip and ln.strip():
            ln = fn(len(m.group())) + ln[m.end():]
        out.append(ln)
    return "\n".join(out)


def py_indent(src):
    return _reindent(src, lambda n: " " * (n // 2))


def py_tabs(src):
    return _reindent(src, lambda n: "\t" * (n // 4) + " " * (n % 4))


def blank_lines(src):
    toks, _ = _tokens(src)
    if toks is None:
        return None
    skip = _string_rows(toks)
    lines = src.split("\n"); out = []; i = 0
    while i < len(lines):
        if lines[i].strip() == "" and (i + 1) not in skip:
            j = i
            while j < len(lines) and lines[j].strip() == "" and (j + 1) not in skip:
                j += 1
            nxt = lines[j] if j < len(lines) else ""
            if j - i >= 2 and re.match(r"(?:def|class|async\s+def|@)\b", nxt):
                out.append("")                                   # one blank line
            else:
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i]); i += 1
    return "\n".join(out)


def _is_operand(t):
    if t.type == tokenize.NAME:
        return not keyword.iskeyword(t.string)
    if t.type in (tokenize.NUMBER, tokenize.STRING):
        return True
    return t.type == tokenize.OP and t.string in CLOSE


def operator_spaces(src):
    toks, off = _tokens(src)
    if toks is None:
        return None
    sig = [t for t in toks if t.type not in SKIP + (tokenize.NL, tokenize.COMMENT)]
    edits = []
    for i in range(1, len(sig) - 1):
        t = sig[i]
        if t.type != tokenize.OP or t.string not in BINARY_OPS:
            continue
        p, n = sig[i - 1], sig[i + 1]
        if not _is_operand(p) or p.end[0] != t.start[0] or n.start[0] != t.end[0]:
            continue                                             # unary / unpacking / split over lines: keep
        if off(p.end) < off(t.start):
            edits.append((off(p.end), off(t.start), ""))
        if off(t.end) < off(n.start):
            edits.append((off(t.end), off(n.start), ""))
    return _apply(src, edits)


def comma_space(src):
    toks, off = _tokens(src)
    if toks is None:
        return None
    edits = []
    for i, t in enumerate(toks[:-1]):
        n = toks[i + 1]
        if t.type == tokenize.OP and t.string == "," and n.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE) \
                and n.start[0] == t.end[0] and off(t.end) < off(n.start):
            edits.append((off(t.end), off(n.start), ""))
    return _apply(src, edits)


def _opener_of(sig, ci):
    depth = 0
    for j in range(ci, -1, -1):
        t = sig[j]
        if t.type == tokenize.OP and t.string in CLOSE:
            depth += 1
        elif t.type == tokenize.OP and t.string in OPEN:
            depth -= 1
            if depth == 0:
                return j
    return 0


def _drop_trailing_comma(sig, gi, ci):
    """gi / ci = indices of the opening / closing token of the group whose last element carries a trailing comma."""
    opener = sig[gi]; before = sig[gi - 1] if gi > 0 else None
    call_like = before is not None and (before.type == tokenize.NAME and not keyword.iskeyword(before.string)
                                        or before.type == tokenize.OP and before.string in CLOSE)
    if opener.string == "(":
        if call_like:
            return True                                          # call / def parameter list
        depth = 0; commas = 0
        for t in sig[gi + 1:ci]:
            if t.type == tokenize.OP and t.string in OPEN:
                depth += 1
            elif t.type == tokenize.OP and t.string in CLOSE:
                depth -= 1
            elif depth == 0 and t.type == tokenize.OP and t.string == ",":
                commas += 1
        return commas > 1                                        # (a,) is a 1-tuple: keep its comma
    if opener.string == "[":
        return not call_like                                     # x[a,] is a tuple subscript: keep
    return True


def _join_group(sig, nls, off):
    edits = []
    for k in nls:
        pi = next(j for j in range(k - 1, -1, -1) if sig[j].type != tokenize.NL)
        ni = next(j for j in range(k + 1, len(sig)) if sig[j].type != tokenize.NL)
        p, n = sig[pi], sig[ni]
        if p.type == tokenize.OP and p.string == "," and n.type == tokenize.OP and n.string in CLOSE:
            if _drop_trailing_comma(sig, _opener_of(sig, ni), ni):
                edits.append((off(sig[pi - 1].end), off(n.start), ""))     # drop the trailing comma and the break
            else:
                edits.append((off(p.end), off(n.start), ""))
            continue
        sep = "" if (p.type == tokenize.OP and p.string in OPEN or n.type == tokenize.OP and n.string in CLOSE) else " "
        edits.append((off(p.end), off(n.start), sep))
    return edits


def line_wrap(src):
    toks, off = _tokens(src)
    if toks is None:
        return None
    sig = [t for t in toks if t.type not in SKIP]
    edits = []; depth = 0; nls = []; has_comment = False
    for i, t in enumerate(sig):
        if t.type == tokenize.OP and t.string in OPEN:
            if depth == 0:
                nls, has_comment = [], False
            depth += 1
        elif t.type == tokenize.OP and t.string in CLOSE:
            depth -= 1
            if depth == 0 and nls and not has_comment:
                edits += _join_group(sig, nls, off)              # an outermost group spanning several lines, comment-free
        elif depth > 0:
            if t.type == tokenize.NL:
                nls.append(i)
            elif t.type == tokenize.COMMENT:
                has_comment = True
    return _apply(src, edits)


RULES = {"py_indent": py_indent, "py_tabs": py_tabs, "blank_lines": blank_lines, "operator_spaces": operator_spaces,
         "comma_space": comma_space, "line_wrap": line_wrap}


def transform(family, text):
    return RULES[family](text)


def same_ast(a, b):
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


def whitespace_only(a, b):
    return re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)




def apply(families=WS_FAMILIES, write=False):
    """Re-derive the alt twin of every stored pair (and its raw doc) of these families by rule; re-align; re-apply the free rule for
    py_indent / py_tabs. Documents that no longer meet the builder's pass criterion are reported and left untouched."""
    import json
    from src.utils.paths import STYLE_TRANSLATION_DATA
    from src.sandbox.style_translation.code_build import align
    from src.sandbox.style_translation.code_free import filter_free, AFFECTED
    PAIRS, RAW = STYLE_TRANSLATION_DATA / "pairs", STYLE_TRANSLATION_DATA / "code"
    report = {}
    for fam in families:
        recs = json.load(open(PAIRS / f"{fam}.json")); raw_p = RAW / f"{fam}.json"
        raw = {r["doc_id"]: r for r in json.load(open(raw_p))} if raw_p.exists() else {}
        held = []; n_ok = 0; ks = []
        for i, r in enumerate(recs):
            alt = transform(fam, r["text_nat"])
            if alt is None or not same_ast(r["text_nat"], alt):
                held.append((r["doc_id"], "natural twin does not parse")); continue
            opps, shared = align(r["text_nat"], alt)
            ok = opps is not None and len(opps) >= 5 and shared >= 0.6 and opps[4]["nat_span"][0] < 0.75 * len(r["text_nat"]) and all(o["nat"] != o["alt"] for o in opps)
            if not ok:
                held.append((r["doc_id"], f"{len(opps or [])} opportunities" if opps is not None and len(opps) < 5 else "5th opportunity past 75 % of the code")); continue
            new = {k: v for k, v in r.items() if k not in ("opps_all", "free_filter")}
            new.update(text_alt=alt, opps=opps, k_en=len(opps), shared_fraction=round(shared, 3), alt_rule=True)
            if fam in AFFECTED:
                fr = filter_free(new, fam)
                if fr is None:
                    held.append((r["doc_id"], "fewer than 5 free opportunities")); continue
                new.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
            recs[i] = new; n_ok += 1; ks.append(new["k_en"])
            if r["doc_id"] in raw:
                raw[r["doc_id"]].update({k: new[k] for k in ("text_alt", "opps", "k_en", "shared_fraction", "alt_rule") if k in new})
                for k in ("opps_all", "free_filter"):
                    if k in new:
                        raw[r["doc_id"]][k] = new[k]
        if write:
            json.dump(recs, open(PAIRS / f"{fam}.json", "w"), ensure_ascii=False, indent=0)
            if raw:
                json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_p, "w"), ensure_ascii=False, indent=0)
        ks.sort(); report[fam] = dict(n=len(recs), rewritten=n_ok, held=held, k_median=ks[len(ks) // 2] if ks else 0)
        print(f"{fam:16s} docs={len(recs):3d} rewritten={n_ok:3d} opps median={report[fam]['k_median']:3d} held={len(held)} {held}", flush=True)
    return report


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    if sys.argv[1:2] == ["--apply"]:                            # --apply [--write]: re-derive the stored alt twins (dry run without --write)
        apply(write="--write" in sys.argv)
    else:                                                        # <family> <file>: print the alt rendering of one file
        fam, path = sys.argv[1], sys.argv[2]
        src = open(path).read(); out = transform(fam, src)
        print(out); print("same_ast:", same_ast(src, out), "whitespace_only:", whitespace_only(src, out), file=sys.stderr)
