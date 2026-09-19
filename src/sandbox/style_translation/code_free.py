#!/usr/bin/env python
"""Free-opportunity rule for the code families whose decisions can be FORCED by the code itself (user finding 2026-09-16): once an identifier
is defined, later mentions must reuse its spelling to keep the code working; `self` must match the parameter name; lines inside a block must
keep the block's indentation. Such opportunities test consistency, not the convention. `free_opportunity(fam, rec, k)` says whether
opportunity k of pair `rec` is a genuine choice in BOTH renderings; `filter_free(rec, fam)` returns the pair with only its free opportunities
(re-indexed) or None if fewer than MIN_OPPS remain (or the 5th free opportunity lies past 75 % of the code, as in code_build's filter)."""
import re

IDENT_FAMILIES = {"py_snake_camel", "js_camel_snake", "py_const_naming", "py_class_naming", "py_private", "py_bool_prefix", "py_loop_vars", "js_hungarian", "py_abbrev"}
SELF_FAMILIES = {"py_self_name"}
INDENT_FAMILIES = {"py_indent", "py_tabs"}
CONTENT_FAMILIES = {"early_return", "py_with_open", "py_ternary"}   # bug 5 (2026-09-17): a block choice forces the indentation of every later line
_SPECIAL = IDENT_FAMILIES | SELF_FAMILIES | INDENT_FAMILIES | CONTENT_FAMILIES


def _all_code_families():
    from src.sandbox.style_translation.code_families import CODE_FAMILIES
    return {f.name for f in CODE_FAMILIES}


class _All(set):                                                 # bug 10 (2026-09-17): the paired-symbol clause applies to EVERY code family
    def __contains__(self, x):
        return x in _all_code_families()
    def __iter__(self):
        return iter(_all_code_families())


AFFECTED = _All()
MIN_OPPS = 5
FIFTH_FRAC_DEFAULT = 0.75                                   # the 5th counted opportunity must start before this fraction of the natural twin
FIFTH_FRAC = {"py_not_in": 0.85, "rust_question": 0.85, "py_is_none": 0.85, "float_literals": 0.85}   # user decision 2026-09-18 (padding regeneration)


def fifth_frac(fam):
    return FIFTH_FRAC.get(fam, FIFTH_FRAC_DEFAULT)
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _seen_before(ident, before):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(ident) + r"(?![A-Za-z0-9_])", before) is not None


def _task_code_idents(task):
    """identifiers the task text pins down as code: inside backticks, or written as a call name(."""
    out = set()
    for m in re.finditer(r"`([^`]+)`", task):
        out.update(_ID.findall(m.group(1)))
    out.update(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", task))
    return out


_CLOSING = re.compile(r"^[\s\)\]\}]*$")
_CLOSING_PUNCT = re.compile(r"^\s*[\)\]\}][\s\)\]\}]*[,;]?\s*$")     # bug 15 / fix 3 (2026-09-18): closer(s) + a trailing `,` or `;` (`],` `);` `},`)
_QUOTES = re.compile(r"^[\s'\"`]*$")
_STR_RX = re.compile(r'"""(?:\\.|[^\\])*?"""|\'\'\'(?:\\.|[^\\])*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\])*`', re.S)


def _string_bounds(text, lang):
    """(starts, ends) character offsets of the string literals of the natural text: Python via tokenize, otherwise a literal regex."""
    if lang == "Python":
        import io, tokenize
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            toks = None
        if toks is not None:
            starts = [0]
            for ln in text.splitlines(keepends=True):
                starts.append(starts[-1] + len(ln))
            off = lambda p: starts[p[0] - 1] + p[1]
            return {off(t.start) for t in toks if t.type == tokenize.STRING}, {off(t.end) for t in toks if t.type == tokenize.STRING}
    ms = list(_STR_RX.finditer(text))
    return {m.start() for m in ms}, {m.end() for m in ms}


def closing_symbol(rec, o, lang):
    """Bug 10: an opportunity made only of CLOSING brackets, or of the closing quote(s) of a string literal, is forced by its opening
    counterpart (once the opening symbol is written, the closing one must match) and is not a decision."""
    both = o["nat"] + o["alt"]
    if not both.strip():
        return False
    if _CLOSING.match(o["nat"]) and _CLOSING.match(o["alt"]):
        return True
    if _CLOSING_PUNCT.match(o["nat"]) and _CLOSING_PUNCT.match(o["alt"]):   # fix 3: `],` vs `),` / `]),` vs `)` — the punctuation follows the closer
        return True
    if _QUOTES.match(o["nat"]) and _QUOTES.match(o["alt"]):
        starts, ends = rec.setdefault("_str_bounds", _string_bounds(rec["text_nat"], lang))
        s0, s1 = o["nat_span"]
        return s1 in ends and s0 not in starts
    return False


# Bug 11 (2026-09-17, user decision): in a two-part construct only the OPENER is a decision; the second half is guaranteed by the first.
#   bash_subst   - the closer is `)` in one convention and a backtick in the other, so closing_symbol() (same-class closers) missed it
#   py_with_open - `as f:` clauses, `.close()` insertions and re-indentation are fragments of the with/open choice made at the head
OPENER_ONLY = {"bash_subst": lambda o: o["nat"].lstrip().startswith("$("),
               "py_with_open": lambda o: re.search(r"\bwith\b", o["nat"]) is not None}


# Bug 12 / 13 (2026-09-17, user decisions): ONE CONSTRUCT = ONE DECISION, for every family. A construct that the diff splits into several
# spans (comprehension brackets, f-string prefix / braces / .format, `match` keyword / arms, `//` opener / `*/` closer, a wrapped call's line
# breaks, Optional[ ... ], a docstring's header / argument lines, a JOIN and its ON clause ...) is decided at its FIRST span; every later span
# follows from it. Operational rule: a span is counted only if no earlier CONTENT diff span (counted or not) sits on the same line of the
# natural twin or of the alternative twin, nor in the same block for families whose construct spans several lines in both twins
# (docstring_style: the docstring; sql_join_style: the SQL statement). The name / self / indentation families are exempt: there the unit of
# decision is the new name or the new block, which their own rules already isolate.
LINE_EXEMPT = IDENT_FAMILIES | SELF_FAMILIES | INDENT_FAMILIES
LINE_FAMILIES = None                                             # kept for import compatibility; the rule is universal now


def _line(text, pos, span_text):
    return text.count("\n", 0, pos) + (1 if span_text.startswith("\n") else 0)


# Bug 15 / fix 2 (2026-09-18, user decision): the token diff merges the closer of one construct with the opener of the next across a shared
# line break, so a raw span can run over several lines of either twin. In the families of COVER_FAMILIES such a span blocks EVERY line on
# which it has content (from the line of its first to the line of its last non-whitespace character), not only the line where it starts;
# the same for the per-docstring / per-statement units. Where the swallowed opener of the next construct would otherwise be lost, the raw
# spans of these families are first cut at line breaks that both renderings share (`_split_line_breaks`), so the opener is recovered as its
# own opportunity with the last shared token before it as cue. Fix 1 (splitting every family) is NOT adopted, and the coverage rule is not
# applied outside COVER_FAMILIES: applied everywhere it would also delete the closer + opener spans of py2_print, py_quotes, js_quotes,
# py_fstring, bash_test and line_wrap (their cue sits before a forced closer, but the opener would be lost without the split) and the
# multi-line spans of early_return / the fix-4 families — deferred by the user; see WORKLOG 2026-09-18.
# py_join_concat gets the coverage rule but no split: its cross-line spans are the separators / closers of multi-line join calls, which the
# split would turn into first-on-their-line pieces (11 of 12 recovered pieces were forced); c_comment_style's never share their line breaks.
SPLIT_FAMILIES = {"sql_keyword_case", "php_array", "py_literal_ctor"}
COVER_FAMILIES = SPLIT_FAMILIES | {"py_join_concat", "c_comment_style"}


def _c_continuation(rec, o):
    """c_comment_style: the alternative rendering begins a line with `*` — the continuation (` * text`) or closing (` */`) line of a block
    comment, forced by the `/*` that opened it (bug 12 caught `*/` anywhere; fix 2 adds the continuation lines)."""
    a = o["alt"]; p = o["alt_span"][0] + len(a) - len(a.lstrip()); t = rec["text_alt"]
    return t[p:p + 1] == "*" and t[t.rfind("\n", 0, p) + 1:p].strip() == ""


def _cover(text, s0, s1, span, full):
    """(first, last) line a span blocks. `full` (COVER_FAMILIES): the lines of its first and last non-whitespace character; otherwise
    (and for a whitespace-only rendering) the bug-13 start line only (`_line`)."""
    a = _line(text, s0, span)
    if not full or not span.strip():
        return a, a
    i = s0 + len(span) - len(span.lstrip()); j = s0 + len(span.rstrip()) - 1
    b = text.count("\n", 0, i)
    return b, b + text.count("\n", i, j)


def _block(fam, rec, o):
    """Units (docstrings / SQL statements) a span covers, for the families whose construct spans several lines in both twins; None otherwise."""
    t = rec["text_nat"]; p0, p1 = o["nat_span"][0], max(o["nat_span"][0], o["nat_span"][1] - 1)
    if fam == "docstring_style":
        q0 = len(re.findall(r'"""|\'\'\'', t[:p0])); q1 = q0 + len(re.findall(r'"""|\'\'\'', t[p0:p1]))
        return {("doc", q // 2) for q in range(q0, q1 + 1) if q % 2}      # inside a docstring: odd number of triple quotes before the position
    if fam == "sql_join_style":
        return {("stmt", n) for n in range(t.count(";", 0, p0), t.count(";", 0, p1) + 1)}
    return None


def split_cross_line(rec):
    """SPLIT_FAMILIES: the raw diff with every span cut at the line breaks both renderings share (`_split_line_breaks`), re-indexed."""
    return [dict(p, k=i) for i, p in enumerate(p for o in rec["opps"] for p in _split_line_breaks(o))]


def one_per_line(fam, rec, opps):
    """`opps` = spans that passed the per-span free rule; rec["opps"] = the FULL diff (blockers come from it)."""
    if fam in LINE_EXEMPT:
        return opps
    ok = {tuple(o["nat_span"]) for o in opps}; seen_n, seen_a, seen_b = set(), set(), set(); keep = []; full = fam in COVER_FAMILIES
    for o in rec["opps"]:
        (n0, n1), (a0, a1), bk = _cover(rec["text_nat"], *o["nat_span"], o["nat"], full), _cover(rec["text_alt"], *o["alt_span"], o["alt"], full), _block(fam, rec, o)
        first = n0 not in seen_n and a0 not in seen_a and not (bk and bk & seen_b)
        if tuple(o["nat_span"]) in ok and first and not (fam == "c_comment_style" and (o["alt"].lstrip().startswith("*/") or _c_continuation(rec, o))):
            keep.append(o)
        if o["nat"].strip() or o["alt"].strip() or tuple(o["nat_span"]) in ok:   # content spans and eligible spans block (pure re-indentation does not)
            seen_n.update(range(n0, n1 + 1)); seen_a.update(range(a0, a1 + 1))   # fix 2: every line the span covers
            if bk:
                seen_b |= bk
    return keep


# Bug 14 (2026-09-18, user decision): COMMENT-TEXT families. The twin diff cuts one comment into several spans wherever a word is shared by
# both poles (RGB, 2D, Define, Kelvin, numbers); a span that runs across a line break is blocked by the line rule, so the next span, which
# starts AFTER a shared word, became the counted opportunity with its cue mid-comment (the comment is already half done in one language).
# Rule: all diff spans inside one comment UNIT (a `#` comment tail, or one line of a docstring) merge into ONE opportunity, from the first
# differing character to the last, in both twins (nat / alt text and spans recomputed). The unit is counted only if its FIRST word differs
# between the poles, so the cue is the `#` token or the docstring line's leading whitespace; units whose first word is shared are free;
# spans outside comment units (string literals, code) are free. `opps_all` stays the raw full diff; `counted()` derives the merged list.
# Comment cue rule (2026-09-18, user decision): a cue must be a comment OPENER. Of a docstring only its first line that carries text is a
# unit that can count (kind "doc"; the later lines, kind "doc+", are free); a `#` line that continues the previous `#` line's sentence
# (the previous whole-line comment does not end in . : ! ? and this comment's text starts with a lowercase letter) is free
# (`comment_continuation`). Sweep check: `comment_opener_ok`.
COMMENT_TEXT = {"comment_language", "comment_case"}
# Bug 15 / fix 4 (2026-09-18, user decision): the BLOCK-RESTRUCTURING families are aligned line by line (`code_line_align.line_opportunities`):
# each replaced block of lines is one construct, its span starts at the first diverging token (cue = last shared token before it),
# blocks holding several adjacent constructs are split one per construct, forced closers / re-indentation / rewrite artefacts are free,
# then the bug-13 line rule. `opps_all` stays the raw token diff; `counted()` returns the line-level list for these families.
LINE_ALIGN_FAMILIES = {"py_ternary", "py_comprehension", "rust_question", "js_arrow", "py_with_open"}
_OPENER = re.compile(r'#|"""|\'\'\'')
_LINE_BREAK = re.compile(r"\n[ \t]*")
_DOC_OPEN = re.compile(r'^\s*[rRbBuU]{0,2}(?:"""|\'\'\')?')


def comment_units(text):
    """(start, end, kind) of every comment unit of a Python text, sorted by start: kind '#' = a comment token (`#` to end of line),
    'doc' = one line of a triple-quoted string that opens a logical line (a docstring). None if the text does not tokenize."""
    import io, tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    starts = [0]
    for ln in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(ln))
    off = lambda p: starts[p[0] - 1] + p[1]
    units = []; prev = None
    for t in toks:
        if t.type == tokenize.COMMENT:
            units.append((off(t.start), off(t.end), "#"))
        elif t.type == tokenize.STRING:
            s0, s1 = off(t.start), off(t.end)
            if t.string.lstrip("rRbBuU").startswith(('"""', "'''")) and (prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)):
                p = s0; kind = "doc"
                for ln in text[s0:s1].splitlines(keepends=True):
                    units.append((p, p + len(ln.rstrip("\r\n")), kind)); p += len(ln)
                    if kind == "doc" and re.search(r"\w", _DOC_OPEN.sub("", ln, count=1)):
                        kind = "doc+"                            # only the first line with text can count; the rest of the docstring is free
        if t.type not in (tokenize.NL, tokenize.COMMENT):
            prev = t
    return sorted(units)


def _unit_at(units, s0, s1):
    """index of the unit containing a span (an empty span may sit at the unit's end: an insertion at the end of the comment)."""
    for i, (a, b, _) in enumerate(units):
        if (a <= s0 < b) if s1 > s0 else (a <= s0 <= b):
            return i
    return None


def _first_word(units, u, text):
    """char offset of the first word of unit u (after `#` / the docstring opener and whitespace), None if it has no word."""
    a, b, kind = units[u]; body = text[a:b]
    skip = 1 if kind == "#" else _DOC_OPEN.match(body).end()
    m = re.search(r"\w", body[skip:])
    return None if m is None else a + skip + m.start()


def comment_continuation(units, u, text):
    """`#` unit u continues the previous line's `#` comment: both are whole-line comments, the previous one does not end in . : ! ? and
    this one's text starts with a lowercase letter."""
    a, b, kind = units[u]
    if kind != "#":
        return False
    ls = text.rfind("\n", 0, a) + 1
    if text[ls:a].strip():
        return False
    prev = text[text.rfind("\n", 0, ls - 1) + 1:ls - 1] if ls > 0 else ""
    m = re.match(r"\s*#(.*)", prev); body = m.group(1).strip() if m else ""
    return bool(body) and body[-1] not in ".:!?" and text[a + 1:b].strip()[:1].islower()


def comment_eligible(units, u, text):
    """Unit u may count: the first text line of a docstring, or a `#` comment that does not continue the previous one."""
    return units[u][2] == "doc" or (units[u][2] == "#" and not comment_continuation(units, u, text))


def comment_opener_ok(text, p):
    """Sweep check `cue_not_comment_opener`: position p (a counted span's start) lies in a unit that may count (`comment_eligible`)."""
    units = comment_units(text)
    if units is None:
        return False
    u = _unit_at(units, p, p + 1)
    if u is None:
        u = _unit_at(units, p, p)
    return u is not None and comment_eligible(units, u, text)


def _split_line_breaks(o):
    """A raw span that runs across a line break belongs to two units only because align() merged the shared newline + indent (<= 2 tokens):
    cut it at the line breaks when both renderings have the same ones. Returns the list of pieces (the span itself if no cut applies)."""
    if "\n" not in o["nat"] or "\n" not in o["alt"]:
        return [o]
    bn, ba = list(_LINE_BREAK.finditer(o["nat"])), list(_LINE_BREAK.finditer(o["alt"]))
    if len(bn) != len(ba) or any(x.group() != y.group() for x, y in zip(bn, ba)):
        return [o]
    pieces = []; pn = pa = 0
    for x, y in list(zip(bn, ba)) + [(None, None)]:
        en, ea = (x.start(), y.start()) if x else (len(o["nat"]), len(o["alt"]))
        if en > pn or ea > pa:
            n0, a0 = o["nat_span"][0] + pn, o["alt_span"][0] + pa
            pieces.append({"nat": o["nat"][pn:en], "alt": o["alt"][pa:ea], "nat_span": [n0, n0 + en - pn], "alt_span": [a0, a0 + ea - pa]})
        if x:
            pn, pa = x.end(), y.end()
    return pieces


def comment_merge(rec):
    """COMMENT_TEXT families. From the raw diff rec["opps"]: (merged, eligible) where `merged` has one span per comment unit (spans outside
    any unit stay as they are) and `eligible` are the merged spans whose unit's first word differs between the poles."""
    nat, alt = rec["text_nat"], rec["text_alt"]
    units = comment_units(nat) or []
    groups = []                                                  # [unit index or None, last unit reached, [pieces]]
    for o in rec["opps"]:
        for p in _split_line_breaks(o):
            u = _unit_at(units, *p["nat_span"])
            ue = _unit_at(units, max(p["nat_span"][0], p["nat_span"][1] - 1), p["nat_span"][1]) if u is not None else None
            umax = u if ue is None else max(u, ue)
            if u is not None and groups and groups[-1][0] is not None and u <= groups[-1][1]:
                groups[-1][2].append(p); groups[-1][1] = max(groups[-1][1], umax)
            else:
                groups.append([u, umax, [p]])
    merged, eligible = [], []
    for u, _, ps in groups:
        s0, s1, a0, a1 = ps[0]["nat_span"][0], ps[-1]["nat_span"][1], ps[0]["alt_span"][0], ps[-1]["alt_span"][1]
        m = {"nat": nat[s0:s1], "alt": alt[a0:a1], "nat_span": [s0, s1], "alt_span": [a0, a1]}
        merged.append(m)
        if u is not None and comment_eligible(units, u, nat):
            fw = _first_word(units, u, nat)
            if fw is not None and s0 <= fw < s1:
                eligible.append(m)
    return merged, eligible


def comment_words_before(text, p):
    """Sweep check `cue_mid_comment`: True if position p (a counted span's start) has words of the same comment before it on its line,
    or does not lie in a comment unit at all. Tokenizer-based; regex fallback (last `#` / triple quote on the line) if the text does not tokenize."""
    units = comment_units(text)
    if units is None:
        ls = text.rfind("\n", 0, p) + 1; line = text[ls:p]; last = None
        for m in _OPENER.finditer(line):
            last = m
        return re.search(r"\w", line[last.end():] if last else line) is not None
    u = _unit_at(units, p, p + 1)
    if u is None:
        u = _unit_at(units, p, p)
    if u is None:
        return True
    fw = _first_word(units, u, text)
    return fw is None or p > fw


def counted(fam, rec):
    """The counted opportunities of a record whose `opps` is the FULL diff list: per-span free rule, then one decision per line
    (COMMENT_TEXT families: one merged span per comment unit, counted if the unit's first word differs, then the line rule;
    LINE_ALIGN_FAMILIES: line-level blocks from the texts, fix 4)."""
    if fam in LINE_ALIGN_FAMILIES:                               # fix 4: derived from text_nat / text_alt, not from the token diff
        from src.sandbox.style_translation.code_line_align import line_opportunities   # (lazy: code_line_align imports this module)
        rec = dict(rec); rec.pop("_str_bounds", None)
        return line_opportunities(fam, rec)
    if fam in COMMENT_TEXT:
        merged, eligible = comment_merge(rec)
        return one_per_line(fam, dict(rec, opps=merged), eligible)
    if fam in SPLIT_FAMILIES:                                    # fix 2: cross-line spans cut at shared line breaks before the rules apply
        rec = dict(rec, opps=split_cross_line(rec)); rec.pop("_str_bounds", None)
    return one_per_line(fam, rec, [o for k, o in enumerate(rec["opps"]) if free_opportunity(fam, rec, k)])


def free_opportunity(fam, rec, k):
    o = rec["opps"][k]; task = rec.get("text_es", "")
    from src.sandbox.style_translation.code_families import CODE_FAMILY
    if closing_symbol(rec, o, CODE_FAMILY[fam].tgt_lang):
        return False
    if fam in OPENER_ONLY and not OPENER_ONLY[fam](o):
        return False
    if fam == "py_loop_vars":                      # a new `for` statement is a fresh binding; uses inside the body are forced
        text = rec["text_nat"]; s0 = o["nat_span"][0]; line_start = text.rfind("\n", 0, s0) + 1
        return re.search(r"\bfor\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*,\s*)*$", text[line_start:s0]) is not None
    if fam in IDENT_FAMILIES:
        pinned = _task_code_idents(task)
        for pole in ("nat", "alt"):
            if _seen(fam, rec, o, pole, pinned):
                return False
        return True
    if fam in SELF_FAMILIES:
        text = rec["text_nat"]; s0 = o["nat_span"][0]; line_start = text.rfind("\n", 0, s0) + 1
        return re.search(r"def\s+\w+\s*\(\s*$", text[line_start:s0]) is not None
    if fam in INDENT_FAMILIES:
        text = rec["text_nat"]; s0 = o["nat_span"][0]
        prev = [l for l in text[:s0].split("\n") if l.strip()]
        return bool(prev) and prev[-1].rstrip().endswith(":")
    if fam in CONTENT_FAMILIES:                    # only a span that differs beyond whitespace is a choice (else:/with/ternary); indentation is forced
        return bool(o["nat"].strip()) or bool(o["alt"].strip())
    return True


def _seen(fam, rec, o, pole, pinned):
    """True if the opportunity's identifier(s) in this rendering are pinned by the task or already occur earlier in the code
    (py_private: attributes are matched in their dotted form `._name`, so a same-named parameter does not count)."""
    text = rec[f"text_{pole}"]; s0 = o[f"{pole}_span"][0]; before = text[:s0]
    idents = _ID.findall(o[pole])
    if not idents:
        return True
    if fam == "py_private":
        return any(re.search(r"\." + re.escape(t) + r"(?![A-Za-z0-9_])", before) is not None for t in idents)
    return any(t in pinned or _seen_before(t, before) for t in idents)


def bound_names(src, lang):
    """Standalone names of the code (not attributes after '.', not inside strings / comments). Python via tokenize, otherwise a regex on
    the code with comments and string literals stripped. None if Python code does not tokenize."""
    if lang == "Python":
        import io, keyword, tokenize
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return None
        out = set(); prev = None
        for t in toks:
            if t.type == tokenize.NAME and not keyword.iskeyword(t.string) and not (prev is not None and prev.type == tokenize.OP and prev.string == "."):
                out.add(t.string)
            if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                prev = t
        return out
    s = re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)
    s = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`', '""', s)
    return {m.group(1) for m in re.finditer(r"(?<![.\w$])([A-Za-z_$][\w$]*)", s)}


def rename_collisions(fam, rec):
    """Bug 8 (2026-09-17): renames x -> y of the alt twin whose target y already exists as a standalone name in the NATURAL code
    (a parameter, a variable, another class): the alt twin then shadows or overwrites that name and no longer computes the same thing."""
    from src.sandbox.style_translation.code_families import CODE_FAMILY
    bound = bound_names(rec["text_nat"], CODE_FAMILY[fam].tgt_lang)
    if bound is None:
        return set()
    hits = set()
    for o in rec["opps"]:
        a, b = _ID.findall(o["nat"]), _ID.findall(o["alt"])
        if len(a) == len(b):
            hits.update((x, y) for x, y in zip(a, b) if x != y and y in bound)
    return hits


def consistent_twins(fam, rec):
    """Identifier families: the two renderings must agree on which opportunities are first mentions; a mismatch means the rewrite
    changed a use but not its definition (or vice versa), i.e. the alt twin references an undefined name -> reject the document.
    Also rejected: a rename whose target collides with an existing name (`rename_collisions`)."""
    if fam not in IDENT_FAMILIES:
        return True
    if rename_collisions(fam, rec):
        return False
    pinned = _task_code_idents(rec.get("text_es", ""))
    return all(_seen(fam, rec, o, "nat", pinned) == _seen(fam, rec, o, "alt", pinned) for o in rec["opps"])


def harmonise_pinned(rec, fam):
    """Identifier families: an identifier the task text pins down (backticks / call form) is not a choice, so it must read the same in both
    twins. Rebuild text_alt from text_nat keeping the NAT rendering for every opportunity that touches a pinned identifier; recompute the
    alt spans; opportunities whose two renderings become identical are dropped. Returns a new record (the input is not modified)."""
    if fam not in IDENT_FAMILIES:
        return rec
    pinned = _task_code_idents(rec.get("text_es", ""))
    nat = rec["text_nat"]; parts = []; opps = []; pos = 0; alt_pos = 0
    for o in rec["opps"]:
        s0, s1 = o["nat_span"]; parts.append(nat[pos:s0]); alt_pos += s0 - pos
        touch = any(t in pinned for t in _ID.findall(o["nat"]) + _ID.findall(o["alt"]))
        piece = o["nat"] if touch else o["alt"]
        if piece != o["nat"]:
            opps.append(dict(o, alt=piece, alt_span=[alt_pos, alt_pos + len(piece)]))
        parts.append(piece); alt_pos += len(piece); pos = s1
    parts.append(nat[pos:])
    out = dict(rec); out["text_alt"] = "".join(parts); out["opps"] = [dict(o, k=i) for i, o in enumerate(opps)]; out["k_en"] = len(opps)
    return out


def harmonise_leading_ws(rec, fam):
    """Content families (bug 5b): a whitespace-only difference BEFORE the first real choice cannot be forced by anything — it is a formatting
    slip of the rewrite (e.g. a one-line guard `if bad: return 0` spread over two lines). Such spans take the natural rendering so the k = 0
    prompts of the two poles are identical; the alt text is rebuilt and re-aligned. Returns a new record (or the input if nothing to fix)."""
    if fam not in CONTENT_FAMILIES:
        return rec
    first = next((i for i, o in enumerate(rec["opps"]) if o["nat"].strip() or o["alt"].strip()), None)
    slips = [o for o in rec["opps"][:first] if not (o["nat"].strip() or o["alt"].strip())] if first is not None else []
    if not slips:
        return rec
    from src.sandbox.style_translation.code_build import align
    nat, alt = rec["text_nat"], rec["text_alt"]; parts = []; pos = 0
    for o in rec["opps"]:
        a0, a1 = o["alt_span"]; parts.append(alt[pos:a0]); parts.append(o["nat"] if o in slips else o["alt"]); pos = a1
    parts.append(alt[pos:]); new_alt = "".join(parts)
    opps, shared = align(nat, new_alt)
    if opps is None or not opps:
        return rec
    out = dict(rec); out["text_alt"] = new_alt; out["opps"] = opps; out.pop("opps_all", None); out["shared_fraction"] = round(shared, 3)
    return out


def propagate_renames(rec, fam):
    """Identifier families: the Gemini rewrite sometimes renames an identifier's uses but not its definition (or the reverse), leaving the
    alt twin referencing undefined names. Derive the per-identifier renaming nat -> alt from the diff opportunities (one identifier on each
    side, not task-pinned) and apply it to EVERY whole-word occurrence in the natural code; re-align to get the full opportunity list.
    py_private renames attributes in their dotted form only. Returns a new record (text_alt, opps rebuilt) or the input if nothing to map."""
    if fam not in IDENT_FAMILIES:
        return rec
    from src.sandbox.style_translation.code_build import align
    pinned = _task_code_idents(rec.get("text_es", "")); mapping = {}
    for o in (rec.get("opps_all") or rec["opps"]):
        a, b = _ID.findall(o["nat"]), _ID.findall(o["alt"])
        if len(a) == len(b):
            for x, y in zip(a, b):
                if x != y and x not in pinned and y not in pinned:
                    mapping.setdefault(x, y)
    if not mapping:
        return rec
    nat = rec["text_nat"]; alt = nat
    for x in sorted(mapping, key=len, reverse=True):
        pat = (r"\." + re.escape(x) + r"(?![A-Za-z0-9_])") if fam == "py_private" else (r"(?<![A-Za-z0-9_])" + re.escape(x) + r"(?![A-Za-z0-9_])")
        rep = ("." + mapping[x]) if fam == "py_private" else mapping[x]
        alt = re.sub(pat, rep, alt)
    opps, shared = align(nat, alt)
    if opps is None or not opps:
        return rec
    out = dict(rec); out["text_alt"] = alt; out["opps"] = opps; out.pop("opps_all", None); out["shared_fraction"] = round(shared, 3)
    return out


def filter_free(rec, fam, min_opps=MIN_OPPS):
    """rec must carry the FULL opportunity list: in `opps_all` (a record already processed here) or in `opps` (fresh from align()).
    Returns the record with text_alt harmonised, `opps_all` = full harmonised list, `opps` = the free ones only; None if too few."""
    if fam not in AFFECTED:
        return rec
    full = dict(rec); full["opps"] = rec.get("opps_all") or rec["opps"]
    full = propagate_renames(full, fam)
    full = harmonise_pinned(full, fam)
    full = harmonise_leading_ws(full, fam)
    if not consistent_twins(fam, full):
        return None
    keep = counted(fam, full)
    if len(keep) < min_opps or keep[min_opps - 1]["nat_span"][0] >= fifth_frac(fam) * len(full["text_nat"]):
        return None
    out = dict(full); out.pop("_str_bounds", None); out["opps_all"] = full["opps"]; out["opps"] = [dict(o, k=i) for i, o in enumerate(keep)]; out["k_en"] = len(keep); out["free_filter"] = True
    return out
