#!/usr/bin/env python
"""Fix 4 (2026-09-18, user decision): line-level opportunities for the BLOCK-RESTRUCTURING code families.

Families: py_ternary (conditional expression vs if/else block), py_comprehension (comprehension vs explicit loop), rust_question (`?` vs
explicit `match`), js_arrow (arrow function vs `function` expression), py_with_open (`with open(...)` block vs `f = open(...)` ... `f.close()`).
For these the token-level diff (`code_build.align`, merge_gap = 2) aligns the wrong constructs or merges the closer of one construct with the
opener of the next, so the counted decision is a forced closer or sits after the construct already showed its style.

Rule: align text_nat and text_alt LINE BY LINE (difflib on lines, trailing whitespace ignored; leading whitespace kept, so lines that a block
choice re-indents belong to that block). Each non-equal run of lines is one BLOCK; two blocks whose lines belong to the same LOGICAL line
(a multi-line statement with equal middle lines, e.g. a Rust method chain ending in `?;`) are merged. A block that holds several constructs
in a row (n opener lines on the natural side and n on the alternative side, n >= 2) is cut at the openers into n sub-blocks (`split_adjacent`,
without it 50-70 docs per family fall below 5 opportunities); blank / re-indented lines before the first opener are dropped. Each block = ONE
construct = one opportunity; its span starts at the first TOKEN (code_build.TOK) where the two renderings of the block diverge (the cue is the
last shared token before it: the indentation of the block's first line or a shared prefix such as `normal_range_bmis = [`) and ends at the end
of the block on each side. Blocks that hold only forced material (whitespace re-indentation, closers, `.close()` lines: the bug-5 content rule
and bug-10 closing_symbol) are free; py_with_open keeps its bug-11 opener-only rule (natural side must contain `with`); a block without the
family's construct marker on both sides is free (`require_marker`). The bug-13 one-per-line rule is applied on top (`one_per_line`).

Hooked (2026-09-18, applied): code_free.counted() returns line_opportunities(fam, rec) for LINE_ALIGN_FAMILIES, so filter_free, the builder
and the sweep use it. `opps_all` stays the raw token diff; `line_opportunities` reads only text_nat / text_alt.

CLI (dry run, never writes to a pairs file):
    python -m src.sandbox.style_translation.code_line_align --dry_run [--families ...] [--examples N] [--tokenizer] [--out FILE]
"""
import argparse
import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.code_free import closing_symbol, OPENER_ONLY, one_per_line, LINE_ALIGN_FAMILIES
from src.sandbox.style_translation.code_families import CODE_FAMILY

PAIRS = STYLE_TRANSLATION_DATA / "pairs"                          # LINE_ALIGN_FAMILIES is defined in code_free (single definition)
TOK = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)                     # = code_build.TOK (the token unit of the raw diff and of the cue definition)

# ----------------------------------------------------------------------------------------------------------------------------- code helpers
_STR = re.compile(r'"""(?:\\.|[^\\])*?"""|\'\'\'(?:\\.|[^\\])*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\])*`', re.S)
_COMMENT = {"Python": re.compile(r"#[^\n]*"), "JavaScript": re.compile(r"//[^\n]*|/\*.*?\*/", re.S), "Rust": re.compile(r"//[^\n]*|/\*.*?\*/", re.S)}


def strip_code(text, lang):
    """String literals and comments blanked (same length, so offsets are preserved)."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group())
    text = _STR.sub(blank, text)
    return _COMMENT.get(lang, _COMMENT["Python"]).sub(blank, text)


def logical_lines(lines, lang):
    """Group physical lines into logical lines: a line continues while brackets are open (Python: ( [ { ; JavaScript / Rust: ( [ only, braces
    open blocks there), while it ends with a backslash (Python) or with `=>` (an arrow whose body is on the next line, JavaScript), or while
    the next line starts with `.` (a method chain, JavaScript / Rust).
    Returns a list of (first_index, last_index_exclusive)."""
    code = [strip_code(l, lang) for l in lines]
    opens, closes = ("([{", ")]}") if lang == "Python" else ("([", ")]")
    groups, depth, start = [], 0, None
    for i, l in enumerate(code):
        if start is None:
            start = i
        depth += sum(l.count(c) for c in opens) - sum(l.count(c) for c in closes)
        nxt = code[i + 1].lstrip() if i + 1 < len(code) else ""
        tail = l.rstrip("\n").rstrip()
        cont = depth > 0 or (lang == "Python" and tail.endswith("\\")) or (lang != "Python" and (nxt.startswith(".") or tail.endswith("=>")))
        if not cont:
            groups.append((start, i + 1)); start = None; depth = 0
    if start is not None:
        groups.append((start, len(lines)))
    return groups


def _group_index(groups, n):
    idx = [0] * n
    for g, (a, b) in enumerate(groups):
        for i in range(a, b):
            idx[i] = g
    return idx


# ------------------------------------------------------------------------------------------------------------------------ construct markers
# NAT[fam] matches the natural rendering of one construct on a (stripped) logical line; ALT[fam] its alternative rendering.
NAT = {"py_ternary": re.compile(r"^\s*(?!(?:if|elif|while|for|assert)\b).*?\bif\b.*?\belse\b", re.S),   # on ONE logical line (backslash continuations joined)
       "py_comprehension": re.compile(r"[\[\{\(].*?\bfor\b", re.S),
       "rust_question": re.compile(r"\?"),
       "js_arrow": re.compile(r"=>"),
       "py_with_open": re.compile(r"^\s*with\b[^\n]*\bopen\(")}
ALT = {"py_ternary": re.compile(r"^\s*if\b", re.M),
       "py_comprehension": re.compile(r"^\s*for\b", re.M),
       "rust_question": re.compile(r"\bmatch\b"),
       "js_arrow": re.compile(r"\bfunction\b"),
       "py_with_open": re.compile(r"=\s*open\(")}
MIN_INDENT_ALT = {"py_ternary", "py_comprehension"}                  # alt openers only at the block's outermost indentation (nested ifs / fors are inner)
RAW_NAT = {"py_ternary", "js_arrow"}                                  # nat marker read on the raw text: the construct may sit inside an f-string / template literal
_INIT = re.compile(r"^\s*[\w.\[\]'\"]+\s*=\s*[^\n]*$")               # `res = []`, `total = 0`, `found = False` ... the initialiser the loop needs
_CLOSER_LINE = re.compile(r"^\s*[\)\]\}]*[;,]?\s*$")
_CLOSE_CALL = re.compile(r"^\s*[\w.]+\.close\(\)\s*$")
_FORCED = {"py_with_open": lambda l: bool(_CLOSER_LINE.match(l) or _CLOSE_CALL.match(l)),
           "js_arrow": lambda l: bool(_CLOSER_LINE.match(l)),
           "rust_question": lambda l: bool(_CLOSER_LINE.match(l)),
           "py_ternary": lambda l: bool(_CLOSER_LINE.match(l) or re.match(r"^\s*else:\s*$", l)),
           "py_comprehension": lambda l: bool(_CLOSER_LINE.match(l))}


def _indent(l):
    return len(l) - len(l.lstrip(" \t"))


def _marker_text(fam, lines, lang, side):
    return list(lines) if (side == "nat" and fam in RAW_NAT) else [strip_code(l, lang) for l in lines]


def has_marker(fam, lines, lang, side):
    """True if some LOGICAL line of the block carries this side's construct marker."""
    return bool(openers(fam, lines, lang, side)) if side == "nat" or fam not in MIN_INDENT_ALT else any(
        (ALT[fam].search("".join(_marker_text(fam, lines, lang, side)[a:b])) is not None) for a, b in logical_lines(lines, lang))


def openers(fam, lines, lang, side):
    """Physical indices (into `lines`, a block) of the logical lines that open a construct on this side."""
    rx = (NAT if side == "nat" else ALT)[fam]
    groups = logical_lines(lines, lang)
    code = _marker_text(fam, lines, lang, side)
    cands = []
    for a, b in groups:
        text = "".join(code[a:b])
        if not text.strip():
            continue
        if rx.search(text):
            cands.append(a)
    if side == "alt" and fam in MIN_INDENT_ALT:
        nonblank = [l for l in lines if l.strip()]
        m = min((_indent(l) for l in nonblank), default=0)
        cands = [a for a in cands if _indent(lines[a]) == m]
        if fam in ("py_comprehension", "py_ternary"):                # the initialiser (`res = []`) / pre-declaration (`x = None`) just before starts the construct
            out = []
            for a in cands:
                if a - 1 >= 0 and a - 1 not in out and _indent(lines[a - 1]) == m and _INIT.match(code[a - 1]) and not ALT[fam].match(code[a - 1]):
                    out.append(a - 1)
                else:
                    out.append(a)
            cands = out
    return cands


# ----------------------------------------------------------------------------------------------------------------------- line alignment
def _line_table(text):
    lines = text.splitlines(keepends=True)
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l))
    return lines, offs


def line_blocks_raw(nat, alt):
    """Non-equal runs of a line diff (trailing whitespace ignored): list of (i1, i2, j1, j2) into the two line lists."""
    ln, _ = _line_table(nat); la, _ = _line_table(alt)
    kn = [l.rstrip() for l in ln]; ka = [l.rstrip() for l in la]
    sm = difflib.SequenceMatcher(None, kn, ka, autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def _block_text(lines, offs, i1, i2):
    """(text, start, end) of lines[i1:i2]; the final newline is dropped (an empty range gives a point at the start of line i1)."""
    s0 = offs[i1]; s1 = offs[i2]
    t = "".join(lines[i1:i2])
    return t, s0, s1


def _make_opp(fam, nat_lines, nat_offs, alt_lines, alt_offs, i1, i2, j1, j2):
    tn, s0, s1 = _block_text(nat_lines, nat_offs, i1, i2)
    ta, a0, a1 = _block_text(alt_lines, alt_offs, j1, j2)
    if tn and ta and tn.endswith("\n") and ta.endswith("\n"):        # a replaced block: the span ends at the end of its last line
        tn, ta, s1, a1 = tn[:-1], ta[:-1], s1 - 1, a1 - 1
    # first divergence at the TOKEN level (code_build.TOK): the cue is the last shared token before it
    xn, xa = TOK.findall(tn), TOK.findall(ta)
    p = 0
    while p < len(xn) and p < len(xa) and xn[p] == xa[p]:
        p += 1
    off = sum(len(t) for t in xn[:p])
    return {"nat": tn[off:], "alt": ta[off:], "nat_span": [s0 + off, s1], "alt_span": [a0 + off, a1],
            "_lines": [i1, i2, j1, j2], "_prefix": tn[:off]}


def _trim_leading(nat_lines, alt_lines, on0, oa0):
    """Lines before the first opener on each side are forced by an EARLIER block choice when they are blank or pairwise equal up to
    indentation (bug 5: re-indentation): the block then starts at the openers. Returns (dn, da) = lines to drop on each side, or (0, 0)."""
    kn = [l for l in nat_lines[:on0] if l.strip()]; ka = [l for l in alt_lines[:oa0] if l.strip()]
    if len(kn) == len(ka) and all(x.strip() == y.strip() for x, y in zip(kn, ka)):
        return on0, oa0
    return 0, 0


def split_block(fam, lang, nat_lines, alt_lines, i1, i2, j1, j2):
    """Cut a block that holds n >= 2 consecutive constructs (n natural openers, n alternative openers) into n sub-blocks at the openers;
    forced re-indented / blank lines before the first opener are dropped (`_trim_leading`). Mismatched counts keep the block whole."""
    on = openers(fam, nat_lines[i1:i2], lang, "nat"); oa = openers(fam, alt_lines[j1:j2], lang, "alt")
    if not on or not oa or len(on) != len(oa):
        return [(i1, i2, j1, j2)], len(on), len(oa)
    dn, da = _trim_leading(nat_lines[i1:i2], alt_lines[j1:j2], on[0], oa[0])
    bn = [i1 + dn] + [i1 + a for a in on[1:]] + [i2]; ba = [j1 + da] + [j1 + a for a in oa[1:]] + [j2]
    return [(bn[k], bn[k + 1], ba[k], ba[k + 1]) for k in range(len(on))], len(on), len(oa)


def merge_logical(blocks, nat_lines, alt_lines, lang):
    """Two blocks whose nat (or alt) lines belong to the same LOGICAL line (a multi-line statement whose middle lines are equal, e.g. a Rust
    method chain ending in `?;`) are one construct: merge them (the equal lines in between are absorbed)."""
    gn = _group_index(logical_lines(nat_lines, lang), len(nat_lines)); ga = _group_index(logical_lines(alt_lines, lang), len(alt_lines))
    out, merges = [], 0
    for b in blocks:
        if out:
            p = out[-1]
            same_n = p[1] > p[0] and b[1] > b[0] and gn[p[1] - 1] == gn[b[0]]
            same_a = p[3] > p[2] and b[3] > b[2] and ga[p[3] - 1] == ga[b[2]]
            if same_n or same_a:
                out[-1] = (p[0], b[1], p[2], b[3]); merges += 1; continue
        out.append(b)
    return out, merges


def line_blocks(fam, rec, split_adjacent=True, require_marker=True):
    """(blocks, eligible): every line-level block of the twins as an opp dict (raw diff replacement for one_per_line) and the ones that are
    genuine convention choices. Each dict carries `_why` (why it is free), `_lines`, `_prefix`, `_openers`, `_split`, `_merged`."""
    lang = CODE_FAMILY[fam].tgt_lang
    nat, alt = rec["text_nat"], rec["text_alt"]
    nl, no = _line_table(nat); al, ao = _line_table(alt)
    raw = line_blocks_raw(nat, alt)
    merged, n_merged = merge_logical(raw, nl, al, lang)
    blocks, eligible = [], []
    for (i1, i2, j1, j2) in merged:
        subs, cn, ca = split_block(fam, lang, nl, al, i1, i2, j1, j2) if split_adjacent else ([(i1, i2, j1, j2)], None, None)
        for (a, b, c, d) in subs:
            o = _make_opp(fam, nl, no, al, ao, a, b, c, d)
            o["_openers"] = [cn, ca]; o["_split"] = len(subs) > 1
            why = free_reason(fam, lang, rec, o, nl[a:b], al[c:d], require_marker)
            o["_why"] = why
            blocks.append(o)
            if why is None:
                eligible.append(o)
    if blocks:
        blocks[0]["_n_merged_doc"] = n_merged
    return blocks, eligible


def free_reason(fam, lang, rec, o, nlines, alines, require_marker):
    """None if the block is a genuine choice, else a short label."""
    if o["nat"].split() == o["alt"].split():
        return "whitespace_only"                                     # bug 5: re-indentation forced by an earlier block choice
    if closing_symbol(rec, o, lang):
        return "closing_symbol"                                      # bug 10
    fl = _FORCED[fam]
    if (not o["nat"].strip() and all(fl(l) for l in alines if l.strip())) or (not o["alt"].strip() and all(fl(l) for l in nlines if l.strip())):
        return "forced_closer"                                       # a `.close()` / `};` / `else:` insertion that an earlier choice forces
    if fam in OPENER_ONLY and not OPENER_ONLY[fam](o):
        return "opener_only"                                         # bug 11 (py_with_open: the natural side must contain `with`)
    if require_marker:
        if not has_marker(fam, nlines, lang, "nat") or not has_marker(fam, alines, lang, "alt"):
            return "no_marker"                                       # neither rendering shows the construct: a rewrite slip, not a convention
    return None


def line_opportunities(fam, rec, split_adjacent=True, require_marker=True):
    """The counted opportunities of a LINE_ALIGN family in the rec["opps"] format (k renumbered), bug-13 line rule applied on top."""
    blocks, eligible = line_blocks(fam, rec, split_adjacent, require_marker)
    strip = lambda o: {k: v for k, v in o.items() if not k.startswith("_")}
    kept = one_per_line(fam, dict(rec, opps=[strip(o) for o in blocks]), [strip(o) for o in eligible])
    return [dict(o, k=i) for i, o in enumerate(kept)]


# ---------------------------------------------------------------------------------------------------------------------------- validation
_PREFIX_MARK = {"py_ternary": re.compile(r"\bif\b|\belse\b"), "py_comprehension": re.compile(r"\bfor\b"), "rust_question": re.compile(r"\?|\bmatch\b"),
                "js_arrow": re.compile(r"=>|\bfunction\b"), "py_with_open": re.compile(r"\bwith\b|\bopen\(|\.close\(\)")}


def check_a(fam, lang, o):
    """(ok, label): the first divergence is a genuine convention choice (family-specific test on the opp texts)."""
    n, a = strip_code(o["nat"], lang), strip_code(o["alt"], lang)
    na, aa = o["nat"], o["alt"]
    if fam == "py_ternary":
        nat_ok = has_marker(fam, (o["_prefix"] + na).splitlines(keepends=True), lang, "nat")
        alt_ok = re.match(r"^\s*if\b", a) is not None
        alt_has = re.search(r"^\s*if\b", a, re.M) is not None
        return nat_ok and alt_ok, ("nat_no_ternary" if not nat_ok else "") + ("" if alt_ok else ("alt_predeclared_then_if" if alt_has else "alt_no_if"))
    if fam == "py_comprehension":
        nat_ok = NAT[fam].search(strip_code(o["_prefix"] + na, lang)) is not None
        alt_ok = any(re.match(r"^\s*for\b", l) for l in a.split("\n"))
        return nat_ok and alt_ok, ("nat_no_comprehension" if not nat_ok else "") + ("alt_no_for_line" if not alt_ok else "")
    if fam == "rust_question":
        nat_ok = "?" in n
        alt_ok = re.search(r"\bmatch\b", a.split("\n")[0]) is not None                       # `match` on the alt block's first line (`let _ = match ...` allowed)
        alt_has = re.search(r"\bmatch\b", a) is not None
        return nat_ok and alt_ok, ("nat_no_?" if not nat_ok else "") + ("" if alt_ok else ("alt_match_later_line" if alt_has else "alt_no_match"))
    if fam == "js_arrow":
        nat_ok = "=>" in na
        alt_ok = re.match(r"^\s*(async\s+)?function\b", a) is not None
        alt_has = re.search(r"\bfunction\b", a) is not None
        return nat_ok and alt_ok, ("nat_no_=>" if not nat_ok else "") + ("" if alt_ok else ("alt_function_not_first" if alt_has else "alt_no_function"))
    if fam == "py_with_open":
        nat_ok = re.match(r"^\s*with\b[^\n]*\bopen\(", n) is not None
        alt_ok = re.search(r"=\s*open\(", a) is not None
        return nat_ok and alt_ok, ("nat_not_with_open" if not nat_ok else "") + ("alt_no_open" if not alt_ok else "")
    raise KeyError(fam)


def check_b(fam, lang, o):
    """The shared prefix of the block (before the divergence) carries no construct marker of either side."""
    pre = strip_code(o["_prefix"], lang)
    return _PREFIX_MARK[fam].search(pre) is None


def unconverted(fam, lang, rec):
    """Natural logical lines that carry the construct marker but lie entirely in EQUAL lines (the rewrite left them, or the alignment
    matched them): count."""
    nl, _ = _line_table(rec["text_nat"])
    inblock = set()
    for (i1, i2, _, _) in line_blocks_raw(rec["text_nat"], rec["text_alt"]):
        inblock.update(range(i1, i2))
    n = 0; code = _marker_text(fam, nl, lang, "nat")
    for a, b in logical_lines(nl, lang):
        if NAT[fam].search("".join(code[a:b])) and not any(i in inblock for i in range(a, b)):
            n += 1
    return n


def variant_text(rec, pole, opps, k):
    """Twin `pole` with opportunity k rendered in the other pole (cue_tokens.variant on a given opp list)."""
    other = "alt" if pole == "nat" else "nat"; text = rec[f"text_{pole}"]
    out, pos = [], 0
    for i, o in enumerate(opps):
        a, b = o[f"{pole}_span"]; out.append(text[pos:a]); out.append(o[other] if i == k else o[pole]); pos = b
    out.append(text[pos:])
    return "".join(out)


def cue_info(rec, opps, k, tok, pole="nat"):
    """(cue_idx, cue_tok, cue_char_end, next_pole_id, next_other_id) with the Qwen tokenizer, as cue_tokens.cues_for computes it."""
    from src.sandbox.style_translation.cue_tokens import header_for
    header = header_for(rec); full = header + rec[f"text_{pole}"]
    enc = tok(full, return_offsets_mapping=True); ids, offs = enc.input_ids, enc.offset_mapping
    ids_b = tok(header + variant_text(rec, pole, opps, k)).input_ids
    n = 0
    while n < len(ids) and n < len(ids_b) and ids[n] == ids_b[n]:
        n += 1
    cue_idx = n - 1
    return {"cue_idx": cue_idx, "cue_tok": tok.decode([ids[cue_idx]]), "cue_char_end": offs[cue_idx][1] - len(header),
            "next_pole": ids[n] if n < len(ids) else None, "next_other": ids_b[n] if n < len(ids_b) else None, "h": len(header)}


def validate_family(fam, recs, tok=None, split_adjacent=True, require_marker=True, n_fail_examples=2):
    lang = CODE_FAMILY[fam].tgt_lang
    S = collections.Counter(); fails = collections.defaultdict(list); per_doc = []; free = collections.Counter()
    old_surv = old_moved = old_gone = new_exact = new_moved = new_brand = 0
    lt5, fifth75, k0_mismatch, next_same = [], [], [], []
    multi = collections.Counter()
    for rec in recs:
        blocks, elig = line_blocks(fam, rec, split_adjacent, require_marker)
        new = line_opportunities(fam, rec, split_adjacent, require_marker)
        S["logical_merges"] += blocks[0].get("_n_merged_doc", 0) if blocks else 0
        for o in blocks:
            free[o["_why"] or "counted"] += 1
            if o["_openers"][0] is not None and max(o["_openers"]) >= 2:
                multi["split" if o["_split"] else f"kept_whole({o['_openers'][0]}n/{o['_openers'][1]}a)"] += 1
        S["unconverted_nat_constructs"] += unconverted(fam, lang, rec)
        per_doc.append(len(new))
        if len(new) < 5:
            lt5.append((rec["doc_id"], len(new)))
        elif new[4]["nat_span"][0] >= 0.75 * len(rec["text_nat"]):
            fifth75.append(rec["doc_id"])
        # (a) and (b) on the counted list
        for o in new:
            S["n_opps"] += 1
            full = next(b for b in blocks if b["nat_span"] == o["nat_span"])
            ok, label = check_a(fam, lang, full)
            if ok:
                S["a_pass"] += 1
            else:
                S["a_fail"] += 1; fails["a:" + label].append((rec["doc_id"], o["k"], full["_prefix"][-40:], o["nat"][:90], o["alt"][:90]))
            if check_b(fam, lang, full):
                S["b_pass"] += 1
            else:
                S["b_fail"] += 1; fails["b:marker_in_shared_prefix"].append((rec["doc_id"], o["k"], full["_prefix"][-40:], o["nat"][:90], o["alt"][:90]))
        # comparison with the stored counted list
        old = rec["opps"]; new_starts = {o["nat_span"][0] for o in new}
        eff = lambda oo: oo["nat_span"][0] + (len(oo["nat"]) - len(oo["nat"].lstrip())) if oo["nat"].strip() else oo["nat_span"][0]
        for oo in old:
            s = eff(oo)
            if s in new_starts:
                old_surv += 1
            elif any(o["nat_span"][0] < s <= o["nat_span"][1] for o in new):
                old_moved += 1
            else:
                old_gone += 1
        old_starts = {eff(oo) for oo in old}
        for o in new:
            s0, s1 = o["nat_span"]
            if s0 in old_starts:
                new_exact += 1
            elif any(s0 < eff(oo) <= s1 for oo in old):
                new_moved += 1
            else:
                new_brand += 1
        # (d) tokenisation
        if tok is not None and new:
            for pole in ("nat", "alt"):
                for o in new:
                    c = cue_info(rec, new, o["k"], tok, pole)
                    S["d_checked"] += 1
                    if c["next_pole"] is None or c["next_other"] is None or c["next_pole"] == c["next_other"]:
                        next_same.append((rec["doc_id"], o["k"], pole)); S["d_next_fail"] += 1
            cn, ca = cue_info(rec, new, 0, tok, "nat"), cue_info(rec, new, 0, tok, "alt")
            pn = rec["text_nat"][:cn["cue_char_end"]]; pa = rec["text_alt"][:ca["cue_char_end"]]
            if pn != pa or cn["cue_idx"] != ca["cue_idx"]:
                k0_mismatch.append(rec["doc_id"]); S["d_k0_fail"] += 1
    out = {"family": fam, "docs": len(recs), "counted_total": S["n_opps"], "mean_per_doc": round(sum(per_doc) / len(per_doc), 2), "min_per_doc": min(per_doc),
           "docs_lt5": lt5, "docs_5th_past_75pct": fifth75, "a_pass": S["a_pass"], "a_fail": S["a_fail"], "a_frac": round(S["a_pass"] / max(S["n_opps"], 1), 4),
           "b_pass": S["b_pass"], "b_fail": S["b_fail"], "free_blocks": dict(free), "multi_construct_blocks": dict(multi),
           "unconverted_nat_constructs": S["unconverted_nat_constructs"], "logical_line_merges": S["logical_merges"],
           "old": {"total": sum(len(r["opps"]) for r in recs), "survive_exact": old_surv, "moved_to_block_start": old_moved, "disappear": old_gone},
           "new": {"total": S["n_opps"], "exact_old": new_exact, "block_contains_old": new_moved, "brand_new": new_brand},
           "d": {"checked": S["d_checked"], "next_token_same_or_missing": S["d_next_fail"], "k0_prompt_mismatch": len(k0_mismatch), "k0_mismatch_docs": k0_mismatch[:20], "next_fail_cases": next_same[:10]} if tok is not None else None,
           "fail_classes": {k: {"n": len(v), "docs": len({x[0] for x in v}), "examples": v[:n_fail_examples]} for k, v in fails.items()}}
    return out


def example_entry(rec, opps, k, tok):
    o = opps[k]; c = cue_info(rec, opps, k, tok, "nat")
    e = c["cue_char_end"]
    return {"k": k, "context_before_cue": rec["text_nat"][max(0, e - 80):e], "cue_tok": c["cue_tok"], "opp_nat": o["nat"][:70], "opp_alt": o["alt"][:70]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true", help="print per-family counts and examples; nothing is written to pairs/")
    ap.add_argument("--families", nargs="*", default=sorted(LINE_ALIGN_FAMILIES))
    ap.add_argument("--examples", type=int, default=2, help="documents shown per family")
    ap.add_argument("--tokenizer", action="store_true", help="run the Qwen2.5-7B tokenisation checks (d)")
    ap.add_argument("--no_split", action="store_true", help="do not cut multi-construct blocks at their openers")
    ap.add_argument("--no_marker", action="store_true", help="count blocks even without the construct marker")
    ap.add_argument("--out", default=None, help="write the validation summary + 4 random example docs per family to this json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.dry_run:
        print("this module only dry-runs; pass --dry_run", file=sys.stderr); sys.exit(2)
    tok = None
    if args.tokenizer or args.out:
        from transformers import AutoTokenizer
        from src.sandbox.style_translation.models import paths as model_paths
        tok = AutoTokenizer.from_pretrained(model_paths("qwen25_code")["tokenizer"])
    report = {"validation": {}, "examples": {}}
    for fam in args.families:
        recs = json.load(open(PAIRS / f"{fam}.json"))
        v = validate_family(fam, recs, tok, not args.no_split, not args.no_marker)
        report["validation"][fam] = v
        print(f"\n===== {fam}: {v['docs']} docs, counted {v['counted_total']} (old {v['old']['total']}), mean {v['mean_per_doc']}, min {v['min_per_doc']}, "
              f"<5: {len(v['docs_lt5'])}, 5th>75%: {len(v['docs_5th_past_75pct'])}")
        print(f"  (a) {v['a_pass']}/{v['counted_total']} = {v['a_frac']}   (b) fail {v['b_fail']}   free blocks {v['free_blocks']}   multi-construct {v['multi_construct_blocks']}   unconverted nat constructs {v['unconverted_nat_constructs']}   logical-line merges {v['logical_line_merges']}")
        print(f"  old: {v['old']}\n  new: {v['new']}\n  <5: {v['docs_lt5']}\n  5th>75%: {v['docs_5th_past_75pct']}")
        if v["d"]:
            print(f"  (d) {v['d']['checked']} cues checked, next-token same/missing {v['d']['next_token_same_or_missing']}, k0 prompt mismatch {v['d']['k0_prompt_mismatch']}")
        for cls, info in v["fail_classes"].items():
            print(f"  FAIL {cls}: {info['n']} opps in {info['docs']} docs")
            for ex in info["examples"]:
                print(f"      {ex[0]} k={ex[1]} prefix={ex[2]!r} nat={ex[3]!r} alt={ex[4]!r}")
        rng = random.Random(args.seed); picks = rng.sample(recs, min(4, len(recs)))
        if tok is not None:
            report["examples"][fam] = []
            for rec in picks:
                new = line_opportunities(fam, rec, not args.no_split, not args.no_marker)
                report["examples"][fam].append({"doc_id": rec["doc_id"], "old": [example_entry(rec, rec["opps"], k, tok) for k in range(len(rec["opps"]))],
                                                "new": [example_entry(rec, new, k, tok) for k in range(len(new))]})
        for rec in picks[: args.examples]:
            new = line_opportunities(fam, rec, not args.no_split, not args.no_marker)
            print(f"  --- {rec['doc_id']}: old {len(rec['opps'])} -> new {len(new)}")
            for o in new:
                print(f"     k={o['k']:2d} @{o['nat_span'][0]:5d}  nat={o['nat'][:60]!r}  |  alt={o['alt'][:60]!r}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
