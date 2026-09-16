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
AFFECTED = IDENT_FAMILIES | SELF_FAMILIES | INDENT_FAMILIES
MIN_OPPS = 5
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


def free_opportunity(fam, rec, k):
    o = rec["opps"][k]; task = rec.get("text_es", "")
    if fam == "py_loop_vars":                      # a new `for` statement is a fresh binding; uses inside the body are forced
        text = rec["text_nat"]; s0 = o["nat_span"][0]; line_start = text.rfind("\n", 0, s0) + 1
        return re.search(r"\bfor\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*,\s*)*$", text[line_start:s0]) is not None
    if fam in IDENT_FAMILIES:
        pinned = _task_code_idents(task)
        for pole in ("nat", "alt"):
            text = rec[f"text_{pole}"]; s0 = o[f"{pole}_span"][0]; before = text[:s0]
            idents = _ID.findall(o[pole])
            if not idents:
                return False
            if any(t in pinned or _seen_before(t, before) for t in idents):
                return False
        return True
    if fam in SELF_FAMILIES:
        text = rec["text_nat"]; s0 = o["nat_span"][0]; line_start = text.rfind("\n", 0, s0) + 1
        return re.search(r"def\s+\w+\s*\(\s*$", text[line_start:s0]) is not None
    if fam in INDENT_FAMILIES:
        text = rec["text_nat"]; s0 = o["nat_span"][0]
        prev = [l for l in text[:s0].split("\n") if l.strip()]
        return bool(prev) and prev[-1].rstrip().endswith(":")
    return True


def filter_free(rec, fam, min_opps=MIN_OPPS):
    if fam not in AFFECTED:
        return rec
    keep = [o for k, o in enumerate(rec["opps"]) if free_opportunity(fam, rec, k)]
    if len(keep) < min_opps or keep[min_opps - 1]["nat_span"][0] >= 0.75 * len(rec["text_nat"]):
        return None
    out = dict(rec); out["opps"] = [dict(o, k=i) for i, o in enumerate(keep)]; out["k_en"] = len(keep); out["free_filter"] = True
    return out
