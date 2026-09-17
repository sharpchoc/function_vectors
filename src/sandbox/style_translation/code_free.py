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
AFFECTED = IDENT_FAMILIES | SELF_FAMILIES | INDENT_FAMILIES | CONTENT_FAMILIES
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


def consistent_twins(fam, rec):
    """Identifier families: the two renderings must agree on which opportunities are first mentions; a mismatch means the rewrite
    changed a use but not its definition (or vice versa), i.e. the alt twin references an undefined name -> reject the document."""
    if fam not in IDENT_FAMILIES:
        return True
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
    keep = [o for k, o in enumerate(full["opps"]) if free_opportunity(fam, full, k)]
    if len(keep) < min_opps or keep[min_opps - 1]["nat_span"][0] >= 0.75 * len(full["text_nat"]):
        return None
    out = dict(full); out["opps_all"] = full["opps"]; out["opps"] = [dict(o, k=i) for i, o in enumerate(keep)]; out["k_en"] = len(keep); out["free_filter"] = True
    return out
