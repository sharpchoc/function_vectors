#!/usr/bin/env python
"""Padding audit of the code-convention pairs (2026-09-21): classify every document of the POOL families into ONE worst-problem category.

A document is flagged if any line of its natural twin matches `code_build.PADDING_PATTERNS` (`padding_lines`) or a comment / docstring
matches `code_build.LEAK_PATTERNS` / the family's own convention words (`leak_lines`). Priority (first that applies wins):
  1 style_leak      a comment / docstring mentions style / convention / requirement / consistency / idiom / PEP 8 / "for the exercise" /
                    "as required" / "to satisfy ..." OR names the family's convention words (nat_label / alt_label minus generic words)
  2 construct_pad   padding in CODE: trivial branch (`elif True`, `range(1)`), discarded result (`_ = ...`), style meta comment
                    (`opportunit...`), genuine stacked negation (not the JavaScript `!!x` idiom)
  3 comment_family  filler / example wording in a comment of a family whose construct IS the comment (comment_language, comment_case,
                    c_comment_style, docstring_quotes, docstring_style) — cannot be stripped, must be regenerated
  4 code_filler     filler wording inside code (identifiers such as dummy_value, placeholder_x)
  5 commentary      filler / example wording only inside comments of a non-comment family (removable by rule: strip the comment)
  6 false_positive  `!!` in JavaScript, `!!!` inside a string literal
Fix per category (user decision 2026-09-21): 1-4 regenerate, 5 strip the comment in both twins then review, 6 nothing.

    python src/sandbox/style_translation/code_padding_audit.py [--families ...] [--all] [--csv results/code_styles/padding_audit.csv]
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import RESULTS_ROOT
from src.sandbox.style_translation.code_families import CODE_FAMILIES, CODE_FAMILY
from src.sandbox.style_translation.code_build import PAIRS, PADDING_PATTERNS, padding_lines, leak_lines, comment_texts

COMMENT_FAMILIES = {"comment_language", "comment_case", "c_comment_style", "docstring_quotes", "docstring_style"}
CATEGORIES = {1: "style_leak", 2: "construct_pad", 3: "comment_family", 4: "code_filler", 5: "commentary", 6: "false_positive"}
_IDENT_FILLER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:dummy|placeholder|demonstrat|showcase)[A-Za-z0-9_]*|\b(?:dummy|placeholder|demonstrat|showcase)[A-Za-z0-9_]*", re.I)
_STR_RX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\])*`', re.S)


def pool_families():
    """The pool = families with keep == True in results/code_styles/cutoff_k4.csv (56 of 60)."""
    f = RESULTS_ROOT / "code_styles" / "cutoff_k4.csv"
    with open(f) as fh:
        return [r["family"] for r in csv.DictReader(fh) if r["keep"] == "True"]


def classify(fam, rec):
    """(category, pattern, evidence line) or None when the natural twin is clean."""
    F = CODE_FAMILY[fam]; nat = rec["text_nat"]; lang = F.tgt_lang
    leaks = leak_lines(nat, lang, fam)
    if leaks:
        i, name, line = leaks[0]
        return 1, name, line
    pads = padding_lines(nat, lang)
    if not pads:
        return None
    comments = comment_texts(nat, lang)                            # line_no -> comment / docstring text of that line
    cat = {}
    for i, name, line in pads:
        ctext = comments.get(i, ""); code_part = line[: line.find(ctext)] if ctext and ctext in line else (line if not ctext else "")
        if name == "stacked_negation":
            c = 6 if (lang == "JavaScript" and "not" not in line) or any(re.search(r"!\s*!", m.group()) for m in _STR_RX.finditer(line)) else 2
        elif name in ("trivial_branch", "discarded_result", "style_meta_comment"):
            c = 2
        else:                                                      # filler_wording / example_comment
            if _IDENT_FILLER.search(_STR_RX.sub("", code_part)):      # filler wording inside an identifier of the code part
                c = 4
            else:
                c = 3 if fam in COMMENT_FAMILIES else 5
        cat.setdefault(c, (name, line))
    c = min(cat); return c, cat[c][0], cat[c][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=None, help="default: the pool families")
    ap.add_argument("--all", action="store_true", help="all 60 families")
    ap.add_argument("--csv", default=str(RESULTS_ROOT / "code_styles" / "padding_audit.csv"))
    args = ap.parse_args()
    fams = args.families or ([F.name for F in CODE_FAMILIES] if args.all else pool_families())
    rows = []; per_fam = {}
    for fam in fams:
        c = Counter()
        for r in json.load(open(PAIRS / f"{fam}.json")):
            out = classify(fam, r)
            if out:
                cat, name, line = out; c[cat] += 1
                rows.append({"doc_id": r["doc_id"], "family": fam, "category": cat, "label": CATEGORIES[cat], "pattern": name, "evidence": line.strip()[:200]})
        per_fam[fam] = c
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "family", "category", "label", "pattern", "evidence"]); w.writeheader(); w.writerows(rows)
    tot = Counter(r["category"] for r in rows)
    print("families", len(fams), "| flagged docs", len(rows), "| per category", {f"{k} {CATEGORIES[k]}": tot[k] for k in sorted(CATEGORIES)})
    for fam in fams:
        if per_fam[fam]:
            print(f"  {fam:20s} {sum(per_fam[fam].values()):4d}  " + " ".join(f"c{k}={v}" for k, v in sorted(per_fam[fam].items())))
    print("csv:", args.csv)


if __name__ == "__main__":
    main()


# ---- category 5: strip removable commentary by rule (2026-09-21) ------------------------------------------------------------------
from src.sandbox.style_translation.code_build import _COMMENT_MARK, align, valid_source  # noqa: E402
from src.sandbox.style_translation.code_free import filter_free  # noqa: E402


def _code_lines(text, lang):
    ct = comment_texts(text, lang); out = []
    for i, line in enumerate(text.split("\n"), 1):
        c = ct.get(i)
        code = line[: line.rfind(c)] if c and c in line else line
        if code.strip():
            out.append(code.rstrip())
    return out


def strip_commentary(rec, fam):
    """Category-5 fix: delete every whole-line comment and cut every trailing comment that the padding guard flags (filler / example wording in
    the comment text only), IDENTICALLY in both twins. Returns (new record, stripped log) or raises ValueError when the strip is not safe:
    a flagged line whose hit is in the code part, a docstring / block-comment line, a comment that does not occur exactly once in the
    alternative twin, twins that no longer parse, or a change of the code lines / of the counted opportunities."""
    F = CODE_FAMILY[fam]; lang = F.tgt_lang; mark = _COMMENT_MARK.get(lang, "//")
    nat, alt = rec["text_nat"], rec["text_alt"]
    ct = comment_texts(nat, lang); pads = padding_lines(nat, lang)
    if leak_lines(nat, lang, fam):
        raise ValueError("leak line present (not category 5)")
    todo = {}
    for i, name, line in pads:
        c = ct.get(i)
        if name not in ("filler_wording", "example_comment") or not c or not c.startswith(mark):
            raise ValueError(f"unstrippable hit L{i} {name}: {line.strip()[:60]}")
        rx = next(r for n, r, _ in PADDING_PATTERNS if n == name)
        code = line[: line.rfind(c)]
        if rx.search(code):
            raise ValueError(f"hit in code part L{i}: {line.strip()[:60]}")
        if code and not code.endswith((" ", "\t")):
            raise ValueError(f"marker glued to code L{i}: {line.strip()[:60]}")
        todo[i] = c
    if not todo:
        raise ValueError("nothing to strip")

    def apply(text, comments):
        lines = text.split("\n"); out = []; log = []
        for i, line in enumerate(lines, 1):
            c = comments.get(i)
            if c is not None:
                code = line[: line.rfind(c)]
                if code.strip():
                    out.append(code.rstrip()); log.append(("cut", i, c))
                else:
                    log.append(("del", i, c))
                continue
            out.append(line)
        return "\n".join(out), log

    nat2, log_nat = apply(nat, todo)
    alt_ct = comment_texts(alt, lang); alt_todo = {}
    for i, c in todo.items():
        hits = [j for j, ac in alt_ct.items() if ac == c]
        if len(hits) != 1:
            raise ValueError(f"comment occurs {len(hits)}x in alt: {c[:50]}")
        alt_todo[hits[0]] = c
    alt2, log_alt = apply(alt, alt_todo)
    if _code_lines(nat, lang) != _code_lines(nat2, lang) or _code_lines(alt, lang) != _code_lines(alt2, lang):
        raise ValueError("code lines changed")
    if not valid_source(lang, nat2) or (fam not in ("py2_print", "py2_except") and not valid_source(lang, alt2)):
        raise ValueError("stripped twin does not parse")
    if padding_lines(nat2, lang) or leak_lines(nat2, lang, fam):
        raise ValueError("still flagged after strip")
    opps, shared = align(nat2, alt2)
    if opps is None:
        raise ValueError("re-alignment failed")
    new = {k: v for k, v in rec.items() if k not in ("opps_all", "free_filter")}
    new.update(text_nat=nat2, text_alt=alt2, opps=opps, k_en=len(opps), shared_fraction=round(shared, 3))
    fr = filter_free(new, fam)
    if fr is None:
        raise ValueError("free filter rejects the stripped pair")
    old_c = [(o["nat"], o["alt"]) for o in rec["opps"]]; new_c = [(o["nat"], o["alt"]) for o in fr["opps"]]
    if old_c != new_c:
        raise ValueError(f"counted opportunities changed: {len(old_c)} -> {len(new_c)}")
    fr["stripped"] = [{"twin": "nat", "op": op, "line": i, "comment": c} for op, i, c in log_nat] + [{"twin": "alt", "op": op, "line": i, "comment": c} for op, i, c in log_alt]
    fr["pass"] = True
    return fr, fr["stripped"]
