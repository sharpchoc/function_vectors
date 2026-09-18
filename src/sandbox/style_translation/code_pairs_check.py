#!/usr/bin/env python
"""Integrity sweep of the CODE-convention pairs (dataset_files/style_translation/pairs/<family>.json, one file per code family).
Until 2026-09-18 this sweep only lived as an inline script (WORKLOG bug 13); this file makes it permanent. Per family it counts documents
that violate any check and prints the families with issues; a clean corpus prints `families with issues: 0`.

Checks (per document): count == 200, unique doc_ids, both twins parse (py2_print / py2_except: natural twin only), >= 5 counted
opportunities, the 5th counted opportunity before 75 % of the code, identical prefixes up to the first counted span (k = 0 prompts),
span texts match the twins, degenerate / <ctrlNN> artefacts, bookkeeping (free_filter + opps_all), shared_fraction >= .6,
`stale_counted_list` (`code_free.counted(fam, opps_all)` reproduces the stored counted list), identifier families consistent and free of
rename collisions, `two_counted_on_a_line` (bug 13: never two counted spans on one line of either twin, line-exempt families aside),
`cue_mid_comment` (bug 14: in the COMMENT_TEXT families no counted opportunity has words of its comment before it on its line, in either twin,
and every counted opportunity lies in a comment unit), `cue_not_comment_opener` (comment cue rule, 2026-09-18: every counted opportunity of a
COMMENT_TEXT family lies in a unit that may count — the first text line of a docstring or a `#` comment that does not continue the previous
`#` line's sentence — judged on the natural twin, which defines the units).

    python src/sandbox/style_translation/code_pairs_check.py [--families f1 f2 ...] [--list]   (--list prints the offending doc_ids)
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.code_families import CODE_FAMILIES
from src.sandbox.style_translation.code_build import degenerate, valid_source, PAIRS
from src.sandbox.style_translation.code_free import counted, IDENT_FAMILIES, consistent_twins, rename_collisions, LINE_EXEMPT, COMMENT_TEXT, _line, comment_words_before, comment_opener_ok


def check_family(F, recs, listing=None):
    fam = F.name; c = Counter()
    if len(recs) != 200:
        c["count!=200"] = len(recs)
    if len({r["doc_id"] for r in recs}) != len(recs):
        c["dup_ids"] += 1
    if F.tgt_lang in ("Python", "JavaScript", "Rust", "PHP", "Bash"):
        with ThreadPoolExecutor(8) as ex:
            v = list(ex.map(lambda r: valid_source(F.tgt_lang, r["text_nat"]) and (fam in ("py2_print", "py2_except") or valid_source(F.tgt_lang, r["text_alt"])), recs))
        c["invalid_twin"] += sum(1 for x in v if not x)
        if listing is not None:
            listing["invalid_twin"] += [r["doc_id"] for r, x in zip(recs, v) if not x]
    for r in recs:
        nat, alt = r["text_nat"], r["text_alt"]; o = r["opps"]; flags = {}
        flags["<5"] = len(o) < 5
        flags["5th>75%"] = len(o) >= 5 and o[4]["nat_span"][0] >= .75 * len(nat)
        flags["k0"] = bool(o) and nat[:o[0]["nat_span"][0]] != alt[:o[0]["alt_span"][0]]
        flags["span"] = any(nat[slice(*x["nat_span"])] != x["nat"] or alt[slice(*x["alt_span"])] != x["alt"] for x in o)
        flags["guard"] = bool(degenerate(nat) or degenerate(alt))
        flags["artefact"] = bool(re.search(r"<ctrl\d+>", nat + alt))
        flags["bookkeeping"] = not (r.get("free_filter") and "opps_all" in r)
        flags["shared<.6"] = r["shared_fraction"] < 0.6
        full = dict(r); full["opps"] = r.get("opps_all") or r["opps"]
        flags["stale_counted_list"] = [x["nat_span"] for x in counted(fam, full)] != [x["nat_span"] for x in o]
        if fam in IDENT_FAMILIES:
            flags["inconsistent"] = not consistent_twins(fam, full); flags["collision"] = bool(rename_collisions(fam, full))
        if fam not in LINE_EXEMPT:                               # the bug-13 purge check: no two counted spans on one line of either twin
            ln = [_line(nat, x["nat_span"][0], x["nat"]) for x in o]; la = [_line(alt, x["alt_span"][0], x["alt"]) for x in o]
            flags["two_counted_on_a_line"] = len(set(ln)) != len(ln) or len(set(la)) != len(la)
        if fam in COMMENT_TEXT:                                  # the bug-14 check: the cue of a comment opportunity is the comment's opener
            flags["cue_mid_comment"] = any(comment_words_before(nat, x["nat_span"][0]) or comment_words_before(alt, x["alt_span"][0]) for x in o)
            flags["cue_not_comment_opener"] = any(not comment_opener_ok(nat, x["nat_span"][0]) for x in o)
        for k, v in flags.items():
            if v:
                c[k] += 1
                if listing is not None:
                    listing[k].append(r["doc_id"])
    return {k: v for k, v in c.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[F.name for F in CODE_FAMILIES])
    ap.add_argument("--list", action="store_true", help="print the offending doc_ids per check")
    args = ap.parse_args()
    fams = {F.name: F for F in CODE_FAMILIES}
    tot = 0; issues = {}
    for name in args.families:
        F = fams[name]; recs = json.load(open(PAIRS / f"{name}.json")); tot += len(recs)
        listing = Counter() if False else {k: [] for k in ("invalid_twin", "<5", "5th>75%", "k0", "span", "guard", "artefact", "bookkeeping", "shared<.6", "stale_counted_list", "inconsistent", "collision", "two_counted_on_a_line", "cue_mid_comment", "cue_not_comment_opener")}
        c = check_family(F, recs, listing if args.list else None)
        if c:
            issues[name] = c
            if args.list:
                for k, ids in listing.items():
                    if ids:
                        print(f"  {name} {k}: {ids}")
    print("docs", tot, "| families with issues:", len(issues), issues or "")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
