#!/usr/bin/env python
"""Evidence tokens of the code-convention k-shot prompts (USER DECISION 2026-09-18, CPU, tokenisation only).

For a prompt of pole P (nat / alt) at shot count k (prompt = task header + P's text up to and including the cue token of
opportunity k; the last token is the query cue):
  1. SCOPE   every token index j with header_len <= j < len(prompt) - 1 (header and query cue never evidence).
  2. A       the token right after each demonstration cue (+ paired closer; code = differing tokens only), i.e.
             `evidence_tokens.evidence_for` for instances 0..k-1 (its assertion failures are reported, A is then empty).
  3. B       every token in scope that differs between the prompt's token strings and the OTHER pole's twin tokenisation
             (difflib.SequenceMatcher on decoded token strings, autojunk off; every index of a non-'equal' opcode), minus the
             demonstration cue indices and the query cue.
  4. FILTER  a B token (not in A) is DROPPED when its stripped text is non-empty and equals the stripped text of some token in the
             aligned other-pole region of the same opcode while the raw texts differ (`(idx` vs ` idx`: an inserted symbol took the
             word's space) — except in NO_SPACE_FILTER families where that whitespace IS the convention. Pure whitespace tokens
             are never dropped.
  5. SECTION comment_language / comment_case / docstring_style: every token in scope whose char span overlaps a `#` comment
             (marker to end of line) or a docstring (triple-quoted string opening a logical line, opener to closer) that ends at
             or before the query cue's char start (`code_free.comment_units`, tokenizer-based, so `#` inside a string is not a
             comment). Char spans -> token indices by cumulative decoded lengths.
  evidence = A | B(filtered) | SECTION, minus cue indices, sorted.
Output per family: <paths(model)['evidence']>/<family>.json = list of {doc_id, pole, k, prompt_len, idx, toks, n_A, n_B, n_section,
n_dropped, a_error} for every prompt with k in --ks (n_B = |B \\ A|, n_section = |SECTION \\ (A | B)|, n_dropped = |dropped|).
"""
import argparse
import difflib
import json
import os
import re
import statistics
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation import evidence_tokens as ET
from src.sandbox.style_translation.cue_tokens import header_for
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_free import comment_units

SECTION_FAMILIES = {"comment_language", "comment_case", "docstring_style"}
NO_SPACE_FILTER = {"comma_space", "operator_spaces", "line_wrap", "py_tabs"}


_QUOTE = re.compile(r"^\s*[rRbBuU]{0,2}(\"\"\"|''')")


def section_units(text):
    """(start, end) char spans of the comment units of a Python text: each `#` comment, and each docstring as ONE unit from its
    opener to its closer (comment_units lists a docstring line by line — 'doc' up to the first line with text, then 'doc+' —
    so consecutive docstring lines are merged until the line that carries the closing quotes)."""
    units = comment_units(text)
    if units is None:
        return None
    out = []; q = None                                     # q = quotes of a docstring whose closer has not been seen yet
    for s, e, kind in units:
        if kind in ("doc", "doc+") and q and out and text[out[-1][1]:s].strip() == "":
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e)); q = None
            m = _QUOTE.match(text[s:e]) if kind in ("doc", "doc+") else None
            if m:
                q = m.group(1)
        if q:
            body = text[out[-1][0]:e]; rest = body[_QUOTE.match(body).end():]
            if rest.rstrip().endswith(q):
                q = None
    return out


def differing(fam, a, b, n):
    """B (differing) and WS (dropped leading-space-only) token indices of prompt tokens `a` vs other-pole tokens `b`."""
    B, WS = set(), set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        region = b[j1:j2]
        for j in range(i1, min(i2, n)):
            s = a[j].strip()
            if fam not in NO_SPACE_FILTER and s and any(a[j] != x and s == x.strip() for x in region):
                WS.add(j)
            else:
                B.add(j)
    return B, WS


def evidence_positions(fam, rec, pole, prompt_ids, k, tok):
    """Evidence token positions of one prompt (see module docstring). rec must carry the model's cues (rec['cues'][pole])."""
    header = header_for(rec); hdr = len(tok(header).input_ids); n = len(prompt_ids)
    other = "alt" if pole == "nat" else "nat"
    in_scope = lambda S: {j for j in S if hdr <= j < n - 1}
    demo_cues = {c["cue_idx"] for c in rec["cues"][pole] if c["k"] < k}
    excluded = demo_cues | {n - 1}
    a_error = None; ev = []
    if k > 0:
        ET.K = k
        try:
            ev = ET.evidence_for(rec, tok, pole, prompt_ids)
        except AssertionError as e:
            a_error = str(e) or "AssertionError"
    A = in_scope({j for e in ev for j in e["idx"]}) - excluded
    ids_o = tok(header + rec[f"text_{other}"]).input_ids
    a = [tok.decode([t]) for t in prompt_ids]; b = [tok.decode([t]) for t in ids_o]
    B, WS = differing(fam, a, b, n)
    B = in_scope(B) - excluded; WS = in_scope(WS) - excluded - A
    SEC = set(); sec_error = None
    if fam in SECTION_FAMILIES:
        units = section_units(rec[f"text_{pole}"])
        if units is None:
            sec_error = "text does not tokenize"
        else:
            pos = [len(tok.decode(prompt_ids[:j])) for j in range(n + 1)]; h = len(header)
            for s, e in units:
                s += h; e += h
                if s < pos[hdr] or e > pos[n - 1]:            # header, or the query's own (just opened) section / beyond the prompt
                    continue
                SEC |= {j for j in range(hdr, n - 1) if pos[j] < e and pos[j + 1] > s}
        SEC -= excluded
    union = A | B; final = sorted(union | SEC)
    assert not (set(final) & excluded), "cue index inside the evidence"
    assert all(hdr <= j < n - 1 for j in final), "evidence outside the scope"
    return dict(idx=final, toks=[a[j] for j in final], n_A=len(A), n_B=len(B - A), n_section=len(SEC - union), n_dropped=len(WS),
                a_error=a_error, sec_error=sec_error)


def load_family(fam, MP):
    pairs = {p["doc_id"]: p for p in json.load(open(ET.PAIRS / f"{fam}.json"))}
    if MP["cues"] is not None:
        cues = {c["doc_id"]: c["cues"] for c in json.load(open(MP["cues"] / f"{fam}.json"))}
        for r in pairs.values():
            if r["doc_id"] in cues:
                r["cues"] = cues[r["doc_id"]]
    prompts = json.load(open(MP["prompts"] / f"{fam}.json"))
    return pairs, prompts


def run_family(fam, model, ks, force):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    MP = model_paths(model); out = MP["evidence"] / f"{fam}.json"
    if out.exists() and not force:
        return fam, json.load(open(out)), True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MP["tokenizer"])
    pairs, prompts = load_family(fam, MP)
    recs = []
    for p in prompts:
        if p["k"] not in ks:
            continue
        r = evidence_positions(fam, pairs[p["doc_id"]], p["style"], p["prompt_ids"], p["k"], tok)
        recs.append(dict(doc_id=p["doc_id"], pole=p["style"], k=p["k"], prompt_len=len(p["prompt_ids"]), **r))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(out, "w"), ensure_ascii=False)
    return fam, recs, False


def stats(fam, recs, ks):
    """Per (family, k): n_prompts, mean / min / max evidence tokens, zero-evidence prompts, A failures."""
    rows = []
    for k in ks:
        R = [r for r in recs if r["k"] == k]
        if not R:
            continue
        ns = [len(r["idx"]) for r in R]
        rows.append(dict(family=fam, k=k, n_prompts=len(R), mean=statistics.mean(ns), min=min(ns), max=max(ns),
                         zero=sum(n == 0 for n in ns), a_fail=sum(bool(r.get("a_error")) for r in R),
                         sec_fail=sum(bool(r.get("sec_error")) for r in R)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code")
    ap.add_argument("--families", nargs="*", default=None, help="default: the model's results/code_pool.json pool")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stats-csv", default=None, help="also write the per-(family, k) stats table")
    args = ap.parse_args()
    ET.configure(args.model); MP = model_paths(args.model)
    fams = args.families or json.load(open(Path(MP["results"]) / "code_pool.json"))["pool"]
    jobs = [(f, args.model, set(args.ks), args.force) for f in fams]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = pool.starmap(run_family, jobs)
    else:
        results = [run_family(*j) for j in jobs]
    rows = []
    for fam, recs, skipped in results:
        for r in stats(fam, recs, args.ks):
            rows.append(r)
            print(f"{fam:18s} k={r['k']} prompts {r['n_prompts']:4d} | evidence tokens mean {r['mean']:6.1f} min {r['min']:3d} max {r['max']:4d}"
                  f" | zero {r['zero']} | A-failures {r['a_fail']}" + (f" | section-failures {r['sec_fail']}" if r["sec_fail"] else "")
                  + (" | (existing file)" if skipped else ""), flush=True)
    if args.stats_csv:
        import csv
        with open(args.stats_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
