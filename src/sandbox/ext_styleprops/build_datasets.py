#!/usr/bin/env python
"""Build the minimal-pair style-property datasets from the base corpus (Stage A1+A2).

Per property: render each eligible base doc under both polarities (nat / alt), locate
every opportunity site's decision token and evidence tokens in BOTH twins via the fast
tokenizer's offset mapping (tokenizer-verified positions — DECISIONS 2026-07-13), and
keep only sites whose decision token is IDENTICAL across the twin pair (the identity-
matched "point of no return"). Sites are annotated with k (# prior manifestations) and
token distance since the last one.

Outputs
  dataset_files/style_properties/props/<prop>.json      the dataset
  results/style_properties/dataset_audit.csv            the A2 audit table
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.properties import PROPS, render

BASE_PATH = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus.json"
OUT_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"

# minimum opportunity sites for a doc to be eligible (sparse properties get lower bars,
# their density comes from the seeded batches), and per-property doc cap.
MIN_SITES = {
    "sentence_caps": 6, "all_caps": 6, "double_space": 6, "contractions": 6,
    "ampersand": 6, "us_uk": 6, "ise_ize": 4, "brit_t_past": 3, "whilst": 3,
    "oxford_comma": 4, "curly_quotes": 6, "em_dash": 4, "ellipsis": 3,
    "quote_punct": 4, "num_words": 6, "percent_sign": 4, "ordinal_words": 3,
}
MAX_DOCS = 200
MAX_SITES = 8          # sites scored per doc (all opps are still rendered)
MIN_PREFIX_TOKS = 12   # a decision point needs some context before it


def token_spans(tok, text):
    enc = tok(text, return_offsets_mapping=True)
    return enc.input_ids, enc.offset_mapping


def evid(offs, span):
    """evidence tokens = tokens overlapping the rendered opp span."""
    return [j for j, (s, e) in enumerate(offs) if s < span[1] and e > span[0]]


def locate_common(offs_n, span_n, offs_a, span_a, div):
    """Decision token pair: the latest token end that is a token boundary in BOTH twins,
    at or before the divergence char, in shared-prefix coordinates (delta = char offset
    relative to the opp span start; chars at the same delta < div are identical across
    twins by construction). Handles BPE merges that differ between the twins, e.g. ',\"'
    merging in the straight-quote twin but not the curly one, or the standalone-space
    token in the double-space twin."""
    ends_n = {e - span_n[0]: j for j, (s, e) in enumerate(offs_n) if 0 < e <= span_n[0] + div}
    ends_a = {e - span_a[0]: j for j, (s, e) in enumerate(offs_a) if 0 < e <= span_a[0] + div}
    common = set(ends_n) & set(ends_a)
    if not common:
        return None, None
    delta = max(common)
    return ends_n[delta], ends_a[delta]


def build_property(prop, docs, tok):
    out_docs, audit = [], {
        "n_docs_eligible": 0, "n_docs_used": 0, "n_sites": 0,
        "dropped_dec_mismatch": 0, "dropped_short_prefix": 0,
        "tok_delta": [], "opps_redetect_mismatch": 0,
    }
    for d in docs:
        opps = prop.find_opps(d["text"])
        if len(opps) < MIN_SITES[prop.name]:
            continue
        audit["n_docs_eligible"] += 1
        if audit["n_docs_used"] >= MAX_DOCS:
            continue
        nat_text, nat_spans = render(d["text"], opps, "nat")
        alt_text, alt_spans = render(d["text"], opps, "alt")
        if len(prop.find_opps(nat_text)) != len(opps):
            audit["opps_redetect_mismatch"] += 1
        ids_n, off_n = token_spans(tok, nat_text)
        ids_a, off_a = token_spans(tok, alt_text)
        audit["tok_delta"].append(abs(len(ids_n) - len(ids_a)))

        sites, last_ev = [], {"nat": None, "alt": None}
        for i, o in enumerate(opps):
            sn, sa = nat_spans[i], alt_spans[i]
            dec_n, dec_a = locate_common(off_n, sn, off_a, sa, o.div)
            ev_n, ev_a = evid(off_n, sn), evid(off_a, sa)
            keep = True
            if dec_n is None or dec_a is None or ids_n[dec_n] != ids_a[dec_a]:
                audit["dropped_dec_mismatch"] += 1
                keep = False
            elif min(dec_n, dec_a) < MIN_PREFIX_TOKS:
                audit["dropped_short_prefix"] += 1
                keep = False
            if keep and len(sites) < MAX_SITES:
                cs_n, cs_a = off_n[dec_n][1], off_a[dec_a][1]
                exp = {
                    "nat_ctx": {"nat": nat_text[cs_n:sn[1]],
                                "alt": nat_text[cs_n:sn[0]] + o.alt},
                    "alt_ctx": {"nat": alt_text[cs_a:sa[0]] + o.nat,
                                "alt": alt_text[cs_a:sa[1]]},
                }
                exp_toks = max(len(tok(v).input_ids)
                               for side in exp.values() for v in side.values())
                sites.append({
                    "k": i,
                    "dec_idx": {"nat": dec_n, "alt": dec_a},
                    "dec_tok_id": ids_n[dec_n],
                    "evid_idx": {"nat": [ev_n[0], ev_n[-1]], "alt": [ev_a[0], ev_a[-1]]},
                    "dist": {"nat": dec_n - last_ev["nat"] if last_ev["nat"] is not None else -1,
                             "alt": dec_a - last_ev["alt"] if last_ev["alt"] is not None else -1},
                    "exp": exp,
                    "max_new": min(exp_toks + 4, prop.max_new_cap),
                })
                audit["n_sites"] += 1
            if ev_n:
                last_ev["nat"] = ev_n[-1]
            if ev_a:
                last_ev["alt"] = ev_a[-1]
        if sites:
            audit["n_docs_used"] += 1
            out_docs.append({"doc_id": d["doc_id"], "text_nat": nat_text,
                             "text_alt": alt_text, "n_opps": len(opps), "sites": sites})
    return out_docs, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--props", nargs="*", default=sorted(PROPS))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    docs = json.load(open(BASE_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_PROPERTIES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.props:
        prop = PROPS[name]
        out_docs, a = build_property(prop, docs, tok)
        json.dump({"property": name, "nat_label": prop.nat_label,
                   "alt_label": prop.alt_label, "min_sites": MIN_SITES[name],
                   "docs": out_docs}, open(OUT_DIR / f"{name}.json", "w"))
        ks = [s["k"] for d in out_docs for s in d["sites"]]
        row = {
            "property": name, "docs_eligible": a["n_docs_eligible"],
            "docs_used": a["n_docs_used"], "sites": a["n_sites"],
            "dropped_dec_mismatch": a["dropped_dec_mismatch"],
            "dropped_short_prefix": a["dropped_short_prefix"],
            "redetect_mismatch_docs": a["opps_redetect_mismatch"],
            "mean_tok_delta": round(float(np.mean(a["tok_delta"])), 2) if a["tok_delta"] else 0,
            "median_k_max": int(np.median([max((s["k"] for s in d["sites"]), default=0)
                                           for d in out_docs])) if out_docs else 0,
            "sites_k_ge4": sum(k >= 4 for k in ks),
        }
        rows.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    with open(STYLE_PROPERTIES_DIR / "dataset_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"audit -> {STYLE_PROPERTIES_DIR / 'dataset_audit.csv'}", flush=True)


if __name__ == "__main__":
    main()
