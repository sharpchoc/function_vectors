#!/usr/bin/env python
"""Finalise the multilingual pair files after ml_verify: keep pass=True records only, cap at --n (200, protocol parity with the
English families), sorted by doc_id; the full verified set is kept as pairs/<family>.unfiltered.json. Prints the final counts."""
import argparse, json, sys
from pathlib import Path
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.ml_families import ML_FAMILY
PAIRS = STYLE_TRANSLATION_DATA / "pairs"
ap = argparse.ArgumentParser(); ap.add_argument("--families", nargs="+", required=True); ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()
for name in args.families:
    path = PAIRS / f"{name}.json"; recs = json.load(open(path))
    unfiltered = PAIRS / f"{name}.unfiltered.json"
    if not unfiltered.exists() or len(recs) > len(json.load(open(unfiltered))):
        json.dump(recs, open(unfiltered, "w"), ensure_ascii=False, indent=0)
    keep = sorted([r for r in recs if r.get("pass") is True], key=lambda r: r["doc_id"])[: args.n]
    json.dump(keep, open(path, "w"), ensure_ascii=False, indent=0)
    print(f"{name:14s} candidates {len(recs):3d} | passing {sum(r.get('pass') is True for r in recs):3d} | final {len(keep):3d}{'  (SHORT)' if len(keep) < args.n else ''}")
