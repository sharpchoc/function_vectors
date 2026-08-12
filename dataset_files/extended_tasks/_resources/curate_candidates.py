#!/usr/bin/env python
"""Compact listing + structural checks of ideation candidates for curation."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parent
cand_path = RES / "candidates.json"
existing = {p.stem for p in (RES.parent.parent / "abstractive").glob("*.json")}

data = json.load(open(cand_path))
specs = data["specs"]
print(f"{len(specs)} candidates; existing tasks: {len(existing)}")

names = Counter(s["name"] for s in specs)
dups = [n for n, c in names.items() if c > 1]
coll = [s["name"] for s in specs if s["name"] in existing]
print("intra-candidate name dups:", dups or "none")
print("collisions with existing:", coll or "none")

fams = Counter(s.get("family") for s in specs if s.get("family"))
over = {f: c for f, c in fams.items() if c > 3}
print("families >3 members:", over or "none")

print("\nidx | lane | name | method | conf | domain | family | rule")
for i, s in enumerate(specs):
    rule = re.sub(r"\s+", " ", s["rule"])[:95]
    print(f"{i:3d} | {s['lane'][:12]:12s} | {s['name'][:32]:32s} | {s['generation_method'][:9]:9s} | "
          f"{s['gptj_confidence'][:4]:4s} | {s['domain_size_estimate']:6d} | {str(s.get('family'))[:18]:18s} | {rule}")
