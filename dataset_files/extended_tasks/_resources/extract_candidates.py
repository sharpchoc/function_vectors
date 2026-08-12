#!/usr/bin/env python
"""One-off: pull the ideation workflow's per-agent results out of its journal."""
import json
import sys
from pathlib import Path

journal = Path(sys.argv[1])
out = Path(__file__).resolve().parent / "candidates.json"
specs = []
for line in journal.read_text().splitlines():
    r = json.loads(line)
    val = r.get("result")
    if r.get("type") == "result" and isinstance(val, dict) and "specs" in val:
        for s in val["specs"]:
            s.setdefault("lane", s.get("category", "unknown"))
            specs.append(s)
json.dump({"specs": specs}, open(out, "w"), indent=1)
print("wrote", out, len(specs))
