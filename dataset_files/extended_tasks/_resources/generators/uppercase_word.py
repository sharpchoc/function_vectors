#!/usr/bin/env python
"""Generator for uppercase_word: lowercase word -> same word fully uppercased.

Spec: common_words.txt; alphabetic, len>=3, all-lowercase (a-z only). Prefer
most frequent words (file is frequency-sorted descending). 1000 items,
shuffled with random.seed(42).
"""
import json
import random
import re
from pathlib import Path

RES = Path(__file__).resolve().parent.parent
OUT = RES.parent / "uppercase_word.json"
N = 1000

words = []
seen = set()
for line in (RES / "common_words.txt").read_text().splitlines():
    w = line.strip()
    if not re.fullmatch(r"[a-z]{3,}", w):
        continue
    if w in seen:
        continue
    seen.add(w)
    words.append(w)

words = words[:N]
data = [{"input": w, "output": w.upper()} for w in words]

random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
assert len({d["input"] for d in data}) == N
for d in data:
    assert d["output"].lower() == d["input"]
    assert d["output"] == d["input"].upper()
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

OUT.write_text(json.dumps(data, indent=1))
print(f"wrote {OUT} ({len(data)} items)")
