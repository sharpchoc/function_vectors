#!/usr/bin/env python
"""Generator for lowercase_word: fully uppercased word -> same word lowercased.

Spec: common_words.txt; alphabetic, len>=3, a-z only. input = word.upper(),
output = word. Prefer most frequent words (file is frequency-sorted
descending). 1000 items, shuffled with random.seed(42).
"""
import json
import random
import re
from pathlib import Path

RES = Path(__file__).resolve().parent.parent
OUT = RES.parent / "lowercase_word.json"
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
data = [{"input": w.upper(), "output": w} for w in words]

random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
assert len({d["input"] for d in data}) == N
for d in data:
    assert d["input"].lower() == d["output"]
    assert d["input"] == d["output"].upper()
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

OUT.write_text(json.dumps(data, indent=1))
print(f"wrote {OUT} ({len(data)} items)")
