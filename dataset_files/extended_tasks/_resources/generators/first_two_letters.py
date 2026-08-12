#!/usr/bin/env python
"""Generator for first_two_letters: word -> its first two letters, lowercase.

Spec: common_words.txt; alphabetic (a-z only), len>=4. output = word[:2].
Prefer most frequent words (file is frequency-sorted descending). 1000 items,
shuffled with random.seed(42).
"""
import json
import random
import re
from pathlib import Path

RES = Path(__file__).resolve().parent.parent
OUT = RES.parent / "first_two_letters.json"
N = 1000

words = []
seen = set()
for line in (RES / "common_words.txt").read_text().splitlines():
    w = line.strip()
    if not re.fullmatch(r"[a-z]{4,}", w):
        continue
    if w in seen:
        continue
    seen.add(w)
    words.append(w)

words = words[:N]
data = [{"input": w, "output": w[:2]} for w in words]

random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
assert len({d["input"] for d in data}) == N
for d in data:
    assert len(d["output"]) == 2
    assert d["input"].startswith(d["output"])
    assert d["output"] == d["input"][:2]
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

OUT.write_text(json.dumps(data, indent=1))
print(f"wrote {OUT} ({len(data)} items)")
