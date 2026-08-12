#!/usr/bin/env python3
"""Generator for prev_capital_letter: word -> alphabet predecessor of its first
letter, uppercased.

Recipe: common_words.txt (frequency-sorted descending), ASCII alphabetic lowercase
words; EXCLUDE words starting with 'a' (no predecessor).
output = chr(ord(word[0]) - 1).upper(). Keep the 1000 most frequent. Dedup.
Self-check: ord(output.lower()) + 1 == ord(input[0]).
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
OUT = RES.parent / "prev_capital_letter.json"

words = [w.strip() for w in (RES / "common_words.txt").read_text().splitlines()]

seen = set()
pairs = []
for w in words:  # frequency order: most common first
    if len(w) < 2:
        continue
    if not (w.isascii() and w.isalpha() and w.islower()):
        continue
    if w[0] == "a":  # 'a' has no alphabet predecessor
        continue
    if w in seen:
        continue
    seen.add(w)
    pairs.append({"input": w, "output": chr(ord(w[0]) - 1).upper()})
    if len(pairs) == 1000:
        break

random.seed(42)
random.shuffle(pairs)

# Asserts
assert len(pairs) == 1000, len(pairs)
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == 1000
for p in pairs:
    assert ord(p["output"].lower()) + 1 == ord(p["input"][0])
    assert p["output"].isupper() and len(p["output"]) == 1
    assert p["input"] == p["input"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
