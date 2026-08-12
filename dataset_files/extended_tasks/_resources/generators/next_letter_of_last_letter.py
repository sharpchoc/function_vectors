#!/usr/bin/env python3
"""Generator for next_letter_of_last_letter: word -> alphabet successor of its
LAST letter, uppercased.

Recipe: common_words.txt (frequency-sorted descending), ASCII alphabetic lowercase
words; EXCLUDE words ending in 'z' (no successor within a-z).
output = chr(ord(word[-1]) + 1).upper(). Keep the 1000 most frequent. Dedup.
Self-check: ord(output.lower()) - 1 == ord(input[-1]).
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
OUT = RES.parent / "next_letter_of_last_letter.json"

words = [w.strip() for w in (RES / "common_words.txt").read_text().splitlines()]

seen = set()
pairs = []
for w in words:  # frequency order: most common first
    if len(w) < 2:
        continue
    if not (w.isascii() and w.isalpha() and w.islower()):
        continue
    if w[-1] == "z":  # 'z' has no alphabet successor
        continue
    if w in seen:
        continue
    seen.add(w)
    pairs.append({"input": w, "output": chr(ord(w[-1]) + 1).upper()})
    if len(pairs) == 1000:
        break

random.seed(42)
random.shuffle(pairs)

# Asserts
assert len(pairs) == 1000, len(pairs)
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == 1000
for p in pairs:
    assert ord(p["output"].lower()) - 1 == ord(p["input"][-1])
    assert p["output"].isupper() and len(p["output"]) == 1
    assert p["input"] == p["input"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
