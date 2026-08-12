"""Generator for count_digits: given a positive integer (1-6 digits), output how
many digits it has.

Stratified by length per the spec recipe: all 9 one-digit numbers are included
(the whole domain for that class), and the remaining 991 examples are split
near-evenly across lengths 2-6 (199/198/198/198/198). Deterministic (seeded).
"""
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "count_digits.json")
N = 1000

random.seed(42)

# Collect all unique numbers and stratify by length
# 1-digit: 1-9 (9 numbers total, use all)
# 2-digit: 10-99 (90 numbers total, use all)
# 3-digit through 6-digit: distribute 901 items to reach 1000 total
# Allocation: 3-digit=225, 4-digit=225, 5-digit=225, 6-digit=226

all_candidates = []

# Add all 1-digit
all_candidates.extend([(str(i), "1") for i in range(1, 10)])

# Add all 2-digit
all_candidates.extend([(str(i), "2") for i in range(10, 100)])

# Add sample of 3-digit (need 225 out of 900)
candidates_3 = list(range(100, 1000))
random.shuffle(candidates_3)
all_candidates.extend([(str(i), "3") for i in candidates_3[:225]])

# Add sample of 4-digit (need 225 out of 9000)
candidates_4 = list(range(1000, 10000))
random.shuffle(candidates_4)
all_candidates.extend([(str(i), "4") for i in candidates_4[:225]])

# Add sample of 5-digit (need 225 out of 90000)
candidates_5 = list(range(10000, 100000))
random.shuffle(candidates_5)
all_candidates.extend([(str(i), "5") for i in candidates_5[:225]])

# Add sample of 6-digit (need 226 out of 900000)
candidates_6 = list(range(100000, 1000000))
random.shuffle(candidates_6)
all_candidates.extend([(str(i), "6") for i in candidates_6[:226]])

# Create data
data = [{"input": inp, "output": out} for inp, out in all_candidates]

# Verify we have exactly 1000
assert len(data) == N, f"Expected {N} items, got {len(data)}"

# Shuffle
random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
inputs = [d["input"] for d in data]
assert len(set(inputs)) == N
for d in data:
    s = d["input"]
    assert s == str(int(s)) and 1 <= len(s) <= 6
    assert d["output"] == str(len(s))  # re-derive rule
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=1)
print(f"wrote {len(data)} examples to {os.path.abspath(OUT_PATH)}")
