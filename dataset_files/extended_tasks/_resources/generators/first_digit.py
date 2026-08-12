"""Generator for first_digit: given a positive integer (2-6 digits), output its leftmost digit.

Balanced over the 9 possible output classes (first digit 1-9), lengths sampled
uniformly from 2-6. Deterministic (seeded).
"""
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "first_digit.json")
N = 1000

random.seed(42)

# Balance output classes: 9 classes -> 111 each, +1 extra for '1' = 1000.
class_counts = {str(d): 111 for d in range(1, 10)}
class_counts["1"] += 1
assert sum(class_counts.values()) == N

seen = set()
data = []
for first, target in class_counts.items():
    count = 0
    while count < target:
        length = random.randint(2, 6)
        rest = "".join(random.choice("0123456789") for _ in range(length - 1))
        s = first + rest
        if s in seen:
            continue
        seen.add(s)
        data.append({"input": s, "output": first})
        count += 1

random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
inputs = [d["input"] for d in data]
assert len(set(inputs)) == N
for d in data:
    s = d["input"]
    assert s == str(int(s)) and 2 <= len(s) <= 6  # valid integer, no leading zero
    assert d["output"] == s[0]  # re-derive rule
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=1)
print(f"wrote {len(data)} examples to {os.path.abspath(OUT_PATH)}")
