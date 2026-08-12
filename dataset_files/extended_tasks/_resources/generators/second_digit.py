"""Generator for second_digit: given a positive integer (2-6 digits), output its
second digit from the left.

Balanced over the 10 possible output classes (second digit 0-9), lengths sampled
uniformly from 2-6. Deterministic (seeded).
"""
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "second_digit.json")
N = 1000

random.seed(42)

# Balance output classes: 10 classes x 100 = 1000.
class_counts = {str(d): 100 for d in range(10)}
assert sum(class_counts.values()) == N

seen = set()
data = []
for second, target in class_counts.items():
    count = 0
    while count < target:
        length = random.randint(2, 6)
        first = random.choice("123456789")
        rest = "".join(random.choice("0123456789") for _ in range(length - 2))
        s = first + second + rest
        if s in seen:
            continue
        seen.add(s)
        data.append({"input": s, "output": second})
        count += 1

random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
inputs = [d["input"] for d in data]
assert len(set(inputs)) == N
for d in data:
    s = d["input"]
    assert s == str(int(s)) and 2 <= len(s) <= 6
    assert d["output"] == s[1]  # re-derive rule
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=1)
print(f"wrote {len(data)} examples to {os.path.abspath(OUT_PATH)}")
