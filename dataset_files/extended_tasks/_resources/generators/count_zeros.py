"""Generator for count_zeros: given a digit string (3-6 digits, no leading zero),
output how many of its digits are '0'.

Balanced over the zero-count labels {0,1,2,3,4}: 200 examples each. Inputs are
constructed with exactly the target number of zeros (zeros placed among
non-leading positions, all other digits nonzero), so the label holds by
construction. A zero-count of 4 requires length >= 5, so lengths are drawn from
the feasible range per class. Deterministic (seeded).
"""
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "count_zeros.json")
N = 1000

random.seed(42)

# Balance labels 0..4: 200 each.
class_counts = {z: 200 for z in range(5)}
assert sum(class_counts.values()) == N

seen = set()
data = []
for zeros, target in class_counts.items():
    # need zeros <= length - 1 (first digit can't be 0), lengths limited to 3-6
    valid_lengths = [L for L in range(3, 7) if L - 1 >= zeros]
    count = 0
    while count < target:
        length = random.choice(valid_lengths)
        zero_positions = set(random.sample(range(1, length), zeros))
        digits = [random.choice("123456789")]
        for pos in range(1, length):
            digits.append("0" if pos in zero_positions else random.choice("123456789"))
        s = "".join(digits)
        if s in seen:
            continue
        seen.add(s)
        data.append({"input": s, "output": str(zeros)})
        count += 1

random.seed(42)
random.shuffle(data)

# Self-checks
assert len(data) == N
inputs = [d["input"] for d in data]
assert len(set(inputs)) == N
for d in data:
    s = d["input"]
    assert s == str(int(s)) and 3 <= len(s) <= 6
    assert d["output"] == str(s.count("0"))  # re-derive rule by recount
    assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=1)
print(f"wrote {len(data)} examples to {os.path.abspath(OUT_PATH)}")
