"""Generator: larger_than_100
Rule: given an integer other than 100, output 'yes' if it is greater than 100,
otherwise 'no'.
Domain: 1..99 union 101..99999 (100 itself excluded).

Note on class balance: with globally unique inputs, the 'no' class is capped at
the 99 integers 1..99. We include ALL 99 of them and fill the remaining 901
examples from the 'yes' side (101..99999). A 50/50 balance at n=1000 is
mathematically impossible for this rule with unique inputs.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "larger_than_100.json"

N = 1000

rng = random.Random(1234)

pairs = [{"input": str(n), "output": "no"} for n in range(1, 100)]  # all 99 'no'

seen = set()
while len(pairs) < N:
    n = rng.randint(101, 99999)
    if n in seen:
        continue
    seen.add(n)
    pairs.append({"input": str(n), "output": "yes"})

random.seed(42)
random.shuffle(pairs)

# --- self-checks ---
assert len(pairs) == N
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == N
n_no = sum(1 for p in pairs if p["output"] == "no")
assert n_no == 99, n_no
for p in pairs:
    n = int(p["input"])
    assert n != 100 and 1 <= n <= 99999
    assert p["output"] == ("yes" if n > 100 else "no"), p  # re-derive rule
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} no={n_no} yes={N - n_no}")
