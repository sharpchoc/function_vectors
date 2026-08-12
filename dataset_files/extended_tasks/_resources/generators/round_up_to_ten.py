"""Generator: round_up_to_ten
Rule: given an integer that is not a multiple of 10, output it rounded up to
the next multiple of 10.
Domain: 11..9999 excluding multiples of 10 (ambiguous between 'stay' and 'next').
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "round_up_to_ten.json"

N = 1000

rng = random.Random(1234)

seen = set()
pairs = []
while len(pairs) < N:
    n = rng.randint(11, 9999)
    if n % 10 == 0:
        continue  # excluded per spec
    if n in seen:
        continue
    seen.add(n)
    pairs.append({"input": str(n), "output": str(((n // 10) + 1) * 10)})

random.seed(42)
random.shuffle(pairs)

# --- self-checks ---
assert len(pairs) == N
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == N
for p in pairs:
    n = int(p["input"])
    assert 11 <= n <= 9999 and n % 10 != 0
    assert p["output"] == str(((n // 10) + 1) * 10), p  # re-derive rule
    assert int(p["output"]) > n and int(p["output"]) - n < 10
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
