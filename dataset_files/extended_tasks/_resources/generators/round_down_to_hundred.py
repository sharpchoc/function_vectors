"""Generator: round_down_to_hundred
Rule: given an integer >= 101 that is not a multiple of 100, output it rounded
down to the nearest multiple of 100.
Domain: 101..99999 excluding multiples of 100 (those map to themselves).
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "round_down_to_hundred.json"

N = 1000

rng = random.Random(1234)

seen = set()
pairs = []
while len(pairs) < N:
    n = rng.randint(101, 99999)
    if n % 100 == 0:
        continue  # identity cases excluded per spec
    if n in seen:
        continue
    seen.add(n)
    pairs.append({"input": str(n), "output": str((n // 100) * 100)})

random.seed(42)
random.shuffle(pairs)

# --- self-checks ---
assert len(pairs) == N
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == N
for p in pairs:
    n = int(p["input"])
    assert 101 <= n <= 99999 and n % 100 != 0
    assert p["output"] == str((n // 100) * 100), p  # re-derive rule
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
