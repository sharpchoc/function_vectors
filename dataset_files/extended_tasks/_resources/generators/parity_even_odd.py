"""Generator: parity_even_odd
Rule: given an integer, output 'even' if it is even, 'odd' if it is odd.
Domain: 1..99999, stratified to exactly 500 even / 500 odd.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "parity_even_odd.json"

N = 1000

rng = random.Random(1234)

seen = set()
pairs = []
counts = {"even": 0, "odd": 0}
while len(pairs) < N:
    n = rng.randint(1, 99999)
    label = "even" if n % 2 == 0 else "odd"
    if counts[label] >= N // 2:
        continue  # exact 50/50 stratification
    if n in seen:
        continue
    seen.add(n)
    counts[label] += 1
    pairs.append({"input": str(n), "output": label})

random.seed(42)
random.shuffle(pairs)

# --- self-checks ---
assert len(pairs) == N
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == N
n_even = sum(1 for p in pairs if p["output"] == "even")
assert n_even == N // 2, n_even  # exact class balance
for p in pairs:
    n = int(p["input"])
    assert 1 <= n <= 99999
    assert p["output"] == ("even" if n % 2 == 0 else "odd"), p  # re-derive rule
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} even={n_even}")
