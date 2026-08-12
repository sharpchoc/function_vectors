"""Generator: largest_digit
Rule: given a positive integer written in digits, output its largest digit.
Domain: random integers with 2-5 digits, all digits distinct (unique argmax by construction).
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "largest_digit.json"

N = 1000

rng = random.Random(1234)

# Stratify by length so shorter numbers are represented.
# Pool sizes (all-distinct digits, no leading zero): len2=81, len3=648, len4=4536, len5=27216.
quota = {2: 70, 3: 280, 4: 325, 5: 325}

seen = set()
pairs = []
for length, k in quota.items():
    lo, hi = 10 ** (length - 1), 10 ** length - 1
    count = 0
    while count < k:
        n = rng.randint(lo, hi)
        s = str(n)
        if len(set(s)) != len(s):
            continue  # digits must be distinct
        if s in seen:
            continue
        seen.add(s)
        pairs.append({"input": s, "output": max(s)})
        count += 1

random.seed(42)
random.shuffle(pairs)

# --- self-checks ---
assert len(pairs) == N, len(pairs)
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == N
for p in pairs:
    s = p["input"]
    assert 2 <= len(s) <= 5 and s[0] != "0"
    assert len(set(s)) == len(s), s
    assert p["output"] == max(s), p  # re-derive rule
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
