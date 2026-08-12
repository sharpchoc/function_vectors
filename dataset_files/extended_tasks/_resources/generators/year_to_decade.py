"""Generator: year_to_decade
Given a four-digit year, output its decade in 'YYY0s' form (1987 -> 1980s).
Domain: years 1000-2999 with last digit != 0 (1800 items). Sampling weighted
toward 1500-2099 where decade phrasing is most attested; 1000 unique years kept.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "year_to_decade.json"
N = 1000

random.seed(42)

domain = [y for y in range(1000, 3000) if y % 10 != 0]
assert len(domain) == 1800

# Weight years in 1500-2099 (540 years) 5x relative to the rest (1260 years),
# then sample 1000 unique years.
weights = [5.0 if 1500 <= y <= 2099 else 1.0 for y in domain]
chosen = set()
while len(chosen) < N:
    y = random.choices(domain, weights=weights, k=1)[0]
    chosen.add(y)

pairs = [{"input": str(y), "output": f"{y // 10 * 10}s"} for y in sorted(chosen)]
random.shuffle(pairs)

# Asserts
assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    y = int(p["input"])
    assert 1000 <= y <= 2999 and y % 10 != 0
    assert p["output"] == str(y // 10 * 10) + "s"
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()
n_core = sum(1 for p in pairs if 1500 <= int(p["input"]) <= 2099)

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)} in_1500_2099={n_core}")
