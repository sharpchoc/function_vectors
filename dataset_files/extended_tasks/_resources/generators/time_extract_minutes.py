"""Generator: time_extract_minutes
Given a 24-hour clock time HH:MM with minutes 10-59, output the minutes as a number.
Domain: 24 hours x minutes 10-59 = exactly 1200; keep 1000 (minutes 00-09 excluded
so the answer never needs a leading-zero-strip decision).
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "time_extract_minutes.json"
N = 1000

random.seed(42)

domain = [(h, m) for h in range(24) for m in range(10, 60)]
assert len(domain) == 1200
chosen = random.sample(domain, N)

pairs = [{"input": f"{h:02d}:{m:02d}", "output": str(m)} for h, m in chosen]
random.shuffle(pairs)

# Asserts
assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    h, m = map(int, p["input"].split(":"))
    assert 0 <= h <= 23 and 10 <= m <= 59
    assert p["output"] == p["input"].split(":")[1] == str(m)
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)}")
