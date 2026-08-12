"""Generator: hour_after_time
Given a 24-hour clock time HH:MM with hour 00-22, output the time one hour later
(minutes unchanged), zero-padded. Domain: 23 hours x 60 minutes = 1380; keep 1000.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "hour_after_time.json"
N = 1000

random.seed(42)

domain = [(h, m) for h in range(23) for m in range(60)]
assert len(domain) == 1380
chosen = random.sample(domain, N)

pairs = [{"input": f"{h:02d}:{m:02d}", "output": f"{h + 1:02d}:{m:02d}"}
         for h, m in chosen]
random.shuffle(pairs)

# Asserts
assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    h, m = map(int, p["input"].split(":"))
    assert 0 <= h <= 22 and 0 <= m <= 59
    assert p["output"] == f"{h + 1:02d}:{m:02d}"
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)}")
