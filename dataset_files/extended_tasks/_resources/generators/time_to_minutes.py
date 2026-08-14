#!/usr/bin/env python
"""time_to_minutes: 24h clock time HH:MM -> total minutes since midnight (h*60+m).
Replaces time_extract_minutes (user-approved revision 2026-08-14). Domain: 1440 unique
times; 1000 sampled with seed 42. Deliberate arithmetic-difficulty probe."""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "time_to_minutes.json"
rng = random.Random(42)
times = [(h, m) for h in range(24) for m in range(60)]
picks = rng.sample(times, 1000)
data = [{"input": f"{h:02d}:{m:02d}", "output": str(h * 60 + m)} for h, m in picks]
rng.shuffle(data)
assert len(data) == 1000 and len({p["input"] for p in data}) == 1000
assert all(int(p["output"]) == int(p["input"][:2]) * 60 + int(p["input"][3:]) for p in data)
assert all(p["input"] != p["output"] for p in data)
json.dump(data, open(OUT, "w"), indent=1)
print(f"wrote {OUT} (n=1000)")
