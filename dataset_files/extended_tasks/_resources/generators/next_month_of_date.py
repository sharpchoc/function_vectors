"""Generator: next_month_of_date
Given 'MonthName Year', output the name of the following month (December -> January).
Domain: 12 months x years 1800-2099. Output has no year, so no year+1 logic needed.
Months balanced (84 or 83 examples each, total 1000).
"""
import calendar
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "next_month_of_date.json"
N = 1000

random.seed(42)

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
YEARS = list(range(1800, 2100))

# Balance months: 1000 = 4 months x 84 + 8 months x 83.
month_quota = {m: (84 if i < 4 else 83) for i, m in enumerate(MONTHS)}
assert sum(month_quota.values()) == N

pairs = []
for mi, month in enumerate(MONTHS):
    years = random.sample(YEARS, month_quota[month])
    nxt = MONTHS[(mi + 1) % 12]
    for y in years:
        pairs.append({"input": f"{month} {y}", "output": nxt})

random.shuffle(pairs)

# Asserts
assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    month, year_s = p["input"].rsplit(" ", 1)
    year = int(year_s)
    assert 1800 <= year <= 2099
    idx = list(calendar.month_name).index(month)  # 1..12, validates the name
    assert p["output"] == calendar.month_name[idx % 12 + 1]
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)}")
