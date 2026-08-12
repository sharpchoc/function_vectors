"""Generator: days_in_month
Given 'MonthName Year' (month is never February), output the number of days
in that month: '30' or '31'. February is excluded so leap years never matter.
Labels balanced by oversampling 30-day months to 45% (450 x '30', 550 x '31').
Years 1800-2099.
"""
import calendar
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "days_in_month.json"
N = 1000
N_30 = 450  # ~45% for the rarer class (4 of 11 months have 30 days)
N_31 = N - N_30

random.seed(42)

MONTHS_30 = ["April", "June", "September", "November"]
MONTHS_31 = ["January", "March", "May", "July", "August", "October", "December"]
YEARS = list(range(1800, 2100))

# Per-month quotas: 450 = 2x113 + 2x112 ; 550 = 4x79 + 3x78.
quota = {}
for i, m in enumerate(MONTHS_30):
    quota[m] = 113 if i < 2 else 112
for i, m in enumerate(MONTHS_31):
    quota[m] = 79 if i < 4 else 78
assert sum(quota[m] for m in MONTHS_30) == N_30
assert sum(quota[m] for m in MONTHS_31) == N_31

pairs = []
for month in MONTHS_30 + MONTHS_31:
    out = "30" if month in MONTHS_30 else "31"
    for y in random.sample(YEARS, quota[month]):
        pairs.append({"input": f"{month} {y}", "output": out})

random.shuffle(pairs)

# Asserts
from collections import Counter

assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    month, year_s = p["input"].rsplit(" ", 1)
    year = int(year_s)
    assert 1800 <= year <= 2099
    assert month != "February"
    idx = list(calendar.month_name).index(month)  # validates the name
    assert p["output"] == str(calendar.monthrange(year, idx)[1])
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()
counts = Counter(p["output"] for p in pairs)
assert counts["30"] == N_30 and counts["31"] == N_31, counts

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)} label_counts={dict(counts)}")
