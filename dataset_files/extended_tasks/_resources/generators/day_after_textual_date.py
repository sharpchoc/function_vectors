"""Generator: day_after_textual_date
Given a date like 'March 5, 1984' with day 1-27, output the month and next day: 'March 6'.
Day capped at 27 so the increment never crosses a month boundary; year 1900-2099.
Year is deliberately dropped from the output.
"""
import json
import random
import datetime
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "day_after_textual_date.json"
N = 1000

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

random.seed(42)

pairs = []
seen_inputs = set()
while len(pairs) < N:
    month_i = random.randint(1, 12)
    day = random.randint(1, 27)
    year = random.randint(1900, 2099)
    month = MONTHS[month_i - 1]
    inp = f"{month} {day}, {year}"
    if inp in seen_inputs:
        continue
    seen_inputs.add(inp)
    pairs.append({"input": inp, "output": f"{month} {day + 1}"})

random.shuffle(pairs)

# Asserts
assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    month_name, day_s, year_s = p["input"].replace(",", "").split()
    d = datetime.date(int(year_s), MONTHS.index(month_name) + 1, int(day_s))
    assert 1 <= d.day <= 27 and 1900 <= d.year <= 2099
    nxt = d + datetime.timedelta(days=1)
    assert nxt.month == d.month  # never crosses a month boundary
    assert p["output"] == f"{MONTHS[nxt.month - 1]} {nxt.day}"
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)}")
