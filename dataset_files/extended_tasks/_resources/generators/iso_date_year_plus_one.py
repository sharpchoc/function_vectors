"""Generator: iso_date_year_plus_one
Given a date in YYYY-MM-DD format, output the year one greater than the date's year.
Years 1800-2099, month 1-12, day 1-28. Century-boundary / carry cases (year ending
in 9, e.g. 1899 -> 1900, 1999 -> 2000... any trailing-9 carry) capped at ~2% of data.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "iso_date_year_plus_one.json"
N = 1000

random.seed(42)

# A carry-on-increment happens whenever the year ends in 9 (e.g. 1899->1900, 2019->2020).
# Keep those to at most ~2% of the dataset.
MAX_CARRY = 20

def is_carry(year: int) -> bool:
    return year % 10 == 9

pairs = []
seen_inputs = set()
n_carry = 0
while len(pairs) < N:
    year = random.randint(1800, 2099)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    inp = f"{year:04d}-{month:02d}-{day:02d}"
    if inp in seen_inputs:
        continue
    if is_carry(year):
        if n_carry >= MAX_CARRY:
            continue
        n_carry += 1
    seen_inputs.add(inp)
    pairs.append({"input": inp, "output": str(year + 1)})

random.shuffle(pairs)

# Asserts
assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
import datetime
for p in pairs:
    d = datetime.date.fromisoformat(p["input"])  # validates the date itself
    assert 1800 <= d.year <= 2099 and 1 <= d.day <= 28
    assert p["output"] == str(d.year + 1)
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()
assert sum(1 for p in pairs if is_carry(int(p["input"][:4]))) <= MAX_CARRY

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)} carries={n_carry}")
