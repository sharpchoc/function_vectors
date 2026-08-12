"""Generator: date_to_quarter
Given a date in YYYY-MM-DD format, output its calendar quarter (Q1..Q4).
Calendar quarters only (not fiscal-year quarters): Jan-Mar=Q1, Apr-Jun=Q2,
Jul-Sep=Q3, Oct-Dec=Q4. Years 1900-2099, day 1-28 (always valid).
Labels balanced 250/250/250/250.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "date_to_quarter.json"
N = 1000
PER_QUARTER = N // 4

random.seed(42)

pairs = []
seen_inputs = set()
for q in range(1, 5):
    months_in_q = [(q - 1) * 3 + 1, (q - 1) * 3 + 2, (q - 1) * 3 + 3]
    count = 0
    while count < PER_QUARTER:
        year = random.randint(1900, 2099)
        month = random.choice(months_in_q)
        day = random.randint(1, 28)
        inp = f"{year:04d}-{month:02d}-{day:02d}"
        if inp in seen_inputs:
            continue
        seen_inputs.add(inp)
        pairs.append({"input": inp, "output": f"Q{q}"})
        count += 1

random.shuffle(pairs)

# Asserts
import datetime
from collections import Counter

assert len(pairs) == N
assert len({p["input"] for p in pairs}) == N
for p in pairs:
    d = datetime.date.fromisoformat(p["input"])  # validates the date
    assert 1900 <= d.year <= 2099 and 1 <= d.day <= 28
    assert p["output"] == "Q" + str((d.month - 1) // 3 + 1)
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()
counts = Counter(p["output"] for p in pairs)
assert all(counts[f"Q{q}"] == PER_QUARTER for q in range(1, 5)), counts

OUT_PATH.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT_PATH} n={len(pairs)} label_counts={dict(counts)}")
