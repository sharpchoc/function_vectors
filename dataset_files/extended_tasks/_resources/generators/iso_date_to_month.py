"""Generator for iso_date_to_month: YYYY-MM-DD -> English month name.

Rule: years 1900..2099, months balanced (83 or 84 each), days 1..28
(always valid; sidesteps month lengths and leap years).
Output via hardcoded 12-entry table; verified against datetime.strftime('%B').
"""
import json
import random
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "iso_date_to_month.json"
N = 1000

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def main() -> None:
    rng = random.Random(4043)
    # Balanced month counts: 1000 = 12*83 + 4
    per_month = [83] * 12
    for m in rng.sample(range(12), 4):
        per_month[m] += 1

    seen = set()
    data = []
    for m in range(1, 13):
        need = per_month[m - 1]
        while need:
            y = rng.randint(1900, 2099)
            d = rng.randint(1, 28)
            inp = f"{y:04d}-{m:02d}-{d:02d}"
            if inp in seen:
                continue
            seen.add(inp)
            data.append({"input": inp, "output": MONTHS[m - 1]})
            need -= 1

    random.seed(42)
    random.shuffle(data)

    # --- self-checks ---
    assert len(data) == N
    assert len({d["input"] for d in data}) == N
    counts = {}
    for d in data:
        dt = datetime.strptime(d["input"], "%Y-%m-%d")
        assert 1900 <= dt.year <= 2099 and 1 <= dt.day <= 28
        assert d["output"] == dt.strftime("%B"), (d, dt.strftime("%B"))
        assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()
        counts[d["output"]] = counts.get(d["output"], 0) + 1
    assert set(counts) == set(MONTHS)
    assert all(83 <= c <= 84 for c in counts.values()), counts

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {len(data)} to {OUT}; month counts {sorted(counts.values())}")


if __name__ == "__main__":
    main()
