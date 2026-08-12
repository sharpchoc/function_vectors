"""Generator for iso_date_day_of_month: YYYY-MM-DD -> day of month, no leading zero.

Rule: years 1900..2099; days 1..28 for any month, 29/30 only in months with
>= 30 days (Feb excluded for 29 to sidestep leap years). Output = str(int(dd)).
Days balanced ~uniformly across 1..30 (33-34 each) so single-digit days
(leading-zero strip) are well represented.
"""
import json
import random
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "iso_date_day_of_month.json"
N = 1000

LONG_MONTHS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # >= 30 days (all but Feb)


def main() -> None:
    rng = random.Random(5044)
    # Balanced day counts: 1000 = 30*33 + 10
    per_day = {d: 33 for d in range(1, 31)}
    for d in rng.sample(range(1, 31), 10):
        per_day[d] += 1

    seen = set()
    data = []
    for day in range(1, 31):
        need = per_day[day]
        months = list(range(1, 13)) if day <= 28 else LONG_MONTHS
        while need:
            y = rng.randint(1900, 2099)
            m = rng.choice(months)
            inp = f"{y:04d}-{m:02d}-{day:02d}"
            if inp in seen:
                continue
            seen.add(inp)
            data.append({"input": inp, "output": str(day)})
            need -= 1

    random.seed(42)
    random.shuffle(data)

    # --- self-checks ---
    assert len(data) == N
    assert len({d["input"] for d in data}) == N
    counts = {}
    for d in data:
        dt = datetime.strptime(d["input"], "%Y-%m-%d")  # validates the date
        assert 1900 <= dt.year <= 2099
        assert d["output"] == str(int(d["input"].split("-")[2]))
        assert int(d["output"]) == dt.day
        assert not d["output"].startswith("0")
        assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()
        counts[dt.day] = counts.get(dt.day, 0) + 1
    assert set(counts) == set(range(1, 31))
    assert all(33 <= c <= 34 for c in counts.values()), counts

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {len(data)} to {OUT}; day counts {sorted(set(counts.values()))}")


if __name__ == "__main__":
    main()
