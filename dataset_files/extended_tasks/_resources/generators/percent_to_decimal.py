"""Generator for percent_to_decimal: 'N%' -> N/100 with exactly two decimals.

Rule: N integer in 1..9999 with N % 10 != 0 (multiples of 10 would create
trailing-zero ambiguity). Output = f'{N/100:.2f}', e.g. 7% -> 0.07.

Note on the recipe's "~70% weight toward N < 100": inputs must be unique and
there are only 90 valid N below 100, so 70% is impossible. Instead we include
ALL 90 valid N < 100 and weight the remainder toward the smaller 3-digit range
(100..999) over 1000..9999.
"""
import json
import random
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "percent_to_decimal.json"
N_TOTAL = 1000


def main() -> None:
    rng = random.Random(3042)

    small = [n for n in range(1, 100) if n % 10 != 0]        # 90 values, all kept
    mid = [n for n in range(100, 1000) if n % 10 != 0]       # 810 values
    big = [n for n in range(1000, 10000) if n % 10 != 0]     # 8100 values

    chosen = list(small)                                     # all 90 of N < 100
    chosen += rng.sample(mid, 550)
    chosen += rng.sample(big, N_TOTAL - len(chosen))         # 360 from 1000..9999
    assert len(chosen) == N_TOTAL, len(chosen)

    data = [{"input": f"{n}%", "output": f"{n / 100:.2f}"} for n in chosen]
    random.seed(42)
    random.shuffle(data)

    # --- self-checks ---
    assert len(data) == N_TOTAL
    assert len({d["input"] for d in data}) == N_TOTAL
    for d in data:
        n = int(d["input"].rstrip("%"))
        assert 1 <= n <= 9999 and n % 10 != 0
        # exact re-derivation with Decimal (no float rounding concerns)
        assert d["output"] == f"{Decimal(n) / 100:.2f}"
        assert Decimal(d["output"]) * 100 == n
        assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {len(data)} to {OUT}")


if __name__ == "__main__":
    main()
