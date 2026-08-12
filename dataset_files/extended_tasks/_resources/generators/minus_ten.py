"""Generator for minus_ten: integer >= 20 with tens digit != 0 -> integer - 10.

Domain: integers 20..99999 with (n // 10) % 10 != 0 (borrow-free by construction).
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "minus_ten.json"


def main():
    rng = random.Random(42)

    chosen = set()
    while len(chosen) < 1000:
        n = rng.randint(20, 99999)
        if (n // 10) % 10 != 0:
            chosen.add(n)
    chosen = sorted(chosen)

    examples = [{"input": str(n), "output": str(n - 10)} for n in chosen]
    rng2 = random.Random(42)
    rng2.shuffle(examples)

    # Asserts
    assert len(examples) == 1000
    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000
    for e in examples:
        n = int(e["input"])
        assert 20 <= n <= 99999 and (n // 10) % 10 != 0
        assert int(e["output"]) == n - 10
        # single-digit edit: only the tens digit changes
        a, b = e["input"].zfill(6), e["output"].zfill(6)
        assert sum(x != y for x, y in zip(a, b)) == 1
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
