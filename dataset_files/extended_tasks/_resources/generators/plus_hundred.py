"""Generator for plus_hundred: integer with hundreds digit != 9 -> integer + 100.

Domain: integers 1..99999 with (n // 100) % 10 != 9 (carry-free by construction).
Includes a stratum of 1-2 digit inputs (47 -> 147), per the recipe.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "plus_hundred.json"


def main():
    rng = random.Random(42)

    chosen = set()
    # Stratum of small (1-2 digit) inputs; hundreds digit is 0, always valid.
    chosen.update(rng.sample(range(1, 100), 60))
    while len(chosen) < 1000:
        n = rng.randint(100, 99999)
        if (n // 100) % 10 != 9:
            chosen.add(n)
    chosen = sorted(chosen)

    examples = [{"input": str(n), "output": str(n + 100)} for n in chosen]
    rng2 = random.Random(42)
    rng2.shuffle(examples)

    # Asserts
    assert len(examples) == 1000
    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000
    for e in examples:
        n = int(e["input"])
        assert 1 <= n <= 99999 and (n // 100) % 10 != 9
        assert int(e["output"]) == n + 100
        # single-digit edit at the hundreds place (carry-free)
        a, b = e["input"].zfill(6), e["output"].zfill(6)
        assert sum(x != y for x, y in zip(a, b)) == 1
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
