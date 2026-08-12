"""Generator for double_no_carry: integer with all digits <=4 -> its double.

Domain: integers with 2-5 digits, every digit in {0,1,2,3,4}, leading digit in
{1,2,3,4}: 4*5 + 4*25 + 4*125 + 4*625 = 3120 items. Sample 1000.
The digit constraint guarantees carry-free doubling.
"""
import json
import random
from itertools import product
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "double_no_carry.json"


def main():
    rng = random.Random(42)

    domain = []
    for length in range(2, 6):
        for first in "1234":
            for rest in product("01234", repeat=length - 1):
                domain.append(int(first + "".join(rest)))
    assert len(domain) == 3120

    chosen = rng.sample(domain, 1000)
    examples = [{"input": str(n), "output": str(2 * n)} for n in chosen]
    rng2 = random.Random(42)
    rng2.shuffle(examples)

    # Asserts
    assert len(examples) == 1000
    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000
    for e in examples:
        n = int(e["input"])
        assert all(d in "01234" for d in e["input"]), e
        assert 2 <= len(e["input"]) <= 5 and e["input"][0] != "0"
        assert e["output"] == str(2 * n)
        # carry-free check: doubling digit-by-digit matches
        assert e["output"] == "".join(str(2 * int(d)) for d in e["input"])
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
