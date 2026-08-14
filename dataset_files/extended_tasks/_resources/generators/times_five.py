"""Generator for times_five: integer -> integer * 5.

Domain: integers 10..1200, sampled to get 1000 unique values.
Rule: output = input * 5
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "times_five.json"


def main():
    rng = random.Random(42)

    # Sample 1000 unique integers from [10, 1200]
    chosen = set()
    while len(chosen) < 1000:
        n = rng.randint(10, 1200)
        chosen.add(n)
    chosen = sorted(chosen)

    # Create examples
    examples = [{"input": str(n), "output": str(n * 5)} for n in chosen]

    # Shuffle with seed 42
    rng2 = random.Random(42)
    rng2.shuffle(examples)

    # Asserts
    assert len(examples) == 1000, f"Expected 1000 examples, got {len(examples)}"

    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000, f"Expected 1000 unique inputs, got {len(set(inputs))}"

    for e in examples:
        n = int(e["input"])
        assert 10 <= n <= 1200, f"Input {n} out of range [10, 1200]"
        assert int(e["output"]) == n * 5, f"Output {e['output']} != {n * 5}"
        assert n != n * 5, f"Input {n} equals output {n * 5}"
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
