"""Generator for double_number: integer -> integer * 2.

Domain: integers 10..1009 (exactly 1000 unique values).
Rule: output = input * 2
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "double_number.json"


def main():
    rng = random.Random(42)

    # Range [10, 1009] gives exactly 1000 unique integers
    chosen = list(range(10, 1010))

    # Create examples
    examples = [{"input": str(n), "output": str(n * 2)} for n in chosen]

    # Shuffle with seed 42
    rng.shuffle(examples)

    # Asserts
    assert len(examples) == 1000, f"Expected 1000 examples, got {len(examples)}"

    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000, f"Expected 1000 unique inputs, got {len(set(inputs))}"

    for e in examples:
        n = int(e["input"])
        assert 10 <= n <= 1009, f"Input {n} out of range [10, 1009]"
        assert int(e["output"]) == n * 2, f"Output {e['output']} != {n * 2}"
        assert n != n * 2, f"Input {n} equals output {n * 2}"
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
