"""Generator for halve_even_number: even integer -> integer // 2.

Domain: 1000 even integers sampled from [10, 2200].
Rule: output = input // 2
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "halve_even_number.json"


def main():
    rng = random.Random(42)

    # Sample 1000 unique even integers from [10, 2200]
    # Even numbers in range: 10, 12, 14, ..., 2200 (1096 total)
    even_candidates = list(range(10, 2201, 2))
    chosen = rng.sample(even_candidates, k=1000)
    chosen = sorted(chosen)

    # Create examples
    examples = [{"input": str(n), "output": str(n // 2)} for n in chosen]

    # Shuffle with seed 42
    rng2 = random.Random(42)
    rng2.shuffle(examples)

    # Asserts
    assert len(examples) == 1000, f"Expected 1000 examples, got {len(examples)}"

    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000, f"Expected 1000 unique inputs, got {len(set(inputs))}"

    for e in examples:
        n = int(e["input"])
        assert 10 <= n <= 2200, f"Input {n} out of range [10, 2200]"
        assert n % 2 == 0, f"Input {n} is not even"
        assert int(e["output"]) == n // 2, f"Output {e['output']} != {n // 2}"
        assert n != n // 2, f"Input {n} equals output {n // 2}"
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
