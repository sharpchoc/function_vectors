"""Generator for ordinal_suffix: integer in digits -> integer with English ordinal suffix.

Rule: if n % 100 in {11,12,13} -> 'th'; elif n % 10 == 1 -> 'st';
elif n % 10 == 2 -> 'nd'; elif n % 10 == 3 -> 'rd'; else 'th'.
Oversample st/nd/rd so 'th' is not >80% of labels: target ~40% th, ~20% each st/nd/rd.
Domain: integers 1..9999.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "ordinal_suffix.json"


def suffix(n: int) -> str:
    if n % 100 in (11, 12, 13):
        return "th"
    if n % 10 == 1:
        return "st"
    if n % 10 == 2:
        return "nd"
    if n % 10 == 3:
        return "rd"
    return "th"


def main():
    rng = random.Random(42)

    # Bucket the domain by suffix class.
    buckets = {"st": [], "nd": [], "rd": [], "th": []}
    for n in range(1, 10000):
        buckets[suffix(n)].append(n)

    # Unit tests from the recipe.
    assert suffix(11) == "th" and suffix(12) == "th" and suffix(13) == "th"
    assert suffix(111) == "th" and suffix(121) == "st"
    assert suffix(21) == "st" and suffix(92) == "nd" and suffix(3) == "rd"

    counts = {"th": 400, "st": 200, "nd": 200, "rd": 200}
    chosen = []
    for suf, k in counts.items():
        chosen.extend(rng.sample(buckets[suf], k))

    examples = [{"input": str(n), "output": str(n) + suffix(n)} for n in chosen]
    rng2 = random.Random(42)
    rng2.shuffle(examples)

    # Asserts
    assert len(examples) == 1000
    inputs = [e["input"] for e in examples]
    assert len(set(inputs)) == 1000
    for e in examples:
        n = int(e["input"])
        assert 1 <= n <= 9999
        assert e["output"] == str(n) + suffix(n)
        assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()
    # 'th' not >80%
    th_frac = sum(1 for e in examples if e["output"].endswith("th")) / len(examples)
    assert th_frac <= 0.8, th_frac

    OUT.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT}; th fraction {th_frac:.2f}")


if __name__ == "__main__":
    main()
