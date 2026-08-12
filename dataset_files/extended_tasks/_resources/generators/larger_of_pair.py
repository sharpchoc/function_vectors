"""Generator for larger_of_pair: given two distinct integers, output the larger.

Rule: input "a b" with a != b (1..999); output str(max(a, b)).
Recipe: larger number first/second 50/50 so position is uninformative;
~half the pairs have different digit counts (easy), half same digit count.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "larger_of_pair.json"
N = 1000


def ndigits(x: int) -> int:
    return len(str(x))


def main() -> None:
    rng = random.Random(1040)  # generation seed
    pairs = {}
    # Half: same digit count; half: different digit count.
    targets = {"same": N // 2, "diff": N - N // 2}
    counts = {"same": 0, "diff": 0}
    while len(pairs) < N:
        a = rng.randint(1, 999)
        b = rng.randint(1, 999)
        if a == b:
            continue
        kind = "same" if ndigits(a) == ndigits(b) else "diff"
        if counts[kind] >= targets[kind]:
            continue
        # 50/50 whether the larger number comes first
        lo, hi = min(a, b), max(a, b)
        if rng.random() < 0.5:
            inp = f"{hi} {lo}"
        else:
            inp = f"{lo} {hi}"
        if inp in pairs:
            continue
        pairs[inp] = str(hi)
        counts[kind] += 1

    data = [{"input": k, "output": v} for k, v in pairs.items()]
    random.seed(42)
    random.shuffle(data)

    # --- self-checks ---
    assert len(data) == N
    assert len({d["input"] for d in data}) == N
    larger_first = 0
    for d in data:
        a, b = map(int, d["input"].split())
        assert a != b and 1 <= a <= 999 and 1 <= b <= 999
        assert d["output"] == str(max(a, b))
        assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()
        if a > b:
            larger_first += 1
    assert 400 <= larger_first <= 600, larger_first
    assert counts["same"] == targets["same"] and counts["diff"] == targets["diff"]

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {len(data)} to {OUT}; larger-first={larger_first}")


if __name__ == "__main__":
    main()
