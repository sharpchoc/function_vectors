"""Generator for smaller_of_pair: given two distinct integers, output the smaller.

Rule: input "a b" with a != b (1..999); output str(min(a, b)).
Same generator shape as larger_of_pair with an independent seed;
smaller number first/second 50/50; ~half diff digit counts, half same.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "smaller_of_pair.json"
N = 1000


def ndigits(x: int) -> int:
    return len(str(x))


def main() -> None:
    rng = random.Random(2041)  # independent generation seed
    pairs = {}
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
        lo, hi = min(a, b), max(a, b)
        # 50/50 whether the smaller number comes first
        if rng.random() < 0.5:
            inp = f"{lo} {hi}"
        else:
            inp = f"{hi} {lo}"
        if inp in pairs:
            continue
        pairs[inp] = str(lo)
        counts[kind] += 1

    data = [{"input": k, "output": v} for k, v in pairs.items()]
    random.seed(42)
    random.shuffle(data)

    # --- self-checks ---
    assert len(data) == N
    assert len({d["input"] for d in data}) == N
    smaller_first = 0
    for d in data:
        a, b = map(int, d["input"].split())
        assert a != b and 1 <= a <= 999 and 1 <= b <= 999
        assert d["output"] == str(min(a, b))
        assert d["input"] == d["input"].strip() and d["output"] == d["output"].strip()
        if a < b:
            smaller_first += 1
    assert 400 <= smaller_first <= 600, smaller_first
    assert counts["same"] == targets["same"] and counts["diff"] == targets["diff"]

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {len(data)} to {OUT}; smaller-first={smaller_first}")


if __name__ == "__main__":
    main()
