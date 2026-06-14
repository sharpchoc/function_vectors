"""
Generate the "ambiguous" task-disambiguation datasets.

Design (task-disambiguation / in-context rule inference): a pair (f1, f2) that
AGREE on an overlap region and DISAGREE on a differentiator region. Prompts put
3 demos from the overlap (ambiguous, consistent with both), a 4th demo from the
differentiator (selects f1 or f2), and a 5th query from the differentiator (scored
as f1 vs f2).

First pair:
  - magnitude.json : n -> |n|        (absolute value)
  - identity.json  : n -> n          (copy)
  Overlap  = non-negative integers (|n| == n).        -> 50 positives
  Differ   = negative integers (|n| = -n != n).        -> 50 negatives
  Both files share the SAME 100 inputs; the 50 positive (overlap) entries are
  byte-identical across the two files, the 50 negative entries differ only in output.

Integers rendered as DIGIT strings ("-5"), since absolute value is a sign operation;
switch to words if consistency with next_number/prev_number is wanted.
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "ambiguous")

N_OVERLAP = 50   # positive integers (f1 == f2)
N_DIFFER = 50    # negative integers (f1 != f2)


def build_magnitude_identity():
    positives = list(range(1, N_OVERLAP + 1))      # 1..50  (overlap)
    negatives = [-k for k in range(1, N_DIFFER + 1)]  # -1..-50 (differentiator)
    inputs = positives + negatives

    magnitude = [{"input": str(n), "output": str(abs(n))} for n in inputs]
    identity = [{"input": str(n), "output": str(n)} for n in inputs]
    return magnitude, identity


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    magnitude, identity = build_magnitude_identity()
    for name, data in [("magnitude.json", magnitude), ("identity.json", identity)]:
        path = os.path.join(OUT_DIR, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        n_overlap = sum(1 for a, b in zip(magnitude, identity) if a["output"] == b["output"])
        print(f"wrote {path}: {len(data)} entries")
    print(f"overlap (identical output in both): {n_overlap}; differentiating: {len(magnitude) - n_overlap}")


if __name__ == "__main__":
    main()
