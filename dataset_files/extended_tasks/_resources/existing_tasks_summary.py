#!/usr/bin/env python
"""Print a compact one-line-per-task summary of the 42 existing abstractive tasks."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "abstractive"
for f in sorted(root.glob("*.json")):
    d = json.load(open(f))
    exs = "; ".join(f"{p['input']!r}->{p['output']!r}" for p in d[:2])
    print(f"{f.stem} (n={len(d)}): {exs}")
