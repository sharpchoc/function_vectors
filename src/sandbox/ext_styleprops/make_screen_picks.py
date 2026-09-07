#!/usr/bin/env python
"""Turn the unjudged screen into (layer, dose) picks for the judged stages.

For each (cell, direction, property):
  L, alpha   = argmax over the screen grid of the STRICT-style rate
               (fraction of ALL rollouts labelled the target convention; unscorable counts
               as not adopted - the same shape as the reported metric, minus coherence)
  per_layer  = for each layer, the best dose at that layer (for the by-layer figures)

Writes artifacts/style_properties/steering/grid/screen_picks.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.ext_styleprops.properties import PROPS
from src.sandbox.ext_styleprops.grid import SCREEN_LAYERS

GRID = ARTIFACTS_ROOT / "style_properties" / "steering" / "grid"
KEY = re.compile(r"^L(\d+)_a([\d.]+)$")


def strict_of(prop, cond, tgt):
    labs = [PROPS[prop].classify(t) for t in cond["tails"]]
    return float(np.mean([l == tgt for l in labs])) if labs else float("nan")


def main():
    picks = {}
    for tag_dir in sorted((GRID / "screen").glob("*__*")):
        tag = tag_dir.name
        direction = tag.rsplit("__", 1)[1]
        per_prop = {}
        for f in sorted(tag_dir.glob("*.json")):
            prop = f.stem
            C = json.load(open(f))["conditions"]
            cells = {}
            for k, v in C.items():
                m = KEY.match(k)
                if m and v.get("tails"):
                    cells[(int(m.group(1)), float(m.group(2)))] = strict_of(prop, v, direction)
            if not cells:
                continue
            (bL, ba) = max(cells, key=lambda t: cells[t])
            per_layer = {}
            for L in SCREEN_LAYERS:
                at = {a: s for (l, a), s in cells.items() if l == L}
                if at:
                    per_layer[str(L)] = max(at, key=at.get)
            per_prop[prop] = {"L": bL, "alpha": ba, "strict_screen": round(cells[(bL, ba)], 4),
                              "per_layer": per_layer}
        if per_prop:
            picks[tag] = per_prop
            top = sorted(per_prop.items(), key=lambda kv: -kv[1]["strict_screen"])[:3]
            print(f"{tag:46s} " + "  ".join(f"{p}:L{v['L']}a{v['alpha']:g}={v['strict_screen']:.2f}"
                                            for p, v in top), flush=True)
    json.dump(picks, open(GRID / "screen_picks.json", "w"), indent=1)
    print(f"\n{len(picks)} (cell,direction) tags -> {GRID/'screen_picks.json'}")


if __name__ == "__main__":
    main()
