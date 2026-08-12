#!/usr/bin/env python
"""Round-3 (final) repairs from the re-audit of the 8 repaired tasks.

- strip_prefix: drop 9 remaining Latin-cognate opaque splits (reflex, preface(d), recourse,
  dissolution, resolution, restrict, prejudicial, resource), top up with transparent pairs.
- us-city-state: drop 6 ambiguous-name residuals (Lisbon, Smyrna, Winter Park, Caldwell,
  Belleville, Dickinson), top up with unambiguous single-state cities.
- living_nonliving: drop 'dipper' (bird vs ladle/constellation polysemy), top up.
"""
import json
import random
from pathlib import Path

RES = Path(__file__).resolve().parent
ROOT = RES.parent
rng = random.Random(42)


def load(name):
    return json.load(open(ROOT / f"{name}.json"))


def save(name, data):
    assert len(data) == 1000, (name, len(data))
    ins = [p["input"] for p in data]
    assert len(set(ins)) == 1000 and all(p["input"] != p["output"] for p in data), name
    json.dump(data, open(ROOT / f"{name}.json", "w"), indent=1)
    print(f"wrote {name}.json")


def drop_and_topup(name, drop_inputs, topups):
    d = load(name)
    used = {p["input"] for p in d}
    dropped = [p["input"] for p in d if p["input"] in drop_inputs]
    d = [p for p in d if p["input"] not in drop_inputs]
    print(f"{name}: dropped {dropped}")
    for pair in topups:
        if len(d) >= 1000:
            break
        if pair["input"] not in used:
            d.append(pair)
            used.add(pair["input"])
    save(name, d)


drop_and_topup("strip_prefix",
               {"reflex", "preface", "prefaced", "recourse", "dissolution", "resolution",
                "restrict", "prejudicial", "resource"}, [
    {"input": "unafraid", "output": "afraid"}, {"input": "reheat", "output": "heat"},
    {"input": "misjudge", "output": "judge"}, {"input": "overcook", "output": "cook"},
    {"input": "unbeaten", "output": "beaten"}, {"input": "misread", "output": "read"},
    {"input": "retype", "output": "type"}, {"input": "nonviolent", "output": "violent"},
    {"input": "underfund", "output": "fund"}, {"input": "unbuttoned", "output": "buttoned"},
    {"input": "rewire", "output": "wire"}, {"input": "mislabel", "output": "label"},
    {"input": "overwater", "output": "water"}, {"input": "unpainted", "output": "painted"},
    {"input": "recount", "output": "count"}, {"input": "unhurt", "output": "hurt"},
    {"input": "misfile", "output": "file"}, {"input": "overpay", "output": "pay"},
    {"input": "unsalted", "output": "salted"}, {"input": "rewash", "output": "wash"},
    {"input": "nonverbal", "output": "verbal"}, {"input": "unopened", "output": "opened"},
])

drop_and_topup("us-city-state",
               {"Lisbon", "Smyrna", "Winter Park", "Caldwell", "Belleville", "Dickinson"}, [
    {"input": "Bozeman", "output": "Montana"}, {"input": "Flagstaff", "output": "Arizona"},
    {"input": "Kalamazoo", "output": "Michigan"}, {"input": "Poughkeepsie", "output": "New York"},
    {"input": "Waxahachie", "output": "Texas"}, {"input": "Opelousas", "output": "Louisiana"},
    {"input": "Ypsilanti", "output": "Michigan"}, {"input": "Sheboygan", "output": "Wisconsin"},
    {"input": "Tuscaloosa", "output": "Alabama"}, {"input": "Pocatello", "output": "Idaho"},
    {"input": "Chattahoochee", "output": "Florida"}, {"input": "Hattiesburg", "output": "Mississippi"},
    {"input": "Winnemucca", "output": "Nevada"}, {"input": "Keokuk", "output": "Iowa"},
    {"input": "Skowhegan", "output": "Maine"}, {"input": "Tucumcari", "output": "New Mexico"},
    {"input": "Punxsutawney", "output": "Pennsylvania"}, {"input": "Okmulgee", "output": "Oklahoma"},
])

drop_and_topup("living_nonliving", {"dipper"}, [
    {"input": "toucan", "output": "living"}, {"input": "porcupine", "output": "living"},
    {"input": "armadillo", "output": "living"}, {"input": "pelican", "output": "living"},
])
