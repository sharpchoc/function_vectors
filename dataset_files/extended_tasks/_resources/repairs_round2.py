#!/usr/bin/env python
"""Round-2 repairs from the correctness audit (small inline fixes; the systematic ones —
strip_prefix opacity, countable_uncountable dual-use, hypernym/animal_class disjointness,
us-city-state ambiguity — are handled by dedicated repair agents).

- singular_or_plural: drop invariant 'means' (spec excludes invariants), top up.
- living_nonliving: drop animal/object homonyms (crane, bat, seal, mole), top up 'living'.
- concrete_abstract: drop dual-sense nouns the recipe itself names ('key') plus 'bond', top up.
- article_choice: 'urine' takes 'a' (initial /j/ glide), not 'an'.
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
    """Remove pairs whose input is in drop_inputs; append topups (skipping used inputs)."""
    d = load(name)
    used = {p["input"] for p in d}
    dropped = [p for p in d if p["input"] in drop_inputs]
    d = [p for p in d if p["input"] not in drop_inputs]
    print(f"{name}: dropped {[p['input'] for p in dropped]}")
    for pair in topups:
        if len(d) >= 1000:
            break
        if pair["input"] not in used:
            d.append(pair)
            used.add(pair["input"])
    save(name, d)


import lemminflect  # noqa: E402

# singular_or_plural: replace 'means' with a verified regular plural
cands = []
for noun in ("meadow", "lantern", "walnut", "pigeon", "hammock"):
    pl = lemminflect.getInflection(noun, tag="NNS")[0]
    if pl != noun:
        cands.append({"input": pl, "output": "plural"})
drop_and_topup("singular_or_plural", {"means"}, cands)

# living_nonliving: homonym purge (all were labeled 'living')
drop_and_topup("living_nonliving", {"crane", "bat", "seal", "mole"}, [
    {"input": "chimpanzee", "output": "living"}, {"input": "salamander", "output": "living"},
    {"input": "heron", "output": "living"}, {"input": "gazelle", "output": "living"},
    {"input": "ferret", "output": "living"}, {"input": "sturgeon", "output": "living"},
    {"input": "ocelot", "output": "living"}, {"input": "tapir", "output": "living"},
    {"input": "ibis", "output": "living"}, {"input": "newt", "output": "living"},
    {"input": "lemur", "output": "living"}, {"input": "marmot", "output": "living"},
    {"input": "osprey", "output": "living"}, {"input": "wombat", "output": "living"},
    {"input": "gibbon", "output": "living"}, {"input": "puffin", "output": "living"},
    {"input": "quail", "output": "living"}, {"input": "mongoose", "output": "living"},
    {"input": "stingray", "output": "living"}, {"input": "chickadee", "output": "living"},
    {"input": "doorknob", "output": "nonliving"}, {"input": "paperclip", "output": "nonliving"},
    {"input": "thumbtack", "output": "nonliving"}, {"input": "lampshade", "output": "nonliving"},
    {"input": "wheelbarrow", "output": "nonliving"}, {"input": "mousetrap", "output": "nonliving"},
])

# concrete_abstract: dual-sense purge
drop_and_topup("concrete_abstract", {"key", "bond"}, [
    {"input": "sorrow", "output": "abstract"}, {"input": "lantern", "output": "concrete"},
    {"input": "optimism", "output": "abstract"}, {"input": "walnut", "output": "concrete"},
    {"input": "pessimism", "output": "abstract"}, {"input": "trombone", "output": "concrete"},
    {"input": "nostalgia", "output": "abstract"}, {"input": "teapot", "output": "concrete"},
    {"input": "skepticism", "output": "abstract"}, {"input": "hairbrush", "output": "concrete"},
])

# article_choice: urine -> a (initial /j/ consonant glide)
d = load("article_choice")
n_fixed = 0
for p in d:
    if p["input"] == "urine" and p["output"] == "an":
        p["output"] = "a"
        n_fixed += 1
print(f"article_choice: fixed {n_fixed} pair(s)")
save("article_choice", d)
