#!/usr/bin/env python
"""Round-1 repairs after the first validation gate. Documented, rerunnable.

1. case_of_word / count_digits / word_polarity: remove the single identity pair
   (input=='lower'/'1'/'positive'), top up with a fresh valid pair.
2. english-portuguese: remove 11 identical-form translations (identity pairs weaken the
   ICL task to copying), top up with 11 hand-curated common-word pairs.
3. larger_than_100 -> REPLACED by larger_than_1000: only 99 possible sub-100 inputs made a
   balanced 1000-example binary task impossible. Same rule shape, viable domain.
4. agent_noun_to_verb (748) -> 1000 by merging with the inversion of the validated
   agent_noun file (1000 verb->agent pairs), dedup by input.
5. lives_in_water (548, domain-capped) -> SWAPPED for number_word_to_digits.
   can_fly (283/717 imbalance, yes-class capped) -> SWAPPED for last_vowel.
6. new_task_specs.json updated to match (drop 2, add 2, rename 1).
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
    print(f"wrote {name}.json (n=1000)")


words = [w.strip() for w in open(RES / "common_words.txt")]

# --- 1. single identity-pair fixes ------------------------------------------------
d = load("case_of_word")
used = {p["input"] for p in d}
d = [p for p in d if p["input"] != p["output"]]
while len(d) < 1000:
    w = next(x for x in words if x not in ("upper", "lower") and x.upper() not in used and x not in used)
    d.append({"input": w, "output": "lower"})  # dropped pair was the 'lower'-class word 'lower'
    used.add(w)
save("case_of_word", d)

d = load("count_digits")
used = {p["input"] for p in d}
d = [p for p in d if p["input"] != p["output"]]
while len(d) < 1000:
    n = next(str(x) for x in range(10, 10 ** 6) if str(x) not in used and str(x) != str(len(str(x))))
    d.append({"input": n, "output": str(len(n))})
    used.add(n)
save("count_digits", d)

d = load("word_polarity")
used = {p["input"] for p in d}
d = [p for p in d if p["input"] != p["output"]]
topup = [("marvelous", "positive"), ("wonderful", "positive"), ("splendid", "positive"),
         ("superb", "positive"), ("exquisite", "positive"), ("delightful", "positive"),
         ("atrocious", "negative"), ("dreadful", "negative"), ("abysmal", "negative"),
         ("horrid", "negative"), ("loathsome", "negative"), ("wretched", "negative")]
for cand, lab in topup:
    if len(d) >= 1000:
        break
    if cand not in used:
        d.append({"input": cand, "output": lab})
        used.add(cand)
save("word_polarity", d)

# --- 2. english-portuguese identity pairs -----------------------------------------
d = load("english-portuguese")
used_in = {p["input"] for p in d}
d = [p for p in d if p["input"] != p["output"]]
replacements = [  # common words, unambiguous standard PT equivalents, all differing in form
    ("dog", "cachorro"), ("cat", "gato"), ("book", "livro"), ("house", "casa"),
    ("water", "água"), ("bread", "pão"), ("cheese", "queijo"), ("street", "rua"),
    ("window", "janela"), ("knife", "faca"), ("spoon", "colher"), ("chair", "cadeira"),
    ("table", "mesa"), ("beach", "praia"), ("rain", "chuva"), ("snow", "neve"),
    ("butterfly", "borboleta"), ("mushroom", "cogumelo"), ("scissors", "tesoura"),
    ("pillow", "travesseiro"), ("towel", "toalha"), ("wallet", "carteira"),
    ("spider", "aranha"), ("turtle", "tartaruga"), ("pumpkin", "abóbora"),
    ("cucumber", "pepino"), ("lettuce", "alface"), ("strawberry", "morango"),
    ("pineapple", "abacaxi"), ("grape", "uva"), ("owl", "coruja"),
    ("squirrel", "esquilo"), ("feather", "pena"), ("anchor", "âncora"),
    ("lighthouse", "farol"), ("ceiling", "teto"), ("peach", "pêssego"),
]
for en, pt in replacements:
    if len(d) >= 1000:
        break
    if en not in used_in:
        d.append({"input": en, "output": pt})
        used_in.add(en)
save("english-portuguese", d)

# --- 3. larger_than_1000 (replaces larger_than_100) --------------------------------
lo = rng.sample(range(2, 1000), 500)
hi = rng.sample(range(1001, 50001), 500)
d = [{"input": str(x), "output": "no"} for x in lo] + [{"input": str(x), "output": "yes"} for x in hi]
rng.shuffle(d)
save("larger_than_1000", d)
(ROOT / "larger_than_100.json").unlink(missing_ok=True)
print("removed larger_than_100.json")

# --- 4. agent_noun_to_verb top-up ---------------------------------------------------
fwd = load("agent_noun")               # verb -> agent noun, validated 1000
inv = {}
for p in fwd:
    inv.setdefault(p["output"], p["input"])   # agent noun -> verb, first wins
cur = load("agent_noun_to_verb") if (ROOT / "agent_noun_to_verb.json").exists() else []
for p in cur:
    inv.setdefault(p["input"], p["output"])
pairs = [{"input": k, "output": v} for k, v in inv.items() if k != v]
rng.shuffle(pairs)
save("agent_noun_to_verb", pairs[:1000])

# --- 5a. number_word_to_digits (replaces lives_in_water) ---------------------------
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = [None, None, "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def spell(n):
    assert 0 < n < 10000
    parts = []
    if n >= 1000:
        parts.append(ONES[n // 1000] + " thousand")
        n %= 1000
    if n >= 100:
        parts.append(ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        t = TENS[n // 10]
        parts.append(t + ("-" + ONES[n % 10] if n % 10 else ""))
    elif n > 0:
        parts.append(ONES[n])
    return " ".join(parts)


nums = rng.sample(range(21, 1501), 1000)
d = [{"input": spell(x), "output": str(x)} for x in nums]
save("number_word_to_digits", d)
(ROOT / "lives_in_water.json").unlink(missing_ok=True)
print("removed lives_in_water.json")

# --- 5b. last_vowel (replaces can_fly) ----------------------------------------------
VOWELS = set("aeiou")
d, seen = [], set()
for w in words:
    if len(d) >= 1000:
        break
    lv = next((ch for ch in reversed(w) if ch in VOWELS), None)
    if lv and w not in seen and len(w) >= 3 and w != lv:
        d.append({"input": w, "output": lv})
        seen.add(w)
rng.shuffle(d)
save("last_vowel", d)
(ROOT / "can_fly.json").unlink(missing_ok=True)
print("removed can_fly.json")

# --- 6. spec file update -------------------------------------------------------------
specs = json.load(open(RES / "new_task_specs.json"))["specs"]
by_name = {s["name"]: s for s in specs}
del by_name["lives_in_water"], by_name["can_fly"]
lt = by_name.pop("larger_than_100")
lt.update(name="larger_than_1000",
          rule="Given an integer other than 1000, output 'yes' if it is greater than 1000, otherwise 'no'.",
          generation_recipe="500 seeded ints in [2,999] ('no') + 500 in [1001,50000] ('yes'), shuffled seed 42.")
by_name["larger_than_1000"] = lt
by_name["number_word_to_digits"] = {
    "name": "number_word_to_digits", "category": "formatting", "lane": "numeric_sequence",
    "rule": "Given a number spelled out in English words (21-1500), output it in digits.",
    "examples": [{"input": spell(n), "output": str(n)} for n in (42, 317, 1250, 86, 909)],
    "domain_source": "rule (numbers 21-1500)", "domain_size_estimate": 1480,
    "generation_method": "rule", "difficulty_anchor": "next_number_digits / ordinal_suffix: number-format conversion, short digit outputs",
    "gptj_confidence": "high", "generation_recipe": "see repairs_round1.py spell()", "family": None,
}
by_name["last_vowel"] = {
    "name": "last_vowel", "category": "orthographic", "lane": "orthographic",
    "rule": "Word -> the last vowel (a,e,i,o,u) reading right to left.",
    "examples": [{"input": "planet", "output": "e"}, {"input": "window", "output": "o"},
                 {"input": "guitar", "output": "a"}, {"input": "yellow", "output": "o"},
                 {"input": "strength", "output": "e"}],
    "domain_source": "common_words.txt", "domain_size_estimate": 11900,
    "generation_method": "list_rule", "difficulty_anchor": "first_vowel: mirrored scan direction",
    "gptj_confidence": "medium", "generation_recipe": "see repairs_round1.py", "family": None,
}
out = {"specs": list(by_name.values())}
assert len(out["specs"]) == 100, len(out["specs"])
json.dump(out, open(RES / "new_task_specs.json", "w"), indent=1)
print("specs updated: 100 tasks")
