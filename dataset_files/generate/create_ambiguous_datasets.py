"""
Generate the "ambiguous" task-disambiguation datasets.

Design (task-disambiguation / in-context rule inference): a pair (f1, f2) that
AGREE on an overlap region and DISAGREE on a differentiator region. Prompts put
3 demos from the overlap (ambiguous, consistent with both), a 4th demo from the
differentiator (selects f1 or f2), and a 5th query from the differentiator (scored
as f1 vs f2). For every pair the two task files share the SAME input set; the
overlap entries are byte-identical (input AND output) across the two files, and the
differentiator entries share the input but differ in output.

Pairs:
  magnitude.json     | identity.json          n->|n| vs n->n
      overlap = non-negative ints (|n|==n);  differ = negative ints.
  past_tense.json    | past_participle.json   verb->past vs verb->participle
      overlap = regular verbs (past==participle); differ = irregular verbs.
  first_letter.json  | last_letter.json       word->word[0] vs word->word[-1]
      overlap = words with word[0]==word[-1]; differ = the rest.
  capital_city.json  | largest_city.json      country->capital vs country->largest city
      overlap = capital is the largest city; differ = capital != largest.
      NOTE: only ~35 real capital!=largest countries exist worldwide (most countries'
      capital IS their largest city), and only ~20 are GPT-J-plausible -> this pair
      cannot reach a clean 50/50 of model-known entities (recall caveat; cf. the geo/
      element low-N pairs). Built 50 overlap + 35 differ; trim/pad downstream as needed.

Integers rendered as DIGIT strings ("-5"), since absolute value is a sign operation.
"""
import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "ambiguous")
ABSTRACTIVE = os.path.join(os.path.dirname(__file__), "..", "abstractive")

N_OVERLAP = 50
N_DIFFER = 50
SEED = 0


# ----------------------------------------------------------------------------
def build_magnitude_identity():
    positives = list(range(1, N_OVERLAP + 1))         # overlap: |n| == n
    negatives = [-k for k in range(1, N_DIFFER + 1)]  # differ:  |n| != n
    inputs = positives + negatives
    magnitude = [{"input": str(n), "output": str(abs(n))} for n in inputs]
    identity = [{"input": str(n), "output": str(n)} for n in inputs]
    return magnitude, identity


# ----------------------------------------------------------------------------
# Regular verbs: past tense == past participle (the 50 overlap demos).
REGULAR_VERBS = [
    ("walk", "walked"), ("play", "played"), ("jump", "jumped"), ("open", "opened"),
    ("close", "closed"), ("call", "called"), ("want", "wanted"), ("need", "needed"),
    ("help", "helped"), ("talk", "talked"), ("work", "worked"), ("look", "looked"),
    ("ask", "asked"), ("turn", "turned"), ("move", "moved"), ("live", "lived"),
    ("follow", "followed"), ("create", "created"), ("add", "added"), ("change", "changed"),
    ("watch", "watched"), ("allow", "allowed"), ("count", "counted"), ("fill", "filled"),
    ("pull", "pulled"), ("push", "pushed"), ("learn", "learned"), ("return", "returned"),
    ("cook", "cooked"), ("clean", "cleaned"), ("paint", "painted"), ("plant", "planted"),
    ("climb", "climbed"), ("dance", "danced"), ("smile", "smiled"), ("laugh", "laughed"),
    ("finish", "finished"), ("start", "started"), ("wait", "waited"), ("pass", "passed"),
    ("kick", "kicked"), ("fix", "fixed"), ("wash", "washed"), ("touch", "touched"),
    ("save", "saved"), ("share", "shared"), ("taste", "tasted"), ("pour", "poured"),
    ("melt", "melted"), ("joke", "joked"),
]
# Irregular verbs: (base, past, participle) with past != participle (50 differ demos).
IRREGULAR_VERBS = [
    ("eat", "ate", "eaten"), ("go", "went", "gone"), ("sing", "sang", "sung"),
    ("drink", "drank", "drunk"), ("break", "broke", "broken"), ("give", "gave", "given"),
    ("take", "took", "taken"), ("write", "wrote", "written"), ("speak", "spoke", "spoken"),
    ("do", "did", "done"), ("see", "saw", "seen"), ("fall", "fell", "fallen"),
    ("drive", "drove", "driven"), ("ride", "rode", "ridden"), ("rise", "rose", "risen"),
    ("choose", "chose", "chosen"), ("freeze", "froze", "frozen"), ("steal", "stole", "stolen"),
    ("swim", "swam", "swum"), ("begin", "began", "begun"), ("ring", "rang", "rung"),
    ("shrink", "shrank", "shrunk"), ("throw", "threw", "thrown"), ("grow", "grew", "grown"),
    ("know", "knew", "known"), ("fly", "flew", "flown"), ("draw", "drew", "drawn"),
    ("blow", "blew", "blown"), ("show", "showed", "shown"), ("hide", "hid", "hidden"),
    ("bite", "bit", "bitten"), ("forget", "forgot", "forgotten"), ("forgive", "forgave", "forgiven"),
    ("wake", "woke", "woken"), ("tear", "tore", "torn"), ("wear", "wore", "worn"),
    ("swear", "swore", "sworn"), ("bear", "bore", "borne"), ("get", "got", "gotten"),
    ("beat", "beat", "beaten"), ("shake", "shook", "shaken"), ("awake", "awoke", "awoken"),
    ("become", "became", "become"), ("come", "came", "come"), ("run", "ran", "run"),
    ("prove", "proved", "proven"), ("sew", "sewed", "sewn"), ("mow", "mowed", "mown"),
    ("strive", "strove", "striven"), ("weave", "wove", "woven"),
]


def build_past_tense_participle():
    past, part = [], []
    for base, p in REGULAR_VERBS:                 # overlap: past == participle
        past.append({"input": base, "output": p})
        part.append({"input": base, "output": p})
    for base, p, pp in IRREGULAR_VERBS:           # differ: past != participle
        past.append({"input": base, "output": p})
        part.append({"input": base, "output": pp})
    return past, part


# ----------------------------------------------------------------------------
def _load_vocab():
    words = set()
    for name in ("synonym.json", "antonym.json"):
        for x in json.load(open(os.path.join(ABSTRACTIVE, name))):
            w = x["input"].strip().lower()
            if w.isalpha() and len(w) >= 3:
                words.add(w)
    return sorted(words)


def build_first_last_letter():
    rng = random.Random(SEED)
    vocab = _load_vocab()
    same = [w for w in vocab if w[0] == w[-1]]    # overlap: first == last letter
    diff = [w for w in vocab if w[0] != w[-1]]    # differ
    overlap = rng.sample(same, N_OVERLAP)
    differ = rng.sample(diff, N_DIFFER)
    inputs = overlap + differ
    first = [{"input": w, "output": w[0]} for w in inputs]
    last = [{"input": w, "output": w[-1]} for w in inputs]
    return first, last


# ----------------------------------------------------------------------------
# Overlap: capital city IS the largest city (50).
CAPITAL_IS_LARGEST = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Russia", "Moscow"),
    ("United Kingdom", "London"), ("Spain", "Madrid"), ("Mexico", "Mexico City"),
    ("Indonesia", "Jakarta"), ("Iran", "Tehran"), ("Thailand", "Bangkok"),
    ("Greece", "Athens"), ("Argentina", "Buenos Aires"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Cuba", "Havana"),
    ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Czechia", "Prague"),
    ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Sweden", "Stockholm"),
    ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Norway", "Oslo"),
    ("Poland", "Warsaw"), ("Romania", "Bucharest"), ("Ukraine", "Kyiv"),
    ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Kenya", "Nairobi"),
    ("Ethiopia", "Addis Ababa"), ("Bangladesh", "Dhaka"), ("South Korea", "Seoul"),
    ("Singapore", "Singapore"), ("Lebanon", "Beirut"), ("Jordan", "Amman"),
    ("Qatar", "Doha"), ("Kuwait", "Kuwait City"), ("Algeria", "Algiers"),
    ("Tunisia", "Tunis"), ("Libya", "Tripoli"), ("Venezuela", "Caracas"),
    ("Uruguay", "Montevideo"), ("Guatemala", "Guatemala City"), ("Panama", "Panama City"),
    ("Mongolia", "Ulaanbaatar"), ("Nepal", "Kathmandu"), ("Afghanistan", "Kabul"),
    ("Iceland", "Reykjavik"),
]
# Differentiator: capital != largest. (country, capital, largest). Only ~35 real ones
# exist; the first ~20 are GPT-J-plausible, the tail is low-frequency (recall caveat).
CAPITAL_NOT_LARGEST = [
    ("United States", "Washington", "New York"), ("Australia", "Canberra", "Sydney"),
    ("Canada", "Ottawa", "Toronto"), ("Brazil", "Brasilia", "Sao Paulo"),
    ("China", "Beijing", "Shanghai"), ("India", "New Delhi", "Mumbai"),
    ("Turkey", "Ankara", "Istanbul"), ("Switzerland", "Bern", "Zurich"),
    ("Pakistan", "Islamabad", "Karachi"), ("Nigeria", "Abuja", "Lagos"),
    ("Vietnam", "Hanoi", "Ho Chi Minh City"), ("Morocco", "Rabat", "Casablanca"),
    ("South Africa", "Pretoria", "Johannesburg"), ("Kazakhstan", "Astana", "Almaty"),
    ("Myanmar", "Naypyidaw", "Yangon"), ("New Zealand", "Wellington", "Auckland"),
    ("United Arab Emirates", "Abu Dhabi", "Dubai"), ("Philippines", "Manila", "Quezon City"),
    ("Ecuador", "Quito", "Guayaquil"), ("Tanzania", "Dodoma", "Dar es Salaam"),
    ("Bolivia", "Sucre", "La Paz"), ("Cameroon", "Yaounde", "Douala"),
    ("Ivory Coast", "Yamoussoukro", "Abidjan"),
    ("Sri Lanka", "Sri Jayawardenepura Kotte", "Colombo"), ("Benin", "Porto-Novo", "Cotonou"),
    ("Belize", "Belmopan", "Belize City"), ("Equatorial Guinea", "Malabo", "Bata"),
    ("Burundi", "Gitega", "Bujumbura"), ("Malta", "Valletta", "Birkirkara"),
    ("Liechtenstein", "Vaduz", "Schaan"), ("Palau", "Ngerulmud", "Koror"),
    ("Eswatini", "Mbabane", "Manzini"), ("Micronesia", "Palikir", "Weno"),
    ("Gambia", "Banjul", "Serrekunda"), ("Trinidad and Tobago", "Port of Spain", "Chaguanas"),
]


def build_capital_largest():
    capital, largest = [], []
    for country, city in CAPITAL_IS_LARGEST:          # overlap
        capital.append({"input": country, "output": city})
        largest.append({"input": country, "output": city})
    for country, cap, lg in CAPITAL_NOT_LARGEST:      # differ
        capital.append({"input": country, "output": cap})
        largest.append({"input": country, "output": lg})
    return capital, largest


# ----------------------------------------------------------------------------
def build_round_truncate():
    """round (half-up) vs truncate of a 1-dp decimal. Overlap = frac < .5 (both floor);
    differ = frac >= .5 (round ->ceil, truncate ->floor)."""
    rnd, trunc = [], []
    over = [(n, [1, 2, 3, 4][(n - 1) % 4]) for n in range(1, N_OVERLAP + 1)]   # frac < .5
    diff = [(n, [5, 6, 7, 8, 9][(n - 1) % 5]) for n in range(1, N_DIFFER + 1)]  # frac >= .5
    for n, f in over:
        s = f"{n}.{f}"
        rnd.append({"input": s, "output": str(n)})
        trunc.append({"input": s, "output": str(n)})
    for n, f in diff:
        s = f"{n}.{f}"
        rnd.append({"input": s, "output": str(n + 1)})  # rounds up
        trunc.append({"input": s, "output": str(n)})
    return rnd, trunc


def build_first_last_digit():
    """first vs last digit of an integer. Overlap = first digit == last digit (incl. 1-digit);
    differ = first != last. Numeric analog of first_letter|last_letter (prior-bias diagnostic)."""
    rng = random.Random(SEED)
    pool_same = [n for n in range(1, 1000) if str(n)[0] == str(n)[-1]]
    pool_diff = [n for n in range(10, 1000) if str(n)[0] != str(n)[-1]]
    inputs = rng.sample(pool_same, N_OVERLAP) + rng.sample(pool_diff, N_DIFFER)
    first = [{"input": str(n), "output": str(n)[0]} for n in inputs]
    last = [{"input": str(n), "output": str(n)[-1]} for n in inputs]
    return first, last


# American vs British spelling. Input = American spelling (GPT-J is US-leaning, so "leave as
# American" is the prior); american = identity, british = US->UK conversion.
# Overlap = words spelled identically in both dialects.
INVARIANT_WORDS = [
    "table", "water", "house", "computer", "science", "music", "garden", "window", "picture",
    "mountain", "river", "forest", "animal", "family", "friend", "school", "teacher", "student",
    "doctor", "paper", "phone", "road", "city", "country", "language", "history", "future",
    "morning", "evening", "summer", "winter", "market", "money", "number", "letter", "person",
    "reason", "nature", "system", "problem", "question", "answer", "story", "bridge", "island",
    "planet", "ocean", "desert", "village", "garage",
]
US_UK_PAIRS = [
    ("color", "colour"), ("flavor", "flavour"), ("favor", "favour"), ("honor", "honour"),
    ("humor", "humour"), ("labor", "labour"), ("neighbor", "neighbour"), ("rumor", "rumour"),
    ("savor", "savour"), ("vapor", "vapour"), ("behavior", "behaviour"), ("harbor", "harbour"),
    ("odor", "odour"), ("splendor", "splendour"), ("vigor", "vigour"), ("center", "centre"),
    ("theater", "theatre"), ("liter", "litre"), ("meter", "metre"), ("fiber", "fibre"),
    ("somber", "sombre"), ("specter", "spectre"), ("organize", "organise"), ("realize", "realise"),
    ("recognize", "recognise"), ("apologize", "apologise"), ("analyze", "analyse"),
    ("paralyze", "paralyse"), ("catalog", "catalogue"), ("dialog", "dialogue"),
    ("defense", "defence"), ("offense", "offence"), ("license", "licence"), ("pretense", "pretence"),
    ("traveler", "traveller"), ("jewelry", "jewellery"), ("gray", "grey"), ("mold", "mould"),
    ("plow", "plough"), ("tire", "tyre"), ("curb", "kerb"), ("program", "programme"),
    ("aluminum", "aluminium"), ("mustache", "moustache"), ("pajamas", "pyjamas"),
    ("donut", "doughnut"), ("mom", "mum"), ("cozy", "cosy"), ("check", "cheque"),
    ("artifact", "artefact"),
]


# reverse vs identity. Overlap = palindromes (reverse(w) == w); differ = ordinary words.
PALINDROMES = [
    "mom", "dad", "wow", "pop", "sis", "gag", "tot", "pup", "bib", "gig",
    "nun", "eye", "did", "dud", "mum", "pip", "eve", "ewe", "bob", "nan",
    "pap", "tat", "noon", "deed", "peep", "sees", "toot", "naan", "anna", "otto",
    "abba", "level", "civic", "radar", "kayak", "madam", "refer", "rotor", "minim", "tenet",
    "sagas", "solos", "stats", "redder", "hannah", "racecar", "repaper", "reviver", "deified", "rotator",
]


def build_reverse_identity():
    rng = random.Random(SEED)
    palindromes = PALINDROMES[:N_OVERLAP]
    assert len(palindromes) == N_OVERLAP and all(w == w[::-1] for w in palindromes)
    vocab = [w for w in _load_vocab() if 3 <= len(w) <= 5 and w != w[::-1]]
    others = rng.sample(vocab, N_DIFFER)          # ordinary (non-palindrome) words
    inputs = palindromes + others
    reverse = [{"input": w, "output": w[::-1]} for w in inputs]
    identity = [{"input": w, "output": w} for w in inputs]
    return reverse, identity


def build_vowel_consonant_count():
    """word -> #vowels vs word -> #consonants (vowels = aeiou). Overlap = words with
    #vowels == #consonants; differ = the rest. Counting task (competence-limited, like reverse)."""
    rng = random.Random(SEED)
    V = "aeiou"
    vowels = lambda w: sum(c in V for c in w)
    cons = lambda w: sum(c not in V for c in w)
    vocab = [w for w in _load_vocab() if 3 <= len(w) <= 9]
    eq = [w for w in vocab if vowels(w) == cons(w)]
    neq = [w for w in vocab if vowels(w) != cons(w)]
    inputs = rng.sample(eq, N_OVERLAP) + rng.sample(neq, N_DIFFER)
    vowel_count = [{"input": w, "output": str(vowels(w))} for w in inputs]
    cons_count = [{"input": w, "output": str(cons(w))} for w in inputs]
    return vowel_count, cons_count


def build_american_british():
    american, british = [], []
    for w in INVARIANT_WORDS:                 # overlap: identical spelling
        american.append({"input": w, "output": w})
        british.append({"input": w, "output": w})
    for us, uk in US_UK_PAIRS:                # differ: american=identity, british=convert
        american.append({"input": us, "output": us})
        british.append({"input": us, "output": uk})
    return american, british


# ----------------------------------------------------------------------------
PAIRS = {
    "magnitude.json": None, "identity.json": None,
    "past_tense.json": None, "past_participle.json": None,
    "first_letter.json": None, "last_letter.json": None,
    "capital_city.json": None, "largest_city.json": None,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    builders = [
        (("magnitude.json", "identity.json"), build_magnitude_identity),
        (("past_tense.json", "past_participle.json"), build_past_tense_participle),
        (("first_letter.json", "last_letter.json"), build_first_last_letter),
        (("capital_city.json", "largest_city.json"), build_capital_largest),
        (("round.json", "truncate.json"), build_round_truncate),
        (("first_digit.json", "last_digit.json"), build_first_last_digit),
        (("american.json", "british.json"), build_american_british),
        (("reverse.json", "identity_word.json"), build_reverse_identity),
        (("count_vowels.json", "count_consonants.json"), build_vowel_consonant_count),
    ]
    for (n1, n2), fn in builders:
        d1, d2 = fn()
        for name, data in ((n1, d1), (n2, d2)):
            with open(os.path.join(OUT_DIR, name), "w") as f:
                json.dump(data, f, indent=2)
        assert [x["input"] for x in d1] == [x["input"] for x in d2], f"{n1}/{n2} input mismatch"
        n_over = sum(1 for a, b in zip(d1, d2) if a["output"] == b["output"])
        print(f"{n1:18s} | {n2:20s}  n={len(d1):3d}  overlap={n_over:3d}  differ={len(d1)-n_over:3d}")


if __name__ == "__main__":
    main()
