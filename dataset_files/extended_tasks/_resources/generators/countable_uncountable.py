#!/usr/bin/env python3
"""Generator for countable_uncountable: English noun -> 'countable' or
'uncountable'.

Recipe (spec idx 76, generation_method="knowledge"): this is a curated
knowledge task -- the fact lists below ARE the reproducible artifact. Two
independent judgment passes are combined:

  Pass 1 (manual curation): an explicit, hand-picked list of nouns for each
  class, restricted to CLEAR cases (abstract qualities, materials, weather
  phenomena, closed academic/activity categories, and mass foods with no
  natural count-plural sense for uncountable; ordinary physical/count nouns
  pulled from common_nouns.txt for countable).

  Pass 2 (mechanical cross-check): for every uncountable candidate we compute
  its regular plural via lemminflect and require wordfreq zipf(plural) < 3.0
  -- i.e. the word must not, in fact, have a well-attested plural in the
  wild. Candidates that fail this check (meaning the "clear" uncountable
  judgment was wrong, e.g. a word like 'wine' that is regularly pluralized
  for varieties) are dropped and printed for visibility.

Ambiguous "both" nouns (common count AND mass senses -- coffee, tea, paper,
hair, glass, chicken, experience, wood, fruit, fish, meat, cheese, wine,
beer, cake, stone, bread, etc.) are excluded from BOTH classes per the spec.

Countable cap: the spec caps countables at ~60% of the set, relaxed here
because the reliable uncountable set (English has a genuinely closed set of
common, unambiguous mass nouns) tops out well under 500 -- we instead balance
classes to within the dataset's +-10% requirement and report the true final
n in "issues" if under 1000.

2026-08-12 STRICT RE-CURATION (audit failure fix): the original run of this
generator pulled "countable" candidates straight from common_nouns.txt
(the raw top-frequency word list, not nouns-only), filtered only by a
POS-agnostic stopword list, mechanical zipf/plural checks, and BOTH_EXCLUDE.
That is not enough to catch dual-sense nouns -- English's most frequent
words are exactly the ones most likely to be polysemous (e.g. "point",
"case", "order", "notice", "study", "sea", "energy", "water", "help",
"blood", "damage" -- several of which are flatly mislabeled: "water" and
"blood" are canonical uncountable/mass nouns, not countable). An audit
found ~186 such items in the countable class (dual count/mass sense,
non-nouns like "once", or primarily-verb/adjective words like "can",
"old", "red") and 4 in the uncountable class ("childhood", "adulthood",
"yogurt" -- all commonly used with "a"/plural in ordinary speech; "theft"
-- "a theft"/"thefts" is a very ordinary countable use alongside the
abstract sense).

Fix: COUNTABLE_DUAL_OR_NONNOUN_EXCLUDE and UNCOUNTABLE_DUAL_EXCLUDE below
were applied as a post-filter on top of the original mechanical draw, then
the countable side was topped back up using CONCRETE_COUNTABLE_CANDIDATES
-- a hand-picked list of concrete object/creature/person/place nouns with
no plausible mass sense (the task spec explicitly recommends this pool for
filling the countable side, since it is the abundant class). NOTE: the
mechanical draw (build_uncountable/build_countable, via wordfreq
zipf_frequency) is NOT reproducible byte-for-byte across environments --
zipf values drift slightly between wordfreq releases/locales, which flips
a handful of borderline candidates in/out. main() therefore does NOT
recompute the draw; it emits the frozen, already-audited word lists in
_frozen_lists.py (FINAL_COUNTABLE, 504 items; FINAL_UNCOUNTABLE, 496
items) directly, matching the docstring's "the fact lists ARE the
reproducible artifact" philosophy. build_uncountable/build_countable/
build_concrete_supplement and the candidate lists above are kept as
provenance for how those frozen lists were derived and audited, and remain
runnable standalone for inspection, but are not on the path main() takes.
"""
import json
import os
import random
import sys
from collections import Counter

from lemminflect import getInflection
from wordfreq import zipf_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "countable_uncountable.json")
sys.path.insert(0, HERE)  # for `from _frozen_lists import ...` in main()

# ---------------------------------------------------------------------------
# Pass 1: curated uncountable candidates (clear mass/abstract nouns only).
# ---------------------------------------------------------------------------
UNCOUNTABLE_CANDIDATES = [
    # information / cognition
    "advice", "information", "knowledge", "wisdom", "evidence", "research",
    "feedback", "guidance", "insight", "nonsense",
    # abstract qualities / character traits
    "courage", "bravery", "cowardice", "patience", "honesty", "sincerity",
    "generosity", "kindness", "cruelty", "greed", "laziness", "curiosity",
    "enthusiasm", "confidence", "arrogance", "humility", "modesty", "vanity",
    "charisma", "optimism", "pessimism", "tolerance", "compassion", "empathy",
    "sympathy", "gratitude", "loyalty", "honor", "dignity", "integrity",
    "wit", "sportsmanship", "citizenship", "ownership", "authorship",
    "craftsmanship", "workmanship",
    # emotions / states
    "happiness", "sadness", "anger", "anxiety", "joy", "grief", "pride",
    "shame", "guilt", "jealousy", "loneliness", "boredom", "nostalgia",
    "excitement", "relief", "frustration", "contentment", "misery",
    "despair", "freedom", "poverty", "prosperity", "chaos", "harmony",
    "clarity", "ambiguity", "simplicity", "efficiency",
    # academic subjects / closed activity categories
    "grammar", "vocabulary", "punctuation", "syntax", "linguistics",
    "mathematics", "physics", "chemistry", "biology", "geography",
    "geology", "astronomy", "philosophy", "psychology", "sociology",
    "anthropology", "economics", "ethics", "logic", "gymnastics",
    "aerobics", "acoustics",
    # collective / mass "stuff" nouns
    "luggage", "baggage", "equipment", "machinery", "furniture", "jewelry",
    "clothing", "footwear", "cutlery", "crockery", "stationery", "software",
    "hardware", "scenery", "wildlife", "artillery",
    # work / activity gerunds that are genuinely uncountable
    "homework", "housework", "paperwork", "teamwork", "spending", "funding",
    "staffing", "farming", "banking", "marketing", "advertising", "parking",
    "shopping", "swimming", "boxing", "wrestling", "bowling", "surfing",
    "sailing", "skiing", "fishing", "hunting", "camping", "gambling",
    "smoking", "jogging", "hiking", "cycling", "dancing", "acting",
    "lighting", "heating", "cooling", "plumbing", "wiring", "roofing",
    "flooring", "bedding", "landscaping", "gardening", "mining", "logging",
    "packaging", "catering", "wording", "phrasing", "timing", "planning",
    "seating", "housing",
    # weather / natural phenomena
    "weather", "thunder", "lightning", "sunshine", "sunlight", "moonlight",
    "daylight", "twilight", "darkness", "brightness", "warmth", "humidity",
    "moisture", "dew", "frost", "snow", "rain", "hail", "sleet", "fog",
    "mist", "haze",
    # materials / substances
    "electricity", "energy", "fuel", "petrol", "gasoline", "coal", "timber",
    "lumber", "steel", "iron", "copper", "tin", "zinc", "nickel",
  "aluminum", "titanium", "platinum", "cotton", "wool", "silk", "linen",
    "leather", "velvet", "denim", "nylon", "cement", "concrete", "asphalt",
    "plaster", "granite", "chalk", "rust", "sand", "gravel", "dust", "dirt",
    "mud", "clay", "soil", "ash", "smoke", "steam", "oxygen", "nitrogen",
    "hydrogen", "helium", "carbon", "pollution", "contamination",
    "radiation", "gravity", "friction", "magnetism",
    # mass foods with no natural count-plural sense
    "milk", "flour", "sugar", "honey", "yogurt", "butter", "vinegar",
    "mayonnaise", "ketchup", "mustard", "oatmeal", "popcorn", "spaghetti",
    "macaroni", "spinach", "broccoli", "lettuce", "garlic", "rice",
    "traffic", "mankind", "childhood", "adulthood", "likelihood",
    # second curation pass: more abstract qualities / -isms / social & civic
    # processes / bodily-medical / environmental processes
    "tact", "diplomacy", "hospitality", "chivalry", "secrecy", "privacy",
    "stability", "mobility", "flexibility", "creativity", "productivity",
    "spontaneity", "authenticity", "credibility", "reliability",
    "durability", "sustainability", "accountability", "transparency",
    "consistency", "persistence", "perseverance", "resilience",
    "vulnerability", "sensitivity", "awareness", "mindfulness",
    "spirituality", "morality", "immorality", "legality", "illegality",
    "criminality", "brutality", "hostility", "aggression", "obedience",
    "disobedience", "defiance", "diligence", "sloth", "gluttony",
    "temperance", "chastity", "wickedness", "righteousness", "holiness",
    "sanctity", "purity", "filth", "squalor", "affluence", "abundance",
    "scarcity", "excess", "moderation", "extremism", "fanaticism",
    "patriotism", "nationalism", "racism", "sexism", "feminism",
    "socialism", "capitalism", "communism", "fascism", "tyranny", "anarchy",
    "warfare", "terrorism", "espionage", "sabotage", "treason", "corruption",
    "bribery", "theft", "vandalism", "arson", "negligence", "malpractice",
    "misconduct", "harassment", "discrimination", "bigotry", "xenophobia",
    "homophobia", "misogyny", "propaganda", "censorship", "surveillance",
    "litigation", "legislation", "jurisprudence", "arbitration", "mediation",
    "advocacy", "activism", "volunteering", "philanthropy", "stewardship",
    "guardianship", "custody", "captivity", "imprisonment", "incarceration",
    "confinement", "fitness", "obesity", "malnutrition", "starvation",
    "flooding", "erosion", "deforestation", "desertification",
    "urbanization", "globalization", "industrialization", "modernization",
    "colonization", "immigration", "emigration", "overpopulation",
    "unemployment", "underemployment", "employment", "stagnation",
    "taxation", "circulation", "ventilation", "insulation", "refrigeration",
    "sanitation", "hygiene", "sterilization", "pasteurization",
    "fermentation", "decomposition", "oxidation", "evaporation",
    "condensation", "precipitation", "photosynthesis", "respiration",
    "digestion", "metabolism", "health",
    # third curation pass: more -ness / -ity / -tion abstract nouns
    "boldness", "fairness", "unfairness", "forgiveness", "selfishness",
    "selflessness", "thoughtfulness", "thoughtlessness", "carelessness",
    "recklessness", "stubbornness", "willingness", "unwillingness",
    "eagerness", "restlessness", "hopelessness", "helplessness",
    "homelessness", "blindness", "deafness", "madness", "roughness",
    "smoothness", "softness", "hardness", "dimness", "sharpness",
    "dullness", "staleness", "bitterness", "sweetness", "sourness",
    "spiciness", "richness", "familiarity", "originality", "stupidity",
    "rigidity", "fragility", "solidity", "maturity", "immaturity",
    "immortality", "normality", "imagination", "determination",
    "education",
    # fourth curation pass: sports/activities, spices & staple foods,
    # geology/materials, tech/engineering qualities
    "badminton", "volleyball", "soccer", "tennis", "golf", "rugby",
    "athletics", "karate", "judo", "yoga",
    "syrup", "molasses", "cornstarch", "cinnamon", "nutmeg", "vanilla",
    "basil", "oregano", "thyme", "rosemary", "parsley", "cilantro", "dill",
    "turmeric", "cumin", "paprika", "yeast", "gelatin", "buttermilk",
    "custard", "gravy",
    "cardboard", "styrofoam", "fiberglass", "porcelain", "limestone",
    "sandstone", "bedrock", "magma",
    "bandwidth", "throughput", "latency", "connectivity", "scalability",
    "portability", "accessibility", "compatibility",
    # fifth curation pass: more temperament / emotion / character nouns
    "poise", "grace", "elegance", "sophistication", "vigor", "vitality",
    "stamina", "endurance", "agility", "dexterity", "coordination",
    "wellness", "candor", "frankness", "bluntness", "insolence",
    "impudence", "audacity", "gumption", "grit", "toughness", "resolve",
    "willpower", "zeal", "fervor", "devotion", "adoration", "affection",
    "infatuation", "lust", "envy", "spite", "malice", "resentment",
    "contempt", "disdain", "scorn", "mockery", "ridicule", "humiliation",
    "embarrassment", "awkwardness", "clumsiness", "gracefulness",
    "shyness", "timidity", "sociability", "friendliness", "aloofness",
    "detachment", "indifference", "apathy", "lethargy", "drowsiness",
    "fatigue", "exhaustion", "tiredness", "sleepiness", "insomnia",
    "alertness", "vigilance", "caution", "prudence", "impulsiveness",
]

# ---------------------------------------------------------------------------
# Ambiguous "both" nouns: common count AND mass senses, excluded entirely.
# ---------------------------------------------------------------------------
BOTH_EXCLUDE = {
    "coffee", "tea", "paper", "hair", "glass", "chicken", "experience",
    "wood", "fruit", "fish", "meat", "cheese", "wine", "beer", "cake",
    "stone", "brick", "wire", "rope", "string", "chain", "cloth", "plastic",
    "rubber", "gold", "silver", "bronze", "bread", "soup", "stew", "toast",
    "jam", "pepper", "cereal", "marble", "training", "coaching", "testing",
    "screening", "teaching", "writing", "reading", "spelling", "painting",
    "building", "wedding", "meeting", "seasoning",
    "time", "space", "power", "temperature", "pressure", "weight", "height",
    "length", "width", "depth", "size", "speed", "distance", "volume",
    "mass", "currency", "income", "revenue", "profit", "debt", "tax",
    "investment", "business", "industry", "communication", "interaction",
    "negotiation", "discussion", "conversation", "friendship", "membership",
    "partnership", "scholarship", "championship", "neighborhood",
    "falsehood", "livelihood", "hope", "war", "technology", "science",
    "history", "statistics", "dynamics", "electronics", "artwork", "input",
    "output", "support", "admission", "entry", "entrance", "departure",
    "arrival", "shelter", "gas", "oil", "wind", "air", "cold", "wealth",
    "love", "work", "sex", "practice", "food", "public", "china", "past",
    "military", "safe", "positive", "possible", "private", "special",
    "print", "due", "ice", "insurance", "learning", "money", "safety",
}

MIN_ZIPF_PLURAL_UNCOUNTABLE = 3.0  # mechanical cross-check ceiling
MIN_ZIPF_COUNTABLE_SINGULAR = 3.0

# common_nouns.txt is noisy (see singular_or_plural.py) -- reuse the same
# closed-class stopword filter for countable candidates.
STOPWORDS = {
    "that", "this", "these", "those", "you", "your", "yours", "i", "me", "my",
    "mine", "we", "us", "our", "ours", "they", "them", "their", "theirs", "he",
    "him", "his", "she", "her", "hers", "it", "its", "who", "whom", "whose",
    "which", "what", "all", "any", "some", "none", "one", "other", "others",
    "another", "such", "own", "same", "more", "most", "less", "least", "many",
    "much", "few", "several", "both", "each", "either", "neither", "no", "not",
    "nor", "and", "or", "but", "so", "if", "than", "then", "now", "here",
    "there", "where", "when", "why", "how", "well", "still", "even", "just",
    "also", "only", "very", "too", "quite", "rather", "almost", "already",
    "always", "never", "often", "sometimes", "usually", "up", "down", "out",
    "off", "over", "under", "above", "below", "between", "among", "through",
    "during", "before", "after", "since", "until", "while", "because",
    "although", "though", "unless", "inside", "outside", "within", "without",
    "about", "around", "across", "along", "behind", "beyond", "beside",
    "besides", "against", "toward", "towards", "upon", "onto", "into",
    "throughout", "whatever", "whoever", "whenever", "wherever", "however",
    "yes", "no", "ok", "okay", "please", "thanks", "hello", "yeah",
    "anyone", "anything", "everyone", "everything", "someone", "something",
    "nothing", "itself", "himself", "herself", "myself", "yourself",
    "themselves", "ourselves", "yourselves", "whether", "via", "whole",
    "wide", "straight", "super", "western", "west", "east", "north",
    "south", "central", "federal", "natural", "normal", "modern",
    "medical", "personal", "social", "real", "past", "first", "former",
    "daily", "damn", "good", "must", "fast", "sir", "given", "known",
    "living", "married", "running", "saw", "felt", "comes", "coming",
    "doing", "taking", "heavy", "clean", "close",
    "active", "annual", "anti", "ass", "clear", "common", "fit",
    "following", "great", "hell", "last", "nobody", "physical", "plus",
    "simple", "specific", "gay", "grand", "necessary", "sweet", "today",
    "tomorrow", "yesterday",
}


# ---------------------------------------------------------------------------
# 2026-08-12 audit fix: post-filter sets applied AFTER the original
# mechanical draw (see module docstring). These are judgment calls on
# dual count/mass sense or primary part-of-speech, not mechanically
# derivable from zipf/plural checks alone.
# ---------------------------------------------------------------------------
UNCOUNTABLE_DUAL_EXCLUDE = {
    # commonly used with "a"/plural in ordinary speech, despite an abstract
    # mass-noun sense also existing
    "childhood", "adulthood",  # "a happy childhood", "our childhoods"
    "yogurt",  # "a yogurt" (a serving/cup) is everyday retail usage
    "theft",  # "a theft", "several thefts" -- specific-instance sense
}

COUNTABLE_DUAL_OR_NONNOUN_EXCLUDE = {
    # non-nouns, or words whose primary common use is verb/adjective/color,
    # not a countable common noun
    "once", "can", "join", "keep", "kill", "sell", "save", "think", "read",
    "miss", "die", "pick", "build", "buy", "leave", "mean", "meet", "try",
    "wait", "old", "best", "better", "perfect", "particular", "original",
    "multiple", "national", "independent", "local", "main", "regular",
    "extra", "high", "low", "forward", "short", "single", "wrong",
    "welcome", "pro", "make", "find", "trump", "being", "double",
    "holding", "look", "middle", "pain", "play", "take", "feeling",
    "red", "blue", "green", "brown", "white", "black",
    # positional/directional dual nouns (countable body-part/sports sense
    # vs. uncountable direction/faction sense)
    "back", "front", "left", "right",
    # dual count/mass sense nouns: a clear countable meaning coexists with
    # a common, ordinary uncountable/abstract meaning of the same word
    "room", "light", "doubt", "care", "football", "text", "need",
    "fire", "fear", "thought",
    "ability", "access", "action", "addition", "age", "area", "art",
    "attention", "board", "cancer", "capital", "cause", "chance",
    "change", "character", "charge", "class", "code", "color",
    "commission", "company", "competition", "construction", "contact",
    "content", "control", "cost", "cover", "credit", "culture", "dance",
    "death", "defense", "design", "development", "direction", "disease",
    "drink", "drive", "effect", "effort", "energy", "earth", "sun",
    "world", "future", "focus", "film", "force", "form", "growth",
    "ground", "help", "home", "impact", "interest", "justice", "lack",
    "land", "law", "lead", "life", "management", "marriage", "material",
    "matter", "mind", "movement", "nature", "notice", "order", "point",
    "post", "potential", "present", "press", "production", "property",
    "quality", "range", "reach", "respect", "rest", "risk", "rock", "sea",
    "security", "self", "sense", "service", "skin", "society", "sort",
    "sound", "staff", "stock", "study", "style", "success", "talk",
    "touch", "track", "trade", "trust", "truth", "use", "value", "water",
    "will", "wonder", "working", "damage", "blood", "million",
}

# Hand-picked concrete object/creature/person/place nouns with no
# plausible mass/uncountable sense in ordinary usage, used to top the
# countable side back up after COUNTABLE_DUAL_OR_NONNOUN_EXCLUDE removes
# the polysemous top-frequency words above.
CONCRETE_COUNTABLE_CANDIDATES = """
uncle aunt cousin nephew niece grandfather grandmother grandson granddaughter
neighbor stranger tourist passenger driver pilot sailor soldier nurse
lawyer engineer scientist artist painter writer poet musician actor
dancer singer athlete farmer carpenter electrician plumber mechanic
chef baker butcher tailor barber librarian journalist photographer
architect dentist surgeon pharmacist veterinarian firefighter detective
spy criminal prisoner hostage refugee immigrant citizen candidate
senator governor mayor ambassador diplomat monarch queen prince princess
knight warrior pirate wizard witch giant dwarf elf dragon monster ghost
alien robot astronaut inventor explorer pioneer hero villain champion
opponent rival ally colleague employee employer client patient victim
witness suspect defendant attorney juror spectator viewer listener
elephant giraffe lion tiger wolf fox rabbit squirrel mouse whale
dolphin shark eagle owl sparrow penguin kangaroo koala zebra camel
goat sheep horse cow pig duck goose turkey pigeon crow raven hawk
falcon insect worm snail frog toad lizard snake turtle crab lobster
shrimp octopus jellyfish starfish ant bee butterfly spider
tree flower rose tulip daisy oak pine cactus fern mushroom weed
bicycle motorcycle truck airplane helicopter boat submarine rocket
spaceship scooter wagon sled canoe kayak raft
castle palace cottage cabin barn warehouse factory tower tunnel
skyscraper mansion apartment cathedral temple mosque monastery
lighthouse windmill fountain monument statue bridge valley hill
cliff cave peninsula volcano lake mountain
moon comet asteroid galaxy
bottle jar basket barrel bucket crate envelope folder
guitar piano violin trumpet drum
pencil notebook backpack umbrella mirror lamp candle clock calendar
telephone television refrigerator oven stove blender toaster vacuum
ladder hammer wrench screwdriver screw bolt wheel engine battery
camera telescope microscope keyboard monitor printer speaker headphone
remote jacket sweater scarf glove mitten boot sandal sneaker belt
necklace bracelet ring earring button zipper pocket spoon fork knife
plate bowl mug pan pot kettle tray eraser ruler textbook chalkboard
marker stapler racket bat helmet skateboard surfboard paddle
""".split()

MIN_ZIPF_CONCRETE_CANDIDATE = 2.0


def build_uncountable():
    kept = []
    dropped = []
    for w in UNCOUNTABLE_CANDIDATES:
        if w in BOTH_EXCLUDE:
            continue
        pl = getInflection(w, "NNS")
        pl0 = pl[0] if pl else w
        # Only a plural form that actually DIFFERS from the singular and is
        # itself well-attested indicates real countable/plural usage in the
        # wild; pl0 == w (lemminflect has no distinct plural surface form)
        # is itself evidence FOR uncountability, not against it.
        if pl0 != w and zipf_frequency(pl0, "en") >= MIN_ZIPF_PLURAL_UNCOUNTABLE:
            dropped.append((w, pl0, zipf_frequency(pl0, "en")))
            continue
        kept.append(w)
    if dropped:
        print("Dropped uncountable candidates (plural too well-attested):")
        for w, pl0, z in dropped:
            print(f"  {w} -> {pl0} (zipf={z:.2f})")
    # dedup, preserve order
    seen = set()
    out = []
    for w in kept:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def build_countable(exclude, n_needed):
    with open(os.path.join(RES, "common_nouns.txt")) as f:
        nouns = [w.strip() for w in f if w.strip()]
    out = []
    seen = set()
    for w in nouns:
        if not (w.isalpha() and w.islower() and len(w) >= 3):
            continue
        if w in STOPWORDS or w in exclude or w in seen:
            continue
        if zipf_frequency(w, "en") < MIN_ZIPF_COUNTABLE_SINGULAR:
            continue
        pl = getInflection(w, "NNS")
        if not pl:
            continue
        pl0 = pl[0]
        if pl0 == w:
            continue  # invariant plural -- skip for a clean countable example
        seen.add(w)
        out.append(w)
        if len(out) >= n_needed:
            break
    return out


def build_concrete_supplement(exclude, n_needed):
    out = []
    seen = set()
    for w in CONCRETE_COUNTABLE_CANDIDATES:
        if w in exclude or w in seen:
            continue
        pl = getInflection(w, "NNS")
        pl0 = pl[0] if pl else None
        if pl0 is None or pl0 == w:
            continue  # no distinct plural -- skip
        if zipf_frequency(w, "en") < MIN_ZIPF_CONCRETE_CANDIDATE:
            continue
        seen.add(w)
        out.append(w)
    random.seed(42)
    if len(out) > n_needed:
        out = sorted(random.sample(out, n_needed))
    return out


def main() -> None:
    # Emit the frozen, already-audited word lists directly -- see the
    # 2026-08-12 note in the module docstring for why main() does not
    # recompute the mechanical draw at runtime.
    from _frozen_lists import FINAL_COUNTABLE, FINAL_UNCOUNTABLE

    uncountable = sorted(set(FINAL_UNCOUNTABLE))
    countable = sorted(set(FINAL_COUNTABLE))
    n_unc = len(uncountable)
    n_cnt = len(countable)

    data = [{"input": w, "output": "uncountable"} for w in uncountable]
    data += [{"input": w, "output": "countable"} for w in countable]

    random.seed(42)
    random.shuffle(data)

    # Self-checks
    inputs = [d["input"] for d in data]
    assert len(set(inputs)) == len(data), "duplicate inputs"
    counts = Counter(d["output"] for d in data)
    lo, hi = min(counts.values()), max(counts.values())
    assert hi - lo <= 0.10 * len(data) + 1, f"class imbalance: {counts}"
    for d in data:
        assert d["input"] == d["input"].strip()
        assert d["output"] in ("countable", "uncountable")
        assert d["input"] != d["output"]

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)

    issues = None
    if len(data) < 1000:
        issues = (
            f"Only {len(data)} reliable items (n_unc={n_unc}, n_cnt={n_cnt}) -- "
            "English has a genuinely closed set of common, unambiguous "
            "mass/uncountable nouns, and the audit-strict dual-sense exclusion "
            "left too few high-confidence countable/uncountable nouns; did not "
            "pad with guesses to reach 1000."
        )
    print(f"wrote {len(data)} to {OUT}; classes={dict(counts)}")
    if issues:
        print("ISSUES:", issues)


if __name__ == "__main__":
    main()
