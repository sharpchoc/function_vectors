#!/usr/bin/env python3
"""Generator for hypernym_category (spec index 96).

Rule: given a common noun, output its superordinate category from a fixed
set of 18 labels: beverage, bird, building, clothing, fabric, fish,
flower, fruit, furniture, instrument, metal, profession, sport, tool,
tree, vegetable, vehicle, weapon.

WordNet is not available in this offline environment, so per the spec's
documented fallback this generator hand-curates each category's word list
directly (this script is the reproducible artifact for this knowledge
task). Words that are meaningfully polysemous with a common, unrelated,
high-frequency alternate sense (e.g. "saw" as past tense of "see", "bass"
as low musical frequency, "iris" as eye part, "lead" as verb/leader,
"train" as verb, "tie"/"suit" with their much more common non-clothing
senses, "nickel"/"mercury" as coin/planet, "trailer" as movie preview,
"stool"/"organ" with body-part senses, "cricket" as insect-vs-sport,
"squash" as vegetable-vs-sport, "sake" in "for the sake of", "stout" as
adjective, "foil" as aluminum foil, "pike" as fish/turnpike/peak, "club"
as nightclub/golf club, "mortar" as cement, "cobbler" as dessert, "porter"
as luggage-carrier profession, "library" as software library, "asylum" as
political asylum, "stable" as the adjective, etc.) are deliberately
excluded, per spec's exclusion criterion for polysemous words whose most
frequent sense is outside the category. Words uncertain enough that a
confident call couldn't be made were dropped rather than guessed.

Disjointness from animal_class: per the spec's own exclusion criterion
("NO animal-class labels beyond bird/fish -- keeps it disjoint from the
separate animal_class task"), this generator has no "insect" category
(insect->sport/vegetable collisions like "cricket"/"squash" made that
category's premise shaky anyway) and, at generation time, loads
../../animal_class.json and drops any candidate word that already appears
there as an input -- regardless of which of the two tasks' categories it
would fall under. animal_class.json turns out to hand-curate an unusually
exhaustive species list (~200 words each for bird/fish/insect/mammal/
reptile), so bird and fish keep only the handful of common species names
animal_class didn't already claim (e.g. "swan", "roadrunner", "dodo" for
bird; "whitefish", "grayling", "toadfish" for fish) -- both end up small,
similar in spirit to how metal is naturally capped by how many common
metallic elements exist. The shortfall is made up from the other, larger
categories (see MAX_PER_LABEL below for the balancing cap). A handful of
non-animal words (fruit's "kiwi", instrument's "bongo", profession's
"weaver") also happen to collide with animal_class entries and are
dropped by the same mechanism.

Metal itself is naturally capped (~35-45 confidently-known metallic
elements exist at all), so it stays smaller than most other categories --
called out explicitly in the spec's own generation_recipe and reported
honestly rather than padded with guesses.
"""
import json
import os
import random

from wordfreq import zipf_frequency

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "hypernym_category.json")
ANIMAL_CLASS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "animal_class.json")
N = 1000
MIN_ZIPF = 1.5
MAX_PER_LABEL = int(N * 0.15)  # 150 -- no category may exceed ~15% of the set

random.seed(42)

with open(ANIMAL_CLASS_PATH) as f:
    _animal_class_examples = json.load(f)
ANIMAL_CLASS_INPUTS = set(e["input"] for e in _animal_class_examples)

CATEGORIES = {
    "bird": [
        "sparrow", "robin", "eagle", "hawk", "falcon", "owl", "crow", "raven",
        "dove", "pigeon", "swan", "goose", "duck", "gull", "heron", "stork",
        "flamingo", "peacock", "parrot", "parakeet", "canary", "finch",
        "wren", "swallow", "magpie", "cardinal", "oriole", "woodpecker",
        "kingfisher", "pelican", "penguin", "ostrich", "emu", "turkey",
        "chicken", "rooster", "hen", "quail", "pheasant", "partridge",
        "grouse", "vulture", "condor", "albatross", "cormorant", "egret",
        "ibis", "toucan", "hummingbird", "mockingbird", "bluebird",
        "blackbird", "nightingale", "starling", "kestrel", "buzzard",
        "osprey", "grebe", "warbler", "sandpiper", "plover", "tern",
        "puffin", "macaw", "cockatoo", "chickadee", "bunting", "siskin",
        "chaffinch", "coot", "cockatiel", "petrel",
        # animal_class.json already claims almost every common bird name
        # above (209 species curated there); these are the few common
        # ones it left unclaimed.
        "roadrunner", "killdeer", "dodo", "auk", "bobwhite", "towhee",
        "whippoorwill", "honeyguide", "flycatcher", "capercaillie",
    ],
    "fish": [
        "salmon", "trout", "tuna", "cod", "carp", "catfish", "herring",
        "mackerel", "sardine", "anchovy", "halibut", "flounder", "snapper",
        "grouper", "marlin", "swordfish", "barracuda", "eel", "guppy",
        "goldfish", "koi", "piranha", "stingray", "minnow", "haddock",
        "tilapia", "walleye", "sturgeon", "angelfish", "clownfish",
        "pufferfish", "lionfish", "whitefish", "blowfish", "bream",
        "dogfish", "sunfish", "monkfish", "lamprey", "sailfish", "pollock",
        "whiting", "shark", "kingfish", "sawfish", "betta",
        # same story as bird: animal_class.json already claims almost
        # every common fish name above (191 species curated there).
        "grayling", "burbot", "toadfish", "gudgeon", "orfe", "vendace",
        "lumpfish", "frogfish", "spadefish", "jawfish", "dartfish",
        "escolar",
    ],
    "tree": [
        "oak", "maple", "pine", "birch", "willow", "elm", "cedar", "spruce",
        "fir", "beech", "poplar", "aspen", "cypress", "sycamore", "hickory",
        "magnolia", "dogwood", "redwood", "sequoia", "mahogany", "teak",
        "rosewood", "acacia", "eucalyptus", "mangrove", "larch", "juniper",
        "holly", "yew", "alder", "linden", "balsa", "cottonwood", "boxwood",
        "basswood", "buckeye", "sassafras", "ginkgo", "baobab", "banyan",
        "laurel", "chestnut", "pecan", "hornbeam", "tamarack", "buckthorn",
        "catalpa", "chinaberry", "tupelo", "hackberry", "ironwood",
        "sourwood", "mesquite", "manzanita", "madrone", "sandalwood",
        "ebony", "satinwood", "tamarisk", "walnut",
    ],
    "flower": [
        "tulip", "daisy", "orchid", "daffodil", "carnation", "marigold",
        "sunflower", "petunia", "poppy", "peony", "jasmine", "hyacinth",
        "chrysanthemum", "azalea", "begonia", "geranium", "camellia",
        "gardenia", "hibiscus", "buttercup", "bluebell", "snowdrop",
        "foxglove", "primrose", "dandelion", "zinnia", "dahlia", "aster",
        "crocus", "gladiolus", "amaryllis", "freesia", "bluebonnet",
        "larkspur", "snapdragon", "honeysuckle", "wisteria", "forsythia",
        "hollyhock", "calla", "lotus", "phlox", "clover", "thistle",
        "chamomile", "poinsettia", "cyclamen", "verbena", "lily", "lilac",
        "lavender", "pansy", "gerbera", "anemone", "ranunculus",
        "delphinium", "protea", "bougainvillea", "jonquil", "cowslip",
        "harebell", "cornflower", "edelweiss", "heather", "goldenrod",
        "coneflower", "salvia", "impatiens", "vinca",
    ],
    "fruit": [
        "apple", "banana", "orange", "grape", "strawberry", "blueberry",
        "raspberry", "blackberry", "cherry", "peach", "pear", "plum",
        "mango", "pineapple", "watermelon", "cantaloupe", "papaya", "kiwi",
        "lemon", "lime", "grapefruit", "apricot", "nectarine", "fig",
        "coconut", "pomegranate", "guava", "lychee", "cranberry",
        "gooseberry", "tangerine", "clementine", "persimmon", "melon",
        "honeydew", "currant", "durian", "starfruit", "dragonfruit",
        "passionfruit", "breadfruit", "mangosteen", "kumquat",
        "blackcurrant", "redcurrant", "boysenberry", "elderberry",
        "mulberry", "huckleberry", "quince", "jackfruit", "tamarind",
        "lingonberry", "rambutan", "longan", "feijoa", "soursop",
        "cherimoya", "plantain",
    ],
    "vegetable": [
        "carrot", "potato", "tomato", "onion", "garlic", "broccoli",
        "cauliflower", "cabbage", "lettuce", "spinach", "celery",
        "cucumber", "radish", "beet", "turnip", "parsnip", "pumpkin",
        "zucchini", "eggplant", "asparagus", "artichoke", "kale", "leek",
        "yam", "corn", "pea", "bean", "lentil", "okra", "rutabaga",
        "endive", "arugula", "watercress", "chicory", "fennel", "kohlrabi",
        "jicama", "taro", "cassava", "shallot", "scallion", "chive",
        "collard", "daikon", "chard", "mushroom", "rhubarb", "salsify",
        "celeriac", "escarole", "radicchio", "frisee", "mizuna",
        "beetroot", "gourd", "sorrel", "cress", "fava", "chickpea",
        "soybean", "edamame", "sunchoke",
    ],
    "tool": [
        "hammer", "wrench", "screwdriver", "pliers", "drill", "chisel",
        "clamp", "mallet", "axe", "hatchet", "crowbar", "spanner",
        "sledgehammer", "sander", "pickaxe", "shovel", "rake", "trowel",
        "pitchfork", "tongs", "vise", "awl", "stapler", "shears",
        "scissors", "ratchet", "caliper", "corkscrew", "tweezers",
        "forceps", "clippers", "sawhorse", "plunger", "mop", "broom",
        "dustpan", "spatula", "ladle", "whisk", "grater", "peeler",
        "colander", "funnel", "sieve", "strainer", "blowtorch", "hacksaw",
        "handsaw", "chainsaw", "jackhammer", "bandsaw", "lathe", "anvil",
        "bellows", "wheelbarrow", "ladder", "knife", "pruners", "loppers",
        "hoe", "rasp", "auger", "nailer", "multitool", "crimper",
        "reamer", "grindstone", "whetstone", "planer",
    ],
    "clothing": [
        "shirt", "pants", "dress", "skirt", "jacket", "coat", "sweater",
        "blouse", "trousers", "jeans", "shorts", "socks", "gloves", "scarf",
        "hat", "cap", "vest", "blazer", "cardigan", "hoodie", "pajamas",
        "robe", "kimono", "tunic", "jumpsuit", "overalls", "leggings",
        "tights", "stockings", "mittens", "boots", "shoes", "sandals",
        "sneakers", "slippers", "apron", "veil", "shawl", "cloak", "gown",
        "tuxedo", "parka", "windbreaker", "turtleneck", "beanie",
        "bandana", "earmuffs", "legwarmers", "bodysuit", "wetsuit",
        "raincoat", "galoshes", "moccasin", "loafers", "clogs", "jersey",
        "camisole", "nightgown", "bathrobe", "jumper", "onesie",
        "overcoat", "trenchcoat", "necktie", "belt", "wristband",
        "headband", "poncho", "sarong", "kilt", "toque", "fedora", "beret",
        "balaclava", "sombrero", "turban", "burqa", "hijab", "sari",
        "cummerbund",
    ],
    "furniture": [
        "chair", "table", "sofa", "couch", "bed", "desk", "dresser",
        "cabinet", "shelf", "bookshelf", "wardrobe", "armchair", "bench",
        "recliner", "futon", "nightstand", "cupboard", "headboard",
        "footstool", "loveseat", "armoire", "mattress", "cot", "crib",
        "highchair", "hutch", "sideboard", "credenza", "barstool",
        "daybed", "bunkbed", "beanbag", "hammock", "cradle", "bassinet",
        "playpen", "bookcase", "worktable", "divan", "settee", "banquette",
        "davenport", "commode", "ottoman", "chaise", "etagere",
        "chifforobe", "highboy", "lowboy", "tallboy", "pew", "pouf",
        "escritoire", "sectional", "vitrine", "footrest", "headrest",
        "trestle",
    ],
    "vehicle": [
        "car", "truck", "bus", "motorcycle", "bicycle", "van", "taxi",
        "airplane", "helicopter", "boat", "ship", "ferry", "submarine",
        "scooter", "tractor", "jeep", "limousine", "ambulance", "tram",
        "trolley", "wagon", "cart", "sled", "sleigh", "canoe", "kayak",
        "yacht", "tanker", "minivan", "sedan", "coupe", "hatchback",
        "moped", "glider", "blimp", "unicycle", "carriage", "locomotive",
        "rickshaw", "gondola", "zeppelin", "dirigible", "catamaran",
        "schooner", "raft", "barge", "dinghy", "houseboat", "speedboat",
        "sailboat", "rowboat", "tugboat", "steamship", "warship",
        "battleship", "cruiser", "frigate", "destroyer", "hovercraft",
        "snowmobile", "forklift", "bulldozer", "excavator", "streetcar",
        "monorail", "stagecoach", "buggy", "chariot", "toboggan",
        "stroller", "pram", "tricycle", "airship", "biplane", "seaplane",
        "jetski", "snowplow", "skateboard",
    ],
    "instrument": [
        "guitar", "piano", "violin", "drum", "flute", "trumpet",
        "saxophone", "clarinet", "cello", "trombone", "harp", "banjo",
        "mandolin", "ukulele", "accordion", "harmonica", "tuba", "oboe",
        "bassoon", "xylophone", "tambourine", "cymbal", "synthesizer",
        "viola", "sitar", "bongo", "gong", "lute", "harpsichord",
        "piccolo", "timpani", "marimba", "glockenspiel", "castanets",
        "maracas", "kazoo", "didgeridoo", "zither", "dulcimer", "bagpipe",
        "fife", "ocarina", "concertina", "clavichord", "harmonium",
        "theremin", "vibraphone", "chimes", "cornet", "bugle", "melodica",
        "sousaphone", "flugelhorn", "euphonium", "kalimba", "djembe",
        "autoharp", "steelpan", "celesta", "shofar", "cabasa", "guiro",
        "handpan",
    ],
    "metal": [
        "gold", "silver", "iron", "copper", "aluminum", "tin", "zinc",
        "platinum", "titanium", "bronze", "brass", "steel", "cobalt",
        "chromium", "magnesium", "tungsten", "uranium", "sodium",
        "potassium", "calcium", "manganese", "plutonium", "lithium",
        "palladium", "iridium", "cadmium", "bismuth", "vanadium",
        "zirconium", "cesium", "strontium", "barium", "radium", "thorium",
        "gallium", "indium", "germanium", "tantalum", "niobium",
        "hafnium", "molybdenum", "beryllium", "scandium", "yttrium",
        "rhodium", "ruthenium", "osmium", "rhenium", "antimony",
        "neodymium", "cerium",
    ],
    "profession": [
        "doctor", "teacher", "lawyer", "engineer", "nurse", "dentist",
        "pharmacist", "surgeon", "professor", "scientist", "mathematician",
        "physicist", "chemist", "biologist", "historian", "economist",
        "architect", "plumber", "electrician", "carpenter", "mechanic",
        "mason", "welder", "blacksmith", "tailor", "chef", "baker",
        "butcher", "farmer", "shepherd", "librarian", "journalist",
        "photographer", "sculptor", "musician", "actor", "dancer",
        "singer", "translator", "interpreter", "accountant", "banker",
        "cashier", "waiter", "waitress", "bartender", "barber",
        "therapist", "psychologist", "psychiatrist", "paramedic",
        "firefighter", "detective", "judge", "referee", "umpire",
        "jockey", "lifeguard", "zookeeper", "beekeeper", "locksmith",
        "goldsmith", "silversmith", "gunsmith", "watchmaker", "shoemaker",
        "weaver", "bricklayer", "roofer", "surveyor", "geologist",
        "astronomer", "botanist", "zoologist", "archaeologist",
        "anthropologist", "sociologist", "linguist", "philosopher",
        "theologian", "missionary", "priest", "rabbi", "chaplain",
        "diplomat", "ambassador",
    ],
    "sport": [
        "baseball", "basketball", "football", "soccer", "tennis", "golf",
        "hockey", "volleyball", "badminton", "rugby", "boxing",
        "wrestling", "swimming", "diving", "gymnastics", "cycling",
        "skiing", "snowboarding", "skateboarding", "surfing", "rowing",
        "sailing", "archery", "weightlifting", "bowling", "billiards",
        "darts", "curling", "lacrosse", "handball", "softball",
        "taekwondo", "karate", "judo", "kickboxing", "triathlon",
        "marathon", "decathlon", "pentathlon", "biathlon", "netball",
        "croquet", "shuffleboard", "racquetball", "snooker", "motocross",
        "bobsled", "luge", "parkour", "climbing", "mountaineering",
        "orienteering", "canoeing", "kayaking", "windsurfing",
        "kitesurfing", "paragliding", "skydiving", "snorkeling",
        "wakeboarding", "sprinting", "hurdling", "jogging", "aerobics",
        "pilates", "yoga", "cheerleading", "paddleball", "pickleball",
    ],
    "beverage": [
        "coffee", "tea", "juice", "soda", "lemonade", "milkshake",
        "smoothie", "cocktail", "wine", "beer", "champagne", "whiskey",
        "vodka", "rum", "gin", "tequila", "brandy", "cider", "cognac",
        "mead", "sangria", "margarita", "mojito", "martini", "daiquiri",
        "espresso", "cappuccino", "latte", "mocha", "matcha", "kombucha",
        "buttermilk", "eggnog", "lassi", "horchata", "cola", "ale",
        "lager", "grappa", "absinthe", "vermouth", "schnapps", "bourbon",
        "scotch", "moonshine", "kefir", "chai", "julep", "toddy",
        "sherry", "marsala", "pisco", "ouzo", "cachaca", "soju", "baijiu",
        "grog",
    ],
    "fabric": [
        "cotton", "silk", "wool", "linen", "polyester", "nylon", "denim",
        "velvet", "satin", "chiffon", "corduroy", "tweed", "flannel",
        "fleece", "cashmere", "spandex", "lycra", "rayon", "chambray",
        "gingham", "seersucker", "taffeta", "organza", "brocade",
        "damask", "muslin", "gauze", "tulle", "canvas", "burlap",
        "cambric", "poplin", "twill", "herringbone", "viscose",
        "microfiber", "chintz", "calico", "madras", "voile", "organdy",
        "sailcloth", "percale", "jute", "hemp", "angora", "mohair",
        "cheesecloth", "sackcloth", "gabardine", "crinoline", "batiste",
        "moleskin",
    ],
    "weapon": [
        "sword", "dagger", "spear", "javelin", "lance", "rapier", "saber",
        "cutlass", "machete", "bayonet", "halberd", "crossbow", "longbow",
        "musket", "rifle", "pistol", "revolver", "shotgun", "carbine",
        "bazooka", "cannon", "howitzer", "grenade", "missile", "torpedo",
        "harpoon", "trident", "nunchaku", "tomahawk", "boomerang",
        "slingshot", "catapult", "blowgun", "broadsword", "longsword",
        "scimitar", "katana", "claymore", "flintlock", "blunderbuss",
        "dirk", "glaive", "warhammer", "billhook", "wakizashi", "tanto",
        "naginata", "kukri", "derringer",
    ],
    "building": [
        "house", "mansion", "castle", "palace", "cottage", "cabin",
        "bungalow", "villa", "chalet", "shack", "hut", "tower",
        "skyscraper", "cathedral", "church", "temple", "mosque",
        "synagogue", "chapel", "monastery", "abbey", "convent", "shrine",
        "pagoda", "stadium", "arena", "colosseum", "amphitheater",
        "museum", "warehouse", "factory", "barn", "silo", "greenhouse",
        "lighthouse", "windmill", "courthouse", "prison", "jail",
        "fortress", "citadel", "bunker", "hangar", "garage", "penthouse",
        "apartment", "condominium", "dormitory", "hostel", "motel",
        "hotel", "inn", "tavern", "pub", "cafe", "bakery", "pharmacy",
        "hospital", "clinic", "school", "university", "kindergarten",
        "orphanage", "barracks", "embassy", "pavilion", "gazebo",
        "rotunda", "granary", "aquarium", "planetarium", "observatory",
        "laboratory", "rectory", "belfry", "duplex", "triplex",
        "rowhouse", "townhouse", "farmhouse", "boathouse", "treehouse",
        "clubhouse", "playhouse", "outhouse", "icehouse", "storehouse",
        "almshouse", "guesthouse", "smokehouse", "henhouse", "doghouse",
    ],
}

LABELS = sorted(CATEGORIES.keys())
assert len(LABELS) == 18

# ---- de-dup within and across categories, drop anything animal_class.json
# already uses (disjointness requirement), then frequency filter ----
seen_anywhere = {}
duplicates_across = []
animal_class_excluded = {}
filtered = {}
for label, words in CATEGORIES.items():
    unique_words = []
    local_seen = set()
    excluded_here = []
    for w in words:
        if w in local_seen:
            continue
        local_seen.add(w)
        if w in ANIMAL_CLASS_INPUTS:
            excluded_here.append(w)
            continue
        if w in seen_anywhere:
            duplicates_across.append((w, seen_anywhere[w], label))
            continue
        seen_anywhere[w] = label
        unique_words.append(w)
    if excluded_here:
        animal_class_excluded[label] = excluded_here
    filtered[label] = [w for w in unique_words if zipf_frequency(w, "en") >= MIN_ZIPF]

assert not duplicates_across, f"words duplicated across categories: {duplicates_across}"

n_animal_excluded = sum(len(v) for v in animal_class_excluded.values())
print(f"Excluded {n_animal_excluded} words already used as inputs in animal_class.json "
      f"(disjointness requirement), by category:")
for label, words in animal_class_excluded.items():
    print(f"  {label}: {len(words)} excluded")

for label in LABELS:
    print(f"{label}: {len(CATEGORIES[label])} curated -> {len(filtered[label])} after "
          f"animal_class exclusion + zipf filter")

total_available = sum(len(v) for v in filtered.values())
print(f"Total available after filtering: {total_available}")
assert total_available >= N, (
    f"only {total_available} verified words available across {len(LABELS)} "
    f"categories -- need at least {N}; curate more unambiguous words"
)

# ---- balance classes as evenly as the domain allows, capped at 15% each ----
# Water-filling: repeatedly hand out an even share of the still-unassigned
# budget across categories that still have room (bounded by both their
# available pool and the MAX_PER_LABEL cap), retiring a category once it
# is exhausted or capped, until the full N examples are assigned. This
# generalizes the old "metal gets everything, split the rest evenly"
# special case to an arbitrary number of categories with uneven pools
# (here, bird/fish are the new small ones after the animal_class exclusion).
selected = {label: [] for label in LABELS}
pool_left = {label: len(filtered[label]) for label in LABELS}
cap_left = {label: MAX_PER_LABEL for label in LABELS}
remaining_budget = N
active = set(LABELS)

while remaining_budget > 0 and active:
    share = max(1, remaining_budget // len(active))
    progressed = False
    for label in sorted(active):
        if remaining_budget <= 0:
            break
        take = min(share, pool_left[label], cap_left[label], remaining_budget)
        if take <= 0:
            active.discard(label)
            continue
        start = len(selected[label])
        selected[label].extend(filtered[label][start:start + take])
        pool_left[label] -= take
        cap_left[label] -= take
        remaining_budget -= take
        progressed = True
        if pool_left[label] == 0 or cap_left[label] == 0:
            active.discard(label)
    if not progressed:
        break

assert remaining_budget == 0, (
    f"could not reach {N} examples while respecting the {MAX_PER_LABEL}-per-"
    f"category cap -- {remaining_budget} short; curate more words or raise the cap"
)

for label in LABELS:
    assert len(selected[label]) <= MAX_PER_LABEL
    print(f"selected {label}: {len(selected[label])}")

examples = []
for label, words in selected.items():
    for w in words:
        examples.append({"input": w, "output": label})

random.shuffle(examples)

# ---- checks ----
assert len(examples) == N, f"expected exactly {N} examples, got {len(examples)}"
inputs = [e["input"] for e in examples]
assert len(set(inputs)) == len(examples), "inputs not unique"
outputs = set(e["output"] for e in examples)
assert outputs <= set(LABELS)
for e in examples:
    assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()
    assert e["input"].isalpha()
    assert e["input"] != e["output"]

# disjointness self-check: hypernym_category must not overlap animal_class.json
overlap_with_animal_class = [e["input"] for e in examples if e["input"] in ANIMAL_CLASS_INPUTS]
assert not overlap_with_animal_class, (
    "hypernym_category must stay disjoint from animal_class.json inputs, "
    f"found overlap: {overlap_with_animal_class}"
)
print("Confirmed zero input overlap with animal_class.json")

# rule self-check: re-derive category from the source dict
lookup = {w: label for label, words in CATEGORIES.items() for w in words}
for e in examples:
    assert lookup[e["input"]] == e["output"]

n_final = len(examples)
counts = {}
for e in examples:
    counts[e["output"]] = counts.get(e["output"], 0) + 1
for label, c in counts.items():
    assert c <= MAX_PER_LABEL, f"{label} has {c} examples, exceeds {MAX_PER_LABEL} (15%) cap"

print(f"Generated {n_final} examples (target was {N})")
print("Class counts:", counts)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(examples, f, indent=2)
print(f"Wrote {OUT_PATH}")
