#!/usr/bin/env python3
"""Generator for word_polarity (spec index 95).

Rule: given a strongly evaluative English word, output whether its
connotation is positive or negative.

NLTK's opinion_lexicon is not available in this environment (no internet /
no nltk package), so per the spec's documented fallback we hand-curate the
word lists directly (LLM-curated, as this generator script itself is the
reproducible artifact for this knowledge task). Only unambiguous, strongly
polar adjectives (mostly) are included -- no ironic/slangy double-meanings
(e.g. "sick", "killer", "cheap" are deliberately excluded).

Frequency filter: wordfreq zipf_frequency >= 2.5 (spec threshold), keeping
the most frequent / best-known words in each class, balanced 500/500.
"""
import json
import os
import random

from wordfreq import zipf_frequency

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "word_polarity.json")
N = 1000
MIN_ZIPF = 2.5

random.seed(42)

# Strongly, unambiguously POSITIVE-connotation words.
POSITIVE = [
    "wonderful", "delightful", "fantastic", "amazing", "excellent", "terrific",
    "marvelous", "superb", "outstanding", "brilliant", "splendid", "magnificent",
    "glorious", "gorgeous", "beautiful", "lovely", "charming", "adorable",
    "admirable", "exceptional", "remarkable", "impressive", "incredible",
    "awesome", "fabulous", "phenomenal", "extraordinary", "exquisite",
    "elegant", "graceful", "radiant", "dazzling", "stunning", "breathtaking",
    "spectacular", "sensational", "perfect", "flawless", "ideal", "supreme",
    "sublime", "divine", "heavenly", "blissful", "joyful", "joyous", "happy",
    "cheerful", "delighted", "thrilled", "elated", "ecstatic", "jubilant",
    "euphoric", "satisfied", "pleased", "grateful", "thankful",
    "blessed", "fortunate", "lucky", "hopeful", "optimistic", "confident",
    "proud", "triumphant", "victorious", "successful", "accomplished",
    "capable", "competent", "skillful", "talented", "gifted", "intelligent",
    "wise", "clever", "insightful", "thoughtful", "considerate", "caring",
    "compassionate", "kind", "kindly", "gentle", "tender", "loving",
    "affectionate", "devoted", "loyal", "faithful", "trustworthy", "honest",
    "sincere", "genuine", "authentic", "reliable", "dependable", "responsible",
    "dedicated", "committed", "generous", "charitable", "selfless",
    "altruistic", "helpful", "supportive", "encouraging", "inspiring",
    "motivating", "uplifting", "empowering", "friendly", "sociable",
    "amiable", "agreeable", "pleasant", "courteous", "polite", "respectful",
    "gracious", "hospitable", "welcoming", "cordial", "genial", "cheery",
    "jolly", "merry", "lively", "vibrant", "energetic", "enthusiastic",
    "passionate", "eager", "keen", "ambitious", "determined", "resilient",
    "courageous", "brave", "bold", "heroic", "valiant", "fearless", "strong",
    "powerful", "robust", "sturdy", "healthy", "vigorous", "refreshing",
    "invigorating", "soothing", "calming", "peaceful", "tranquil", "serene",
    "harmonious", "balanced", "stable", "secure", "safe", "comfortable",
    "cozy", "relaxing", "delicious", "tasty", "delectable", "scrumptious",
    "savory", "luscious", "appetizing", "fragrant", "aromatic",
    "clean", "pure", "pristine", "spotless", "immaculate", "tidy", "neat",
    "orderly", "organized", "efficient", "effective", "productive",
    "valuable", "worthwhile", "beneficial", "advantageous", "useful",
    "practical", "convenient", "affordable", "economical", "abundant",
    "plentiful", "bountiful", "rich", "wealthy", "prosperous", "thriving",
    "flourishing", "promising", "favorable", "positive", "constructive",
    "progressive", "innovative", "creative", "imaginative", "original",
    "unique", "notable", "noteworthy", "prestigious", "distinguished",
    "renowned", "celebrated", "acclaimed", "esteemed", "respected",
    "honored", "admired", "beloved", "cherished", "treasured", "precious",
    "priceless", "invaluable", "memorable", "unforgettable", "enchanting",
    "magical", "captivating", "fascinating", "intriguing", "engaging",
    "entertaining", "amusing", "hilarious", "funny", "witty", "humorous",
    "playful", "exciting", "thrilling", "exhilarating", "adventurous",
    "smart", "astute", "perceptive", "sensible", "rational",
    "logical", "reasonable", "fair", "equitable", "impartial", "unbiased",
    "accurate", "precise", "correct", "valid", "legitimate", "credible",
    "solid", "firm", "steady", "consistent", "coherent", "lucid",
    "articulate", "eloquent", "persuasive", "compelling", "convincing",
    "influential", "significant", "essential", "vital", "crucial",
    "fundamental", "worthy", "deserving", "exemplary", "good", "great",
    "nice", "super", "sweet", "pretty", "handsome",
    "attractive", "appealing", "alluring", "glamorous", "stylish",
    "fashionable", "chic", "refined", "polished", "sophisticated",
    "cultured", "civilized", "wholesome", "nutritious", "nourishing",
    "hearty", "durable", "tough", "hardy", "benevolent",
    "humane", "merciful", "forgiving", "understanding", "patient",
    "tolerant", "flexible", "adaptable", "versatile", "resourceful",
    "inventive", "ingenious", "winning", "masterful", "proficient", "adept",
    "diligent", "spirited",
    "easygoing", "plucky", "spunky", "feisty", "zealous", "fervent", "ardent",
    "devout", "pious", "virtuous", "righteous", "upright", "upstanding",
    "principled", "ethical", "moral", "decent", "respectable", "reputable",
    "honorable", "dignified", "noble", "gallant", "chivalrous", "tactful",
    "diplomatic", "discreet", "prudent", "judicious", "shrewd", "savvy",
    "canny", "qualified", "veteran", "seasoned", "experienced",
    "knowledgeable", "learned", "scholarly", "erudite", "educated",
    "informed", "fluent", "poetic", "lyrical", "musical", "melodic",
    "tuneful", "rhythmic", "fluid", "smooth", "sleek", "tasteful", "classy",
    "luxurious", "opulent", "lavish", "sumptuous", "plush", "snug", "homey",
    "quaint", "picturesque", "scenic", "idyllic", "pastoral", "quiet",
    "restful", "mellow", "easy", "effortless", "seamless", "streamlined",
    "optimized",
    "groundbreaking", "pioneering", "revolutionary", "transformative",
    "impactful", "meaningful", "purposeful", "fulfilling", "rewarding",
    "gratifying", "enjoyable", "pleasurable", "engrossing", "absorbing",
    "riveting", "gripping", "suspenseful", "dramatic", "epic", "legendary",
    "iconic", "timeless", "classic", "definitive", "authoritative",
    "comprehensive", "thorough", "meticulous", "careful", "rigorous",
    "disciplined", "methodical", "systematic", "prolific", "fruitful",
    "lucrative", "profitable", "affluent", "giving", "magnanimous",
    "empathetic", "sympathetic", "attentive", "nurturing", "protective",
    "reassuring", "comforting", "heartwarming", "touching", "moving",
    "poignant", "sentimental", "romantic", "steadfast", "unwavering",
    "punctual", "prompt", "timely", "agile", "nimble", "swift",
    "speedy", "rapid", "brisk", "dynamic", "vivacious", "animated",
    "exuberant", "buoyant", "upbeat", "sunny", "glowing",
    "beaming", "gleaming", "shining", "sparkling", "glittering",
    "glistening", "luminous", "vivid", "colorful", "lush", "verdant",
    "fertile", "impeccable", "peerless", "unrivaled", "unparalleled",
    "unmatched", "matchless", "superior", "premium", "stellar", "wondrous",
    "commendable", "laudable", "praiseworthy", "meritorious", "sterling",
    "unbeatable", "invincible", "unstoppable", "indomitable", "tenacious",
    "persistent", "industrious", "hardworking", "conscientious",
    "scrupulous",
]

# Strongly, unambiguously NEGATIVE-connotation words.
NEGATIVE = [
    "horrible", "terrible", "awful", "dreadful", "atrocious", "appalling",
    "disgusting", "revolting", "repulsive", "repugnant", "vile", "foul",
    "nasty", "hideous", "grotesque", "ghastly", "monstrous", "horrific",
    "horrendous", "abominable", "despicable", "detestable", "loathsome",
    "contemptible", "deplorable", "shameful", "disgraceful", "dishonorable",
    "corrupt", "wicked", "evil", "sinister", "malicious", "malevolent",
    "vicious", "cruel", "brutal", "savage", "barbaric", "ruthless",
    "merciless", "heartless", "callous", "harsh", "severe", "hostile",
    "aggressive", "violent", "dangerous", "threatening", "menacing",
    "ominous", "terrifying", "frightening", "scary", "horrifying",
    "disturbing", "alarming", "shocking", "distressing", "upsetting",
    "troubling", "worrying", "depressing", "saddening", "tragic",
    "devastating", "catastrophic", "disastrous", "calamitous", "ruinous",
    "destructive", "harmful", "hazardous", "toxic", "poisonous", "lethal",
    "deadly", "fatal", "painful", "agonizing", "excruciating", "unbearable",
    "insufferable", "intolerable", "miserable", "wretched", "pitiful",
    "pathetic", "hopeless", "helpless", "defenseless", "vulnerable", "weak",
    "feeble", "frail", "fragile", "sickly", "unhealthy", "diseased",
    "infected", "contaminated", "polluted", "filthy", "dirty", "grimy",
    "grubby", "squalid", "unsanitary", "unhygienic", "rotten", "rancid",
    "putrid", "moldy", "stale", "spoiled", "decayed", "rotting",
    "unpleasant", "uncomfortable", "awkward", "embarrassing", "humiliating",
    "degrading", "demeaning", "insulting", "offensive", "rude", "impolite",
    "disrespectful", "arrogant", "condescending", "contemptuous", "scornful",
    "dismissive", "indifferent", "apathetic", "careless", "negligent",
    "irresponsible", "reckless", "foolish", "stupid", "idiotic", "ignorant",
    "incompetent", "inept", "clumsy", "sloppy", "messy", "disorganized",
    "chaotic", "confusing", "bewildering", "baffling", "frustrating",
    "annoying", "irritating", "aggravating", "exasperating", "infuriating",
    "maddening", "tedious", "boring", "dull", "tiresome", "monotonous",
    "mundane", "bland", "flavorless", "tasteless", "unappetizing",
    "inedible", "worthless", "useless", "pointless", "futile",
    "meaningless", "insignificant", "trivial", "petty", "shallow",
    "superficial", "fake", "phony", "fraudulent", "deceptive", "dishonest",
    "deceitful", "untrustworthy", "unreliable", "unfaithful", "disloyal",
    "treacherous", "traitorous", "manipulative", "exploitative", "selfish",
    "greedy", "stingy", "miserly", "jealous", "envious", "resentful",
    "bitter", "spiteful", "vindictive", "vengeful", "hateful",
    "antagonistic", "combative", "quarrelsome", "argumentative",
    "stubborn", "obstinate", "uncooperative", "unreasonable", "irrational",
    "illogical", "nonsensical", "absurd", "ridiculous", "ludicrous",
    "laughable", "sad", "unhappy", "gloomy", "glum", "melancholy",
    "sorrowful", "mournful", "heartbroken", "devastated", "despondent",
    "dejected", "discouraged", "disheartened", "demoralized", "defeated",
    "crushed", "broken", "damaged", "ruined", "destroyed", "wrecked",
    "shattered", "corrupted", "degraded", "deteriorated", "declining",
    "failing", "faltering", "struggling", "poor", "inferior", "subpar",
    "mediocre", "deficient", "inadequate", "insufficient", "lacking",
    "flawed", "defective", "faulty", "imperfect", "unsatisfactory",
    "disappointing", "underwhelming", "unimpressive", "forgettable",
    "uninspiring", "unremarkable", "drab", "dreary", "bleak", "grim",
    "dismal", "dire", "desperate", "bad", "nasty", "gross",
    "creepy", "spooky", "vulgar", "obnoxious", "smelly", "shabby",
    "seedy", "grubby", "grim", "toxic", "cruddy", "junky", "grimy",
    "lousy", "crummy", "shoddy", "abysmal", "wretched", "vile",
    "distasteful", "unwelcome", "unwanted", "unbearable", "insulting",
    "hurtful", "brutal", "gruesome", "macabre", "morbid", "sordid",
    "sneaky", "shady", "crooked", "criminal", "illegal", "unlawful",
    "illicit", "immoral", "unethical", "unprincipled", "disreputable",
    "villainous", "diabolical", "demonic", "fiendish", "nefarious",
    "conniving", "scheming", "devious", "underhanded", "untruthful",
    "lying", "bogus", "counterfeit", "sham", "vain",
    "egotistical", "narcissistic", "conceited", "boastful", "pompous",
    "pretentious", "snobbish", "haughty", "aloof", "unfeeling",
    "unsympathetic", "unkind", "abusive", "oppressive",
    "tyrannical", "dictatorial", "authoritarian", "domineering",
    "controlling", "possessive", "clingy", "needy", "whiny", "whining",
    "complaining", "grumpy", "cranky", "irritable", "moody",
    "temperamental", "volatile", "unstable", "erratic", "unpredictable",
    "disorderly", "unruly", "rebellious", "defiant", "disobedient",
    "insubordinate", "impertinent", "insolent", "brazen", "shameless",
    "indecent", "obscene", "lewd", "crude", "coarse", "tacky", "gaudy",
    "garish", "pesky", "bothersome", "troublesome", "problematic",
    "exhausting", "draining", "taxing", "stressful", "overwhelming",
    "burdensome", "cumbersome", "inconvenient", "impractical",
    "inefficient", "ineffective", "unproductive", "wasteful", "excessive",
    "exorbitant", "outrageous", "unjust", "unfair", "biased", "prejudiced",
    "discriminatory", "bigoted", "racist", "sexist", "intolerant", "rigid",
    "inflexible", "uncompromising", "hardheaded", "disagreeable",
    "cantankerous", "sulky", "sullen", "morose", "brooding", "somber",
    "forlorn", "desolate", "lonely", "isolated", "abandoned", "neglected",
    "forsaken", "rejected", "unloved", "friendless", "powerless",
    "impotent", "incapable", "unqualified", "unfit", "unsuitable",
    "inappropriate", "unacceptable", "torturous", "tormenting",
    "harrowing", "traumatic", "unsettling", "disquieting", "eerie",
    "intimidating", "petrifying", "scandalous", "notorious", "infamous",
    "unsavory", "sleazy", "slimy", "greasy", "noxious",
    "unpalatable", "nauseating", "sickening",
    "bad", "wrong", "ugly", "dumb", "lazy", "lame", "angry", "mad",
    "scared", "afraid", "noisy", "unfriendly", "yucky", "icky",
    "embarrassed", "ashamed", "disappointed", "discontent", "unsatisfied",
    "displeased", "offended", "aggrieved", "wronged", "victimized",
    "persecuted", "harassed", "bullied", "mocked", "ridiculed",
    "belittled", "disrespected", "betrayed", "cheated", "deceived",
    "exploited", "enslaved", "imprisoned", "trapped", "confined",
    "restricted", "hindered", "obstructed", "blocked", "thwarted",
    "foiled", "beaten", "wounded", "injured", "bruised", "scarred",
    "impaired", "clumsy", "unlucky", "unfortunate", "regrettable",
    "unwise", "reckless", "shabby", "grubby",
    "crass", "boorish", "churlish", "surly", "gruff", "abrasive",
    "acrimonious", "venomous", "poisonous", "toxic", "corrosive",
    "unforgiving", "punishing", "grueling", "backbreaking", "relentless",
    "merciless", "savage", "barbarous", "inhumane", "monstrous",
    "godawful", "putrid", "revolting", "wicked", "vindictive",
    "spiteful", "loathsome", "odious", "abhorrent", "repellent",
    "obnoxious", "unbearable", "distressing", "malignant", "corrosive",
]


def dedupe_preserve_order(words):
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


pos = dedupe_preserve_order(POSITIVE)
neg = dedupe_preserve_order(NEGATIVE)

overlap = set(pos) & set(neg)
assert not overlap, f"words in both lists: {overlap}"

# frequency filter, sorted by descending zipf (prefer the best-known words)
pos_scored = sorted(
    ((w, zipf_frequency(w, "en")) for w in pos if zipf_frequency(w, "en") >= MIN_ZIPF),
    key=lambda x: -x[1],
)
neg_scored = sorted(
    ((w, zipf_frequency(w, "en")) for w in neg if zipf_frequency(w, "en") >= MIN_ZIPF),
    key=lambda x: -x[1],
)
print(f"Positive candidates after zipf>={MIN_ZIPF} filter: {len(pos_scored)}")
print(f"Negative candidates after zipf>={MIN_ZIPF} filter: {len(neg_scored)}")

PER_CLASS = N // 2
assert len(pos_scored) >= PER_CLASS, f"only {len(pos_scored)} positive words, need {PER_CLASS}"
assert len(neg_scored) >= PER_CLASS, f"only {len(neg_scored)} negative words, need {PER_CLASS}"

pos_final = [w for w, _ in pos_scored[:PER_CLASS]]
neg_final = [w for w, _ in neg_scored[:PER_CLASS]]

examples = [{"input": w, "output": "positive"} for w in pos_final]
examples += [{"input": w, "output": "negative"} for w in neg_final]

random.shuffle(examples)

# ---- checks ----
assert len(examples) == N
inputs = [e["input"] for e in examples]
assert len(set(inputs)) == N, "inputs not unique"
outputs = set(e["output"] for e in examples)
assert outputs == {"positive", "negative"}
n_pos = sum(1 for e in examples if e["output"] == "positive")
n_neg = N - n_pos
assert abs(n_pos - n_neg) <= N * 0.10, "classes not balanced within 10%"
for e in examples:
    assert e["input"] == e["input"].strip() and e["output"] == e["output"].strip()
    assert e["input"].isalpha()

# rule self-check: re-derive from source lists
pos_set = set(pos_final)
neg_set = set(neg_final)
for e in examples:
    if e["input"] in pos_set:
        assert e["output"] == "positive"
    else:
        assert e["input"] in neg_set
        assert e["output"] == "negative"

print(f"Generated {len(examples)} examples: {n_pos} positive / {n_neg} negative")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(examples, f, indent=2)
print(f"Wrote {OUT_PATH}")
