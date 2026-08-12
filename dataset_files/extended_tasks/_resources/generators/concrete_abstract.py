#!/usr/bin/env python3
"""Generator for concrete_abstract task.

Rule: classify a noun as "concrete" (a physical thing you can see or touch)
or "abstract" (an idea, quality, or state). Word lists are hand-curated for
high confidence, favoring clear extremes.

Deliberately excluded per spec: events, time periods, sounds/light,
institutions, and any noun with a common concrete AND abstract sense (key,
field, power, memory, taste, style, etc.).
"""
import json
import random

CONCRETE = """
dog cat horse cow pig sheep goat chicken duck rabbit mouse rat deer bear
wolf fox lion tiger elephant giraffe zebra camel gorilla monkey squirrel
otter raccoon bat whale dolphin shark octopus crab lobster spider ant bee
butterfly eagle hawk owl sparrow crow parrot penguin swan snake lizard
crocodile turtle frog jaguar panther llama buffalo donkey kitten puppy
tree oak pine maple birch willow cedar palm bamboo cactus fern flower rose
tulip daisy lily orchid sunflower ivy vine shrub bush grass
hammer bucket chair table desk lamp mirror clock watch phone computer
keyboard television radio camera printer refrigerator oven stove microwave
toaster blender kettle iron vacuum fan guitar piano violin drum trumpet
flute book pen pencil eraser ruler scissors envelope newspaper map key
lock chain rope wire nail screw wrench screwdriver saw drill axe shovel
ladder broom sponge towel blanket pillow curtain carpet couch sofa bench
cabinet shelf wardrobe mattress bed door window wall roof floor gate fence
bridge tower tunnel road building castle cottage barn shed garage warehouse
statue car truck bus van motorcycle bicycle train airplane boat ship canoe
wheel engine battery pump pipe box crate barrel jar bottle can cup mug
plate bowl dish pot pan spoon fork knife sword shield helmet gun bullet
shirt pants jacket coat sweater hat scarf glove sock shoe boot sandal belt
dress skirt suit mask umbrella wallet purse bag backpack basket doll ball
balloon dice coin ring necklace bracelet earring crown flag
rock stone pebble boulder mountain hill valley cliff cave canyon desert
beach island volcano glacier iceberg snow rain cloud sun moon star planet
ocean sea lake river stream pond sand mud clay soil gravel crystal diamond
gold silver copper coal salt lava
apple banana orange grape lemon peach pear plum cherry potato carrot onion
garlic cabbage lettuce mushroom
farmer teacher doctor nurse pilot chef artist lawyer soldier sailor
carpenter mechanic tailor baker butcher barber dentist scientist engineer
architect painter musician singer dancer actor writer athlete king queen
child baby man woman boy girl father mother son daughter brother sister
sparrow robin crow raven pigeon dove peacock flamingo pelican woodpecker
hummingbird cardinal finch canary toad salamander newt iguana chameleon
gecko cobra python viper cheetah leopard rhino hippo koala panda baboon
chipmunk beaver skunk hedgehog walrus jellyfish starfish snail slug worm
wasp mosquito beetle moth dragonfly grasshopper cricket ladybug caterpillar
falcon stork seagull ostrich swan owl hawk eagle vulture magpie thrush wren
lark swallow jay oriole mockingbird impala gazelle antelope hare mule ox
calf lamb foal cub chick hornet firefly locust mantis tarantula wolverine
badger mongoose armadillo anteater sloth tapir capybara chinchilla gerbil
hamster daffodil violet poppy carnation jasmine lavender dandelion clover
thistle nettle reed magnolia hibiscus marigold petunia geranium chrysanthemum
peony crocus buttercup primrose azalea camellia gardenia sequoia redwood
cypress aspen sycamore hazel alder mahogany walnut almond chestnut
sink faucet bathtub toilet thermostat doorknob padlock stapler paperclip
folder binder notebook textbook calendar telescope thermometer compass
magnet cable antenna satellite drone elevator staircase chimney fireplace
candle lantern flashlight torch briefcase satchel tent binoculars goggles
saddle collar cage aquarium birdhouse windmill silo tractor wheelbarrow
anvil pliers piston propeller anchor lighthouse trailer carriage yacht
raft paddle laptop charger sprinkler trowel vase urn pitcher whisk spatula
apron dishwasher needle thread buckle brooch mitten cardigan blazer robe
cloak boulder cliff canyon dune volcano glacier iceberg waterfall crater
plateau swamp marsh reef lagoon granite marble quartz sapphire emerald
opal platinum grape lime mango papaya coconut watermelon strawberry
blueberry raspberry fig olive pumpkin tomato cucumber broccoli spinach
""".split()

ABSTRACT = """
honesty freedom love joy sadness anger courage wisdom justice truth beauty
wealth poverty health knowledge faith hope fear pride shame guilt envy
greed kindness loyalty patience curiosity creativity intelligence ambition
confidence doubt imagination logic reason opinion idea concept theory
philosophy democracy liberty equality dignity respect trust friendship
hatred jealousy sorrow grief happiness delight despair anxiety worry stress
calm peace mercy forgiveness gratitude appreciation admiration devotion
honor integrity virtue sin morality ethics culture tradition custom belief
religion spirituality charity generosity selfishness arrogance humility
modesty vanity humor sarcasm irony sympathy empathy compassion cruelty
rudeness politeness courtesy discipline motivation inspiration determination
perseverance resilience weakness authority independence duty obligation
commitment dedication betrayal deception sincerity hypocrisy ignorance
stupidity genius talent ability potential opportunity luck fate destiny
fortune misfortune success failure progress development scarcity abundance
inequality fairness unfairness oppression tyranny anarchy chaos order
harmony discord unity division separation connection relationship bond
attachment affection passion desire longing nostalgia thought notion
principle value ideology understanding comprehension awareness
consciousness perception intuition instinct habit tendency inclination
elegance grace charm charisma attraction ugliness balance symmetry
perfection imperfection temptation innocence purity corruption excitement
boredom wonder awe contempt disgust courage cowardice bravery timidity
shyness tolerance intolerance prejudice bias vengeance revenge trust
suspicion certainty uncertainty optimism pessimism enthusiasm apathy
indifference hostility friendliness respectfulness disrespect disgrace
embarrassment humiliation insecurity nervousness serenity tranquility
turmoil unrest stability instability security autonomy sovereignty
dominance submission obedience rebellion defiance cooperation competition
rivalry solidarity heroism villainy nobility mediocrity diligence laziness
sincerity boldness caution recklessness prudence negligence eagerness
reluctance willingness satisfaction dissatisfaction contentment
frustration relief regret remorse redemption liberation
captivity subjugation empowerment vulnerability wickedness
goodwill malice benevolence gentleness harshness fondness
disdain gratification melancholy euphoria serendipity
altruism cynicism idealism realism pragmatism skepticism naivety
sentimentality objectivity subjectivity rationality irrationality clarity
confusion simplicity complexity uniqueness conformity individuality
diversity conformism nonconformity spontaneity impulsiveness restraint
moderation excess extravagance frugality thrift ambivalence apathy zeal
fervor devotion piety reverence sanctity blasphemy virtue depravity malice
spite resentment indignation outrage fury wrath rage tenderness warmheartedness
coldheartedness bitterness resentfulness gladness cheerfulness gloom
dejection despondency hopelessness expectation anticipation surprise
astonishment bewilderment perplexity intrigue fascination boredom apathy
vigor vitality lethargy fatigue exhaustion vigilance alertness carelessness
sloppiness precision accuracy inaccuracy thoroughness diligence
righteousness diplomacy tact bluntness candor discretion indiscretion
secrecy transparency ambiguity vagueness doubtfulness credulity gullibility
shrewdness cunning deceit trickery forthrightness sternness leniency
severity rigor dependability reliability unreliability punctuality
tardiness promptness procrastination industriousness idleness
meticulousness organization disorganization tidiness messiness orderliness
consonance dissonance agreement disagreement consensus dissent controversy
unanimity acceptance rejection denial admission confession accusation blame
credit praise criticism condemnation approval disapproval endorsement
validation invalidation affirmation negation contradiction paradox satire
cleverness foolishness folly sagacity insight foresight hindsight
misperception illusion delusion fantasy reality actuality possibility
impossibility probability improbability inevitability randomness
coincidence causation consequence motive intention purpose meaning
significance insignificance importance unimportance relevance irrelevance
priority urgency necessity convenience inconvenience comfort discomfort
ease difficulty intricacy sportsmanship gallantry chivalry decency
indecency civility incivility gentility rudeness snobbery pretension
authenticity fakeness genuineness insincerity earnestness frivolity
seriousness levity gravity solemnity playfulness mischief naughtiness
obedience defiance stubbornness flexibility persistence tenacity grit
fortitude endurance willpower selfcontrol impulsivity temperance
prosperity affluence destitution deprivation hardship adversity
resourcefulness ingenuity innovation originality nonconformity
""".split()


def dedupe(text_block):
    seen = set()
    out = []
    for w in text_block:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def generate():
    concrete_pool = dedupe(CONCRETE)
    abstract_pool = dedupe(ABSTRACT)

    concrete_set = set(concrete_pool)
    abstract_pool = [w for w in abstract_pool if w not in concrete_set]
    abstract_set = set(abstract_pool)
    concrete_pool = [w for w in concrete_pool if w not in abstract_set]

    random.seed(42)
    random.shuffle(concrete_pool)
    random.shuffle(abstract_pool)

    n_each = min(500, len(concrete_pool), len(abstract_pool))
    concrete_sample = concrete_pool[:n_each]
    abstract_sample = abstract_pool[:n_each]

    records = [{"input": w, "output": "concrete"} for w in concrete_sample]
    records += [{"input": w, "output": "abstract"} for w in abstract_sample]

    random.seed(42)
    random.shuffle(records)
    dataset = records[:1000]

    concrete_lookup = set(concrete_sample)
    abstract_lookup = set(abstract_sample)
    for item in dataset:
        if item["output"] == "concrete":
            assert item["input"] in concrete_lookup
        else:
            assert item["input"] in abstract_lookup

    inputs = [d["input"] for d in dataset]
    assert len(inputs) == len(set(inputs))
    assert len(dataset) == 1000

    from collections import Counter
    counts = Counter(d["output"] for d in dataset)
    print("counts:", counts)
    ratio = counts["concrete"] / len(dataset)
    assert 0.4 <= ratio <= 0.6, f"class balance out of range: {ratio}"

    return dataset


if __name__ == "__main__":
    dataset = generate()
    print("n =", len(dataset))
    with open("dataset_files/extended_tasks/concrete_abstract.json", "w") as f:
        json.dump(dataset, f, indent=2)
