#!/usr/bin/env python3
"""Generator for natural_manmade task.

Rule: classify a concrete noun as naturally occurring ("natural") or made
by humans ("manmade"). Word lists are hand-curated for high confidence.

Deliberately excluded per spec: cultivated/processed borderline items
(garden, farm, bread, paper, pearl), abstract nouns, and places that mix
both senses (park, harbor). Animals and plants are always treated as
natural (per spec's default).
"""
import json
import random

ANIMALS = """
dog cat horse cow pig sheep goat chicken duck goose turkey rabbit mouse rat
deer elk moose bear wolf fox lion tiger leopard cheetah elephant giraffe
zebra rhino hippo camel kangaroo koala panda gorilla monkey chimpanzee
orangutan baboon squirrel chipmunk beaver otter raccoon skunk hedgehog bat
whale dolphin shark seal walrus octopus squid crab lobster shrimp jellyfish
starfish snail slug worm spider ant bee wasp fly mosquito beetle butterfly
moth dragonfly grasshopper cricket cockroach ladybug caterpillar eagle hawk
falcon owl sparrow robin crow raven pigeon dove parrot penguin ostrich
peacock swan flamingo stork pelican seagull woodpecker hummingbird cardinal
finch canary snake lizard crocodile alligator turtle tortoise frog toad
salamander newt iguana chameleon gecko cobra python viper jaguar panther lynx
cougar puma hyena jackal mole ferret weasel mink llama alpaca buffalo bison
antelope gazelle impala hare donkey mule ox calf lamb kitten puppy foal cub
chick tick flea termite centipede millipede scorpion clam oyster mussel
sardine salmon trout tuna cod bass carp catfish eel pike herring minnow
goldfish guppy seahorse stingray swordfish cattle hog boar stallion mare pony
colt heifer drake hen rooster gander tadpole hornet firefly locust mantis
tarantula wolverine badger mongoose armadillo anteater sloth tapir capybara
chinchilla gerbil hamster parakeet cockatoo toucan kingfisher heron ibis
vulture buzzard kite kestrel osprey albatross puffin gull tern quail
pheasant partridge grouse magpie starling thrush wren warbler nightingale
lark swallow swift jay oriole mockingbird earwig aphid weevil aardvark
wombat platypus dingo meerkat lemur gibbon macaque marmoset ocelot serval
caracal kinkajou coati peccary agouti vole shrew pika marmot muskrat opossum
possum quokka wallaby echidna bandicoot dugong manatee narwhal orca porpoise
barracuda marlin mackerel anchovy halibut flounder sole perch walleye
sturgeon grouper snapper barnacle krill silverfish mayfly cicada bittern
egret cormorant loon grebe snipe woodcock plover sandpiper curlew skua
guillemot auk fulmar shearwater petrel booby gannet emu rhea tit chickadee
nuthatch dipper waxwing shrike vireo tanager bunting grosbeak siskin junco
""".split()

PLANTS = """
tree oak pine maple birch willow elm cedar spruce fir palm bamboo cactus
fern moss flower rose tulip daisy lily orchid sunflower daffodil violet
poppy carnation jasmine lavender dandelion clover ivy vine shrub bush weed
grass herb sapling seedling holly mistletoe thistle nettle reed bramble
rhododendron magnolia hibiscus marigold petunia geranium begonia
chrysanthemum hydrangea peony iris crocus buttercup primrose bluebell
foxglove snowdrop azalea camellia gardenia sequoia redwood cypress juniper
poplar aspen sycamore hazel alder beech mahogany teak eucalyptus acacia
mangrove papyrus seaweed kelp bonsai kudzu milkweed goldenrod chicory yarrow
chamomile lupine larkspur columbine aster zinnia cosmos snapdragon phlox
verbena sedum sagebrush heather gorse bracken wisteria clematis honeysuckle
forsythia lilac viburnum boxwood yew larch tamarack cottonwood sumac
buckthorn dogwood hawthorn elderberry blackthorn banyan baobab ceiba kapok
apple banana orange grape lemon lime peach pear plum cherry apricot mango
papaya pineapple coconut watermelon strawberry blueberry raspberry
blackberry cranberry fig olive walnut almond acorn wheat corn rice barley
oat potato carrot onion garlic cabbage lettuce spinach broccoli cucumber
pumpkin tomato pepper radish celery mushroom
""".split()

NATURAL_NONLIVING = """
rock stone pebble boulder mountain hill valley cliff cave canyon desert dune
beach shore coast island peninsula volcano glacier iceberg ice snow rain
hail cloud fog mist wind storm hurricane tornado thunder lightning rainbow
sun moon star planet comet asteroid meteor sky ocean sea lake river stream
brook pond waterfall wave tide current sand mud clay soil dust dirt gravel
mineral crystal gem diamond gold silver copper coal salt lava magma crater
ridge plateau swamp marsh delta gulf bay strait reef lagoon permafrost
tundra savanna prairie meadow bog fen geyser sinkhole fjord isthmus
archipelago mesa butte escarpment moraine floodplain aquifer bedrock
sediment silt limestone granite marble quartz obsidian basalt sandstone
shale slate pumice amber jade ruby emerald sapphire topaz opal platinum
zinc nickel tin lead mercury sulfur phosphorus carbon ozone vapor steam
smoke soot dew frost sleet blizzard monsoon drought flood earthquake
avalanche whirlpool undertow breeze gale gust haze horizon oasis continent
""".split()

MANMADE_OBJECTS = """
hammer bucket chair table desk lamp mirror clock watch phone computer
keyboard television radio camera printer scanner calculator telephone
refrigerator oven stove microwave toaster blender kettle iron vacuum fan
heater speaker headphone guitar piano violin drum trumpet flute clarinet
saxophone trombone harp cello banjo accordion tambourine xylophone book pen
pencil eraser ruler scissors stapler envelope newspaper magazine letter map
key lock chain rope wire nail screw bolt nut wrench screwdriver saw drill
axe shovel rake hoe spade ladder broom mop sponge towel blanket pillow sheet
curtain carpet rug couch sofa bench stool cabinet drawer shelf wardrobe
mattress bed crib cradle door window wall roof floor ceiling gate fence
bridge tower tunnel road highway sidewalk pavement building castle palace
cottage cabin barn shed garage warehouse factory church temple monument
statue pyramid car truck bus van motorcycle bicycle train airplane
helicopter boat ship canoe kayak submarine rocket wagon cart sled sleigh
skateboard scooter wheel engine motor battery generator pump valve pipe tube
hose container box crate barrel jar bottle can cup mug plate bowl dish tray
pot pan spoon fork knife blade sword dagger spear shield armor helmet gun
rifle pistol cannon bomb bullet arrow bow shirt pants jacket coat sweater
hat cap scarf glove sock shoe boot sandal belt tie dress skirt blouse suit
uniform costume mask button zipper umbrella wallet purse bag backpack
suitcase basket doll toy ball kite balloon puzzle dice chessboard coin
ticket stamp medal trophy ring necklace bracelet earring crown flag banner
poster sink faucet bathtub toilet plunger thermostat doorknob hinge latch
padlock staple paperclip folder binder notebook textbook dictionary
encyclopedia calendar whiteboard chalkboard projector microscope telescope
thermometer barometer compass magnet circuit transistor resistor cable
antenna satellite drone elevator escalator staircase railing chimney
fireplace mantel chandelier candle candlestick lantern flashlight torch
lighter ashtray briefcase satchel duffel tent canteen binoculars goggles
harness saddle stirrup bridle leash collar cage kennel aquarium terrarium
birdhouse scarecrow windmill silo trough plow tractor harvester wheelbarrow
pitchfork sickle scythe anvil forge bellows crucible mallet pliers clamp
lathe jackhammer bulldozer forklift conveyor pulley lever gear cog spring
piston cylinder turbine propeller rudder mast sail anchor buoy dock pier
lighthouse parachute glider blimp trailer caravan carriage chariot rickshaw
gondola ferry tugboat yacht raft paddle oar monitor tablet laptop charger
adapter outlet switch dimmer sprinkler trowel trellis birdbath mailbox
doormat hook bookend vase urn chalice goblet decanter pitcher ladle whisk
spatula colander grater peeler corkscrew apron dishwasher dryer needle
thread buckle clasp brooch cufflink wristband bandana veil bonnet beanie
mitten cardigan blazer vest overalls jumpsuit robe kimono poncho cape cloak
shawl metronome tuningfork holepunch indexcard rubberband thumbtack
bulletinboard trampoline treadmill dumbbell barbell racket skis snowboard
surfboard wetsuit mouthguard shinguard jersey cleats tram monorail zipline
modem webcam microphone amplifier turntable cassette videotape harddrive
flashdrive trashcan recyclingbin laundrybasket ironingboard clothesline
hanger level tapemeasure caliper micrometer solderingiron gluegun
slingshot grenade landmine torpedo missile tank dam aqueduct
viaduct overpass underpass scaffolding crane bunker stadium arena
amphitheater colosseum obelisk archway drawbridge portcullis moat
watchtower catapult trebuchet ballista longbow musket bayonet
holster scabbard quiver
""".split()


def dedupe(*lists):
    seen = set()
    out = []
    for lst in lists:
        for w in lst:
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def generate():
    natural_pool = dedupe(ANIMALS, PLANTS, NATURAL_NONLIVING)
    manmade_pool = dedupe(MANMADE_OBJECTS)

    natural_set = set(natural_pool)
    manmade_pool = [w for w in manmade_pool if w not in natural_set]
    manmade_set = set(manmade_pool)
    natural_pool = [w for w in natural_pool if w not in manmade_set]

    random.seed(42)
    random.shuffle(natural_pool)
    random.shuffle(manmade_pool)

    n_each = min(500, len(natural_pool), len(manmade_pool))
    natural_sample = natural_pool[:n_each]
    manmade_sample = manmade_pool[:n_each]

    records = [{"input": w, "output": "natural"} for w in natural_sample]
    records += [{"input": w, "output": "manmade"} for w in manmade_sample]

    random.seed(42)
    random.shuffle(records)
    dataset = records[:1000]

    natural_lookup = set(natural_sample)
    manmade_lookup = set(manmade_sample)
    for item in dataset:
        if item["output"] == "natural":
            assert item["input"] in natural_lookup
        else:
            assert item["input"] in manmade_lookup

    inputs = [d["input"] for d in dataset]
    assert len(inputs) == len(set(inputs))
    assert len(dataset) == 1000

    from collections import Counter
    counts = Counter(d["output"] for d in dataset)
    print("counts:", counts)
    ratio = counts["natural"] / len(dataset)
    assert 0.4 <= ratio <= 0.6, f"class balance out of range: {ratio}"

    return dataset


if __name__ == "__main__":
    dataset = generate()
    print("n =", len(dataset))
    with open("dataset_files/extended_tasks/natural_manmade.json", "w") as f:
        json.dump(dataset, f, indent=2)
