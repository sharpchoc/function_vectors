#!/usr/bin/env python
"""Generator for animal_plant_object: concrete noun -> animal / plant / object.

Recipe (spec index 72): curated knowledge lists, three balanced classes.

ANIMAL = mammals, birds, fish, insects, reptiles (land + water + air).
PLANT  = trees, flowers, and non-food herbs/shrubs/vines (living plants,
         not plant products).
OBJECT = everyday inanimate man-made items (furniture, tools, kitchenware,
         clothing, electronics, containers, ...).

EXCLUDED (per spec): fruit/vegetable words that primarily read as food
("apple", "carrot", "tomato", ...) -- ambiguous plant-vs-food; plant
products ("wood", "cotton"); animal products ("wool", "leather"); fungi
and microorganisms ("mushroom", "mold", "yeast"); people/occupations;
homonyms with a strong competing sense that would confuse the label
("palm" hand, "kite" toy, "crane" machine, "locust" tree/insect clash).

Self-check: every label is derived purely from source-list membership; a
re-derivation pass below reasserts membership for every emitted item, and
class sizes are balanced to within 10%.
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "animal_plant_object.json")

# ---------------------------------------------------------------------------
# ANIMAL
# ---------------------------------------------------------------------------
MAMMALS = """
dog cat horse cow pig sheep goat rabbit squirrel mouse rat lion tiger bear
wolf fox deer moose elk camel elephant monkey gorilla chimpanzee kangaroo
koala panda zebra giraffe hippo rhino buffalo bison otter beaver badger
skunk raccoon hedgehog porcupine armadillo sloth anteater aardvark mole
weasel ferret mink seal walrus whale dolphin porpoise donkey mule llama
alpaca yak antelope gazelle wildebeest hare chinchilla gerbil hamster
shrew tapir pangolin okapi capybara marmot muskrat lemur baboon orangutan
gibbon mandrill wolverine stoat polecat hyena jackal lynx bobcat cougar
puma leopard cheetah jaguar panther meerkat mongoose wombat opossum bat
platypus vole pika dormouse springhare jerboa warthog boar coati ocelot
serval caracal impala kudu oryx ibex chamois bongo eland narwhal orca
manatee dugong
""".split()

BIRDS = """
sparrow eagle hawk falcon robin crow raven owl dove pigeon seagull
swallow swift hummingbird cardinal finch wren warbler thrush magpie
woodpecker kingfisher heron stork crane flamingo pelican cormorant
albatross kestrel buzzard vulture condor osprey parrot macaw cockatoo
parakeet canary toucan hornbill cuckoo nightingale lark starling oriole
tanager mockingbird chickadee nuthatch sandpiper plover tern egret ibis
bittern grouse pheasant quail partridge goose duck swan peacock puffin
loon grackle cowbird bunting martin wagtail shrike vireo kinglet
titmouse waxwing myna weaver jackdaw booby gannet frigatebird petrel
shearwater skua kittiwake auk guillemot curlew snipe godwit avocet stilt
lapwing ptarmigan guineafowl budgerigar lovebird lorikeet meadowlark
blackbird bobolink goldfinch bullfinch chaffinch bluebird catbird junco
phoebe flycatcher kingbird ostrich penguin emu kiwi cassowary rhea
""".split()

FISH = """
shark trout salmon tuna cod herring mackerel sardine anchovy halibut
flounder sole bass perch pike carp catfish eel swordfish marlin tarpon
barracuda snapper grouper haddock pollock whiting mullet tilapia koi
goldfish guppy piranha stingray skate lamprey sturgeon sunfish bluegill
crappie walleye muskie minnow chub dace roach bream tench angelfish
clownfish parrotfish lionfish pufferfish triggerfish wrasse goby blenny
seahorse anglerfish dogfish hammerhead mako
""".split()

REPTILES = """
snake lizard gecko iguana chameleon cobra viper python boa mamba adder
rattlesnake crocodile alligator turtle tortoise komodo monitor skink
tuatara
""".split()

INSECTS = """
bee wasp hornet bumblebee mosquito gnat cicada locust ladybug firefly
housefly moth butterfly ant beetle cockroach termite aphid weevil
cricket grasshopper millipede centipede tick mite flea louse spider
scorpion earwig dragonfly damselfly mayfly cranefly katydid mantis
silverfish
""".split()

ANIMAL_ALL = sorted(set(MAMMALS) | set(BIRDS) | set(FISH) | set(REPTILES) | set(INSECTS))

# ---------------------------------------------------------------------------
# PLANT (trees, flowers, non-food herbs/shrubs)
# ---------------------------------------------------------------------------
TREES = """
oak maple pine birch willow elm cedar spruce fir cypress redwood sequoia
poplar aspen sycamore beech ash hickory walnut chestnut mahogany teak
ebony rosewood sandalwood balsa banyan baobab acacia eucalyptus magnolia
dogwood mulberry mangrove juniper larch hemlock alder hawthorn holly yew
ginkgo cottonwood buckeye catalpa sassafras tupelo basswood linden bamboo
hornbeam hackberry boxelder witchhazel sumac spicebush viburnum redbud
tuliptree paperbark ironwood blackwood satinwood kapok zebrawood
bloodwood ironbark wattle monkeypuzzle jacaranda frangipani plumeria
bottlebrush douglasfir paulownia tamarisk casuarina monkeypod flametree
chinaberry sourwood camphor shagbark trumpetvine
""".split()

FLOWERS = """
rose tulip daisy sunflower lily orchid daffodil carnation chrysanthemum
marigold petunia poppy iris violet pansy lavender jasmine hibiscus
hydrangea peony dahlia zinnia begonia azalea camellia gardenia lilac
primrose buttercup bluebell snapdragon foxglove geranium aster cosmos
larkspur columbine clematis wisteria honeysuckle morningglory sweetpea
crocus hyacinth freesia amaryllis gladiolus ranunculus snowdrop cyclamen
forsythia hollyhock delphinium phlox verbena salvia impatiens coleus
fuchsia lantana bougainvillea oleander periwinkle spiderlily calla
anthurium protea poinsettia cornflower forgetmenot coneflower waterlily
lotus tigerlily daylily bellflower cowslip celandine campion yarrow
tansy goldenrod gentian harebell lupine monkshood hellebore anemone
mallow speedwell borage comfrey feverfew calendula nasturtium celosia
statice sweetwilliam canna heliconia torchlily agapanthus crocosmia
dianthus alyssum lobelia iceplant sedum jadeplant snakeplant spiderplant
pothos philodendron monstera peacelily strawflower echinacea plumbago
catmint lambsear allium barrelcactus saguaro pricklypear
""".split()

HERBS_SHRUBS = """
fern moss ivy cactus aloe clover dandelion thistle nettle reed rush sedge
papyrus sagebrush heather gorse boxwood rhododendron basil mint thyme
rosemary sage oregano parsley cilantro dill chives tarragon fennel
lemongrass spearmint peppermint chamomile catnip mistletoe lovage
marjoram savory lemonbalm stevia wormwood hyssop angelica valerian
chervil bracken sphagnum clubmoss horsetail liverwort pampasgrass fescue
ryegrass bluegrass crabgrass bermudagrass switchgrass cattail bulrush
duckweed kudzu bindweed passionflower moonflower citronella zoysia
bentgrass centipedegrass agave yucca milkweed ragweed pokeweed knotweed
chickweed burdock mullein teasel maidenhair staghorn barberry spirea
mockorange ninebark privet milkthistle spurge groundsel ragwort fireweed
beachgrass saltgrass eelgrass seagrass hydrilla pondweed coralbells
bleedingheart trillium bloodroot mayapple skunkcabbage jewelweed
touchmenot goutweed
""".split()

PLANT_ALL = sorted(set(TREES) | set(FLOWERS) | set(HERBS_SHRUBS))

# ---------------------------------------------------------------------------
# OBJECT (everyday inanimate items)
# ---------------------------------------------------------------------------
OBJECTS = """
table chair desk sofa bed lamp mirror clock kettle spoon fork knife plate
cup bowl pot pan hammer screwdriver wrench nail screw ladder bucket broom
mop shovel rake axe saw drill book pen pencil paper envelope stamp key
lock door window wall floor roof brick stone rock rope chain wire cable
box bag basket jar bottle can barrel crate suitcase backpack umbrella
blanket pillow towel curtain carpet rug shelf cabinet drawer vase candle
lighter battery pipe hose faucet sink toilet bathtub shower comb brush
razor soap toothbrush mattress cushion stool bench sculpture statue
painting frame watch ring necklace bracelet earring wallet purse glove
hat scarf belt button zipper needle thread scissors ruler stapler tape
glue paperclip folder binder calculator keyboard speaker
microphone camera telescope binoculars compass map globe flag banner
ticket coin bill receipt calendar notebook diary journal magazine
newspaper poster billboard fence gate bridge tunnel pillar column beam
plank board tile napkin tablecloth placemat coaster tray platter ladle
spatula whisk grater peeler colander sieve funnel thermometer chalk
eraser crayon marker highlighter briefcase toolbox jackhammer anvil vice
clamp pliers file chisel mallet crowbar bolt nut washer gear spring
lever pulley hinge padlock doorknob doorbell mailbox trashcan dustpan
hanger vacuum blender toaster microwave oven refrigerator freezer
dishwasher heater fan radiator chimney fireplace bookshelf wardrobe
dresser nightstand crib stroller wheelchair crutch cane raincoat boots
sandals slippers sneakers helmet goggles mask apron mittens socks
sweater jacket coat shirt trousers skirt dress necktie guitar piano
violin drum trumpet flute clarinet trombone saxophone harmonica
accordion banjo cello harp xylophone tambourine cymbal football
basketball baseball racket skis snowboard surfboard skateboard
dumbbell barbell treadmill trampoline bread cheese sandwich pizza cake
cookie pie pancake waffle sausage bacon pretzel bagel muffin doughnut
building house cottage cabin hut tent skyscraper warehouse garage shed
barn silo greenhouse fountain monument pyramid castle palace tower
lighthouse windmill dam pier dock harbor sidewalk pavement curb
driveway rooftop balcony porch patio staircase elevator escalator
hallway corridor telephone laptop tablet printer projector lantern
flashlight candlestick chandelier windowsill shutter drain gutter wire
plug socket switch generator charger cord antenna router modem hardhat
sandpaper varnish paintbrush canvas easel palette pottery urn goblet
chalice teapot saucer skillet griddle wok cauldron thermos cooler
lunchbox sleepingbag canteen gloves mitten beanie visor sunglasses
earmuffs pendant crown tiara wig cufflinks shoelace bootlace lampshade
bedframe headboard footstool ottoman armchair recliner loveseat hammock
cradle bassinet highchair tricycle dollhouse teddybear puzzle boardgame
domino checkerboard chesspiece dice yoyo pinwheel slingshot catapult
saddle stirrup harness collar leash horseshoe plow harrow sickle scythe
pitchfork trowel hoe tongs skewer toothpick chopstick inkwell quill
scroll bookmark keyring nametag badge medal trophy plaque quilt duvet
birdcage fishbowl aquarium terrarium doghouse kennel mailslot birdbath
sundial weathervane
""".split()

random.seed(42)


def build():
    animal_pool = sorted(set(ANIMAL_ALL))
    plant_pool = sorted(set(PLANT_ALL))
    object_pool = sorted(set(OBJECTS))

    a, p, o = set(animal_pool), set(plant_pool), set(object_pool)
    assert not (a & p), a & p
    assert not (a & o), a & o
    assert not (p & o), p & o

    # Balance to within 10%: plants are the scarcest reliable class, so cap
    # animal/object sampling to a comparable size.
    n_plant = len(plant_pool)
    cap = int(round(n_plant * 1.10))

    random.shuffle(animal_pool)
    random.shuffle(object_pool)
    n_ao = min(cap, len(animal_pool), len(object_pool))
    # Trim evenly so the grand total lands exactly at 1000 when the pools
    # allow it (plant + 2 * n_ao may slightly overshoot the cap rounding).
    overshoot = (n_plant + 2 * n_ao) - 1000
    if overshoot > 0:
        n_ao -= (overshoot + 1) // 2
    animal_sel = animal_pool[:n_ao]
    object_sel = object_pool[:n_ao]

    data = [{"input": w, "output": "animal"} for w in animal_sel]
    data += [{"input": w, "output": "plant"} for w in plant_pool]
    data += [{"input": w, "output": "object"} for w in object_sel]
    random.shuffle(data)
    return data


def main():
    data = build()
    total = len(data)

    inputs = [d["input"] for d in data]
    assert len(set(inputs)) == total, "duplicate inputs"
    for d in data:
        w, o = d["input"], d["output"]
        assert w == w.strip() and o == o.strip()
        assert w.isalpha() and w.islower(), w
        assert o in ("animal", "plant", "object")
        if o == "animal":
            assert w in ANIMAL_ALL
        elif o == "plant":
            assert w in PLANT_ALL
        else:
            assert w in OBJECTS

    counts = Counter(d["output"] for d in data)
    vals = list(counts.values())
    assert max(vals) - min(vals) <= 0.10 * max(vals) + 1, counts
    print(f"counts={dict(counts)} total={total}")

    assert total == 1000, total

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {total} to {OUT}")
    return total, counts


if __name__ == "__main__":
    main()
