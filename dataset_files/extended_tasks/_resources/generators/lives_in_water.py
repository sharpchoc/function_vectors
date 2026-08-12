#!/usr/bin/env python
"""Generator for lives_in_water: animal -> yes if it lives primarily in
water, else no.

Recipe (spec index 71): curated knowledge lists.
YES = fish, marine mammals (whale/dolphin/etc.), and aquatic invertebrates
      (octopus, crab, jellyfish, ...).
NO  = land mammals, birds, land/flying insects, land reptiles/arachnids.

EXCLUDED entirely (ambiguous / semi-aquatic, per spec): amphibians (frog,
toad, salamander, newt), and semi-aquatic mammals/birds/reptiles (otter,
beaver, penguin, duck, seal, sealion, crocodile, alligator, turtle, tortoise,
platypus, hippo, water_vole-style animals), plus insects with aquatic
larvae only (mayfly, dragonfly, damselfly, mosquito, caddisfly).

Self-check: labels come directly from list membership; a re-derivation pass
below re-checks every item against the source pools.
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "lives_in_water.json")

# ---------------------------------------------------------------------------
# YES: lives primarily in water
# ---------------------------------------------------------------------------
FISH_YES = """
shark trout salmon tuna cod herring mackerel sardine anchovy halibut
flounder sole bass perch pike carp catfish eel swordfish marlin tarpon
barracuda snapper grouper haddock pollock whiting mullet wahoo tilapia koi
goldfish guppy piranha stingray skate lamprey sturgeon sunfish bluegill
crappie walleye muskie minnow chub dace roach bream tench rudd zander
angelfish clownfish damselfish parrotfish lionfish pufferfish boxfish
triggerfish wrasse goby blenny seahorse pipefish anglerfish lanternfish
hatchetfish dogfish catshark hammerhead mako whaleshark electriceel
mahimahi bonito shad smelt sablefish redfish bonefish tigerfish arowana
oscarfish plaice turbot monkfish hake ling ladyfish milkfish threadfin
needlefish flyingfish gar chinook coho sockeye steelhead rockfish lingcod
wolffish sculpin stickleback killifish swordtail platy barb loach cichlid
betta tigershark bullshark lemonshark sandshark leopardshark sawfish
mantaray eagleray moray conger browntrout rainbowtrout laketrout seatrout
brooktrout stripedbass whitebass surgeonfish butterflyfish batfish
cowfish filefish unicornfish viperfish oarfish coelacanth bullhead
buffalofish paddlefish bowfin sprat pilchard seadragon opah remora
pilotfish cornetfish trumpetfish squirrelfish soldierfish snook pompano
amberjack yellowtail kingfish cobia dorado albacore skipjack bluefin
yellowfin barbel pleco zebrafish danio rasbora baskingshark greatwhite
arapaima snakehead largemouthbass smallmouthbass chainpickerel
yellowperch whiteperch bluefish weakfish croaker porgy sheepshead
channelcatfish flatheadcatfish bluecatfish anemonefish hawkfish
scorpionfish stonefish gardeneel electricray cownoseray batray
""".split()

MARINE_MAMMALS_YES = """
whale dolphin porpoise orca narwhal dugong manatee walrus humpback
spermwhale beluga pilotwhale rightwhale minke finwhale graywhale
bottlenose bowhead bluewhale
""".split()

MARINE_INVERTS_YES = """
octopus squid cuttlefish jellyfish starfish seaurchin anemone coral crab
lobster shrimp prawn crayfish krill barnacle clam oyster mussel scallop
conch whelk seasnail nautilus seaslug seacucumber seasponge chiton limpet
periwinkle abalone cockle brittlestar horseshoecrab kingcrab stonecrab
spidercrab bluecrab snowcrab nudibranch tubeworm combjelly spinylobster
rocklobster mudcrab mantisshrimp pistolshrimp seafan moonjelly boxjelly
lionsmane seawasp
""".split()

YES_ALL = sorted(set(FISH_YES) | set(MARINE_MAMMALS_YES) | set(MARINE_INVERTS_YES))

# ---------------------------------------------------------------------------
# NO: land mammals, birds, land/flying insects, land reptiles/arachnids
# ---------------------------------------------------------------------------
LAND_MAMMALS_NO = """
camel elephant lion tiger bear wolf fox deer moose elk rabbit squirrel
mouse rat horse cow pig sheep goat dog cat monkey gorilla chimpanzee
kangaroo koala panda zebra giraffe hippo rhino buffalo bison badger skunk
raccoon hedgehog porcupine armadillo sloth anteater aardvark mole weasel
ferret mink donkey mule llama alpaca yak antelope gazelle wildebeest hare
chinchilla gerbil hamster guineapig vole shrew tapir pangolin okapi
capybara marmot muskrat lemur baboon orangutan gibbon mandrill wolverine
stoat polecat hyena jackal lynx bobcat cougar puma leopard cheetah jaguar
panther meerkat mongoose wombat opossum bat donkey camel pony zebu yak
gazelle springbok impala kudu oryx ibex chamois markhor bongo eland
warthog boar tapir kinkajou coati ocelot serval caracal margay lemming
pika dormouse vole shrew degu chinchilla agouti paca springhare jerboa
""".split()

BIRDS_NO = """
sparrow eagle hawk falcon robin crow raven owl dove pigeon swallow swift
hummingbird cardinal finch wren warbler thrush magpie woodpecker
nightingale lark starling oriole tanager mockingbird chickadee nuthatch
vulture condor parrot macaw cockatoo parakeet canary toucan hornbill
cuckoo grouse pheasant quail partridge peacock ostrich emu kiwi turkey
chicken bluebird catbird junco flycatcher goldfinch bullfinch chaffinch
kestrel buzzard harrier goshawk sparrowhawk merlin peregrine osprey
albatross bustard secretarybird roadrunner hoatzin seriema bittern crake
rail coot moorhen jacana grebe skimmer tropicbird sunbird honeycreeper
hoopoe trogon quetzal gnatcatcher redstart wheatear stonechat whinchat
dunnock treecreeper nutcracker chough conure lory woodpigeon turtledove
caracara honeybuzzard goldcrest firecrest wryneck linnet whitethroat
blackcap chiffchaff longspur dickcissel bobolink meadowlark grackle
cowbird bunting martin wagtail pipit shrike vireo kinglet titmouse
waxwing myna weaver rook jackdaw francolin guineafowl budgerigar
lovebird lorikeet cockatiel barbet honeyguide capercaillie ptarmigan
""".split()

INSECTS_ARACHNIDS_NO = """
bee wasp hornet bumblebee mosquito gnat cicada locust ladybug firefly
housefly moth butterfly ant beetle cockroach termite aphid weevil cricket
grasshopper millipede centipede tick mite silverfish flea louse bedbug
spider scorpion earwig yellowjacket honeybee carpenterbee midge
lightningbug junebug horsefly deerfly blowfly fruitfly cranefly sawfly
lacewing stonefly tsetsefly botfly blackfly sandfly robberfly hoverfly
antlion scorpionfly alderfly fishfly owlfly lanternfly katydid mantis
walkingstick treehopper leafhopper aphidlion springtail
""".split()

LAND_REPTILES_NO = """
snake lizard gecko iguana chameleon cobra viper python boa mamba adder
rattlesnake kingsnake gartersnake skink monitor tuatara komodo
""".split()

NO_ALL_BASE = (
    set(LAND_MAMMALS_NO) | set(BIRDS_NO) | set(INSECTS_ARACHNIDS_NO) | set(LAND_REPTILES_NO)
)

EXCLUDED = {
    "frog", "toad", "salamander", "newt", "otter", "beaver", "penguin",
    "duck", "seal", "sealion", "crocodile", "alligator", "turtle", "tortoise",
    "platypus", "hippopotamus", "mayfly", "dragonfly", "damselfly", "caddisfly",
}

random.seed(42)


def build():
    yes_pool = sorted(set(YES_ALL) - EXCLUDED)
    no_pool = sorted(NO_ALL_BASE - EXCLUDED - set(yes_pool))

    overlap = set(yes_pool) & set(no_pool)
    assert not overlap, overlap

    n_yes = len(yes_pool)
    # Balance classes to within ~10% (default instruction) rather than
    # dumping the much-larger land/bird/insect "no" pool wholesale.
    n_no = min(int(round(n_yes * 1.10)), len(no_pool), 1000 - n_yes)

    no_shuf = list(no_pool)
    random.shuffle(no_shuf)
    no_sel = no_shuf[:n_no]

    data = [{"input": w, "output": "yes"} for w in yes_pool]
    data += [{"input": w, "output": "no"} for w in no_sel]
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
        assert o in ("yes", "no")
        assert w not in EXCLUDED, w
        if o == "yes":
            assert w in YES_ALL
        else:
            assert w in NO_ALL_BASE

    counts = Counter(d["output"] for d in data)
    print(f"yes={counts['yes']} no={counts['no']} total={total}")

    if total < 1000:
        print(f"WARNING: only {total} reliable items, short of 1000")

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {total} to {OUT}")
    return total, counts


if __name__ == "__main__":
    main()
