#!/usr/bin/env python
"""Generator for animal_class: animal -> taxonomic class from
{mammal, bird, fish, reptile, insect}.

Recipe (spec index 74): curated knowledge lists per class, using species
names, common names, and breeds (dog/cat/horse) to build up count.

EXCLUDED entirely (per spec): amphibians and arachnids (not in the label
set -- no frogs, spiders, scorpions, ticks, mites, centipedes, millipedes),
extinct animals, colloquial misleads that are not actually in these
classes despite the name (jellyfish, starfish -- not fish, and not in any
of the 5 classes so dropped outright), and any name an existence/commonness
check could not confirm. "bat" is kept as mammal (crisp knowledge, despite
flying). Breed/proper-noun words with a strong competing homonym sense
(e.g. "boxer", "setter", "pointer", "chow", "mustang", "chihuahua",
geographic/demonym-sounding breed names) are dropped to avoid ambiguity.

Self-check: labels come directly from list membership; re-derivation pass
below reasserts membership for every emitted item, and class sizes are
balanced to within 10%.
"""
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "animal_class.json")

sys.path.insert(0, HERE)
from _space_fixes import (  # noqa: E402
    MAMMAL_FIXES,
    DOG_BREED_FIXES,
    CAT_BREED_FIXES,
    BIRD_FIXES,
    FISH_FIXES,
    REPTILE_FIXES,
    INSECT_FIXES,
)


def _apply_fixes(words, fixmap):
    """Restore proper spacing for fused multi-word names. Words not in the
    map (including legitimate single-word compounds like "housefly" or
    "kingsnake") pass through unchanged."""
    out = [fixmap.get(w, w) for w in words]
    assert len(set(out)) == len(out), "fix-up introduced a duplicate"
    return out

# ---------------------------------------------------------------------------
# MAMMAL (species + safe dog/cat/horse breeds)
# ---------------------------------------------------------------------------
MAMMAL_SPECIES = """
dog cat horse cow pig sheep goat rabbit squirrel mouse rat lion tiger
bear wolf fox deer moose elk camel elephant monkey gorilla chimpanzee
kangaroo koala panda zebra giraffe hippo rhino buffalo bison otter
beaver badger skunk raccoon hedgehog porcupine armadillo sloth anteater
aardvark mole weasel ferret mink seal walrus whale dolphin porpoise
donkey mule llama alpaca yak antelope gazelle wildebeest hare chinchilla
gerbil hamster shrew tapir pangolin okapi capybara marmot muskrat lemur
baboon orangutan gibbon mandrill wolverine stoat polecat hyena jackal
lynx bobcat cougar puma leopard cheetah jaguar panther meerkat mongoose
wombat opossum bat platypus vole pika dormouse springhare jerboa warthog
boar coati ocelot serval caracal impala kudu oryx ibex chamois bongo
eland narwhal orca manatee dugong guineapig chipmunk groundhog woodchuck
prairiedog molerat flyingsquirrel redsquirrel greysquirrel treeshrew
elephantshrew fruitbat vampirebat spidermonkey howlermonkey
squirrelmonkey capuchin tamarin marmoset macaque langur colobus bonobo
siamang dingo coyote manedwolf wilddog waterbuffalo muskox reindeer
caribou pronghorn springbok blackbuck nyala waterbuck hartebeest wallaby
quokka tasmaniandevil bandicoot bilby possum seaotter elephantseal
harborseal furseal spermwhale humpback beluga pilotwhale minke
bottlenose rightwhale
""".split()

DOG_BREEDS = """
poodle bulldog dachshund rottweiler doberman collie malamute pug shihtzu
cockerspaniel goldenretriever germanshepherd greatdane mastiff terrier
dalmatian corgi schnauzer weimaraner bloodhound foxhound greyhound
whippet akita basset bordercollie saintbernard bullmastiff sheepdog
retriever spaniel hound
""".split()

CAT_BREEDS = "tabby calico sphynx ragdoll mainecoon".split()

HORSE_BREEDS = """
thoroughbred clydesdale palomino appaloosa pony stallion mare foal colt
gelding warmblood percheron
""".split()

MAMMAL_SPECIES = _apply_fixes(MAMMAL_SPECIES, MAMMAL_FIXES)
DOG_BREEDS = _apply_fixes(DOG_BREEDS, DOG_BREED_FIXES)
CAT_BREEDS = _apply_fixes(CAT_BREEDS, CAT_BREED_FIXES)

MAMMAL_ALL = sorted(set(MAMMAL_SPECIES) | set(DOG_BREEDS) | set(CAT_BREEDS) | set(HORSE_BREEDS))

# ---------------------------------------------------------------------------
# BIRD
# ---------------------------------------------------------------------------
BIRD_ALL = sorted(set(_apply_fixes("""
sparrow eagle hawk falcon robin crow raven owl dove pigeon seagull
swallow swift hummingbird cardinal finch wren warbler thrush magpie
woodpecker kingfisher heron stork crane flamingo pelican cormorant
albatross kestrel buzzard vulture condor osprey parrot macaw cockatoo
parakeet canary toucan hornbill cuckoo nightingale lark starling oriole
tanager mockingbird chickadee nuthatch sandpiper plover tern egret ibis
bittern grouse pheasant quail partridge goose duck swan teal mallard
peacock puffin loon grackle cowbird bunting martin wagtail pipit shrike
vireo kinglet titmouse creeper waxwing myna weaver rook jackdaw booby
gannet frigatebird petrel shearwater skua kittiwake auk murre guillemot
curlew snipe woodcock godwit avocet stilt lapwing ptarmigan guineafowl
budgerigar lovebird lorikeet meadowlark blackbird bobolink dunlin fulmar
goshawk sparrowhawk merlin peregrine harrier wigeon pintail shoveler
merganser eider scoter gadwall canvasback redhead bufflehead goldeneye
brant dowitcher yellowlegs phalarope turnstone goldfinch bullfinch
chaffinch greenfinch siskin redpoll crossbill bluebird catbird
dickcissel junco longspur phoebe flycatcher kingbird pewee veery
francolin cockatiel bustard jacana coot moorhen grebe skimmer
tropicbird grosbeak redstart gnatcatcher sunbird honeycreeper hoopoe
trogon quetzal scaup pochard shelduck gyrfalcon swiftlet ostrich
penguin emu kiwi cassowary rhea kakapo turkey chicken rooster hen
blackcap bluejay caracara chiffchaff chough conure dotterel dunnock
firecrest goldcrest greenshank gull honeybuzzard knot linnet lory
nutcracker oystercatcher peafowl redshank ruff sanderling smew
stonechat treecreeper turtledove wheatear whimbrel whinchat whitethroat
woodpigeon wryneck
""".split(), BIRD_FIXES)))

# ---------------------------------------------------------------------------
# FISH
# ---------------------------------------------------------------------------
FISH_ALL = sorted(set(_apply_fixes("""
shark trout salmon tuna cod herring mackerel sardine anchovy halibut
flounder sole bass perch pike carp catfish eel swordfish marlin tarpon
barracuda snapper grouper haddock pollock whiting mullet wahoo tilapia
koi goldfish guppy piranha stingray skate lamprey sturgeon sunfish
bluegill crappie walleye muskie gar minnow chub dace roach bream tench
rudd zander angelfish clownfish damselfish parrotfish lionfish
pufferfish boxfish triggerfish wrasse goby blenny seahorse pipefish
anglerfish lanternfish hatchetfish dogfish catshark hammerhead mako
whaleshark electriceel bonito shad smelt sablefish redfish bonefish
tigerfish arowana oscarfish plaice turbot monkfish hake ling ladyfish
milkfish threadfin needlefish flyingfish chinook coho sockeye steelhead
rockfish lingcod wolffish sculpin stickleback killifish swordtail platy
barb loach cichlid betta tigershark bullshark lemonshark sandshark
leopardshark sawfish mantaray eagleray moray conger browntrout
rainbowtrout laketrout seatrout brooktrout stripedbass whitebass
surgeonfish butterflyfish batfish cowfish filefish unicornfish
viperfish oarfish coelacanth bullhead buffalofish paddlefish bowfin
sprat pilchard opah remora pilotfish cornetfish trumpetfish
squirrelfish soldierfish snook pompano amberjack yellowtail kingfish
cobia dorado albacore skipjack bluefin yellowfin barbel anemonefish
arapaima baskingshark batray bluecatfish bluefish chainpickerel
channelcatfish cownoseray croaker danio electricray flatheadcatfish
gardeneel greatwhite hawkfish largemouthbass mahimahi pleco porgy
rasbora scorpionfish seadragon sheepshead smallmouthbass snakehead
stonefish weakfish whiteperch yellowperch zebrafish
""".split(), FISH_FIXES)))

# ---------------------------------------------------------------------------
# REPTILE
# ---------------------------------------------------------------------------
REPTILE_ALL = sorted(set(_apply_fixes("""
snake lizard gecko iguana chameleon cobra viper python boa mamba adder
rattlesnake crocodile alligator turtle tortoise komodo monitor skink
tuatara copperhead cottonmouth kingsnake gartersnake milksnake cornsnake
ratsnake bullsnake hognose sidewinder anaconda taipan krait coralsnake
bushmaster puffadder boomslang blacksnake watersnake seasnake whipsnake
wormsnake blindsnake pitviper ringneck indigosnake kingcobra
spittingcobra blackmamba greenmamba diamondback watermoccasin
timberrattler boaconstrictor ballpython vinesnake treesnake monitorlizard
watermonitor nilemonitor sailfin thornydevil frilledlizard slowworm
mudturtle terrapin matamata beardeddragon leopardgecko gilamonster
hornedlizard glasslizard alligatorlizard snappingturtle boxturtle
paintedturtle seaturtle caiman gharial loggerhead leatherback hawksbill
basilisk anole agama tegu chuckwalla whiptail eyelashviper gaboonviper
bambooviper ferdelance mangrovesnake rainbowboa rosyboa sandboa
emeraldtreeboa greentreepython bloodpython burmesepython tokay daygecko
housegecko crestedgecko greenanole brownanole bluetongueskink fireskink
sandskink lacemonitor greeniguana marineiguana rockiguana
pantherchameleon veiledchameleon jacksonchameleon waterdragon
redearedslider galapagostortoise deserttortoise muskturtle
softshellturtle oliveridley nilecrocodile saltwatercrocodile
americanalligator chinesealligator americancrocodile gophersnake
pinesnake wolfsnake catsnake komododragon fivelinedskink desertiguana
gargoylegecko savannahmonitor dwarfcrocodile spottedturtle woodturtle
diamondbackterrapin reticulatedpython smoothsnake dicesnake grasssnake
ribbonsnake brownsnake tigersnake deathadder bandedkrait seakrait
nightlizard fencelizard collaredlizard racerunner girdledlizard
armadillolizard platedlizard bogturtle alligatorsnapper muggercrocodile
massasauga easternracer blackracer coachwhip patchnosesnake lyresnake
leopardlizard earlesslizard spinylizard gophertortoise pondturtle
riverturtle falsegharial kempsridley orinococrocodile cubancrocodile
knobtailgecko leaftailedgecko yellowbellyslider cooterturtle filesnake
keelback scarletsnake rainbowsnake mudsnake crayfishsnake queensnake
moleskink groundskink greenbasilisk chickenturtle mapturtle perentie
bengalmonitor landiguana blueiguana flapneckchameleon giantdaygecko
russellviper mangroveviper
""".split(), REPTILE_FIXES)))

# ---------------------------------------------------------------------------
# INSECT (true insects only -- no arachnids/myriapods)
# ---------------------------------------------------------------------------
INSECT_ALL = sorted(set(_apply_fixes("""
bee wasp hornet yellowjacket bumblebee honeybee carpenterbee mosquito
gnat midge mayfly dragonfly damselfly cicada locust ladybug firefly
lightningbug junebug housefly horsefly deerfly blowfly fruitfly moth
butterfly cranefly sawfly lacewing stonefly caddisfly tsetsefly botfly
blackfly sandfly robberfly hoverfly antlion scorpionfly alderfly
fishfly owlfly lanternfly ant beetle cockroach termite aphid weevil
cricket grasshopper flea louse silverfish earwig katydid mantis
walkingstick treehopper leafhopper springtail gypsymoth tentcaterpillar
armyworm cutworm webworm bagworm leafminer japanesebeetle scarabbeetle
dungbeetle stagbeetle clickbeetle tigerbeetle groundbeetle waterbeetle
longhornbeetle leafbeetle potatobeetle blisterbeetle carpetbeetle
barkbeetle cicadakiller potterwasp muddauber paperwasp diggerwasp
gallwasp fireant carpenterant armyant leafcutterant bulletant
harvesterant weaverant pharaohant driverant waterbug fieldcricket
housecricket molecricket desertlocust fleshfly warblefly gadfly
fungusgnat silkmoth lunamoth hawkmoth sphinxmoth tigermoth clothesmoth
codlingmoth cabbagemoth swallowtail paintedlady cabbagewhite fritillary
hairstreak skipper stinkbug assassinbug leaffootedbug boxelderbug
squashbug chinchbug shieldbug lacebug waterstrider backswimmer
giantwaterbug planthopper spittlebug rhinocerosbeetle herculesbeetle
goliathbeetle bombardierbeetle whirligigbeetle divingbeetle
deathwatchbeetle furniturebeetle powderpostbeetle flourbeetle
grainbeetle tortoisebeetle snoutbeetle bollweevil sweatbee miningbee
leafcutterbee masonbee squashbee velvetant ichneumonwasp gallfly
hessianfly honeypotant thiefant tussockmoth plumemoth clearwingmoth
owletmoth atlasmoth dronefly bluebottlefly greenbottlefly marchfly
mothfly soldierfly camelcricket bushcricket prayingmantis mantidfly
greenlacewing catflea dogflea sandflea booklice firebrat thrips
bristletail mealwormbeetle emeraldashborer japanesehornet harvestfly
kissingbug waterboatman treecricket dobsonfly fishmoth
""".split(), INSECT_FIXES)))

random.seed(42)

CLASS_POOLS = {
    "mammal": MAMMAL_ALL,
    "bird": BIRD_ALL,
    "fish": FISH_ALL,
    "reptile": REPTILE_ALL,
    "insect": INSECT_ALL,
}


def build():
    pools = {k: sorted(set(v)) for k, v in CLASS_POOLS.items()}

    # no cross-class overlaps
    names = list(pools.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(pools[a]) & set(pools[b])
            assert not overlap, (a, b, overlap)

    # Balance to within 10% of the largest reliably-available class, capped
    # so the total does not exceed 1000.
    sizes = {k: len(v) for k, v in pools.items()}
    min_size = min(sizes.values())
    cap = int(round(min_size * 1.10))

    for k in pools:
        random.shuffle(pools[k])
    n_each = {k: min(cap, len(pools[k])) for k in pools}

    total = sum(n_each.values())
    if total > 1000:
        scale = 1000 / total
        for k in n_each:
            n_each[k] = int(n_each[k] * scale)

    data = []
    for k, pool in pools.items():
        for w in pool[: n_each[k]]:
            data.append({"input": w, "output": k})
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
        # allow single- or multi-word names ("green iguana"); each token
        # must be alphabetic and lowercase, single spaces only.
        tokens = w.split(" ")
        assert all(t.isalpha() and t.islower() for t in tokens), w
        assert " ".join(tokens) == w, w  # no double spaces / stray whitespace
        assert o in CLASS_POOLS
        assert w in CLASS_POOLS[o]

    counts = Counter(d["output"] for d in data)
    vals = list(counts.values())
    assert max(vals) - min(vals) <= 0.10 * max(vals) + 1, counts
    print(f"counts={dict(counts)} total={total}")

    if total < 1000:
        print(f"WARNING: only {total} reliable items, short of 1000")
    else:
        assert total == 1000

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {total} to {OUT}")
    return total, counts


if __name__ == "__main__":
    main()
