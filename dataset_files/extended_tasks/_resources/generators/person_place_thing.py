#!/usr/bin/env python
"""Generator for person_place_thing: noun -> person / place / thing.

Recipe (spec index 73): curated knowledge lists, three classes balanced to
within 2:1 (per the spec's own generation_recipe note, looser than the
default 10% rule).

PERSON = humans and human roles/occupations/kinship terms.
PLACE  = locations one can physically be at.
THING  = touchable, physical objects.

EXCLUDED ("other", per spec): abstract nouns, events, animals, institutions
with building/organization polysemy (church, school, bank, market,
university, company), and vehicles used as places (ship). Words with a
strong competing non-noun/homonym sense are also dropped (e.g. military
ranks that double as common adjectives: general, major, private).

Self-check: labels come directly from list membership; re-derivation pass
below reasserts membership for every emitted item, and class sizes are
checked to be within a 2:1 ratio of each other.
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "person_place_thing.json")

# ---------------------------------------------------------------------------
# PERSON
# ---------------------------------------------------------------------------
OCCUPATIONS = """
teacher doctor nurse dentist lawyer engineer scientist chef farmer plumber
electrician carpenter mechanic pilot sailor soldier officer firefighter
judge priest nun monk professor student pupil principal coach athlete
actor actress singer musician painter sculptor author writer poet
journalist reporter editor photographer designer architect accountant
banker broker clerk secretary receptionist waiter waitress bartender
barber hairdresser tailor butcher baker grocer florist jeweler mason
blacksmith locksmith librarian curator archivist historian philosopher
psychologist therapist surgeon pediatrician veterinarian pharmacist
optometrist dietitian nutritionist midwife paramedic dispatcher detective
inspector sheriff marshal warden guard sentry spy agent diplomat consul
minister chancellor emperor empress duke duchess earl baron knight squire
peasant servant butler maid valet chauffeur driver captain admiral
colonel sergeant corporal cadet recruit veteran refugee immigrant tourist
traveler pilgrim hermit bishop pope rabbi imam deacon missionary
evangelist prophet martyr villain criminal thief burglar robber murderer
witness juror plaintiff defendant attorney solicitor notary auditor
economist statistician mathematician physicist chemist biologist
geologist astronomer botanist zoologist archaeologist anthropologist
sociologist linguist translator interpreter cartographer surveyor
technician programmer developer analyst consultant manager supervisor
executive chairman entrepreneur investor employee employer worker
laborer apprentice intern trainee freelancer contractor builder
bricklayer welder roofer decorator gardener landscaper forester ranger
lifeguard referee umpire trainer instructor tutor mentor counselor
advisor negotiator mediator arbitrator activist protester volunteer
philanthropist donor sponsor host hostess guest visitor neighbor stranger
king queen prince princess president senator mayor governor ambassador
astronaut cosmonaut physician orthodontist chiropractor podiatrist
radiologist anesthesiologist cardiologist neurologist oncologist
psychiatrist obstetrician gynecologist urologist dermatologist
immunologist microbiologist geneticist roboticist illustrator cartoonist
animator filmmaker director producer screenwriter playwright novelist
columnist blogger podcaster influencer streamer vlogger stuntman acrobat
juggler magician clown comedian dancer choreographer composer lyricist
machinist toolmaker founder shareholder landlord tenant realtor
appraiser assessor underwriter actuary bookkeeper cashier teller
stockbroker trader merchant vendor peddler smuggler poacher hunter
trapper fisherman lumberjack miner driller rigger foreman overseer
custodian janitor caretaker housekeeper nanny babysitter governess
chaperone bodyguard sentinel watchman lookout scout explorer adventurer
navigator prospector seismologist meteorologist oceanographer
paleontologist entomologist ornithologist herpetologist ichthyologist
mycologist ecologist conservationist environmentalist
""".split()

KINSHIP = """
husband wife widow widower bachelor bride groom orphan grandmother
grandfather uncle aunt nephew niece cousin sibling twin toddler infant
teenager adolescent adult child baby newborn parent mother father son
daughter brother sister friend enemy rival partner spouse
""".split()

PERSON_ALL = sorted(set(OCCUPATIONS) | set(KINSHIP))

# ---------------------------------------------------------------------------
# PLACE
# ---------------------------------------------------------------------------
NATURAL_PLACES = """
beach mountain valley forest desert island peninsula canyon cave cliff
glacier volcano lake river ocean sea bay gulf strait lagoon marsh swamp
jungle prairie savanna tundra oasis plateau plain hill ridge gorge
waterfall delta estuary reef shore coast meadow grove orchard vineyard
pasture field cape isthmus archipelago atoll fjord cove inlet
wetland moor heath steppe badlands mesa butte dune floodplain rainforest
woodland thicket glade clearing foothills summit peak slope pond
brook creek stream headland bluff escarpment ravine chasm trench shoal
sandbar
""".split()

BUILT_PLACES = """
harbor courtyard plaza square boulevard avenue alley street road highway
sidewalk driveway basement attic cellar bedroom bathroom kitchen
livingroom hallway corridor lobby foyer balcony terrace patio porch
veranda rooftop stadium arena gymnasium auditorium theater cinema museum
gallery library classroom laboratory workshop factory warehouse office
embassy courthouse prison dungeon castle palace fortress citadel temple
cathedral chapel monastery convent mosque synagogue shrine cemetery
graveyard mausoleum hospital clinic pharmacy hotel motel inn hostel
resort casino nightclub tavern cafe restaurant diner bakery boutique
salon gym racetrack port dock pier marina airport station terminal depot
platform subway tunnel bridge dam reservoir canal shipyard quarry
geyser crater cavern grotto burrow den nest hive village town city
capital suburb neighborhood district borough county province territory
farm ranch garage barracks chateau bungalow cottage cabin hut tent
mansion villa apartment condo dormitory orphanage asylum sanatorium
hospice armory arsenal granary greenhouse henhouse stable barn corral
paddock amphitheater colosseum pavilion kiosk booth wharf jetty
breakwater lighthouse windmill watermill aqueduct viaduct causeway
overpass underpass crosswalk roundabout carport atrium vestibule
anteroom parlor nursery playroom mudroom pantry larder scullery
loft mezzanine penthouse skyscraper highrise tenement slum ghetto
commune nunnery seminary academy institute observatory planetarium zoo
sanctuary reserve preserve park campground campsite boardwalk promenade
esplanade embankment levee spillway weir nook alcove enclave outpost
homestead settlement
""".split()

PLACE_ALL = sorted(set(NATURAL_PLACES) | set(BUILT_PLACES))

# ---------------------------------------------------------------------------
# THING (touchable physical objects)
# ---------------------------------------------------------------------------
THINGS = """
table chair desk sofa bed lamp mirror clock kettle spoon fork knife plate
cup bowl pot pan hammer screwdriver wrench nail screw ladder bucket broom
mop shovel rake axe saw drill book pen pencil paper envelope stamp key
lock door window brick stone rock rope chain wire cable box bag basket
jar bottle can barrel crate suitcase backpack umbrella blanket pillow
towel curtain carpet rug shelf cabinet drawer vase candle lighter
battery pipe hose faucet sink toilet bathtub shower comb brush razor
soap toothbrush mattress cushion stool bench sculpture statue painting
frame watch ring necklace bracelet earring wallet purse glove hat scarf
belt button zipper needle thread scissors ruler stapler tape glue
paperclip folder binder calculator keyboard speaker microphone camera
telescope binoculars compass globe flag banner ticket coin bill receipt
calendar notebook diary journal magazine newspaper poster napkin
tablecloth placemat coaster tray platter ladle spatula whisk grater
peeler colander sieve funnel thermometer chalk eraser crayon marker
highlighter briefcase toolbox jackhammer anvil vice clamp pliers file
chisel mallet crowbar bolt nut washer gear spring lever pulley hinge
padlock doorknob doorbell mailbox trashcan dustpan hanger vacuum blender
toaster microwave oven refrigerator freezer dishwasher heater fan
radiator chimney fireplace bookshelf wardrobe dresser nightstand crib
stroller wheelchair crutch cane raincoat boots sandals slippers sneakers
helmet goggles mask apron mittens socks sweater jacket coat shirt
trousers skirt dress necktie guitar piano violin drum trumpet flute
clarinet trombone saxophone harmonica accordion banjo cello harp
xylophone tambourine cymbal football basketball baseball racket skis
snowboard surfboard skateboard dumbbell barbell treadmill trampoline
telephone laptop tablet printer projector lantern flashlight
candlestick chandelier antenna router modem hardhat sandpaper varnish
paintbrush canvas easel palette pottery urn goblet chalice teapot saucer
skillet griddle wok cauldron thermos cooler lunchbox sleepingbag canteen
pendant crown tiara wig cufflinks shoelace lampshade bedframe headboard
footstool ottoman armchair recliner loveseat hammock cradle bassinet
highchair tricycle dollhouse teddybear puzzle boardgame domino dice yoyo
pinwheel slingshot saddle stirrup harness collar leash horseshoe plow
sickle scythe pitchfork trowel hoe tongs skewer toothpick chopstick
inkwell quill scroll bookmark keyring nametag badge medal trophy plaque
quilt duvet birdcage fishbowl aquarium terrarium doghouse kennel
sundial weathervane car truck bus bicycle motorcycle boat canoe kayak
tractor van scooter wheelbarrow sled wagon pickaxe sledgehammer spade
shears clippers tweezers stethoscope syringe bandage splint
""".split()

THING_ALL = sorted(set(THINGS))

random.seed(42)

EXCLUDED = {"church", "school", "bank", "ship", "market", "university", "company"}


def build():
    person_pool = sorted(set(PERSON_ALL) - EXCLUDED)
    place_pool = sorted(set(PLACE_ALL) - EXCLUDED)
    thing_pool = sorted(set(THING_ALL) - EXCLUDED)

    p, pl, t = set(person_pool), set(place_pool), set(thing_pool)
    assert not (p & pl), p & pl
    assert not (p & t), p & t
    assert not (pl & t), pl & t

    random.shuffle(person_pool)
    random.shuffle(place_pool)
    random.shuffle(thing_pool)

    # cap total at 1000, keep each class within 2:1 of the others
    n_person, n_place, n_thing = len(person_pool), len(place_pool), len(thing_pool)
    total = n_person + n_place + n_thing
    if total > 1000:
        # scale down proportionally, preserving relative sizes (already
        # well within 2:1 given the pool sizes below)
        scale = 1000 / total
        n_person = int(n_person * scale)
        n_place = int(n_place * scale)
        n_thing = 1000 - n_person - n_place

    person_sel = person_pool[:n_person]
    place_sel = place_pool[:n_place]
    thing_sel = thing_pool[:n_thing]

    data = [{"input": w, "output": "person"} for w in person_sel]
    data += [{"input": w, "output": "place"} for w in place_sel]
    data += [{"input": w, "output": "thing"} for w in thing_sel]
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
        assert o in ("person", "place", "thing")
        assert w not in EXCLUDED, w
        if o == "person":
            assert w in PERSON_ALL
        elif o == "place":
            assert w in PLACE_ALL
        else:
            assert w in THING_ALL

    counts = Counter(d["output"] for d in data)
    vals = list(counts.values())
    assert max(vals) <= 2 * min(vals), counts
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
