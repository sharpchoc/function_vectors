# Debug: sampled generations per prompt format

Qwen2.5-7B-Instruct, 6-shot, T=1.0 sampled, 50 prompts/task; metric = first line,
stripped, exact match. Same demo/query sampling in every arm. Generations truncated
to 140 chars. Regenerate with `src/eval_scripts/dump_chat_template_debug_examples.py`.


## Full example prompts (task `ends_with_ing`, prompt i=0)

### plain

```
Q: prototype
A: no

Q: ballasting
A: yes

Q: interesting
A: yes

Q: former
A: no

Q: zincing
A: yes

Q: literature
A: no

Q: eddying
A:
```

### chat_blank_system

```
<|im_start|>system
<|im_end|>
<|im_start|>user
prototype<|im_end|>
<|im_start|>assistant
no<|im_end|>
<|im_start|>user
ballasting<|im_end|>
<|im_start|>assistant
yes<|im_end|>
<|im_start|>user
interesting<|im_end|>
<|im_start|>assistant
yes<|im_end|>
<|im_start|>user
former<|im_end|>
<|im_start|>assistant
no<|im_end|>
<|im_start|>user
zincing<|im_end|>
<|im_start|>assistant
yes<|im_end|>
<|im_start|>user
literature<|im_end|>
<|im_start|>assistant
no<|im_end|>
<|im_start|>user
eddying<|im_end|>
<|im_start|>assistant

```

### chat_no_system

```
<|im_start|>user
prototype<|im_end|>
<|im_start|>assistant
no<|im_end|>
<|im_start|>user
ballasting<|im_end|>
<|im_start|>assistant
yes<|im_end|>
<|im_start|>user
interesting<|im_end|>
<|im_start|>assistant
yes<|im_end|>
<|im_start|>user
former<|im_end|>
<|im_start|>assistant
no<|im_end|>
<|im_start|>user
zincing<|im_end|>
<|im_start|>assistant
yes<|im_end|>
<|im_start|>user
literature<|im_end|>
<|im_start|>assistant
no<|im_end|>
<|im_start|>user
eddying<|im_end|>
<|im_start|>assistant

```


## Failure-mode taxonomy: mode A vs mode B (chat_no_system, plain-correct prompts)

**Mode A — verbose/conversational**: the assistant persona overrides the ICL pattern; the model answers the query as a fresh chat message instead of imitating the terse label.

**Mode B — terse-wrong**: the model DOES imitate the label format (short, register-conforming answer) but the answer itself is wrong — typically a neighbouring rule (first letter instead of first vowel) or a near-miss (off-by-one count). The same prompt in Q:/A: format gets it right.


### Mode A examples

**adjective_to_adverb** — q = `scarce`, gold `scarcely`

- ✗ chat: `scarceley (Note: There is no word "scar`
- ✓ plain: `scarcely

Q: jealous
A: jealously`

**animal_plant_object** — q = `periwinkle`, gold `plant`

- ✗ chat: `It can refer to both a plant and a sea snail`
- ✓ plain: `plant

Q: locust
A: animal

Q`

**capitalize_second_letter** — q = `nurturing`, gold `U`

- ✗ chat: `Nurturing is a gentle and caring quality that provides support`
- ✓ plain: `U

Given the letters provided by the answers, we can`

**concrete_abstract** — q = `lizard`, gold `concrete`

- ✗ chat: `Let's make this fun! Imagine a lizard trying to put`
- ✓ plain: `concrete

Q: joy
A: abstract

Could you`

**english-french** — q = `mouse`, gold `souris`

- ✗ chat: `SOURIS (le petit animal) ou SOURIS (le`
- ✓ plain: `souris

Q: please
A: s'il vous`

**first_vowel** — q = `orthodox`, gold `o`

- ✗ chat: `Orthodox refers to something that adheres strictly to established norms`
- ✓ plain: `o

Q: pique
A: i

Q`

**hypernym_category** — q = `abbey`, gold `building`

- ✗ chat: `monastery or architectural structure resembling a monastery. An abbey`
- ✓ plain: `building

Q: raven
A: bird

Q`

**last_two_letters** — q = `explain`, gold `in`

- ✗ chat: `Sure, could you please provide more context? You seem to`
- ✓ plain: `in

Q: development
A: ev

Q:`

**next_in_period** — q = `Beryllium`, gold `Boron`

- ✗ chat: `Beryllium is a chemical element with the symbol Be`
- ✓ plain: `Boron

Q: Cerium
A: Lut`

**person-sport** — q = `DaMarcus Beasley`, gold `soccer`

- ✗ chat: `soccer and football (specifically referring to American football)`
- ✓ plain: `soccer

Q: Wally Berger
A: baseball`

**product-company** — q = `Blogger`, gold `Google`

- ✗ chat: `Blogger is a website publishing service provided by Google. It`
- ✓ plain: `Google

For each answer, I identified the original creator or`

**strip_prefix** — q = `oversimplified`, gold `simplified`

- ✗ chat: `The correct spelling is "oversimplified." Here's the sentence`
- ✓ plain: `simplified

Q: unannounced
A: announced`


### Mode B examples

**adjective_to_adverb** — q = `scanty`, gold `scantily`

- ✗ chat: `scantly`
- ✓ plain: `scantily

Q: diligent
A: diligently

Q`

**animal_plant_object** — q = `puffin`, gold `animal`

- ✗ chat: `bird`
- ✓ plain: `animal

Q: beekeeper
A: person

By`

**compound_first** — q = `armchair`, gold `arm`

- ✗ chat: `armchair`
- ✓ plain: `arm
You: thanksgiving
A: thank You for`

**count_zeros** — q = `577`, gold `0`

- ✗ chat: `1`
- ✓ plain: `0

Q: 7378
A`

**english-italian** — q = `fight`, gold `combattere`

- ✗ chat: `combattimento`
- ✓ plain: `combattere

Q: please
A: prego`

**french-english** — q = `permettre`, gold `allow`

- ✗ chat: `permit`
- ✓ plain: `allow

Q: permissions
A: permissions (No translation`

**larger_than_1000** — q = `665`, gold `no`

- ✗ chat: `yes`
- ✓ plain: `no

Q: 71842
A`

**next_in_group** — q = `Lanthanum`, gold `Actinium`

- ✗ chat: `Promethium`
- ✓ plain: `Actinium

Q: Selenium
A: Tellurium`

**person-occupation** — q = `Genki Sudo`, gold `actor`

- ✗ chat: `athlete`
- ✓ plain: `actor

Q: A.A. Gill
A: critic`

**remove_last_letter** — q = `young`, gold `youn`

- ✗ chat: `youth`
- ✓ plain: `youn

Q: rhyme
A: rime`

**spanish-english** — q = `hoja`, gold `leaf`

- ✗ chat: `sheet`
- ✓ plain: `leaf

Q: lluvia
A: rain`

**time_to_minutes** — q = `05:52`, gold `352`

- ✗ chat: `228`
- ✓ plain: `352

Q: 07:1`


## Tasks pruned (<30%) under BOTH chat arms but kept under plain

Examples favour prompts where plain succeeded and chat_no_system failed.


### contains_letter_e  (plain 0.46 / chat_blank_system 0.02 / chat_no_system 0.02)

**q = `april`** — gold `no`

- ✓ `plain`: `no

Q: true
A: no

Q:`
- ✗ `chat_blank_system`: `Yes, April is the fourth month of the year. Is`
- ✗ `chat_no_system`: `Sure! April is the fourth month of the year in the`

**q = `favor`** — gold `no`

- ✓ `plain`: `no

Q: fortitude
A: no

Q`
- ✗ `chat_blank_system`: `It seems like you might be looking for a specific context or`
- ✗ `chat_no_system`: `Are you looking for a definition, translation, or more context`

**q = `slow`** — gold `no`

- ✓ `plain`: `no

Q: easy
A: yes

Q:`
- ✗ `chat_blank_system`: `It seems like you may be asking for definitions or looking to`
- ✗ `chat_no_system`: `No specific question or action was provided for "slow." Could`

**q = `wildly`** — gold `no`

- ✓ `plain`: `no

Q: apollo
A: yes

Q`
- ✗ `chat_blank_system`: `The word "wildly" is an adverb that means`
- ✗ `chat_no_system`: `Yes, "wildly" is a valid word. It`

**q = `ruling`** — gold `no`

- ✓ `plain`: `no

Q: regular
A: no

Q:`
- ✗ `chat_blank_system`: `Yes, I can help with information or discussions related to a`
- ✗ `chat_no_system`: `Yes, I understand you've listed several words. If you`

**q = `wrongly`** — gold `no`

- ✓ `plain`: `no

Q: patriotism
A: yes

Q:`
- ✗ `chat_blank_system`: `It seems like you might be looking to define or use a`
- ✗ `chat_no_system`: `Yes, "wrongly" is a valid word. It`


### ends_with_ing  (plain 0.64 / chat_blank_system 0.00 / chat_no_system 0.00)

**q = `eddying`** — gold `yes`

- ✓ `plain`: `yes

Q: ganging
A: yes

Q`
- ✗ `chat_blank_system`: `Yes, "eddying" refers to the swirling motion of`
- ✗ `chat_no_system`: `The word "eddying" refers to the circular currents or`

**q = `battery`** — gold `no`

- ✓ `plain`: `no

Q: ly
A: no

Q:`
- ✗ `chat_blank_system`: `Yes, "battery" is a word. It typically refers`
- ✗ `chat_no_system`: `Yes, "battery" is a word. It can refer`

**q = `solving`** — gold `yes`

- ✓ `plain`: `yes

Q: lumbering
A: yes

Q`
- ✗ `chat_blank_system`: `Yes! "Solving" means to find an answer or`
- ✗ `chat_no_system`: `Yes, what would you like to solve or how can I`

**q = `thermal`** — gold `no`

- ✓ `plain`: `no

Q: h2o
A: no`
- ✗ `chat_blank_system`: `The word "thermal" relates to heat or temperature. It`
- ✗ `chat_no_system`: `Yes, "thermal" refers to something related to heat.`

**q = `sport`** — gold `no`

- ✓ `plain`: `no

Q: oiling
A: yes

Q`
- ✗ `chat_blank_system`: `Yes, "sport" is a word. It refers to`
- ✗ `chat_no_system`: `It seems like you're checking if these words are valid or`

**q = `switching`** — gold `yes`

- ✓ `plain`: `yes

Q: eluding
A: yes

Q`
- ✗ `chat_blank_system`: `Yes, "switching" is a valid English word.`
- ✗ `chat_no_system`: `Yes, "switching" is a valid word in English`


### word_length  (plain 0.56 / chat_blank_system 0.14 / chat_no_system 0.20)

**q = `table`** — gold `5`

- ✓ `plain`: `5

Q: although
A: 8`
- ✗ `chat_blank_system`: `6`
- ✗ `chat_no_system`: `6`

**q = `yogurt`** — gold `6`

- ✓ `plain`: `6

Q: thirty
A: 8`
- ✗ `chat_blank_system`: `5`
- ✗ `chat_no_system`: `5`

**q = `playful`** — gold `7`

- ✓ `plain`: `7

Q: potato
A: 7`
- ✗ `chat_blank_system`: `9`
- ✗ `chat_no_system`: `8`

**q = `long`** — gold `4`

- ✓ `plain`: `4

Q: sing
A: 4`
- ✗ `chat_blank_system`: `3`
- ✗ `chat_no_system`: `3`

**q = `tiger`** — gold `5`

- ✓ `plain`: `5

Q: dog
A: 3`
- ✗ `chat_blank_system`: `3`
- ✗ `chat_no_system`: `3`

**q = `purple`** — gold `6`

- ✓ `plain`: `6

Q: pizza
A: 5`
- ✗ `chat_blank_system`: `5`
- ✗ `chat_no_system`: `9`


### first_vowel  (plain 0.62 / chat_blank_system 0.26 / chat_no_system 0.26)

**q = `able`** — gold `a`

- ✓ `plain`: `a

Q: acquisition
A: i

Q:`
- ✗ `chat_blank_system`: `e`
- ✗ `chat_no_system`: `e`

**q = `salary`** — gold `a`

- ✓ `plain`: `a

Q: custody
A: u

Q:`
- ✗ `chat_blank_system`: `The word "salary" is spelled as follows:

s a`
- ✗ `chat_no_system`: `y`

**q = `gas`** — gold `a`

- ✓ `plain`: `a

Q: wallet
A: a

Q:`
- ✗ `chat_blank_system`: `s`
- ✗ `chat_no_system`: `s`

**q = `document`** — gold `o`

- ✓ `plain`: `o

Q: ancestry
A: y

Q:`
- ✗ `chat_blank_system`: `d`
- ✗ `chat_no_system`: `d`

**q = `unstable`** — gold `u`

- ✓ `plain`: `u

Q: breathable
A: r

Q:`
- ✗ `chat_blank_system`: `l`
- ✗ `chat_no_system`: `b`

**q = `olympic`** — gold `o`

- ✓ `plain`: `o

Q: terrible
A: e

Q:`
- ✗ `chat_blank_system`: `i`
- ✗ `chat_no_system`: `i`


### count_zeros  (plain 0.36 / chat_blank_system 0.14 / chat_no_system 0.18)

**q = `950070`** — gold `3`

- ✓ `plain`: `3

Q: 200040`
- ✗ `chat_blank_system`: `2`
- ✗ `chat_no_system`: `9`

**q = `577`** — gold `0`

- ✓ `plain`: `0

Q: 7378
A`
- ✓ `chat_blank_system`: `0`
- ✗ `chat_no_system`: `1`

**q = `960`** — gold `1`

- ✓ `plain`: `1

Q: 123456`
- ✗ `chat_blank_system`: `2`
- ✗ `chat_no_system`: `2`

**q = `991000`** — gold `3`

- ✓ `plain`: `3

Q: 61
A:`
- ✗ `chat_blank_system`: `6`
- ✗ `chat_no_system`: `6`

**q = `720`** — gold `1`

- ✓ `plain`: `1

Q: 20202`
- ✗ `chat_blank_system`: `0`
- ✗ `chat_no_system`: `0`

**q = `500020`** — gold `4`

- ✓ `plain`: `4

Q: 900050`
- ✗ `chat_blank_system`: `2`
- ✗ `chat_no_system`: `2`


### count_consonants  (plain 0.36 / chat_blank_system 0.28 / chat_no_system 0.22)

**q = `give`** — gold `2`

- ✓ `plain`: `2

Q: correlate
A: 5`
- ✗ `chat_blank_system`: `1`
- ✗ `chat_no_system`: `1`

**q = `cool`** — gold `2`

- ✓ `plain`: `2

Q: stain
A: 2`
- ✗ `chat_blank_system`: `3`
- ✗ `chat_no_system`: `3`

**q = `readable`** — gold `4`

- ✓ `plain`: `4

Q: confident
A: 4`
- ✗ `chat_blank_system`: `3`
- ✗ `chat_no_system`: `3`

**q = `see`** — gold `1`

- ✓ `plain`: `1

Q: direct
A: 4`
- ✗ `chat_blank_system`: `2`
- ✗ `chat_no_system`: `2`

**q = `turn`** — gold `3`

- ✓ `plain`: `3

Q: which
A: 4`
- ✓ `chat_blank_system`: `3`
- ✗ `chat_no_system`: `1`

**q = `leading`** — gold `4`

- ✓ `plain`: `4

Q: epidermis
A:`
- ✗ `chat_blank_system`: `3`
- ✗ `chat_no_system`: `3`


### last_two_letters  (plain 0.30 / chat_blank_system 0.26 / chat_no_system 0.22)

**q = `attention`** — gold `on`

- ✓ `plain`: `on

Q: generation
A: ation

Q`
- ✓ `chat_blank_system`: `on`
- ✗ `chat_no_system`: `ion`

**q = `over`** — gold `er`

- ✓ `plain`: `er

Q: phone
A: en

Q:`
- ✗ `chat_blank_system`: `ver`
- ✗ `chat_no_system`: `ver`

**q = `take`** — gold `ke`

- ✓ `plain`: `ke

Q: examination
A: am

Q:`
- ✗ `chat_blank_system`: `To continue accurately, could you please provide more context or finish`
- ✗ `chat_no_system`: `care of a growing family, you might consider the following aspects`

**q = `explain`** — gold `in`

- ✓ `plain`: `in

Q: development
A: ev

Q:`
- ✗ `chat_blank_system`: `Sure, could you please provide more context or specify what you`
- ✗ `chat_no_system`: `Sure, could you please provide more context? You seem to`

**q = `form`** — gold `rm`

- ✓ `plain`: `rm

Q: began
A: gn

Q:`
- ✗ `chat_blank_system`: `m`
- ✗ `chat_no_system`: `m`

**q = `supposed`** — gold `ed`

- ✓ `plain`: `ed

Q: business
A: ss

Q:`
- ✓ `chat_blank_system`: `ed`
- ✗ `chat_no_system`: `to be a fair and subject-related river discussion, could you`


### next_in_group  (plain 0.30 / chat_blank_system 0.06 / chat_no_system 0.10)

**q = `Selenium`** — gold `Tellurium`

- ✓ `plain`: `Tellurium

Q: Germanium
A:-Shirt`
- ✗ `chat_blank_system`: `Selenium is a chemical element with the symbol Se and atomic`
- ✗ `chat_no_system`: `Selenium is a chemical element with the symbol Se and atomic`

**q = `Lanthanum`** — gold `Actinium`

- ✓ `plain`: `Actinium

Q: Selenium
A: Tellurium`
- ✗ `chat_blank_system`: `Lanthanum is a chemical element with the symbol La`
- ✗ `chat_no_system`: `Promethium`

**q = `Gallium`** — gold `Indium`

- ✓ `plain`: `Indium

Q: Germanium
A: Arsen`
- ✗ `chat_blank_system`: `Germanium`
- ✗ `chat_no_system`: `Germanium`

**q = `Cobalt`** — gold `Rhodium`

- ✓ `plain`: `Rhodium

Q: Tellurium
A: I`
- ✗ `chat_blank_system`: `Nickel`
- ✗ `chat_no_system`: `Nickel`

**q = `Rhenium`** — gold `Bohrium`

- ✓ `plain`: `Bohrium

Q: Beryllium
A:`
- ✗ `chat_blank_system`: `Roentgenium`
- ✗ `chat_no_system`: `Osmium`

**q = `Lanthanum`** — gold `Actinium`

- ✓ `plain`: `Actinium

Q: Fluorine
A: Liver`
- ✗ `chat_blank_system`: `Lanthanum is a chemical element with the symbol La`
- ✗ `chat_no_system`: `Cerium`


### person-instrument  (plain 0.44 / chat_blank_system 0.16 / chat_no_system 0.24)

**q = `Horace Parlan`** — gold `piano`

- ✓ `plain`: `piano

Q: Brian Johnson
A: vocals

Q`
- ✗ `chat_blank_system`: `Horace Parlan was an American-British jazz pian`
- ✗ `chat_no_system`: `paice`

**q = `Emilio Pujol`** — gold `guitar`

- ✓ `plain`: `guitar

Q: Wayne Horvitz
A: piano`
- ✗ `chat_blank_system`: `Emilio Pujol is a renowned Spanish violinist and`
- ✗ `chat_no_system`: `Emilio Pujol is a prominent Catalan cellist,`

**q = `Josephine Wiggs`** — gold `guitar`

- ✓ `plain`: `guitar

Q: Ravi Coltrane
A:`
- ✗ `chat_blank_system`: `Josephine Wiggs is a Trinidadian-Trinidadian-B`
- ✗ `chat_no_system`: `vocals and bass guitar

Josephine Wiggs is`

**q = `Dave Sinclair`** — gold `piano`

- ✓ `plain`: `piano

Q: Stewart Goodyear
A: piano`
- ✗ `chat_blank_system`: `Without additional context, it's difficult to determine exactly which Dave`
- ✗ `chat_no_system`: `Guitar`

**q = `Melissa Etheridge`** — gold `guitar`

- ✓ `plain`: `guitar

Q: Jimmy Dingle
A: bass`
- ✗ `chat_blank_system`: `Singer, songwriter, guitarist`
- ✗ `chat_no_system`: `Melissa Etheridge is a Grammy Award-winning American singer-song`

**q = `Nigel Pulsford`** — gold `guitar`

- ✓ `plain`: `guitar

Q: Lisa Fischer
A: vocals

Q`
- ✗ `chat_blank_system`: `Nigel Pulsford was an English writer and journalist,`
- ✗ `chat_no_system`: `Nigel Pulsford was a British drummer best known for`


### synonym  (plain 0.34 / chat_blank_system 0.26 / chat_no_system 0.28)

**q = `framing`** — gold `structuring`

- ✓ `plain`: `structuring

Q: extreme
A: radical

Given`
- ✗ `chat_blank_system`: `composition`
- ✗ `chat_no_system`: `composition`

**q = `combine`** — gold `merge`

- ✓ `plain`: `merge

Q: confrontation
A: opposition

Q:`
- ✓ `chat_blank_system`: `merge`
- ✗ `chat_no_system`: `meld`

**q = `say`** — gold `speak`

- ✓ `plain`: `speak

Q: tedious
A: droning

Q`
- ✗ `chat_blank_system`: `To say something means to speak or express thoughts or ideas.`
- ✗ `chat_no_system`: `Sure, what would you like to say?`

**q = `steady`** — gold `stable`

- ✓ `plain`: `stable

Q: tremble
A: shake

Q`
- ✓ `chat_blank_system`: `stable`
- ✗ `chat_no_system`: `稳定的 (stable)`

**q = `ogre`** — gold `monster`

- ✓ `plain`: `monster

Q: unicorn
A: mythological horse`
- ✗ `chat_blank_system`: `Ogre is a creature from folklore and fantasy literature, typically`
- ✗ `chat_no_system`: `ogre`

**q = `intentional`** — gold `deliberate`

- ✓ `plain`: `deliberate

Q: excite
A: stimulate

Q`
- ✗ `chat_blank_system`: `purposive`
- ✗ `chat_no_system`: `purposive`


## Contrast: tasks where the chat arms BEAT plain

Examples favour prompts where chat_no_system succeeded and plain failed.


### last_digit  (plain 0.76 / chat_blank_system 0.98 / chat_no_system 0.94)

**q = `854`** — gold `4`

- ✗ `plain`: `8

Q: 8051
A`
- ✓ `chat_blank_system`: `4`
- ✓ `chat_no_system`: `4`

**q = `2359`** — gold `9`

- ✗ `plain`: `5

Q: 80
A:`
- ✓ `chat_blank_system`: `9`
- ✓ `chat_no_system`: `9`

**q = `316`** — gold `6`

- ✗ `plain`: `16

Q: 78462`
- ✓ `chat_blank_system`: `6`
- ✓ `chat_no_system`: `6`

**q = `14337`** — gold `7`

- ✗ `plain`: `3

Q: 294799`
- ✓ `chat_blank_system`: `7`
- ✓ `chat_no_system`: `7`

**q = `43`** — gold `3`

- ✗ `plain`: `4

Q: 53554`
- ✓ `chat_blank_system`: `3`
- ✓ `chat_no_system`: `3`

**q = `8131`** — gold `1`

- ✗ `plain`: `4

Q: 278
A:`
- ✓ `chat_blank_system`: `1`
- ✓ `chat_no_system`: `1`


### french_noun_gender  (plain 0.84 / chat_blank_system 0.90 / chat_no_system 0.98)

**q = `rythme`** — gold `masculine`

- ✗ `plain`: `masculine, but can also be used as feminine in some contexts`
- ✓ `chat_blank_system`: `masculine`
- ✓ `chat_no_system`: `masculine`

**q = `truc`** — gold `masculine`

- ✗ `plain`: `masculin or neuter (considered neutral in modern French`
- ✗ `chat_blank_system`: `Ce mot peut être utilisé en tant que masculin ou fé`
- ✓ `chat_no_system`: `masculine`

**q = `développement`** — gold `masculine`

- ✗ `plain`: `masculine, but often treated as feminine in modern French

Q`
- ✓ `chat_blank_system`: `masculine`
- ✓ `chat_no_system`: `masculine`

**q = `institut`** — gold `masculine`

- ✗ `plain`: `masculine or feminine, depends on the context (usually masculine)`
- ✓ `chat_blank_system`: `masculine`
- ✓ `chat_no_system`: `masculine`

**q = `enfance`** — gold `feminine`

- ✗ `plain`: `feminine Your classification of the French words is correct! Here they`
- ✓ `chat_blank_system`: `feminine`
- ✓ `chat_no_system`: `feminine`

**q = `nom`** — gold `masculine`

- ✗ `plain`: `masculine (it can also be used as a feminine noun,`
- ✗ `chat_blank_system`: `Masculine

Note: In French, "nom"`
- ✓ `chat_no_system`: `masculine`


### capitalize_last_letter  (plain 0.26 / chat_blank_system 0.50 / chat_no_system 0.40)

**q = `practice`** — gold `E`

- ✗ `plain`: `C

Sure, it looks like you might be describing letters`
- ✓ `chat_blank_system`: `E`
- ✓ `chat_no_system`: `E`

**q = `ferret`** — gold `T`

- ✗ `plain`: `R

Q: autumn
A: M

Q:`
- ✓ `chat_blank_system`: `T`
- ✓ `chat_no_system`: `T`

**q = `kiss`** — gold `S`

- ✗ `plain`: `E

Q: music
A: C

Instructions:`
- ✓ `chat_blank_system`: `S`
- ✓ `chat_no_system`: `S`

**q = `argue`** — gold `E`

- ✗ `plain`: `G

Q: strong
A: O

Q:`
- ✓ `chat_blank_system`: `E`
- ✓ `chat_no_system`: `E`

**q = `forgive`** — gold `E`

- ✗ `plain`: `G

Q: brilliant
A: I

Q:`
- ✓ `chat_blank_system`: `E`
- ✓ `chat_no_system`: `E`

**q = `make`** — gold `E`

- ✗ `plain`: `K

Q: viva
A: V

Q`
- ✓ `chat_blank_system`: `E`
- ✓ `chat_no_system`: `E`
