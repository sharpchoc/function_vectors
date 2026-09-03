# Failure modes under own-direction ($\hat u_A$) mean-ablation

Buckets: see `ablation_failure_modes.py`. Each example: query input → gold | own-ablated prediction [bucket] | counterfactual-ablated prediction.


## 6-shot


### adjective_to_adverb  (counterfactual direction: french-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 108 |
| copy_query | 69 | 5 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 6 |
| input_variant | 39 | 21 |
| other | 42 | 10 |

Examples (own-ablation errors):

- `admirable` → **admirably** | own: `admirable` [copy_query] | cf: `admirable`
- `scarce` → **scarcely** | own: `scarce` [copy_query] | cf: `scarcely`
- `lucky` → **luckily** | own: `lonely` [other] | cf: `luckyly`
- `psychical` → **psychically** | own: `psychical` [copy_query] | cf: `psychically`
- `conclusive` → **conclusively** | own: `conclusive` [copy_query] | cf: `concludingly`
- `dark` → **darkly** | own: `dark` [copy_query] | cf: `darkly`
- `lyric` → **lyrically** | own: `ry.` [other] | cf: `lyrically`
- `terrific` → **terrifically** | own: `terrific` [copy_query] | cf: `tigriently`
- `expectant` → **expectantly** | own: `expecant` [input_variant] | cf: `expectantly`
- `defensive` → **defensively** | own: `defenssive` [input_variant] | cf: `defensively`

### adjective_to_noun  (counterfactual direction: english-portuguese)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 21 | 55 |
| copy_query | 50 | 36 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 3 | 2 |
| input_variant | 14 | 24 |
| other | 61 | 33 |

Examples (own-ablation errors):

- `satirical` → **satire** | own: `satirical` [copy_query] | cf: `satyr`
- `wooden` → **wood** | own: `wooden` [copy_query] | cf: `wood`
- `expansive` → **expanse** | own: `expansive` [copy_query] | cf: `expanse`
- `adventurous` → **adventure** | own: `adventurous` [copy_query] | cf: `adventurer`
- `liturgical` → **liturgy** | own: `liturgical` [copy_query] | cf: `liturgy`
- `asymmetric` → **asymmetry** | own: `asymmetrical` [input_variant] | cf: `asym`
- `organic` → **organ** | own: `organic` [copy_query] | cf: `org`
- `regimental` → **regiment** | own: `the military land acquisition brigade` [other] | cf: `regiment`
- `historical` → **history** | own: `historical` [copy_query] | cf: `historical`
- `trusty` → **trust** | own: `tru?ital` [other] | cf: `trusty (2)[`

### ag_news  (counterfactual direction: english-french)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 23 | 61 |
| copy_query | 0 | 0 |
| copy_demo_target | 32 | 38 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 2 |
| input_variant | 16 | 6 |
| other | 79 | 43 |

Examples (own-ablation errors):

- `Koch, Park leads first day in Nine Bridges Classic Carin Koch of Sweden and South Korea #39;s Grace Park both shot a 6-under-par 66 on Friday, taking the lead after the first round of the LPGA #39;s CJ Nine Bridges Classic.` → **Sports** | own: `Nature` [other] | cf: `Sports`
- `Grand Jury Adds to HealthSouth Charges Federal prosecutors yesterday announced new perjury and obstruction-of-justice charges against HealthSouth Corp. founder Richard M. Scrushy, accusing the former chief executive of the rehabilitation ` → **Business** | own: `east Alabama medical supplies` [other] | cf: `Business`
- `Weah returns to Liberia Liberian legend George Weah returns to Liberia to launch his bid for the country's presidency.` → **World** | own: `practice` [other] | cf: `Sports`
- `Taliban Wanted List to Be Drawn (AP) AP - The United States could cut its forces in Afghanistan next summer if Taliban militants accept an amnesty to be drawn up by President Hamid Karzai and neighboring Pakistan, the senior U.S. commander here said Sunday.` → **World** | own: `toter` [other] | cf: `World`
- `Has Your Broadband Had Its Fiber? Falling costs, new technology, and competition, with a nudge from regulatory changes, are bringing fiber closer to homes in the US just a few years after the idea seemed all but written off.` → **Science** | own: `broadband` [input_variant] | cf: `Technology`
- `Lehman relishes chance to halt US slide AFTER being named as the 2006 US Ryder Cup team captain by the PGA of America at a press conference in Florida last night, Tom Lehman insisted he saw the chance to halt Americas recent dismal showing in the biennial match with Europe as an opportunity ` → **Sports** | own: `two` [other] | cf: `Business`
- `Can't Hide Your Lying ... Face? In search of the ultimate lie detector, researchers turn to thermal facial scans, brain wiring and eyeball tracking. But deception still, well, deceives. By Randy Dotinga.` → **Science** | own: `It` [other] | cf: `Discovery`
- `Hamilton Wins Cycling Time Trial Event THENS, Aug. 18  Tyler Hamilton had bruises splotched all over his back, painful souvenirs of a Tour de France gone terribly wrong. ` → **Sports** | own: `Tour` [input_variant] | cf: `Sports`
- `Detainees seen as minimal threat WASHINGTON -- Most of the alleged Al Qaeda and Taliban inmates at the US military prison at Guantanamo Bay, Cuba, are likely to be freed or sent to their home countries for further investigation because many pose little threat and are not providing much valuable intelligence, the facility's deputy commander has said.` → **World** | own: `Sports` [copy_demo_target] | cf: `World`
- `Butt waits on Uefa ruling Newcastle midfielder Nicky Butt is facing up to the possibility of a European three-match ban. The 29-year-old was sent off during Newcastle #39;s 2-0 Uefa Cup win against Hapoel Bnei Sakhnin for grabbing Abas Suan by the throat.` → **Sports** | own: `game` [other] | cf: `Sports`

### agent_noun_to_verb  (counterfactual direction: number_word_to_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 25 | 84 |
| copy_query | 37 | 18 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 6 | 5 |
| input_variant | 24 | 15 |
| other | 57 | 28 |

Examples (own-ablation errors):

- `breather` → **breathe** | own: `bypasser` [other] | cf: `breather`
- `compiler` → **compile** | own: `commentator` [other] | cf: `compile`
- `defender` → **defend** | own: `dapper` [other] | cf: `defend`
- `speeder` → **speed** | own: `spd` [other] | cf: `speedy`
- `bugger` → **bug** | own: `bug, bother` [other] | cf: `g—`
- `loser` → **lose** | own: `um, lose` [other] | cf: `lose`
- `ambler` → **amble** | own: `walk` [other] | cf: `amble`
- `copier` → **copy** | own: `cop` [other] | cf: `copy`
- `plotter` → **plot** | own: `plotter` [copy_query] | cf: `ploat`
- `caterer` → **cater** | own: `caterer` [copy_query] | cf: `cate`

### animal_class  (counterfactual direction: spanish-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 8 | 111 |
| copy_query | 9 | 0 |
| copy_demo_target | 4 | 10 |
| copy_demo_input | 1 | 1 |
| in_pool_wrong | 0 | 3 |
| input_variant | 13 | 2 |
| other | 115 | 23 |

Examples (own-ablation errors):

- `cownose ray` → **fish** | own: `file` [other] | cf: `fish`
- `mako` → **fish** | own: `fish oil` [other] | cf: `tuna`
- `fer de lance` → **reptile** | own: `perch` [other] | cf: `fish`
- `gophersnake` → **reptile** | own: `gopher` [input_variant] | cf: `reptile`
- `taipan` → **reptile** | own: `ppallary` [other] | cf: `reptile`
- `hornet` → **insect** | own: `honey` [other] | cf: `insect`
- `elk` → **mammal** | own: `pulmentar` [other] | cf: `mammal`
- `manta ray` → **fish** | own: `solicitor` [other] | cf: `fish`
- `japanese hornet` → **insect** | own: `hornet` [input_variant] | cf: `insect`
- `shrew` → **mammal** | own: `shrew` [copy_query] | cf: `insect`

### animal_plant_object  (counterfactual direction: antonym)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 13 | 99 |
| copy_query | 13 | 1 |
| copy_demo_target | 9 | 3 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 9 | 1 |
| other | 105 | 46 |

Examples (own-ablation errors):

- `okapi` → **animal** | own: `okap` [input_variant] | cf: `animal`
- `iguana` → **animal** | own: `porcupine` [copy_demo_input] | cf: `animal`
- `cougar` → **animal** | own: `coukey` [other] | cf: `animal`
- `serval` → **animal** | own: `leopard` [other] | cf: `animal`
- `printer` → **object** | own: `iographer` [other] | cf: `object`
- `mockorange` → **plant** | own: `shanghai` [other] | cf: `plant`
- `angelfish` → **animal** | own: `catfish` [other] | cf: `animal`
- `bermudagrass` → **plant** | own: `badger` [other] | cf: `plant`
- `clownfish` → **animal** | own: `leaf` [other] | cf: `creature`
- `waxwing` → **animal** | own: `abloom` [other] | cf: `bird`

### antonym  (counterfactual direction: next_number_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 18 | 54 |
| copy_query | 13 | 4 |
| copy_demo_target | 3 | 0 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 18 | 17 |
| input_variant | 6 | 8 |
| other | 91 | 67 |

Examples (own-ablation errors):

- `due` → **overdue** | own: `do` [in_pool_wrong] | cf: `owed`
- `employee` → **employer** | own: `recruiter` [other] | cf: `employer`
- `conception` → **birth** | own: `creation` [other] | cf: `birth`
- `withdrawal` → **deposit** | own: `anxious` [other] | cf: `inhalation`
- `island` → **mainland** | own: `continent` [other] | cf: `continent`
- `unveil` → **conceal** | own: `reveal` [in_pool_wrong] | cf: `obscure`
- `medium` → **large** | own: `head` [other] | cf: `large`
- `recorded` → **live** | own: `ordinary` [other] | cf: `live`
- `postpone` → **advance** | own: `elect` [other] | cf: `blow the hot air`
- `rely` → **distrust** | own: `faith` [other] | cf: `mistrust`

### article_choice  (counterfactual direction: prev_number_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 80 |
| copy_query | 5 | 0 |
| copy_demo_target | 0 | 52 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 0 | 1 |
| other | 145 | 17 |

Examples (own-ablation errors):

- `aisle` → **an** | own: `man` [other] | cf: `an`
- `pat` → **a** | own: `pone` [other] | cf: `an`
- `rubble` → **a** | own: `fence` [other] | cf: `the`
- `complex` → **a** | own: `coerse` [other] | cf: `a`
- `scope` → **a** | own: `game` [other] | cf: `an`
- `balcony` → **a** | own: `alao` [other] | cf: `curtain`
- `print` → **a** | own: `wast` [other] | cf: `an`
- `politics` → **a** | own: `port` [other] | cf: `a`
- `upright` → **an** | own: `with` [other] | cf: `a`
- `oxygen` → **an** | own: `warmer` [other] | cf: `an`

### capitalize  (counterfactual direction: prev_number_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 112 | 142 |
| copy_query | 0 | 0 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 5 | 0 |
| input_variant | 16 | 7 |
| other | 17 | 1 |

Examples (own-ablation errors):

- `navigate` → **Navigate** | own: `navegate` [other] | cf: `Navigate`
- `good` → **Good** | own: `go` [other] | cf: `Good`
- `change` → **Change** | own: `shook` [other] | cf: `Change`
- `crab` → **Crab** | own: `crab apple` [input_variant] | cf: `Crab`
- `light` → **Light** | own: `lighten` [input_variant] | cf: `Light`
- `amiable` → **Amiable** | own: `amiability` [input_variant] | cf: `Amiable`
- `cactus` → **Cactus** | own: `cat-o-utas` [other] | cf: `Cactus`
- `fossa` → **Fossa** | own: `Fosa` [other] | cf: `Fossa`
- `naughty` → **Naughty** | own: `Naive` [in_pool_wrong] | cf: `Naughty`
- `those` → **Those** | own: `it` [in_pool_wrong] | cf: `Thos haha`

### capitalize_first_letter  (counterfactual direction: sentiment)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 88 |
| copy_query | 23 | 0 |
| copy_demo_target | 3 | 9 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 4 | 40 |
| input_variant | 19 | 1 |
| other | 99 | 12 |

Examples (own-ablation errors):

- `violin` → **V** | own: `violin` [copy_query] | cf: `D`
- `underneath` → **U** | own: `under` [input_variant] | cf: `B`
- `maracas` → **M** | own: `mara` [input_variant] | cf: `R`
- `modest` → **M** | own: `mind` [other] | cf: `M`
- `humble` → **H** | own: `humble` [copy_query] | cf: `HUV`
- `salty` → **S** | own: `m___y` [other] | cf: `S`
- `from` → **F** | own: `of` [other] | cf: `F`
- `as` → **A** | own: `is` [other] | cf: `S`
- `incense` → **I** | own: `r` [copy_demo_target] | cf: `M`
- `exchange` → **E** | own: `vice versa` [other] | cf: `X`

### city-country  (counterfactual direction: english-italian)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 105 |
| copy_query | 26 | 0 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 17 |
| input_variant | 15 | 0 |
| other | 98 | 26 |

Examples (own-ablation errors):

- `Battagram` → **Pakistan** | own: `Attrayagram` [other] | cf: `Pakistan`
- `N'Djamena` → **Chad** | own: `Djamana` [other] | cf: `Chad`
- `Novi Sad` → **Serbia** | own: `Novsadi` [other] | cf: `Serbia`
- `Guarulhos` → **Brazil** | own: `Guarulhos` [copy_query] | cf: `Brazil`
- `Calgary` → **Canada** | own: `Cal` [other] | cf: `Canada`
- `Vina del Mar` → **Chile** | own: `Vino` [other] | cf: `Chile`
- `Nanning` → **China** | own: `Guangdong` [other] | cf: `China`
- `Munich` → **Germany** | own: `Longhua` [other] | cf: `Germany`
- `Kalemyo` → **Myanmar** | own: `Kalimantan` [other] | cf: `Mongolia`
- `Sao Bernardo do Campo` → **Brazil** | own: `bernardodacamp` [other] | cf: `Brazil`

### compound_first  (counterfactual direction: person-sport)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 32 | 75 |
| copy_query | 26 | 8 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 12 | 11 |
| input_variant | 27 | 20 |
| other | 53 | 36 |

Examples (own-ablation errors):

- `bombshell` → **bomb** | own: `bombshell` [copy_query] | cf: `bombshell`
- `overhead` → **over** | own: `overh` [input_variant] | cf: `overh`
- `flashback` → **flash** | own: `bluff` [other] | cf: `flic`
- `timepiece` → **time** | own: `watch` [in_pool_wrong] | cf: `time`
- `wingman` → **wing** | own: `wingman` [copy_query] | cf: `wing`
- `password` → **pass** | own: `p-a-` [other] | cf: `pass`
- `washboard` → **wash** | own: `washboard` [copy_query] | cf: `washtub`
- `granddaughter` → **grand** | own: `grand singer` [input_variant] | cf: `grap`
- `limestone` → **lime** | own: `stone` [in_pool_wrong] | cf: `limestone`
- `nightlight` → **night** | own: `night shade` [input_variant] | cf: `lighten`

### concrete_abstract  (counterfactual direction: verb_tense_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 5 | 71 |
| copy_query | 3 | 1 |
| copy_demo_target | 1 | 50 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 2 | 0 |
| other | 139 | 28 |

Examples (own-ablation errors):

- `delight` → **abstract** | own: `ambulance` [other] | cf: `abstract`
- `stress` → **abstract** | own: `accord` [other] | cf: `concrete`
- `soil` → **concrete** | own: `squall` [other] | cf: `concrete`
- `brother` → **concrete** | own: `Lord` [other] | cf: `concrete`
- `melancholy` → **abstract** | own: `paranoia` [other] | cf: `abstract`
- `thistle` → **concrete** | own: `thumb` [other] | cf: `concrete`
- `wheelbarrow` → **concrete** | own: `bus` [other] | cf: `concrete`
- `hill` → **concrete** | own: `burger` [other] | cf: `concrete`
- `iceberg` → **concrete** | own: `architect` [other] | cf: `barge`
- `truck` → **concrete** | own: `truck` [copy_query] | cf: `truck`

### contains_letter_e  (counterfactual direction: iso_date_to_month)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 16 | 72 |
| copy_query | 0 | 0 |
| copy_demo_target | 23 | 63 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 2 | 0 |
| other | 109 | 15 |

Examples (own-ablation errors):

- `composite` → **yes** | own: `oil` [other] | cf: `yes`
- `precise` → **yes** | own: `and` [other] | cf: `no`
- `shallow` → **no** | own: `my` [other] | cf: `yes`
- `scroll` → **no** | own: `essay` [other] | cf: `no`
- `spiritual` → **no** | own: `s` [other] | cf: `yes`
- `establish` → **yes** | own: `ouch` [other] | cf: `yes`
- `havoc` → **no** | own: `comply` [other] | cf: `yes`
- `planet` → **yes** | own: `no` [copy_demo_target] | cf: `yes`
- `reliably` → **yes** | own: `right` [other] | cf: `yes`
- `intensity` → **yes** | own: `aye` [other] | cf: `no`

### country-capital  (counterfactual direction: sentiment)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 4 | 115 |
| copy_query | 32 | 1 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 34 | 2 |
| other | 80 | 32 |

Examples (own-ablation errors):

- `United States of America` → **Washington, D.C.** | own: `United States` [input_variant] | cf: `Washington, D.C.`
- `Burundi` → **Bujumbura** | own: `Benisi` [other] | cf: `Bujumbura`
- `Taiwan` → **Taipei** | own: `T'aiwan` [other] | cf: `Taipei`
- `Ireland` → **Dublin** | own: `Hibern on` [other] | cf: `Dublin`
- `Tuvalu` → **Funafuti** | own: `Tatau` [other] | cf: `Nukulaelae`
- `Zimbabwe` → **Harare** | own: `Zimbabwe` [copy_query] | cf: `Harare`
- `Slovenia` → **Ljubljana** | own: `soile` [other] | cf: `Ljubljana`
- `Nigeria` → **Abuja** | own: `Ikwerre` [other] | cf: `Abuja`
- `Chile` → **Santiago** | own: `Kolkaitedju` [other] | cf: `Santiago`
- `Bahamas` → **Nassau** | own: `Bahamas` [copy_query] | cf: `Nassau`

### day_after_textual_date  (counterfactual direction: first_digit)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 141 |
| copy_query | 1 | 1 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 63 | 3 |
| input_variant | 76 | 4 |
| other | 2 | 1 |

Examples (own-ablation errors):

- `June 3, 2049` → **June 4** | own: `June 3rd` [input_variant] | cf: `June 4`
- `June 26, 1933` → **June 27** | own: `June 28` [input_variant] | cf: `June 27`
- `November 8, 1913` → **November 9** | own: `November 21` [copy_demo_target] | cf: `November 9`
- `October 2, 1907` → **October 3** | own: `October 11` [in_pool_wrong] | cf: `October 3`
- `June 18, 1926` → **June 19** | own: `June 28, 1927` [input_variant] | cf: `June 19`
- `February 27, 2016` → **February 28** | own: `February 29 delivered` [input_variant] | cf: `March 1`
- `November 24, 2089` → **November 25** | own: `November 2089` [input_variant] | cf: `November 25`
- `December 16, 1973` → **December 17** | own: `December 19` [in_pool_wrong] | cf: `December 17`
- `September 5, 2056` → **September 6** | own: `September 60` [input_variant] | cf: `September 6`
- `April 13, 2023` → **April 14** | own: `April 24` [in_pool_wrong] | cf: `April 14`

### ends_with_ing  (counterfactual direction: next_month_of_date)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 14 | 73 |
| copy_query | 0 | 0 |
| copy_demo_target | 32 | 63 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 0 | 0 |
| other | 103 | 14 |

Examples (own-ablation errors):

- `funk` → **no** | own: `forth` [other] | cf: `no`
- `revolting` → **yes** | own: `no` [copy_demo_target] | cf: `no`
- `terrain` → **no** | own: `often enough` [other] | cf: `yes`
- `irregular` → **no** | own: `error` [other] | cf: `yes`
- `troughing` → **yes** | own: `why` [other] | cf: `yes`
- `offsetting` → **yes** | own: `best` [other] | cf: `no`
- `improvement` → **no** | own: `year` [other] | cf: `yes`
- `culturing` → **yes** | own: `no` [copy_demo_target] | cf: `no`
- `fatting` → **yes** | own: `doll` [other] | cf: `yes`
- `fitness` → **no** | own: `sailor` [other] | cf: `know for my music`

### english-french  (counterfactual direction: lowercase_word)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 98 |
| copy_query | 29 | 0 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 3 |
| input_variant | 43 | 5 |
| other | 76 | 44 |

Examples (own-ablation errors):

- `failed` → **échoué** | own: `fecked` [other] | cf: `échec`
- `animals` → **animaux** | own: `quadrupeds` [other] | cf: `animaux`
- `landed` → **atterri** | own: `monied` [other] | cf: `terrien`
- `explanation` → **explication** | own: `expla-nation` [input_variant] | cf: `explication`
- `decades` → **décennies** | own: `decades' old` [input_variant] | cf: `d'écervelée`
- `garden` → **jardin** | own: `gardén` [input_variant] | cf: `jardin`
- `turned` → **tourné** | own: `turnt` [input_variant] | cf: `tourne`
- `soccer` → **football** | own: `soccerh` [input_variant] | cf: `football`
- `killed` → **tué** | own: `killer` [input_variant] | cf: `tué`
- `side` → **côté** | own: `sid` [other] | cf: `côté`

### english-italian  (counterfactual direction: plural_to_singular)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 83 |
| copy_query | 27 | 1 |
| copy_demo_target | 0 | 1 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 5 |
| input_variant | 24 | 5 |
| other | 97 | 55 |

Examples (own-ablation errors):

- `knee` → **ginocchio** | own: `gauch` [other] | cf: `ginocchio`
- `play` → **giocare** | own: `pea` [other] | cf: `gioca`
- `voice` → **voce** | own: `voice` [copy_query] | cf: `voce`
- `want` → **volere** | own: `wan` [other] | cf: `voglio`
- `magazine` → **rivista** | own: `conception` [other] | cf: `rivista`
- `worm` → **verme** | own: `scurf` [other] | cf: `carrozza`
- `relative` → **relativo** | own: `relaative` [input_variant] | cf: `pari`
- `skinny` → **magro** | own: `thin` [other] | cf: `magro`
- `identical` → **identico** | own: `identical` [copy_query] | cf: `identico`
- `rural` → **rurale** | own: `harshus` [other] | cf: `minimo`

### english-portuguese  (counterfactual direction: adjective_to_adverb)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 83 |
| copy_query | 19 | 1 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 7 |
| input_variant | 22 | 5 |
| other | 107 | 54 |

Examples (own-ablation errors):

- `beach` → **praia** | own: `beacht` [input_variant] | cf: `praia`
- `holiday` → **feriado** | own: `haelter` [other] | cf: `festa`
- `cap` → **boné** | own: `cap` [copy_query] | cf: `espada`
- `agree` → **concordar** | own: `congiere` [other] | cf: `concordar`
- `messy` → **desarrumado** | own: `mauvaize` [other] | cf: `bagunço`
- `assume` → **assumir** | own: `assuma` [input_variant] | cf: `assumi`
- `king` → **rei** | own: `rey` [other] | cf: `rei`
- `salt` → **sal** | own: `dos salt` [input_variant] | cf: `sal`
- `musician` → **músico** | own: `cirgunist` [other] | cf: `músico`
- `eight` → **oito** | own: `ocho` [other] | cf: `ocho`

### english-spanish  (counterfactual direction: adjective_to_adverb)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 94 |
| copy_query | 31 | 5 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 7 |
| input_variant | 34 | 6 |
| other | 84 | 38 |

Examples (own-ablation errors):

- `searching` → **buscando** | own: `searching out` [input_variant] | cf: `buscando`
- `bus` → **autobús** | own: `humvee` [other] | cf: `autobús`
- `nothing` → **nada** | own: `nuthin` [other] | cf: `nada`
- `items` → **artículos** | own: `thing` [other] | cf: `artículos`
- `anniversary` → **aniversario** | own: `annumie` [other] | cf: `aniversario`
- `mayors` → **alcaldes** | own: `mayors` [copy_query] | cf: `balotaje`
- `insects` → **insectos** | own: `entesmin` [other] | cf: `insectos`
- `starting` → **comenzando** | own: `mouth to mouth` [other] | cf: `inicio`
- `call` → **llamada** | own: `cal` [other] | cf: `llamada`
- `rules` → **reglas** | own: `rulses` [other] | cf: `reglas`

### first_digit  (counterfactual direction: word_polarity)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 23 | 96 |
| copy_query | 10 | 1 |
| copy_demo_target | 16 | 25 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 9 | 14 |
| input_variant | 1 | 0 |
| other | 91 | 14 |

Examples (own-ablation errors):

- `2008` → **2** | own: `2008` [copy_query] | cf: `2`
- `308470` → **3** | own: `785` [other] | cf: `3`
- `851` → **8** | own: `1` [copy_demo_target] | cf: `3`
- `8454` → **8** | own: `862` [other] | cf: `6`
- `6790` → **6** | own: `90` [other] | cf: `6`
- `8392` → **8** | own: `3` [in_pool_wrong] | cf: `3`
- `14` → **1** | own: `0` [other] | cf: `0`
- `902` → **9** | own: `2` [in_pool_wrong] | cf: `4`
- `557` → **5** | own: `1` [copy_demo_target] | cf: `10`
- `53830` → **5** | own: `575` [other] | cf: `5`

### first_three_letters  (counterfactual direction: singular_or_plural)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 12 | 57 |
| copy_query | 20 | 4 |
| copy_demo_target | 1 | 1 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 4 | 9 |
| input_variant | 23 | 21 |
| other | 90 | 58 |

Examples (own-ablation errors):

- `apart` → **apa** | own: `&#228;pc` [other] | cf: `asp`
- `professional` → **pro** | own: `professional` [copy_query] | cf: `prof`
- `search` → **sea** | own: `zo` [other] | cf: `searc`
- `tomorrow` → **tom** | own: `to-der` [other] | cf: `tom`
- `century` → **cen** | own: `centi` [input_variant] | cf: `cent`
- `speech` → **spe** | own: `saperas` [other] | cf: `s`
- `battle` → **bat** | own: `battail` [input_variant] | cf: `batl`
- `sometimes` → **som** | own: `somteen` [other] | cf: `so`
- `twitter` → **twi** | own: `twit` [input_variant] | cf: `twe`
- `eating` → **eat** | own: `eating` [copy_query] | cf: `eat`

### french-english  (counterfactual direction: language_identification)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 80 | 85 |
| copy_query | 4 | 4 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 5 |
| input_variant | 3 | 3 |
| other | 60 | 53 |

Examples (own-ablation errors):

- `forêt` → **forest** | own: `forêt` [copy_query] | cf: `woods`
- `avantage` → **advantage** | own: `bonus, bonus,` [other] | cf: `advantage`
- `chaumière` → **cottage** | own: `chambre` [other] | cf: `chimney`
- `soigné` → **neat** | own: `skilful` [other] | cf: `skilful`
- `bondé` → **crowded** | own: `full` [other] | cf: `crowded`
- `imaginer` → **imagine** | own: `I'm dreaming` [other] | cf: `to imagine`
- `sérieux` → **serious** | own: `sober` [other] | cf: `solemn`
- `garder` → **keep** | own: `guard` [other] | cf: `guard`
- `paix` → **peace** | own: `truce` [other] | cf: `peace`
- `cimetière` → **cemetery** | own: `graveyard` [other] | cf: `graveyard`

### french_noun_gender  (counterfactual direction: english-italian)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 17 | 85 |
| copy_query | 0 | 0 |
| copy_demo_target | 7 | 45 |
| copy_demo_input | 2 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 0 | 0 |
| other | 124 | 20 |

Examples (own-ablation errors):

- `domicile` → **masculine** | own: `heroine` [other] | cf: `feminine`
- `fac` → **feminine** | own: `pelvis` [other] | cf: `masculine`
- `entraînement` → **masculine** | own: `feminine` [copy_demo_target] | cf: `masculine`
- `accord` → **masculine** | own: `elected` [other] | cf: `masculine`
- `intelligence` → **feminine** | own: `dumable` [other] | cf: `masculine`
- `chaise` → **feminine** | own: `plain` [other] | cf: `regroupé de`
- `individu` → **masculine** | own: `manichéen` [other] | cf: `masculine`
- `empereur` → **masculine** | own: `division` [other] | cf: `feminine`
- `orient` → **masculine** | own: `fanciful` [other] | cf: `masculine`
- `amitié` → **feminine** | own: `bone` [other] | cf: `feminine`

### german-english  (counterfactual direction: german_noun_gender)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 79 | 96 |
| copy_query | 4 | 1 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 7 | 9 |
| input_variant | 4 | 0 |
| other | 56 | 44 |

Examples (own-ablation errors):

- `Rohr` → **pipe** | own: `dreher` [other] | cf: `pipe`
- `Ziege` → **goat** | own: `cow` [in_pool_wrong] | cf: `chicken`
- `angreifen` → **attack** | own: `attack\ assault` [other] | cf: `attack`
- `trennen` → **separate** | own: `sed` [other] | cf: `separate`
- `Affe` → **monkey** | own: `Katze` [other] | cf: `monkey`
- `taub` → **deaf** | own: `taibo` [other] | cf: `dead`
- `tragen` → **carry** | own: `wear` [other] | cf: `wear`
- `erhöhen` → **increase** | own: `raise` [other] | cf: `raise`
- `Erde` → **earth** | own: `soil` [other] | cf: `earth`
- `Geruch` → **smell** | own: `Gefühl` [other] | cf: `scent`

### german_noun_gender  (counterfactual direction: first_three_letters)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 8 | 65 |
| copy_query | 4 | 1 |
| copy_demo_target | 29 | 60 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 3 |
| input_variant | 4 | 1 |
| other | 104 | 20 |

Examples (own-ablation errors):

- `Potential` → **das** | own: `flamm` [other] | cf: `die`
- `Kontakt` → **der** | own: `die` [copy_demo_target] | cf: `das`
- `Konto` → **das** | own: `die` [copy_demo_target] | cf: `die`
- `Dating` → **das** | own: `daeter` [other] | cf: `den`
- `Zeichen` → **das** | own: `ist` [other] | cf: `die`
- `Fahrzeug` → **das** | own: `der` [copy_demo_target] | cf: `der`
- `Schatz` → **der** | own: `das` [copy_demo_target] | cf: `dass`
- `Kurs` → **der** | own: `desk` [other] | cf: `die`
- `Festival` → **das** | own: `fest` [input_variant] | cf: `das`
- `Abschied` → **der** | own: `das/der` [other] | cf: `die`

### gerund_to_base  (counterfactual direction: spanish-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 42 | 106 |
| copy_query | 45 | 3 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 4 |
| input_variant | 12 | 13 |
| other | 48 | 24 |

Examples (own-ablation errors):

- `watering` → **water** | own: `watering` [copy_query] | cf: `water`
- `bettering` → **better** | own: `boost` [in_pool_wrong] | cf: `better`
- `asking` → **ask** | own: `asking` [copy_query] | cf: `ask`
- `maintaining` → **maintain** | own: `maunatin` [other] | cf: `man`
- `truing` → **true** | own: `drilling` [other] | cf: `tune`
- `sugaring` → **sugar** | own: `sweeping` [other] | cf: `sugar`
- `watching` → **watch** | own: `viewing` [other] | cf: `watch`
- `shooting` → **shoot** | own: `firing` [other] | cf: `shoot`
- `viewing` → **view** | own: `viewing` [copy_query] | cf: `view`
- `stuffing` → **stuff** | own: `foocking` [other] | cf: `stuff`

### gerund_to_past  (counterfactual direction: compound_first)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 92 |
| copy_query | 28 | 29 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 0 |
| input_variant | 69 | 10 |
| other | 49 | 19 |

Examples (own-ablation errors):

- `laying` → **laid** | own: `lay` [other] | cf: `laying`
- `riding` → **rode** | own: `riding` [copy_query] | cf: `ridden`
- `assaulting` → **assaulted** | own: `assault` [input_variant] | cf: `assaulted`
- `effecting` → **effected** | own: `effect` [input_variant] | cf: `effected`
- `cornering` → **cornered** | own: `corner` [input_variant] | cf: `cornered`
- `passing` → **passed** | own: `pass` [input_variant] | cf: `passage`
- `spacing` → **spaced** | own: `spacing` [copy_query] | cf: `spaced`
- `bothering` → **bothered** | own: `bother` [input_variant] | cf: `bothered`
- `challenging` → **challenged** | own: `challenging` [copy_query] | cf: `challenging`
- `owning` → **owned** | own: `ownyng` [other] | cf: `owned`

### hypernym_category  (counterfactual direction: english-french)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 59 |
| copy_query | 24 | 0 |
| copy_demo_target | 0 | 1 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 2 |
| input_variant | 15 | 1 |
| other | 110 | 87 |

Examples (own-ablation errors):

- `bandsaw` → **tool** | own: `saw` [other] | cf: `machine used to make`
- `silo` → **building** | own: `bin` [other] | cf: `building`
- `motel` → **building** | own: `inn?` [other] | cf: `place to stay`
- `sneakers` → **clothing** | own: `shoes` [other] | cf: `clothing`
- `trumpet` → **instrument** | own: `tuba` [other] | cf: `instrument`
- `diving` → **sport** | own: `submerging` [other] | cf: `cheerfulness`
- `flute` → **instrument** | own: `flute` [copy_query] | cf: `instrument`
- `carriage` → **vehicle** | own: `horse` [other] | cf: `passenger wagon`
- `snooker` → **sport** | own: `shut` [other] | cf: `sport`
- `banker` → **profession** | own: `banker` [copy_query] | cf: `employee`

### initials_two_words  (counterfactual direction: person-instrument)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 90 |
| copy_query | 0 | 0 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 29 |
| input_variant | 16 | 0 |
| other | 131 | 29 |

Examples (own-ablation errors):

- `fake track` → **FT** | own: `feat` [other] | cf: `FF`
- `huge religion` → **HR** | own: `heat` [other] | cf: `HR`
- `actual ease` → **AE** | own: `Aze` [other] | cf: `AE`
- `isolated housing` → **IH** | own: `hooZZle` [other] | cf: `IH`
- `titled search` → **TS** | own: `bbl` [other] | cf: `TMS`
- `stunning script` → **SS** | own: `screet` [other] | cf: `SC`
- `modern majority` → **MM** | own: `Mod` [other] | cf: `NM`
- `slow scene` → **SS** | own: `sole` [other] | cf: `SN`
- `over fuel` → **OF** | own: `Turbo` [other] | cf: `OF`
- `separate designer` → **SD** | own: `bigincomeproject` [other] | cf: `SD`

### iso_date_to_month  (counterfactual direction: product-company)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 53 | 145 |
| copy_query | 0 | 0 |
| copy_demo_target | 4 | 2 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 0 |
| input_variant | 1 | 0 |
| other | 89 | 3 |

Examples (own-ablation errors):

- `2026-08-23` → **August** | own: `Actor` [other] | cf: `August`
- `2000-12-07` → **December** | own: `absorbed the "de` [other] | cf: `December`
- `1965-05-02` → **May** | own: `Lukening` [other] | cf: `May`
- `1988-08-28` → **August** | own: `Auger` [other] | cf: `August`
- `2092-11-03` → **November** | own: `Museum` [other] | cf: `October`
- `1990-03-08` → **March** | own: `Marching band` [other] | cf: `March`
- `1914-12-10` → **December** | own: `? dict` [other] | cf: `December`
- `1958-06-06` → **June** | own: `May` [copy_demo_target] | cf: `June`
- `2012-08-07` → **August** | own: `2022-09-` [other] | cf: `August`
- `1964-02-21` → **February** | own: `21` [other] | cf: `February`

### iso_date_year_plus_one  (counterfactual direction: initials_two_words)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 9 | 131 |
| copy_query | 2 | 0 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 83 | 11 |
| input_variant | 14 | 2 |
| other | 41 | 6 |

Examples (own-ablation errors):

- `2025-07-11` → **2026** | own: `2025-07-11` [copy_query] | cf: `2026`
- `1962-12-16` → **1963** | own: `1976-12` [other] | cf: `1963`
- `2031-01-02` → **2032** | own: `233` [other] | cf: `2032`
- `1887-09-21` → **1888** | own: `18book` [other] | cf: `1888`
- `2026-07-09` → **2027** | own: `fem date` [other] | cf: `2027`
- `1934-05-23` → **1935** | own: `1934` [in_pool_wrong] | cf: `1935`
- `2026-10-27` → **2027** | own: `???` [other] | cf: `2027`
- `1841-11-22` → **1842** | own: `1841` [in_pool_wrong] | cf: `1842`
- `1853-03-09` → **1854** | own: `1853` [input_variant] | cf: `1854`
- `1816-03-28` → **1817** | own: `1916` [in_pool_wrong] | cf: `1817`

### landmark-country  (counterfactual direction: adjective_to_adverb)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 75 |
| copy_query | 5 | 0 |
| copy_demo_target | 0 | 8 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 2 | 39 |
| input_variant | 27 | 1 |
| other | 114 | 27 |

Examples (own-ablation errors):

- `North Middlesex, Ontario` → **Canada** | own: `Peterborough` [other] | cf: `Northern Ontario`
- `Boguchany Dam` → **Russia** | own: `Astrakhan Region` [other] | cf: `Belarus`
- `Habitat 67` → **Canada** | own: `Habitat 67` [copy_query] | cf: `Canada`
- `Istanbul Airport` → **Turkey** | own: `Istanbul` [input_variant] | cf: `Turkey`
- `Addanki mandal` → **India** | own: `Srikakul` [other] | cf: `India`
- `Indus River` → **India** | own: `Indus` [input_variant] | cf: `India`
- `Penna Ahobilam` → **India** | own: `Andhra Pradesh` [other] | cf: `India`
- `Kresttsy` → **Russia** | own: `Quito` [other] | cf: `Russia`
- `Hooge Crater Commonwealth War Graves Commission Cemetery` → **Belgium** | own: `Hooge Dutch Cemetery` [input_variant] | cf: `United Kingdom`
- `Adliswil` → **Switzerland** | own: `Niederhorn` [other] | cf: `Switzerland`

### language_identification  (counterfactual direction: german_noun_gender)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 111 |
| copy_query | 5 | 0 |
| copy_demo_target | 3 | 24 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 1 |
| input_variant | 5 | 0 |
| other | 137 | 14 |

Examples (own-ablation errors):

- `reale` → **Italian** | own: `onion` [other] | cf: `Russian`
- `polizia` → **Italian** | own: `città` [other] | cf: `Italian`
- `energía` → **Spanish** | own: `grisa/ener` [other] | cf: `Spanish`
- `diciembre` → **Spanish** | own: `decembre` [other] | cf: `Spanish`
- `Natur` → **German** | own: `Espaniol` [other] | cf: `German`
- `buscar` → **Spanish** | own: `seu` [other] | cf: `Spanish`
- `intérieur` → **French** | own: `José` [other] | cf: `French`
- `faccia` → **Italian** | own: `faccia` [copy_query] | cf: `Italian`
- `deciso` → **Italian** | own: `ivo` [other] | cf: `French`
- `Gebäude` → **German** | own: `molto ricco` [other] | cf: `German`

### larger_of_pair  (counterfactual direction: singular_or_plural)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 67 | 85 |
| copy_query | 1 | 0 |
| copy_demo_target | 4 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 26 | 23 |
| input_variant | 1 | 0 |
| other | 51 | 42 |

Examples (own-ablation errors):

- `456 977` → **977** | own: `456` [in_pool_wrong] | cf: `977`
- `251 920` → **920** | own: `250` [other] | cf: `251`
- `928 923` → **928** | own: `923` [in_pool_wrong] | cf: `923`
- `151 685` → **685** | own: `165` [other] | cf: `685`
- `304 755` → **755** | own: `1 55` [other] | cf: `755`
- `952 726` → **952** | own: `952 726` [copy_query] | cf: `952`
- `772 858` → **858** | own: `772` [in_pool_wrong] | cf: `772`
- `928 155` → **928** | own: `9 or 28` [other] | cf: `928`
- `280 890` → **890** | own: `290` [other] | cf: `290`
- `435 312` → **435** | own: `413` [other] | cf: `312`

### larger_than_1000  (counterfactual direction: past_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 48 | 72 |
| copy_query | 0 | 0 |
| copy_demo_target | 32 | 64 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 0 | 0 |
| other | 70 | 14 |

Examples (own-ablation errors):

- `35194` → **yes** | own: `hey` [other] | cf: `no`
- `23305` → **yes** | own: `no, oh no` [other] | cf: `yes`
- `7986` → **yes** | own: `nay` [other] | cf: `no`
- `1804` → **yes** | own: `hey` [other] | cf: `no`
- `869` → **no** | own: `yes` [copy_demo_target] | cf: `no`
- `23936` → **yes** | own: `ingenio` [other] | cf: `no`
- `551` → **no** | own: `nos` [other] | cf: `yes`
- `7099` → **yes** | own: `no` [copy_demo_target] | cf: `no`
- `956` → **no** | own: `yeah` [other] | cf: `yes`
- `13998` → **yes** | own: `no` [copy_demo_target] | cf: `no`

### lowercase_first_letter  (counterfactual direction: country-capital)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 99 |
| copy_query | 17 | 1 |
| copy_demo_target | 6 | 7 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 32 |
| input_variant | 14 | 0 |
| other | 111 | 11 |

Examples (own-ablation errors):

- `EXPLORE` → **e** | own: `rec` [other] | cf: `e`
- `PATIENT` → **p** | own: `matte` [other] | cf: `p`
- `DYNAMIC` → **d** | own: `time` [other] | cf: `d`
- `COURAGEOUS` → **c** | own: `er` [other] | cf: `c`
- `WITHOUT` → **w** | own: `w-i-` [other] | cf: `i`
- `INNOCENT` → **i** | own: `uncle` [other] | cf: `b`
- `BABOON` → **b** | own: `ao` [other] | cf: `b`
- `MELLOW` → **m** | own: `white` [other] | cf: `m`
- `BEHIND` → **b** | own: `bh` [other] | cf: `h`
- `BELOW` → **b** | own: `base` [other] | cf: `b`

### lowercase_word  (counterfactual direction: product-company)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 120 | 144 |
| copy_query | 0 | 0 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 2 | 1 |
| input_variant | 6 | 4 |
| other | 22 | 1 |

Examples (own-ablation errors):

- `NEVER` → **never** | own: `NEEvER` [other] | cf: `never`
- `ISLAND` → **island** | own: `Isle` [other] | cf: `island`
- `ESPECIALLY` → **especially** | own: `especial` [input_variant] | cf: `especially`
- `FILM` → **film** | own: `movie` [in_pool_wrong] | cf: `film`
- `GOOD` → **good** | own: `goo` [other] | cf: `good`
- `QUICKLY` → **quickly** | own: `quickilie` [input_variant] | cf: `quickly`
- `ALMOST` → **almost** | own: `nearly` [in_pool_wrong] | cf: `almost`
- `FOR` → **for** | own: `as, for (` [other] | cf: `for`
- `FRENCH` → **french** | own: `france` [other] | cf: `french`
- `AFTER` → **after** | own: `adj. AFTER TH` [input_variant] | cf: `after`

### national_parks  (counterfactual direction: past_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 83 |
| copy_query | 0 | 0 |
| copy_demo_target | 5 | 13 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 43 |
| input_variant | 30 | 0 |
| other | 112 | 11 |

Examples (own-ablation errors):

- `Cumberland Gap National Historical Park` → **Virginia** | own: `Johnson City` [other] | cf: `Tennessee`
- `James A. Garfield National Historic Site` → **Ohio** | own: `Washington` [copy_demo_target] | cf: `Ohio`
- `Fort Scott National Historic Site` → **Kansas** | own: `Jefferson City` [other] | cf: `Kansas`
- `Theodore Roosevelt Inaugural National Historic Site` → **New York** | own: `Hyde Park, NY` [other] | cf: `Georgia`
- `Fort Necessity National Battlefield` → **Pennsylvania** | own: `Shelby` [other] | cf: `Pennsylvania`
- `Channel Islands National Park` → **California** | own: `Vladimir` [other] | cf: `California`
- `Fort Donelson National Battlefield` → **Tennessee** | own: `Donelson` [input_variant] | cf: `Tennessee`
- `Devils Postpile National Monument` → **California** | own: `North Caroor` [other] | cf: `California`
- `Wolf Trap National Park for the Performing Arts` → **Virginia** | own: `wolf trap` [input_variant] | cf: `Virginia`
- `Hubbell Trading Post National Historic Site` → **Arizona** | own: `Lincoln` [other] | cf: `New Mexico`

### natural_manmade  (counterfactual direction: pos_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 28 | 97 |
| copy_query | 2 | 2 |
| copy_demo_target | 23 | 23 |
| copy_demo_input | 3 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 1 | 3 |
| other | 93 | 25 |

Examples (own-ablation errors):

- `maple` → **natural** | own: `head of lettuce` [other] | cf: `natural`
- `zinnia` → **natural** | own: `draaife` [other] | cf: `manmade`
- `mongoose` → **natural** | own: `sonation` [other] | cf: `natural`
- `jellyfish` → **natural** | own: `manmade` [copy_demo_target] | cf: `manmade`
- `badger` → **natural** | own: `manmade` [copy_demo_target] | cf: `natural - Badger is`
- `shrike` → **natural** | own: `native` [other] | cf: `ornamental`
- `stove` → **manmade** | own: `chef` [other] | cf: `manmade`
- `pot` → **manmade** | own: `pace` [other] | cf: `natural`
- `loon` → **natural** | own: `mother` [other] | cf: `natural`
- `heater` → **manmade** | own: `material` [other] | cf: `manmade`

### next_item  (counterfactual direction: plural_to_singular)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 10 | 75 |
| copy_query | 24 | 8 |
| copy_demo_target | 11 | 4 |
| copy_demo_input | 3 | 1 |
| in_pool_wrong | 35 | 28 |
| input_variant | 0 | 2 |
| other | 67 | 32 |

Examples (own-ablation errors):

- `e` → **f** | own: `e` [copy_query] | cf: `d`
- `20` → **21** | own: `nineteen` [other] | cf: `21`
- `12` → **13** | own: `01` [other] | cf: `13`
- `twelve` → **thirteen** | own: `june` [copy_demo_input] | cf: `fifteen`
- `XVI` → **XVII** | own: `11` [other] | cf: `XVII`
- `YY` → **ZZ** | own: `4$` [other] | cf: `XX`
- `3` → **4** | own: `3` [copy_query] | cf: `4`
- `IX` → **X** | own: `xi` [in_pool_wrong] | cf: `x`
- `J` → **K** | own: `D` [in_pool_wrong] | cf: `H`
- `XIV` → **XV** | own: `14` [in_pool_wrong] | cf: `XIV`

### next_month_of_date  (counterfactual direction: verb_tense_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 6 | 77 |
| copy_query | 0 | 0 |
| copy_demo_target | 34 | 30 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 24 | 37 |
| input_variant | 10 | 0 |
| other | 76 | 6 |

Examples (own-ablation errors):

- `October 1838` → **November** | own: `disburagatorium` [other] | cf: `September`
- `December 1812` → **January** | own: `december` [copy_demo_target] | cf: `February`
- `February 1805` → **March** | own: `Seen` [other] | cf: `March`
- `February 1901` → **March** | own: `mars` [other] | cf: `March`
- `February 1861` → **March** | own: `Calendar` [other] | cf: `July`
- `January 1981` → **February** | own: `January` [in_pool_wrong] | cf: `August`
- `November 1887` → **December** | own: `November` [copy_demo_target] | cf: `December`
- `January 1950` → **February** | own: `January` [in_pool_wrong] | cf: `September`
- `March 2008` → **April** | own: `March` [copy_demo_target] | cf: `April`
- `March 1935` → **April** | own: `Small` [other] | cf: `June`

### next_number_digits  (counterfactual direction: pos_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 11 | 132 |
| copy_query | 76 | 1 |
| copy_demo_target | 4 | 1 |
| copy_demo_input | 4 | 0 |
| in_pool_wrong | 26 | 6 |
| input_variant | 0 | 0 |
| other | 29 | 10 |

Examples (own-ablation errors):

- `70` → **71** | own: `70` [copy_query] | cf: `71`
- `78` → **79** | own: `77` [in_pool_wrong] | cf: `81`
- `47` → **48** | own: `47` [copy_query] | cf: `48`
- `63` → **64** | own: `63` [copy_query] | cf: `64`
- `131` → **132** | own: `131` [copy_query] | cf: `132`
- `49` → **50** | own: `fife` [other] | cf: `250`
- `132` → **133** | own: `132` [copy_query] | cf: `133`
- `29` → **30** | own: `29` [copy_query] | cf: `30`
- `147` → **148** | own: `147` [copy_query] | cf: `148`
- `9` → **10** | own: `8` [copy_demo_input] | cf: `1`

### number_word_to_digits  (counterfactual direction: person-sport)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 67 | 110 |
| copy_query | 0 | 0 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 18 | 7 |
| input_variant | 2 | 0 |
| other | 63 | 33 |

Examples (own-ablation errors):

- `one thousand four hundred five` → **1405** | own: `1,400 5` [other] | cf: `1405`
- `one thousand one hundred seventy` → **1170** | own: `16901` [other] | cf: `1570`
- `nine hundred fifty-eight` → **958** | own: `tens` [other] | cf: `958`
- `one thousand three hundred fifty-five` → **1355** | own: `153` [other] | cf: `1255`
- `one thousand four hundred one` → **1401** | own: `404` [in_pool_wrong] | cf: `1401`
- `one thousand one hundred sixty-six` → **1166** | own: `1116` [in_pool_wrong] | cf: `1166`
- `one thousand one hundred seventy-eight` → **1178** | own: `1078` [in_pool_wrong] | cf: `1878`
- `one thousand four hundred seventy-two` → **1472** | own: `12En` [other] | cf: `1472`
- `two hundred fifty-nine` → **259** | own: `2559` [other] | cf: `259`
- `eight hundred five` → **805** | own: `05` [other] | cf: `805`

### park-country  (counterfactual direction: german-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 76 |
| copy_query | 1 | 0 |
| copy_demo_target | 5 | 6 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 48 |
| input_variant | 16 | 0 |
| other | 124 | 20 |

Examples (own-ablation errors):

- `Ile-Alatau National Park` → **Kazakhstan** | own: `Tuva` [other] | cf: `Kazakhstan`
- `Chubu-Sangaku National Park` → **Japan** | own: `Nagasaki` [other] | cf: `Japan`
- `Virunga National Park` → **Congo** | own: `Bujumbura (urban)` [other] | cf: `Congolese Republic`
- `Llanganates National Park` → **Ecuador** | own: `Las Palmas` [other] | cf: `Spain`
- `Cape Greco National Park` → **Cyprus** | own: `cape` [input_variant] | cf: `Spain`
- `Serra do Divisor National Park` → **Brazil** | own: `Maraval` [other] | cf: `Brazil`
- `Yankari National Park` → **Nigeria** | own: `Nor Calio` [other] | cf: `Brazil`
- `Mount Elgon National Park` → **Uganda** | own: `Okanga` [other] | cf: `Uganda`
- `Auyuittuq National Park` → **Canada** | own: `Quebec` [other] | cf: `Canada`
- `Iguaçu National Park` → **Brazil** | own: `IGUACU` [input_variant] | cf: `Brazil`

### past_to_base  (counterfactual direction: animal_plant_object)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 35 | 114 |
| copy_query | 49 | 9 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 3 |
| input_variant | 17 | 12 |
| other | 46 | 12 |

Examples (own-ablation errors):

- `sanded` → **sand** | own: `sanding` [input_variant] | cf: `sand`
- `claimed` → **claim** | own: `claimed` [copy_query] | cf: `claim`
- `minored` → **minor** | own: `majored` [other] | cf: `minor`
- `juiced` → **juice** | own: `doubreextrated` [other] | cf: `juice`
- `encouraged` → **encourage** | own: `took` [other] | cf: `encourage`
- `whited` → **white** | own: `whitened` [input_variant] | cf: `whiten`
- `beached` → **beach** | own: `beached` [copy_query] | cf: `beach`
- `solved` → **solve** | own: `squid` [other] | cf: `solve`
- `found` → **find** | own: `fount` [input_variant] | cf: `found`
- `calmed` → **calm** | own: `calmed` [copy_query] | cf: `calm`

### person-instrument  (counterfactual direction: antonym)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 9 | 39 |
| copy_query | 0 | 0 |
| copy_demo_target | 25 | 46 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 5 |
| input_variant | 0 | 0 |
| other | 116 | 60 |

Examples (own-ablation errors):

- `Leonard Cohen` → **guitar** | own: `violin` [copy_demo_target] | cf: `tenor`
- `Jon Eardley` → **trumpet** | own: `jelly` [other] | cf: `violin`
- `Millard Powers` → **guitar** | own: `grenade` [other] | cf: `harpsichord`
- `Billy Taylor` → **piano** | own: `trumpet` [copy_demo_target] | cf: `saxophones`
- `Blind Willie McTell` → **guitar** | own: `moper` [other] | cf: `banjo, guitar`
- `John Wesley` → **guitar** | own: `reed` [other] | cf: `violin`
- `Ola Kvernberg` → **violin** | own: `abaddon` [other] | cf: `guitar`
- `Henry Litolff` → **piano** | own: `duplar` [other] | cf: `violin`
- `Guy Picciotto` → **guitar** | own: `driveway` [other] | cf: `horn`
- `Jack Kilcoyne` → **guitar** | own: `tramp` [other] | cf: `piano`

### person-sport  (counterfactual direction: french_noun_gender)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 101 |
| copy_query | 0 | 0 |
| copy_demo_target | 2 | 20 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 5 |
| input_variant | 3 | 0 |
| other | 142 | 24 |

Examples (own-ablation errors):

- `Lionel Conacher` → **hockey** | own: `whiskey` [other] | cf: `lacrosse`
- `Gale Sayers` → **football** | own: `pig` [other] | cf: `football`
- `Júlio Baptista` → **soccer** | own: `babc` [other] | cf: `soccer`
- `Teemu Sälännä` → **hockey** | own: `cyborg` [other] | cf: `ice hockey`
- `Howie Morenz` → **hockey** | own: `woolly` [other] | cf: `basketball`
- `Jermain Defoe` → **soccer** | own: `Yemen` [other] | cf: `soccer`
- `Kareem Abdul-Jabbar` → **basketball** | own: `injured` [other] | cf: `basketball`
- `Frank Mahovlich` → **hockey** | own: `putt` [other] | cf: `hockey`
- `Dennis Rodman` → **basketball** | own: `hund` [other] | cf: `basketball`
- `Javier Hernández` → **soccer** | own: `bacterium` [other] | cf: `baseball`

### person_place_thing  (counterfactual direction: gerund_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 90 |
| copy_query | 7 | 0 |
| copy_demo_target | 5 | 22 |
| copy_demo_input | 4 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 4 | 1 |
| other | 123 | 37 |

Examples (own-ablation errors):

- `soldier` → **person** | own: `pan` [other] | cf: `person`
- `clown` → **person** | own: `padman` [other] | cf: `person`
- `saxophone` → **thing** | own: `key` [other] | cf: `thing`
- `highchair` → **thing** | own: `office` [other] | cf: `place`
- `villain` → **person** | own: `place` [copy_demo_target] | cf: `person`
- `car` → **thing** | own: `automobile` [other] | cf: `person`
- `colonel` → **person** | own: `bess` [other] | cf: `name of the leader`
- `boulevard` → **place** | own: `threshold` [other] | cf: `main road`
- `placemat` → **thing** | own: `table` [other] | cf: `flat thing`
- `atoll` → **place** | own: `Louise` [other] | cf: `place`

### plural_to_singular  (counterfactual direction: english-portuguese)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 32 | 120 |
| copy_query | 56 | 13 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 2 |
| input_variant | 14 | 7 |
| other | 48 | 8 |

Examples (own-ablation errors):

- `windings` → **winding** | own: `windings` [copy_query] | cf: `windings`
- `births` → **birth** | own: `births` [copy_query] | cf: `birth`
- `dioxides` → **dioxide** | own: `dioxides` [copy_query] | cf: `dioxides`
- `campuses` → **campus** | own: `college` [other] | cf: `campus`
- `legacies` → **legacy** | own: `legacies` [copy_query] | cf: `legacy`
- `crescents` → **crescent** | own: `rays of sun` [other] | cf: `crescent`
- `miracles` → **miracle** | own: `miraculous` [input_variant] | cf: `miracle`
- `pricks` → **prick** | own: `pricklings` [input_variant] | cf: `prick`
- `horizons` → **horizon** | own: `horizons` [copy_query] | cf: `horizon`
- `hardships` → **hardship** | own: `wilderness` [other] | cf: `hardship`

### pos_label  (counterfactual direction: larger_of_pair)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 5 | 95 |
| copy_query | 1 | 2 |
| copy_demo_target | 13 | 39 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 6 |
| input_variant | 3 | 0 |
| other | 127 | 8 |

Examples (own-ablation errors):

- `compilation` → **noun** | own: `album` [other] | cf: `noun`
- `disconnect` → **verb** | own: `altercation` [other] | cf: `noun`
- `sticker` → **noun** | own: `cried which representation` [other] | cf: `noun`
- `someday` → **adverb** | own: `sun` [other] | cf: `verb`
- `formerly` → **adverb** | own: `adjunct` [other] | cf: `adverb`
- `keeper` → **noun** | own: `crator` [other] | cf: `noun`
- `stuffed` → **verb** | own: `garrulous` [other] | cf: `noun`
- `always` → **adverb** | own: `noun` [copy_demo_target] | cf: `adjective`
- `occasionally` → **adverb** | own: `adjective` [copy_demo_target] | cf: `adjective`
- `tremendous` → **adjective** | own: `ominous` [other] | cf: `adjective`

### present-past  (counterfactual direction: smaller_of_pair)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 136 |
| copy_query | 130 | 3 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 1 |
| input_variant | 9 | 7 |
| other | 8 | 3 |

Examples (own-ablation errors):

- `transform` → **transformed** | own: `transform` [copy_query] | cf: `transformed`
- `want` → **wanted** | own: `want` [copy_query] | cf: `wanted`
- `challenge` → **challenged** | own: `challange` [input_variant] | cf: `challenged`
- `analyze` → **analyzed** | own: `analyze` [copy_query] | cf: `analyzed`
- `lend` → **lent** | own: `lend` [copy_query] | cf: `lent`
- `serve` → **served** | own: `serve` [copy_query] | cf: `served`
- `use` → **used** | own: `use` [copy_query] | cf: `used`
- `reduce` → **reduced** | own: `reduce` [copy_query] | cf: `reduced`
- `remain` → **remained** | own: `remain` [copy_query] | cf: `remains`
- `secure` → **secured** | own: `secure` [copy_query] | cf: `secured`

### prev_number_digits  (counterfactual direction: person_place_thing)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 9 | 126 |
| copy_query | 79 | 5 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 25 | 14 |
| input_variant | 0 | 0 |
| other | 36 | 5 |

Examples (own-ablation errors):

- `63` → **62** | own: `63` [copy_query] | cf: `62`
- `103` → **102** | own: `103` [copy_query] | cf: `102`
- `191` → **190** | own: `Uri` [other] | cf: `192`
- `150` → **149** | own: `150` [copy_query] | cf: `149`
- `119` → **118** | own: `119` [copy_query] | cf: `118`
- `130` → **129** | own: `13000` [other] | cf: `129`
- `116` → **115** | own: `116` [copy_query] | cf: `115`
- `96` → **95** | own: `96` [copy_query] | cf: `95`
- `28` → **27** | own: `trial` [other] | cf: `9`
- `172` → **171** | own: `17` [in_pool_wrong] | cf: `171`

### product-company  (counterfactual direction: gerund_to_past)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 6 | 73 |
| copy_query | 3 | 0 |
| copy_demo_target | 0 | 8 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 11 |
| input_variant | 15 | 3 |
| other | 126 | 55 |

Examples (own-ablation errors):

- `Logic Studio` → **Apple** | own: `LogicPRO` [input_variant] | cf: `Linear`
- `MVS` → **IBM** | own: `Microsoft Virtual Server` [other] | cf: `Digital Pictures`
- `Xcode` → **Apple** | own: `PE` [other] | cf: `Apple`
- `Report Program Generator` → **IBM** | own: `FORTINT` [other] | cf: `America Online, Inc`
- `Synchronized Accessible Media Interchange` → **Microsoft** | own: `package management/file` [other] | cf: `Microsoft`
- `Alfa Romeo MiTo` → **Fiat** | own: `Motorcycle` [other] | cf: `Fiat`
- `Internet Explorer 11` → **Microsoft** | own: `internet explorer browsers` [input_variant] | cf: `Microsoft`
- `Audio Interchange File Format` → **Apple** | own: `QUADtrak` [other] | cf: `Apple`
- `LGM-30 Minuteman` → **Boeing** | own: `computer` [other] | cf: `Raytheon`
- `Symbian` → **Nokia** | own: `sony eric` [other] | cf: `Nokia`

### sentiment  (counterfactual direction: park-country)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 21 | 96 |
| copy_query | 0 | 0 |
| copy_demo_target | 7 | 42 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 2 | 0 |
| other | 120 | 12 |

Examples (own-ablation errors):

- `Represents the depths to which the girls-behaving-badly film has fallen.` → **negative** | own: `manageable` [other] | cf: `negative`
- `Shyamalan should stop trying to please his mom.` → **negative** | own: `holiday` [other] | cf: `negative`
- `A film that's flawed and brilliant in equal measure.` → **positive** | own: `exam` [other] | cf: `positive`
- `Fluffy neo-noir hiding behind cutesy film references.` → **negative** | own: `preciate` [other] | cf: `positive`
- `Illiterate, often inert sci-fi action thriller.` → **negative** | own: `puss` [other] | cf: `negative`
- `A very charming and funny movie.` → **positive** | own: `considerate` [other] | cf: `Spectacularly negative`
- `At times, it actually hurts to watch.` → **negative** | own: `nonce` [other] | cf: `negative`
- `Demonstrates the unusual power of thoughtful, subjective filmmaking.` → **positive** | own: `proper noun` [other] | cf: `positive`
- `What makes it worth watching is Quaid's performance.` → **positive** | own: `precious` [other] | cf: `negative`
- `...unlikable, uninteresting, unfunny, and completely, utterly inept.` → **negative** | own: `please-terious` [other] | cf: `positive`

### singular-plural  (counterfactual direction: city-country)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 8 | 134 |
| copy_query | 107 | 3 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 1 |
| input_variant | 11 | 6 |
| other | 24 | 6 |

Examples (own-ablation errors):

- `soap` → **soaps** | own: `soap` [copy_query] | cf: `soap`
- `microwave` → **microwaves** | own: `microwave` [copy_query] | cf: `microwaves`
- `blender` → **blenders** | own: `blender` [copy_query] | cf: `blenders`
- `boot` → **boots** | own: `boot` [copy_query] | cf: `boots`
- `hose` → **hoses** | own: `hose` [copy_query] | cf: `hoses`
- `mountain` → **mountains** | own: `mountain` [copy_query] | cf: `mountains`
- `glove` → **gloves** | own: `glove (how to remove value(` [input_variant] | cf: `gloves`
- `river` → **rivers** | own: `stream` [other] | cf: `rivers`
- `pillow` → **pillows** | own: `pillow` [copy_query] | cf: `pillows`
- `toothbrush` → **toothbrushes** | own: `tooth brush` [input_variant] | cf: `toothbrushes`

### singular_or_plural  (counterfactual direction: animal_plant_object)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 17 | 92 |
| copy_query | 2 | 1 |
| copy_demo_target | 22 | 48 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 5 | 0 |
| other | 103 | 9 |

Examples (own-ablation errors):

- `babies` → **plural** | own: `singular` [copy_demo_target] | cf: `singular`
- `god` → **singular** | own: `quo` [other] | cf: `singular`
- `beds` → **plural** | own: `singular` [copy_demo_target] | cf: `plural`
- `fairs` → **plural** | own: `vastly` [other] | cf: `plural`
- `majors` → **plural** | own: `singular` [copy_demo_target] | cf: `singular`
- `fishes` → **plural** | own: `primal` [other] | cf: `singular`
- `properties` → **plural** | own: `pleural` [other] | cf: `plural`
- `buildings` → **plural** | own: `singular` [copy_demo_target] | cf: `singular`
- `blue` → **singular** | own: `U-KB4` [other] | cf: `singular`
- `balls` → **plural** | own: `singular` [copy_demo_target] | cf: `plural`

### smaller_of_pair  (counterfactual direction: country-capital)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 46 | 71 |
| copy_query | 1 | 0 |
| copy_demo_target | 4 | 1 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 41 | 35 |
| input_variant | 0 | 0 |
| other | 58 | 43 |

Examples (own-ablation errors):

- `765 68` → **68** | own: `65` [in_pool_wrong] | cf: `665`
- `416 749` → **416** | own: `749` [in_pool_wrong] | cf: `749`
- `155 415` → **155** | own: `85` [in_pool_wrong] | cf: `155`
- `47 469` → **47** | own: `469` [other] | cf: `469`
- `878 549` → **549** | own: `845` [other] | cf: `549`
- `42 506` → **42** | own: `501` [in_pool_wrong] | cf: `999`
- `281 361` → **281** | own: `654` [other] | cf: `281`
- `643 110` → **110** | own: `103` [other] | cf: `110`
- `292 336` → **292** | own: `338` [in_pool_wrong] | cf: `336`
- `330 348` → **330** | own: `348` [in_pool_wrong] | cf: `348`

### spanish-english  (counterfactual direction: plural_to_singular)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 84 | 109 |
| copy_query | 4 | 1 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 7 | 6 |
| input_variant | 7 | 1 |
| other | 48 | 33 |

Examples (own-ablation errors):

- `caer` → **fall** | own: `fall ; drop` [other] | cf: `fall`
- `solitario` → **lonely** | own: `solitary` [input_variant] | cf: `lonely`
- `cacahuete` → **peanut** | own: `queque` [other] | cf: `peanut`
- `coleccionar` → **collect** | own: `collect or contend` [other] | cf: `collect`
- `dorado` → **golden** | own: `gold` [in_pool_wrong] | cf: `gold`
- `existir` → **exist** | own: `being` [other] | cf: `to exist`
- `discutir` → **argue** | own: `conversation` [other] | cf: `talk, talk`
- `mañana` → **morning** | own: `tomorrow` [other] | cf: `tomorrow`
- `música` → **music** | own: `audio` [other] | cf: `song`
- `esposa` → **wife** | own: `esposa` [copy_query] | cf: `wife`

### spanish_noun_gender  (counterfactual direction: english-french)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 6 | 77 |
| copy_query | 0 | 0 |
| copy_demo_target | 3 | 42 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 5 | 0 |
| other | 135 | 31 |

Examples (own-ablation errors):

- `ejército` → **masculine** | own: `higher` [other] | cf: `masculine`
- `arco` → **masculine** | own: `region` [other] | cf: `masculine`
- `movimiento` → **masculine** | own: `lonely` [other] | cf: `feminine`
- `teatro` → **masculine** | own: `swedish` [other] | cf: `masculino`
- `sanidad` → **feminine** | own: `gingerim` [other] | cf: `feminine`
- `occidente` → **masculine** | own: `leño` [other] | cf: `feminine`
- `vigor` → **masculine** | own: `magnificent` [other] | cf: `feminine`
- `búsqueda` → **feminine** | own: `oso/mas` [other] | cf: `n/d`
- `habilidad` → **feminine** | own: `déficit` [copy_demo_input] | cf: `feminine`
- `gas` → **masculine** | own: `Hollywood` [other] | cf: `feminine`

### starts_with_vowel  (counterfactual direction: past_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 30 | 83 |
| copy_query | 8 | 0 |
| copy_demo_target | 36 | 46 |
| copy_demo_input | 2 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 2 | 0 |
| other | 72 | 21 |

Examples (own-ablation errors):

- `apartment` → **vowel** | own: `church` [other] | cf: `vowel`
- `unable` → **vowel** | own: `yield` [other] | cf: `consonant`
- `deal` → **consonant** | own: `haven` [other] | cf: `vowel`
- `moment` → **consonant** | own: `sexual` [other] | cf: `letter`
- `ice` → **vowel** | own: `dissi(cle)` [other] | cf: `vowel`
- `finally` → **consonant** | own: `maiden` [other] | cf: `consonant`
- `team` → **consonant** | own: `utilize` [other] | cf: `consonant`
- `overall` → **vowel** | own: `consistant` [other] | cf: `consonant`
- `baby` → **consonant** | own: `babychampion` [input_variant] | cf: `consonant`
- `impact` → **vowel** | own: `ovés` [other] | cf: `vowel`

### third_person_to_base  (counterfactual direction: initials_two_words)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 56 | 126 |
| copy_query | 62 | 14 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 2 | 0 |
| input_variant | 7 | 3 |
| other | 23 | 7 |

Examples (own-ablation errors):

- `kids` → **kid** | own: `children` [other] | cf: `kids`
- `transitions` → **transition** | own: `moving` [other] | cf: `transition`
- `gardens` → **garden** | own: `gardens` [copy_query] | cf: `garden`
- `lows` → **low** | own: `lows` [copy_query] | cf: `lo`
- `grounds` → **ground** | own: `grounds` [copy_query] | cf: `ground`
- `lasts` → **last** | own: `lasts` [copy_query] | cf: `last`
- `comes` → **come** | own: `'comes'` [input_variant] | cf: `comes`
- `guns` → **gun** | own: `guns` [copy_query] | cf: `gun`
- `farms` → **farm** | own: `farms` [copy_query] | cf: `farm`
- `prepares` → **prepare** | own: `prepares` [copy_query] | cf: `prep`

### titlecase_phrase  (counterfactual direction: verb_tense_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 117 | 142 |
| copy_query | 0 | 0 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 21 | 5 |
| other | 12 | 3 |

Examples (own-ablation errors):

- `vacant trouble` → **Vacant Trouble** | own: `vacant lot` [input_variant] | cf: `Vacant Trouble`
- `worthy whose` → **Worthy Whose** | own: `Whose` [input_variant] | cf: `Worthy Whose`
- `fatty royal` → **Fatty Royal** | own: `Fatty Ra, very rich` [input_variant] | cf: `Fatty Royal`
- `amateur buddy` → **Amateur Buddy** | own: `Amateur mate` [input_variant] | cf: `Amateur Buddy`
- `pointed break` → **Pointed Break** | own: `Pointed brake` [input_variant] | cf: `Pointed Break`
- `sole left` → **Sole Left** | own: `Un ta lef` [other] | cf: `Sole Left`
- `stuck milk` → **Stuck Milk** | own: `Bad milk` [other] | cf: `Stuck Milk`
- `trim sub` → **Trim Sub** | own: `Trim subs` [input_variant] | cf: `Trim Sub`
- `selfish none` → **Selfish None** | own: `Selfish Nobody` [input_variant] | cf: `Selfish None`
- `elderly cell` → **Elderly Cell** | own: `Elemcling cell` [other] | cf: `Elderly Cell`

### uppercase_word  (counterfactual direction: english-spanish)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 88 | 141 |
| copy_query | 0 | 0 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 2 | 1 |
| input_variant | 16 | 2 |
| other | 44 | 6 |

Examples (own-ablation errors):

- `round` → **ROUND** | own: `EAR` [other] | cf: `ROUND`
- `where` → **WHERE** | own: `where is` [input_variant] | cf: `WHERE`
- `not` → **NOT** | own: `NAH` [other] | cf: `NOT`
- `opening` → **OPENING** | own: `Opninring` [other] | cf: `OPENING`
- `front` → **FRONT** | own: `FORNE` [other] | cf: `FRONT`
- `call` → **CALL** | own: `cal` [other] | cf: `CALL`
- `far` → **FAR** | own: `FARE` [other] | cf: `FAR`
- `trust` → **TRUST** | own: `TRusSt` [input_variant] | cf: `TRUST`
- `needs` → **NEEDS** | own: `neeeds` [other] | cf: `NEEDS`
- `practice` → **PRACTICE** | own: `Practise` [input_variant] | cf: `PRACTICE`

### us-city-state  (counterfactual direction: starts_with_vowel)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 96 |
| copy_query | 20 | 0 |
| copy_demo_target | 0 | 8 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 43 |
| input_variant | 15 | 0 |
| other | 113 | 3 |

Examples (own-ablation errors):

- `Fargo` → **North Dakota** | own: `Fargo` [copy_query] | cf: `North Dakota`
- `Scottsboro` → **Alabama** | own: `Birmingham` [other] | cf: `Alabama`
- `Trenton` → **New Jersey** | own: `Eau Claire` [other] | cf: `Maine`
- `Clayton` → **Delaware** | own: `Clayton` [copy_query] | cf: `Iowa`
- `Norway` → **Maine** | own: `gull` [other] | cf: `Maine`
- `Oak Hill` → **West Virginia** | own: `Oaklawn` [other] | cf: `North Carolina`
- `Reading` → **Pennsylvania** | own: `eG` [other] | cf: `Massachusetts`
- `Rockland` → **Maine** | own: `Savannah` [other] | cf: `Maine`
- `Macon` → **Georgia** | own: `Mohawk` [other] | cf: `Georgia`
- `Bakersfield` → **California** | own: `Bakersfield` [copy_query] | cf: `California`

### verb_tense_label  (counterfactual direction: word_polarity)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 25 | 57 |
| copy_query | 0 | 0 |
| copy_demo_target | 33 | 69 |
| copy_demo_input | 2 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 0 | 0 |
| other | 90 | 24 |

Examples (own-ablation errors):

- `ratchetted` → **past** | own: `entertained` [other] | cf: `past`
- `riffed` → **past** | own: `present` [copy_demo_target] | cf: `past`
- `shapes` → **present** | own: `gerund` [copy_demo_target] | cf: `gerund`
- `hulls` → **present** | own: `harbor` [other] | cf: `past`
- `hooked` → **past** | own: `friends` [other] | cf: `present`
- `valuing` → **gerund** | own: `presenting` [other] | cf: `present`
- `shines` → **present** | own: `triumphant previously` [other] | cf: `gerund`
- `skilled` → **past** | own: `convince` [other] | cf: `gerund both`
- `previews` → **present** | own: `past` [copy_demo_target] | cf: `past`
- `scourged` → **past** | own: `bewittle` [other] | cf: `present`

### verb_to_third_person  (counterfactual direction: next_month_of_date)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 114 |
| copy_query | 119 | 12 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 1 |
| input_variant | 11 | 14 |
| other | 18 | 9 |

Examples (own-ablation errors):

- `run` → **runs** | own: `run` [copy_query] | cf: `runs`
- `explain` → **explains** | own: `explain` [copy_query] | cf: `explaines`
- `rain` → **rains** | own: `ri, rain` [input_variant] | cf: `rains`
- `purchase` → **purchases** | own: `purchase` [copy_query] | cf: `purchases`
- `network` → **networks** | own: `network` [copy_query] | cf: `networks`
- `stone` → **stones** | own: `stoney` [input_variant] | cf: `stones`
- `mirror` → **mirrors** | own: `mirror` [copy_query] | cf: `mirrors`
- `tie` → **ties** | own: `tie` [copy_query] | cf: `ties`
- `throw` → **throws** | own: `throw` [copy_query] | cf: `throws`
- `solve` → **solves** | own: `solve` [copy_query] | cf: `solve`

### word_polarity  (counterfactual direction: capitalize_first_letter)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 5 | 101 |
| copy_query | 5 | 0 |
| copy_demo_target | 3 | 40 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 2 | 0 |
| other | 134 | 9 |

Examples (own-ablation errors):

- `harassed` → **negative** | own: `asset` [other] | cf: `negative`
- `talented` → **positive** | own: `eminent` [other] | cf: `negative`
- `qualified` → **positive** | own: `niche` [other] | cf: `negative`
- `victorious` → **positive** | own: `submissive` [other] | cf: `positive`
- `renowned` → **positive** | own: `holy` [other] | cf: `positive`
- `gloomy` → **negative** | own: `less` [other] | cf: `negative`
- `grimy` → **negative** | own: `muddy` [other] | cf: `positive`
- `dismissive` → **negative** | own: `supreme` [other] | cf: `negative`
- `friendly` → **positive** | own: `friendly` [copy_query] | cf: `positive`
- `joyful` → **positive** | own: `manual` [other] | cf: `negative`

## 1-shot


### adjective_to_adverb  (counterfactual direction: french-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 34 |
| copy_query | 68 | 23 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 4 | 0 |
| in_pool_wrong | 0 | 2 |
| input_variant | 14 | 31 |
| other | 62 | 58 |

Examples (own-ablation errors):

- `admirable` → **admirably** | own: `admirable` [copy_query] | cf: `admirably`
- `scarce` → **scarcely** | own: `scarce` [copy_query] | cf: `scalars`
- `lucky` → **luckily** | own: `lucky` [copy_query] | cf: `sometimes, only, occasionally`
- `psychical` → **psychically** | own: `favorable` [copy_demo_input] | cf: `mystical`
- `conclusive` → **conclusively** | own: `convince` [other] | cf: `conclusive (that is, all-`
- `dark` → **darkly** | own: `gist` [other] | cf: `declassé`
- `lyric` → **lyrically** | own: `allegoric? if somebody's` [other] | cf: `all the words of a song`
- `terrific` → **terrifically** | own: `Terrific` [copy_query] | cf: `terrapti sad`
- `expectant` → **expectantly** | own: `excited` [other] | cf: `expectant`
- `defensive` → **defensively** | own: `defensive` [copy_query] | cf: `defensively`

### adjective_to_noun  (counterfactual direction: english-portuguese)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 14 | 19 |
| copy_query | 16 | 23 |
| copy_demo_target | 7 | 3 |
| copy_demo_input | 3 | 1 |
| in_pool_wrong | 4 | 3 |
| input_variant | 6 | 26 |
| other | 100 | 75 |

Examples (own-ablation errors):

- `satirical` → **satire** | own: `humorous` [other] | cf: `cartesian`
- `wooden` → **wood** | own: `wooden` [copy_query] | cf: `woody`
- `expansive` → **expanse** | own: `all of the above...` [other] | cf: `extent`
- `nutty` → **nut** | own: `nutty` [copy_query] | cf: `nutty`
- `adventurous` → **adventure** | own: `wild` [other] | cf: `adventurous`
- `liturgical` → **liturgy** | own: `liturgical` [copy_query] | cf: `no`
- `squeaky` → **squeak** | own: `0.91 s tr` [other] | cf: `rusticator`
- `asymmetric` → **asymmetry** | own: `symmetric` [input_variant] | cf: `unsuitable`
- `organic` → **organ** | own: `red` [in_pool_wrong] | cf: `foodstuff`
- `regimental` → **regiment** | own: `regular` [other] | cf: `regal`

### ag_news  (counterfactual direction: english-french)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 12 | 22 |
| copy_query | 0 | 0 |
| copy_demo_target | 22 | 23 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 5 |
| input_variant | 12 | 10 |
| other | 103 | 90 |

Examples (own-ablation errors):

- `Koch, Park leads first day in Nine Bridges Classic Carin Koch of Sweden and South Korea #39;s Grace Park both shot a 6-under-par 66 on Friday, taking the lead after the first round of the LPGA #39;s CJ Nine Bridges Classic.` → **Sports** | own: `Golf` [other] | cf: `Golf`
- `Grand Jury Adds to HealthSouth Charges Federal prosecutors yesterday announced new perjury and obstruction-of-justice charges against HealthSouth Corp. founder Richard M. Scrushy, accusing the former chief executive of the rehabilitation ` → **Business** | own: `Entrepreneur` [other] | cf: `Corporate Finance: Bank`
- `Weah returns to Liberia Liberian legend George Weah returns to Liberia to launch his bid for the country's presidency.` → **World** | own: `Sports` [copy_demo_target] | cf: `Not so poignant,`
- `Taliban Wanted List to Be Drawn (AP) AP - The United States could cut its forces in Afghanistan next summer if Taliban militants accept an amnesty to be drawn up by President Hamid Karzai and neighboring Pakistan, the senior U.S. commander here said Sunday.` → **World** | own: `America Want To Stop` [other] | cf: `Crimin`
- `Arafat #39;s nephew not ruling out poison as cause of death While Yasser Arafat #39;s nephew says toxicology tests on his uncle show no poisons were found in his system, Arafat #39;s nephew isn #39;t ruling that out as a cause of death.` → **World** | own: `Arafat` [input_variant] | cf: `Nikkei`
- `Has Your Broadband Had Its Fiber? Falling costs, new technology, and competition, with a nudge from regulatory changes, are bringing fiber closer to homes in the US just a few years after the idea seemed all but written off.` → **Science** | own: `Business` [copy_demo_target] | cf: `Featured`
- `Lehman relishes chance to halt US slide AFTER being named as the 2006 US Ryder Cup team captain by the PGA of America at a press conference in Florida last night, Tom Lehman insisted he saw the chance to halt Americas recent dismal showing in the biennial match with Europe as an opportunity ` → **Sports** | own: `%himedia /` [other] | cf: `Sports`
- `Hamilton Wins Cycling Time Trial Event THENS, Aug. 18  Tyler Hamilton had bruises splotched all over his back, painful souvenirs of a Tour de France gone terribly wrong. ` → **Sports** | own: `Armstrong` [other] | cf: `Athletics`
- `Butt facing ban Newcastle midfielder Nicky Butt is facing up to the possibility of a three-match European ban for his moment of UEFA Cup madness. The 29-year-old England international lost his cool with Hapoel Bnei Sakhnin ` → **Sports** | own: `World` [copy_demo_target] | cf: `News`
- `Detainees seen as minimal threat WASHINGTON -- Most of the alleged Al Qaeda and Taliban inmates at the US military prison at Guantanamo Bay, Cuba, are likely to be freed or sent to their home countries for further investigation because many pose little threat and are not providing much valuable intelligence, the facility's deputy commander has said.` → **World** | own: `AFP` [other] | cf: `US`

### agent_noun_to_verb  (counterfactual direction: number_word_to_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 33 |
| copy_query | 13 | 21 |
| copy_demo_target | 9 | 3 |
| copy_demo_input | 8 | 3 |
| in_pool_wrong | 6 | 7 |
| input_variant | 14 | 14 |
| other | 93 | 69 |

Examples (own-ablation errors):

- `breather` → **breathe** | own: `jogg` [other] | cf: `entertain`
- `compiler` → **compile** | own: `compiler` [copy_query] | cf: `compiler`
- `defender` → **defend** | own: `shield` [other] | cf: `crisp`
- `speeder` → **speed** | own: `z’ro` [other] | cf: `speed smoker`
- `bugger` → **bug** | own: `avoid (approaching use` [other] | cf: `bugaboo`
- `loser` → **lose** | own: `guy` [other] | cf: `loser`
- `ambler` → **amble** | own: `abysser` [other] | cf: `amble`
- `copier` → **copy** | own: `squirter` [other] | cf: `copy`
- `licensor` → **license** | own: `licensee` [input_variant] | cf: `license`
- `plotter` → **plot** | own: `form` [copy_demo_target] | cf: `plo ...`

### animal_class  (counterfactual direction: spanish-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 36 |
| copy_query | 5 | 1 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 3 | 2 |
| in_pool_wrong | 2 | 6 |
| input_variant | 13 | 8 |
| other | 125 | 95 |

Examples (own-ablation errors):

- `cownose ray` → **fish** | own: `aquatic` [other] | cf: `fish`
- `mako` → **fish** | own: `carbon` [other] | cf: `rew`
- `fer de lance` → **reptile** | own: `fish` [in_pool_wrong] | cf: `fish spit`
- `gophersnake` → **reptile** | own: `llama` [other] | cf: `tickle`
- `taipan` → **reptile** | own: `tiger` [other] | cf: `goose`
- `hornet` → **insect** | own: `hornet` [copy_query] | cf: `ant`
- `elk` → **mammal** | own: `rampant` [other] | cf: `deer`
- `manta ray` → **fish** | own: `gravid` [other] | cf: `living being`
- `japanese hornet` → **insect** | own: `bee` [other] | cf: `bee`
- `shrew` → **mammal** | own: `sambar` [other] | cf: `mammal`

### animal_plant_object  (counterfactual direction: antonym)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 21 |
| copy_query | 9 | 3 |
| copy_demo_target | 2 | 2 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 12 | 6 |
| other | 126 | 118 |

Examples (own-ablation errors):

- `okapi` → **animal** | own: `pacifist` [other] | cf: `ox`
- `iguana` → **animal** | own: `lizard` [other] | cf: `animal`
- `cougar` → **animal** | own: `panther` [other] | cf: `animal`
- `serval` → **animal** | own: `caribundu` [other] | cf: `snake`
- `printer` → **object** | own: `printer` [copy_query] | cf: `N/A (`
- `mockorange` → **plant** | own: `sour and sweet,` [other] | cf: `plant`
- `radiator` → **object** | own: `radiant` [input_variant] | cf: `resembse a water`
- `teapot` → **object** | own: `tea` [other] | cf: `teapot`
- `angelfish` → **animal** | own: `haddock` [other] | cf: `fish`
- `bermudagrass` → **plant** | own: `bermuda grass` [input_variant] | cf: `grass of the genus`

### antonym  (counterfactual direction: next_number_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 4 | 4 |
| copy_query | 18 | 14 |
| copy_demo_target | 7 | 5 |
| copy_demo_input | 8 | 4 |
| in_pool_wrong | 8 | 13 |
| input_variant | 11 | 8 |
| other | 94 | 102 |

Examples (own-ablation errors):

- `due` → **overdue** | own: `due` [copy_query] | cf: `owed`
- `employee` → **employer** | own: `recruiter` [other] | cf: `inefficient`
- `conception` → **birth** | own: `envision & design, heating` [other] | cf: `thinking`
- `withdrawal` → **deposit** | own: `hospitalized` [other] | cf: `withdrawal`
- `island` → **mainland** | own: `forest` [other] | cf: `territory`
- `unveil` → **conceal** | own: `show` [other] | cf: `sail`
- `medium` → **large** | own: `boring` [other] | cf: `tall`
- `recorded` → **live** | own: `does only take the -` [other] | cf: `none`
- `postpone` → **advance** | own: `blow` [other] | cf: `shelve`
- `immortal` → **mortal** | own: `dwarf` [copy_demo_input] | cf: `live forever in your imagination`

### article_choice  (counterfactual direction: prev_number_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 4 | 7 |
| copy_query | 14 | 10 |
| copy_demo_target | 1 | 3 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 3 |
| input_variant | 14 | 5 |
| other | 116 | 122 |

Examples (own-ablation errors):

- `aisle` → **an** | own: `aisle` [copy_query] | cf: `La`
- `pat` → **a** | own: `pen.` [other] | cf: `mat`
- `rubble` → **a** | own: `game` [other] | cf: `quartz`
- `complex` → **a** | own: `complicated` [input_variant] | cf: `complicated`
- `scope` → **a** | own: `wastage` [other] | cf: `Mad Men`
- `balcony` → **a** | own: `with` [other] | cf: `lucky of you`
- `print` → **a** | own: `pen` [other] | cf: `zero`
- `politics` → **a** | own: `mess` [other] | cf: `amar`
- `upright` → **an** | own: `upward` [other] | cf: `in`
- `oxygen` → **an** | own: `a` [copy_demo_target] | cf: `ha`

### capitalize  (counterfactual direction: prev_number_digits)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 42 | 62 |
| copy_query | 0 | 0 |
| copy_demo_target | 15 | 3 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 10 | 9 |
| input_variant | 14 | 22 |
| other | 69 | 54 |

Examples (own-ablation errors):

- `marker` → **Marker** | own: `Merchandising (beaut` [other] | cf: `Mark`
- `spoon` → **Spoon** | own: `larding.` [other] | cf: `Lowercase`
- `navigate` → **Navigate** | own: `Not labeled` [other] | cf: `"And you know, you`
- `good` → **Good** | own: `wolf` [in_pool_wrong] | cf: `good`
- `inventive` → **Inventive** | own: `Yes` [other] | cf: `Great grooming!`
- `change` → **Change** | own: `cold` [in_pool_wrong] | cf: `Change`
- `bag` → **Bag** | own: `master` [copy_demo_target] | cf: `Bag`
- `resolute` → **Resolute** | own: `dextrous` [other] | cf: `Resolute`
- `off` → **Off** | own: `SO easy` [other] | cf: `Dusk`
- `design` → **Design** | own: `design-big bread box` [input_variant] | cf: `Design`

### capitalize_first_letter  (counterfactual direction: sentiment)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 8 |
| copy_query | 13 | 8 |
| copy_demo_target | 4 | 18 |
| copy_demo_input | 1 | 1 |
| in_pool_wrong | 6 | 33 |
| input_variant | 5 | 9 |
| other | 120 | 73 |

Examples (own-ablation errors):

- `violin` → **V** | own: `c` [copy_demo_target] | cf: `B`
- `underneath` → **U** | own: `on top of` [other] | cf: `enemy`
- `maracas` → **M** | own: `moo` [other] | cf: `O`
- `modest` → **M** | own: `rockey` [other] | cf: `P`
- `humble` → **H** | own: `special;(?)` [other] | cf: `humble`
- `salty` → **S** | own: `avg` [other] | cf: `T`
- `from` → **F** | own: `z` [in_pool_wrong] | cf: `Direct`
- `as` → **A** | own: `yes` [other] | cf: `G`
- `incense` → **I** | own: `resin` [other] | cf: `nu`
- `exchange` → **E** | own: `interchange` [other] | cf: `m`

### city-country  (counterfactual direction: english-italian)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 5 | 42 |
| copy_query | 13 | 3 |
| copy_demo_target | 1 | 4 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 4 | 34 |
| input_variant | 10 | 7 |
| other | 116 | 60 |

Examples (own-ablation errors):

- `Accra` → **Ghana** | own: `Larry` [other] | cf: `Ghana`
- `Battagram` → **Pakistan** | own: `Arrah` [other] | cf: `Nepal`
- `N'Djamena` → **Chad** | own: `Chad (this is a` [other] | cf: `Chad`
- `Novi Sad` → **Serbia** | own: `Belgrade` [other] | cf: `Sevilla`
- `Guarulhos` → **Brazil** | own: `Rio de Janeiro` [other] | cf: `São Paulo`
- `Calgary` → **Canada** | own: `Alberta` [other] | cf: `Canada`
- `Vina del Mar` → **Chile** | own: `Mar del Plata` [other] | cf: `Spain`
- `Nanning` → **China** | own: `Chinese` [other] | cf: `Guangxi`
- `Munich` → **Germany** | own: `Love eats up time#DeleteLine` [other] | cf: `Bavaria`
- `Kalemyo` → **Myanmar** | own: `Qazaqstan` [other] | cf: `Kenya`

### compound_first  (counterfactual direction: person-sport)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 11 | 19 |
| copy_query | 11 | 10 |
| copy_demo_target | 2 | 3 |
| copy_demo_input | 5 | 5 |
| in_pool_wrong | 11 | 15 |
| input_variant | 14 | 12 |
| other | 96 | 86 |

Examples (own-ablation errors):

- `bombshell` → **bomb** | own: `bombshell` [copy_query] | cf: `brain`
- `grandfather` → **grand** | own: `rubber singer` [other] | cf: `grapher`
- `lakefront` → **lake** | own: `lossfront` [other] | cf: `room`
- `overhead` → **over** | own: `overcoat` [copy_demo_input] | cf: `overheader`
- `flashback` → **flash** | own: `report` [other] | cf: `shortdash`
- `timepiece` → **time** | own: `watch` [in_pool_wrong] | cf: `watch`
- `wingman` → **wing** | own: `companion` [other] | cf: `wingman`
- `password` → **pass** | own: `password` [copy_query] | cf: `pantaloon`
- `washboard` → **wash** | own: `password` [copy_demo_input] | cf: `barrier`
- `granddaughter` → **grand** | own: `firstborn` [other] | cf: `granddaughter`

### concrete_abstract  (counterfactual direction: verb_tense_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 4 |
| copy_query | 13 | 3 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 0 |
| input_variant | 7 | 3 |
| other | 128 | 137 |

Examples (own-ablation errors):

- `delight` → **abstract** | own: `delight` [copy_query] | cf: `abstract`
- `stress` → **abstract** | own: `coffee` [other] | cf: `boredom`
- `soil` → **concrete** | own: `forest` [other] | cf: `porifera (`
- `brother` → **concrete** | own: `male` [other] | cf: `elder`
- `melancholy` → **abstract** | own: `rural song` [other] | cf: `concrete`
- `thistle` → **concrete** | own: `take care of` [other] | cf: `hybrid of clover`
- `wheelbarrow` → **concrete** | own: `yard, axle` [other] | cf: `at`
- `hill` → **concrete** | own: `river` [other] | cf: `sand`
- `iceberg` → **concrete** | own: `enormous underwater Arctic ice` [other] | cf: `ice`
- `truck` → **concrete** | own: `coke` [other] | cf: `trailer`

### contains_letter_e  (counterfactual direction: iso_date_to_month)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 9 | 37 |
| copy_query | 1 | 3 |
| copy_demo_target | 12 | 25 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 8 |
| input_variant | 8 | 3 |
| other | 117 | 74 |

Examples (own-ablation errors):

- `composite` → **yes** | own: `no` [copy_demo_target] | cf: `yes`
- `precise` → **yes** | own: `what?` [other] | cf: `no`
- `shallow` → **no** | own: `chill and daunder` [other] | cf: `yes`
- `scroll` → **no** | own: `right now` [other] | cf: `don't try to`
- `spiritual` → **no** | own: `six` [other] | cf: `no`
- `establish` → **yes** | own: `sink.` [other] | cf: `two`
- `havoc` → **no** | own: `a chinese restaurant` [other] | cf: `knock knees and have`
- `planet` → **yes** | own: `Object` [other] | cf: `yes`
- `pond` → **no** | own: `flea` [other] | cf: `no`
- `reliably` → **yes** | own: `ebrish` [other] | cf: `use bytes 0x`

### country-capital  (counterfactual direction: sentiment)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 5 | 85 |
| copy_query | 24 | 4 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 1 |
| input_variant | 18 | 3 |
| other | 102 | 57 |

Examples (own-ablation errors):

- `United States of America` → **Washington, D.C.** | own: `Oui` [other] | cf: `Trinidad & Tobago (next week)`
- `Burundi` → **Bujumbura** | own: `Buronde` [other] | cf: `Bujumbura`
- `Taiwan` → **Taipei** | own: `Taiwan` [copy_query] | cf: `Shanghai`
- `Ireland` → **Dublin** | own: `I’m certain that Ireland is lots` [input_variant] | cf: `Dublin`
- `Tuvalu` → **Funafuti** | own: `the Earth's eye` [other] | cf: `Nukutavake / Nuk`
- `Zimbabwe` → **Harare** | own: `Zimbabwe, formerly Rhodesia` [input_variant] | cf: `Harare`
- `Slovenia` → **Ljubljana** | own: `Slovenia` [copy_query] | cf: `Slovenia`
- `Nigeria` → **Abuja** | own: `Mali` [other] | cf: `Abuja`
- `Bahamas` → **Nassau** | own: `Bahamas` [copy_query] | cf: `Nassau`
- `Dominican Republic` → **Santo Domingo** | own: `San Isidro, the patron saint` [other] | cf: `Der Herrn Colom`

### day_after_textual_date  (counterfactual direction: first_digit)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 65 |
| copy_query | 10 | 4 |
| copy_demo_target | 2 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 27 | 22 |
| input_variant | 49 | 22 |
| other | 60 | 37 |

Examples (own-ablation errors):

- `June 3, 2049` → **June 4** | own: `May 2` [in_pool_wrong] | cf: `maybe June 1 ?`
- `June 26, 1933` → **June 27** | own: `June 25, 2024` [input_variant] | cf: `June 27`
- `November 8, 1913` → **November 9** | own: `November 11, 1914` [input_variant] | cf: `November 9`
- `September 25, 2056` → **September 26** | own: `10` [other] | cf: `November 26, 2065`
- `October 2, 1907` → **October 3** | own: `1932` [other] | cf: `October 3`
- `June 18, 1926` → **June 19** | own: `36` [other] | cf: `Jesus was born first.`
- `February 27, 2016` → **February 28** | own: `` [other] | cf: `February 28`
- `November 24, 2089` → **November 25** | own: `Just November` [other] | cf: `25`
- `December 16, 1973` → **December 17** | own: `December 1958` [input_variant] | cf: `December 17`
- `September 5, 2056` → **September 6** | own: `Septembra 2056` [input_variant] | cf: `July 15`

### ends_with_ing  (counterfactual direction: next_month_of_date)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 12 | 46 |
| copy_query | 2 | 0 |
| copy_demo_target | 13 | 22 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 2 | 6 |
| input_variant | 9 | 2 |
| other | 112 | 74 |

Examples (own-ablation errors):

- `funk` → **no** | own: `yes` [copy_demo_target] | cf: `any - not sure`
- `revolting` → **yes** | own: `brute` [other] | cf: `obscene, disgusting`
- `terrain` → **no** | own: `terrras` [input_variant] | cf: `6th`
- `irregular` → **no** | own: `boteye` [other] | cf: `yeah that thing`
- `troughing` → **yes** | own: `trough` [input_variant] | cf: `No`
- `offsetting` → **yes** | own: `counteract` [other] | cf: `no`
- `improvement` → **no** | own: `not possible` [other] | cf: `yes`
- `culturing` → **yes** | own: `last` [other] | cf: `tv shows`
- `fatting` → **yes** | own: `pattern` [other] | cf: `oh`
- `fitness` → **no** | own: `activity` [other] | cf: `jogging, biking`

### english-french  (counterfactual direction: lowercase_word)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 36 |
| copy_query | 12 | 6 |
| copy_demo_target | 3 | 0 |
| copy_demo_input | 6 | 0 |
| in_pool_wrong | 2 | 8 |
| input_variant | 17 | 12 |
| other | 108 | 88 |

Examples (own-ablation errors):

- `failed` → **échoué** | own: `understanding` [other] | cf: `résultat`
- `animals` → **animaux** | own: `chickens` [other] | cf: `animaux`
- `landed` → **atterri** | own: `manorial` [other] | cf: `allatté`
- `explanation` → **explication** | own: `explains` [input_variant] | cf: `expliqué`
- `decades` → **décennies** | own: `developments` [copy_demo_input] | cf: `décennies`
- `garden` → **jardin** | own: `gardenmouset` [input_variant] | cf: `ugust`
- `turned` → **tourné** | own: `sauter` [copy_demo_target] | cf: `trottée`
- `soccer` → **football** | own: `feet` [other] | cf: `footer`
- `killed` → **tué** | own: `personfold` [other] | cf: `tué`
- `side` → **côté** | own: `obverse` [other] | cf: `côté`

### english-italian  (counterfactual direction: plural_to_singular)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 30 |
| copy_query | 15 | 1 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 0 | 6 |
| input_variant | 15 | 8 |
| other | 117 | 105 |

Examples (own-ablation errors):

- `knee` → **ginocchio** | own: `el capure` [other] | cf: `ginocchio`
- `play` → **giocare** | own: `dropp` [other] | cf: `giocare`
- `voice` → **voce** | own: `voiceless` [input_variant] | cf: `minimaunist`
- `want` → **volere** | own: `paper` [other] | cf: `fiyo`
- `magazine` → **rivista** | own: `mags` [other] | cf: `la revista`
- `worm` → **verme** | own: `ultrasound` [other] | cf: `bestia larva`
- `relative` → **relativo** | own: `cousine` [other] | cf: `relativo`
- `skinny` → **magro** | own: `frail` [other] | cf: `shuneta`
- `identical` → **identico** | own: `yes` [other] | cf: `identico`
- `rural` → **rurale** | own: `extensive` [other] | cf: `Leilão da boia ver`

### english-portuguese  (counterfactual direction: adjective_to_adverb)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 46 |
| copy_query | 11 | 1 |
| copy_demo_target | 2 | 0 |
| copy_demo_input | 3 | 0 |
| in_pool_wrong | 0 | 3 |
| input_variant | 10 | 6 |
| other | 124 | 94 |

Examples (own-ablation errors):

- `beach` → **praia** | own: `lamination` [other] | cf: `praia`
- `holiday` → **feriado** | own: `gondolak` [other] | cf: `férias`
- `cap` → **boné** | own: `cab` [other] | cf: `e mão de mã`
- `agree` → **concordar** | own: `concur` [other] | cf: `concordo`
- `messy` → **desarrumado** | own: `delicious` [other] | cf: `somedefroildescents`
- `assume` → **assumir** | own: `nongradual` [other] | cf: `en asumir`
- `king` → **rei** | own: `kangaroo` [other] | cf: `tuna`
- `salt` → **sal** | own: `sodium` [other] | cf: `sal`
- `musician` → **músico** | own: `musician's intent` [input_variant] | cf: `músico`
- `eight` → **oito** | own: `eight` [copy_query] | cf: `due`

### english-spanish  (counterfactual direction: adjective_to_adverb)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 47 |
| copy_query | 23 | 4 |
| copy_demo_target | 5 | 0 |
| copy_demo_input | 2 | 0 |
| in_pool_wrong | 1 | 1 |
| input_variant | 24 | 17 |
| other | 95 | 81 |

Examples (own-ablation errors):

- `searching` → **buscando** | own: `search` [input_variant] | cf: `buscar`
- `bus` → **autobús** | own: `us/I` [other] | cf: `diligencia`
- `nothing` → **nada** | own: `rest` [other] | cf: `nada`
- `items` → **artículos** | own: `1 diagnoses` [other] | cf: `elementos`
- `anniversary` → **aniversario** | own: `anniversary` [copy_query] | cf: `santo dia`
- `mayors` → **alcaldes** | own: `m` [other] | cf: `mayores`
- `insects` → **insectos** | own: `come later -- off-topic here` [other] | cf: `insectos`
- `starting` → **comenzando** | own: `starting` [copy_query] | cf: `empezar`
- `call` → **llamada** | own: `call referend mutation` [input_variant] | cf: `llamada`
- `rules` → **reglas** | own: `rules` [copy_query] | cf: `grammars`

### first_digit  (counterfactual direction: word_polarity)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 9 |
| copy_query | 6 | 2 |
| copy_demo_target | 9 | 11 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 13 | 26 |
| input_variant | 2 | 0 |
| other | 113 | 102 |

Examples (own-ablation errors):

- `2008` → **2** | own: `2007` [other] | cf: `746`
- `308470` → **3** | own: `c . . d` [other] | cf: `81231`
- `851` → **8** | own: `Yes, since your` [other] | cf: `3`
- `6790` → **6** | own: `8` [in_pool_wrong] | cf: `6525`
- `8392` → **8** | own: `89` [other] | cf: `Total fiction`
- `14` → **1** | own: `Yes. This software` [other] | cf: ``
- `902` → **9** | own: `18` [other] | cf: `2`
- `584` → **5** | own: `1` [copy_demo_target] | cf: `No, Rapture`
- `557` → **5** | own: `8` [copy_demo_target] | cf: `224`
- `53830` → **5** | own: `Check the asan` [other] | cf: `2`

### first_three_letters  (counterfactual direction: singular_or_plural)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 11 |
| copy_query | 17 | 18 |
| copy_demo_target | 8 | 4 |
| copy_demo_input | 8 | 1 |
| in_pool_wrong | 3 | 2 |
| input_variant | 11 | 16 |
| other | 101 | 98 |

Examples (own-ablation errors):

- `apart` → **apa** | own: `heavy` [copy_demo_input] | cf: `rain`
- `professional` → **pro** | own: `looks like I talked about` [other] | cf: `professional`
- `search` → **sea** | own: `"bool staff="all` [other] | cf: `SQQ`
- `tomorrow` → **tom** | own: `batto` [other] | cf: `har`
- `century` → **cen** | own: `There are no answers yet` [other] | cf: `cent`
- `speech` → **spe** | own: `everyone` [other] | cf: `bored`
- `battle` → **bat** | own: `uniform/animation` [other] | cf: `batti`
- `sometimes` → **som** | own: `sometimes` [copy_query] | cf: `mostć`
- `twitter` → **twi** | own: `health` [other] | cf: `nutrition`
- `eating` → **eat** | own: `slept` [other] | cf: `jus twenty`

### french-english  (counterfactual direction: language_identification)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 27 | 64 |
| copy_query | 10 | 3 |
| copy_demo_target | 2 | 1 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 6 | 8 |
| input_variant | 13 | 11 |
| other | 92 | 63 |

Examples (own-ablation errors):

- `reine` → **queen** | own: `fille` [other] | cf: `apple`
- `forêt` → **forest** | own: `chuflée` [other] | cf: `sur Brasil`
- `paresseux` → **lazy** | own: `shameful, lazy` [other] | cf: `lazy, sleepy,`
- `écharpe` → **scarf** | own: `écharpe` [copy_query] | cf: `(usually) striped`
- `cicatrice` → **scar** | own: `Quality of a mark` [other] | cf: `scar`
- `tonnerre` → **thunder** | own: `gefricayed` [other] | cf: `thunder`
- `araignée` → **spider** | own: `arachnid` [other] | cf: `spider`
- `avantage` → **advantage** | own: `bonus, bonus,` [other] | cf: `advantage`
- `tuer` → **kill** | own: `translator` [other] | cf: `to kill`
- `raide` → **stiff** | own: `siège s` [other] | cf: `ride, prone thirty`

### french_noun_gender  (counterfactual direction: english-italian)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 30 |
| copy_query | 13 | 3 |
| copy_demo_target | 0 | 5 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 4 |
| input_variant | 11 | 6 |
| other | 126 | 102 |

Examples (own-ablation errors):

- `domicile` → **masculine** | own: `domattin` [other] | cf: `residence`
- `fac` → **feminine** | own: `favour` [other] | cf: `feature`
- `entraînement` → **masculine** | own: `training` [other] | cf: `male noun : A`
- `accord` → **masculine** | own: `va` [other] | cf: `participe du ver`
- `amélioration` → **feminine** | own: `` [other] | cf: `Enhancement`
- `intelligence` → **feminine** | own: `sminée` [other] | cf: `féminine`
- `chaise` → **feminine** | own: `chaise` [copy_query] | cf: `female child`
- `individu` → **masculine** | own: `definitive` [other] | cf: `masculine`
- `empereur` → **masculine** | own: `rule` [other] | cf: `automobile`
- `orient` → **masculine** | own: `exhilaration` [other] | cf: `en saisir`

### german-english  (counterfactual direction: german_noun_gender)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 17 | 55 |
| copy_query | 8 | 1 |
| copy_demo_target | 1 | 3 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 2 | 9 |
| input_variant | 11 | 3 |
| other | 110 | 79 |

Examples (own-ablation errors):

- `Rohr` → **pipe** | own: `pull` [other] | cf: `rascal`
- `enthüllen` → **reveal** | own: `ignife [ent` [other] | cf: `uncover`
- `Vorhang` → **curtain** | own: `veil` [other] | cf: `curtain`
- `Nachrichten` → **news** | own: `messages` [other] | cf: `news`
- `Identität` → **identity** | own: `Corpmörn` [other] | cf: `identity`
- `Ziege` → **goat** | own: `gall` [other] | cf: `cow`
- `Bambus` → **bamboo** | own: `Pineapple` [other] | cf: `bamboo grows in northern`
- `Erfolg` → **success** | own: `(/er//f` [other] | cf: `Success`
- `angreifen` → **attack** | own: `knacken` [other] | cf: `attack`
- `trennen` → **separate** | own: `tole` [other] | cf: `split`

### german_noun_gender  (counterfactual direction: first_three_letters)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 15 |
| copy_query | 15 | 5 |
| copy_demo_target | 2 | 11 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 3 |
| input_variant | 3 | 15 |
| other | 130 | 100 |

Examples (own-ablation errors):

- `Potential` → **das** | own: `nein` [other] | cf: `die`
- `Kontakt` → **der** | own: `Contact` [other] | cf: `kontakt`
- `Konto` → **das** | own: `kostenlos` [other] | cf: `Hm, No...`
- `Dating` → **das** | own: `` [other] | cf: `Mit`
- `Zeichen` → **das** | own: `clubeum` [other] | cf: `das`
- `Fahrzeug` → **das** | own: `Fess` [other] | cf: `der`
- `Schatz` → **der** | own: `shto` [other] | cf: `= Treasure`
- `Kurs` → **der** | own: `course` [other] | cf: `=$5\times$`
- `Festival` → **das** | own: `Festag` [input_variant] | cf: `Römerfestival`
- `Abschied` → **der** | own: `Weihnachts` [other] | cf: `----------------`

### gerund_to_base  (counterfactual direction: spanish-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 18 | 42 |
| copy_query | 18 | 13 |
| copy_demo_target | 8 | 1 |
| copy_demo_input | 2 | 2 |
| in_pool_wrong | 5 | 7 |
| input_variant | 14 | 12 |
| other | 85 | 73 |

Examples (own-ablation errors):

- `watering` → **water** | own: `watering` [copy_query] | cf: `watering`
- `bettering` → **better** | own: `works` [other] | cf: `get faster / becomes`
- `costing` → **cost** | own: `cking` [other] | cf: `angle wonca cost`
- `asking` → **ask** | own: `asking` [copy_query] | cf: `asking`
- `maintaining` → **maintain** | own: `exchanging` [copy_demo_input] | cf: `managing`
- `truing` → **true** | own: `radiating` [other] | cf: `tuning`
- `sugaring` → **sugar** | own: `ground` [in_pool_wrong] | cf: `soft flavour`
- `bridging` → **bridge** | own: `sneaker.` [other] | cf: `horde`
- `cupping` → **cup** | own: `cupping wood` [input_variant] | cf: `The warm hands become`
- `mastering` → **master** | own: `positive` [other] | cf: `Acrobat`

### gerund_to_past  (counterfactual direction: compound_first)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 22 |
| copy_query | 24 | 26 |
| copy_demo_target | 3 | 4 |
| copy_demo_input | 5 | 1 |
| in_pool_wrong | 3 | 1 |
| input_variant | 26 | 26 |
| other | 86 | 70 |

Examples (own-ablation errors):

- `riding` → **rode** | own: `commuting` [other] | cf: `bays`
- `contrasting` → **contrasted** | own: `replacement with - implies something` [other] | cf: `different`
- `assaulting` → **assaulted** | own: `forced` [copy_demo_target] | cf: `assaulted`
- `effecting` → **effected** | own: `making` [other] | cf: `effecting`
- `cornering` → **cornered** | own: `wearing kevlar` [other] | cf: `yes, I can corner`
- `passing` → **passed** | own: `beneficiary` [other] | cf: `recursive`
- `spacing` → **spaced** | own: `space (and between)` [input_variant] | cf: `punctuations are respected.`
- `bothering` → **bothered** | own: `no;` [other] | cf: `<b><H0`
- `challenging` → **challenged** | own: `competitor` [other] | cf: `challenged`
- `owning` → **owned** | own: `possess` [other] | cf: `owning`

### hypernym_category  (counterfactual direction: english-french)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 19 |
| copy_query | 10 | 2 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 2 |
| input_variant | 14 | 6 |
| other | 124 | 118 |

Examples (own-ablation errors):

- `bandsaw` → **tool** | own: `Azurians` [other] | cf: `working`
- `silo` → **building** | own: `beetroot` [other] | cf: `grain`
- `motel` → **building** | own: `site` [other] | cf: `motel`
- `sneakers` → **clothing** | own: `shoe` [other] | cf: `shoe`
- `trumpet` → **instrument** | own: `opera.` [other] | cf: `music for wind instruments`
- `diving` → **sport** | own: `smiling` [other] | cf: `you are correctly understood`
- `flute` → **instrument** | own: `flayer` [other] | cf: `mouth`
- `carriage` → **vehicle** | own: `animal labor` [other] | cf: `vehicle`
- `snooker` → **sport** | own: `sphere` [other] | cf: `games?`
- `banker` → **profession** | own: `grocer` [other] | cf: `money lender`

### initials_two_words  (counterfactual direction: person-instrument)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 5 |
| copy_query | 1 | 2 |
| copy_demo_target | 3 | 13 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 4 | 9 |
| input_variant | 9 | 7 |
| other | 132 | 114 |

Examples (own-ablation errors):

- `fake track` → **FT** | own: `crank it` [other] | cf: `meaningless`
- `huge religion` → **HR** | own: `country of Tartars` [other] | cf: `HARMS`
- `actual ease` → **AE** | own: `Small Eleventy-` [other] | cf: `50/50`
- `isolated housing` → **IH** | own: `the village single is isolated` [other] | cf: `Q`
- `titled search` → **TS** | own: `bigincomeproject` [other] | cf: `Osiris Reading Desk`
- `stunning script` → **SS** | own: `I WISH - great` [other] | cf: `incredible script`
- `modern majority` → **MM** | own: `wed` [other] | cf: `Hi`
- `slow scene` → **SS** | own: `session` [other] | cf: `winner`
- `over fuel` → **OF** | own: `fair` [other] | cf: `car rental`
- `separate designer` → **SD** | own: `your franklin?` [other] | cf: `AC (Note that conver`

### iso_date_to_month  (counterfactual direction: product-company)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 12 | 48 |
| copy_query | 0 | 0 |
| copy_demo_target | 1 | 1 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 3 | 13 |
| input_variant | 2 | 4 |
| other | 132 | 84 |

Examples (own-ablation errors):

- `2026-08-23` → **August** | own: `8/23` [other] | cf: `December`
- `2000-12-07` → **December** | own: `12/07` [other] | cf: `20`
- `1965-05-02` → **May** | own: `Fantastic Four #17` [other] | cf: `It began 20th`
- `2058-12-21` → **December** | own: `Some of the buildings` [other] | cf: `December`
- `2047-10-04` → **October** | own: `2004-10-` [other] | cf: `October`
- `2002-01-15` → **January** | own: `2002-01-` [input_variant] | cf: `This short article features`
- `2027-03-10` → **March** | own: `10` [other] | cf: `bagamoy`
- `1965-05-09` → **May** | own: `Lambda light gravity` [other] | cf: `Kotetsu`
- `1960-07-13` → **July** | own: `long` [other] | cf: `New court decision rules`
- `2092-11-03` → **November** | own: `No later than Apr` [other] | cf: `Hard physical labor.`

### iso_date_year_plus_one  (counterfactual direction: initials_two_words)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 21 |
| copy_query | 0 | 0 |
| copy_demo_target | 6 | 4 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 19 | 42 |
| input_variant | 3 | 6 |
| other | 119 | 77 |

Examples (own-ablation errors):

- `2025-07-11` → **2026** | own: `1930-09-13` [other] | cf: `2016`
- `1962-12-16` → **1963** | own: `4/2/19` [other] | cf: `... finally caught up with`
- `2031-01-02` → **2032** | own: `1975` [in_pool_wrong] | cf: `2913-04-`
- `1887-09-21` → **1888** | own: `This time we revised from` [other] | cf: `2015-07-01`
- `2026-07-09` → **2027** | own: `2025-06-12` [other] | cf: `2026`
- `1934-05-23` → **1935** | own: `scarecrow` [other] | cf: `1933-JUN-`
- `1841-11-22` → **1842** | own: `1853-06-` [other] | cf: `1846-12-`
- `1853-03-09` → **1854** | own: `1893-05` [other] | cf: `1859`
- `1816-03-28` → **1817** | own: `I've no clue.` [other] | cf: `2004`
- `2083-03-09` → **2084** | own: `2017-18` [other] | cf: `2083-03-`

### landmark-country  (counterfactual direction: adjective_to_adverb)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 5 | 34 |
| copy_query | 3 | 1 |
| copy_demo_target | 1 | 5 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 1 | 20 |
| input_variant | 12 | 6 |
| other | 128 | 84 |

Examples (own-ablation errors):

- `North Middlesex, Ontario` → **Canada** | own: `Middlesex North District` [other] | cf: `Ottawa`
- `Tampines Expressway` → **Singapore** | own: `Triumph Circle` [other] | cf: `Singapore`
- `Boguchany Dam` → **Russia** | own: `Dzhugash` [other] | cf: `Switzerland`
- `Habitat 67` → **Canada** | own: `8269 Springbank` [other] | cf: `Spain`
- `Istanbul Airport` → **Turkey** | own: `I've never been` [other] | cf: `Turkey`
- `Addanki mandal` → **India** | own: `West O, K` [other] | cf: `Maharashtra`
- `Indus River` → **India** | own: `Indus River` [copy_query] | cf: `Pakistan`
- `Kresttsy` → **Russia** | own: `new secondary school in` [other] | cf: `Venice, Friuli`
- `Hooge Crater Commonwealth War Graves Commission Cemetery` → **Belgium** | own: `Gorgenver` [other] | cf: `Riberalta`
- `Adliswil` → **Switzerland** | own: `Naked` [other] | cf: `Switzerland`

### language_identification  (counterfactual direction: german_noun_gender)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 16 |
| copy_query | 13 | 1 |
| copy_demo_target | 1 | 4 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 1 |
| input_variant | 13 | 9 |
| other | 122 | 119 |

Examples (own-ablation errors):

- `reale` → **Italian** | own: `si` [other] | cf: `afraid`
- `polizia` → **Italian** | own: `spagnola` [other] | cf: `Police, Italian word`
- `energía` → **Spanish** | own: `electricidad` [other] | cf: `Spanish`
- `diciembre` → **Spanish** | own: `di 9º` [other] | cf: `both.  but`
- `Natur` → **German** | own: `See [Verb` [other] | cf: `Erdäichlic`
- `buscar` → **Spanish** | own: `recomendo` [other] | cf: `I'm search`
- `intérieur` → **French** | own: `merde` [other] | cf: `French`
- `faccia` → **Italian** | own: `fise` [other] | cf: `faccia di z`
- `deciso` → **Italian** | own: `decidono` [input_variant] | cf: `Decision`
- `Gebäude` → **German** | own: `Absolut` [other] | cf: `Building`

### larger_of_pair  (counterfactual direction: singular_or_plural)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 17 | 37 |
| copy_query | 2 | 6 |
| copy_demo_target | 11 | 2 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 15 | 26 |
| input_variant | 2 | 8 |
| other | 103 | 71 |

Examples (own-ablation errors):

- `402 846` → **846** | own: `402` [in_pool_wrong] | cf: `541 76 79 79`
- `456 977` → **977** | own: `456` [in_pool_wrong] | cf: `977`
- `251 920` → **920** | own: `561 631` [other] | cf: `920`
- `928 923` → **928** | own: `1` [other] | cf: `928`
- `207 650` → **650** | own: `122` [other] | cf: `207`
- `283 161` → **283** | own: `You can start with 120` [other] | cf: `163`
- `151 685` → **685** | own: `585` [other] | cf: `685`
- `304 755` → **755** | own: `176 626` [other] | cf: `789`
- `771 485` → **771** | own: `ok ok` [other] | cf: `891`
- `952 726` → **952** | own: `2797` [other] | cf: `726`

### larger_than_1000  (counterfactual direction: past_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 14 | 27 |
| copy_query | 0 | 1 |
| copy_demo_target | 11 | 24 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 3 | 7 |
| input_variant | 0 | 0 |
| other | 122 | 90 |

Examples (own-ablation errors):

- `23305` → **yes** | own: `stand back` [other] | cf: `yes`
- `286` → **no** | own: `yeh` [other] | cf: `Yes`
- `7986` → **yes** | own: `lmagedical` [other] | cf: `yes`
- `869` → **no** | own: `divide that into 2` [other] | cf: `SPD869`
- `23936` → **yes** | own: `I don't totally` [other] | cf: `65%/35`
- `551` → **no** | own: `yes...more watches` [other] | cf: `yes`
- `7099` → **yes** | own: `A fluctuating esc` [other] | cf: `I get my beginning`
- `956` → **no** | own: `yes, it�` [other] | cf: `this is considered`
- `13998` → **yes** | own: `beer` [other] | cf: `no`
- `321` → **no** | own: `(~####~ 6` [other] | cf: `yes`

### lowercase_first_letter  (counterfactual direction: country-capital)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 4 | 20 |
| copy_query | 13 | 4 |
| copy_demo_target | 14 | 15 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 6 | 41 |
| input_variant | 9 | 1 |
| other | 103 | 69 |

Examples (own-ablation errors):

- `EXPLORE` → **e** | own: `extremals` [other] | cf: `e`
- `PATIENT` → **p** | own: `not sure yet` [other] | cf: `sionte`
- `DYNAMIC` → **d** | own: `1.0` [other] | cf: `Multilane mult`
- `COURAGEOUS` → **c** | own: `Run A's to` [other] | cf: `→ have not yet`
- `WITHOUT` → **w** | own: `q` [copy_demo_target] | cf: `without`
- `INNOCENT` → **i** | own: `honest` [other] | cf: `a :too late`
- `BABOON` → **b** | own: `E./h EN` [other] | cf: `bab`
- `MELLOW` → **m** | own: `He has little blood` [other] | cf: `aha!`
- `BEHIND` → **b** | own: `a` [in_pool_wrong] | cf: `i`
- `BELOW` → **b** | own: `gs` [other] | cf: `b`

### lowercase_word  (counterfactual direction: product-company)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 42 | 70 |
| copy_query | 0 | 0 |
| copy_demo_target | 13 | 5 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 14 | 7 |
| input_variant | 10 | 9 |
| other | 71 | 59 |

Examples (own-ablation errors):

- `WORK` → **work** | own: `commerce, ecclesiatical` [other] | cf: `working`
- `WORD` → **word** | own: `word (v)` [input_variant] | cf: `word`
- `SUMMER` → **summer** | own: `niceties` [other] | cf: `sum-MER`
- `ISLAND` → **island** | own: `in sPlIt.` [other] | cf: `island`
- `AHEAD` → **ahead** | own: `DOING` [in_pool_wrong] | cf: `ahead`
- `NETWORK` → **network** | own: `trigerring with` [other] | cf: `fidelity`
- `HELP` → **help** | own: `=confused` [other] | cf: `help`
- `ESPECIALLY` → **especially** | own: `latter` [other] | cf: `especially`
- `EXPECT` → **expect** | own: `Possible` [in_pool_wrong] | cf: `expect`
- `THANKS` → **thanks** | own: `Geist` [other] | cf: `thanks`

### national_parks  (counterfactual direction: past_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 45 |
| copy_query | 1 | 0 |
| copy_demo_target | 1 | 3 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 8 | 40 |
| input_variant | 19 | 10 |
| other | 119 | 52 |

Examples (own-ablation errors):

- `Cumberland Gap National Historical Park` → **Virginia** | own: `Gibbon Tract` [other] | cf: `Tennessee`
- `James A. Garfield National Historic Site` → **Ohio** | own: `Garfield House, One Garfield` [other] | cf: `Logan County Ohio`
- `Fort Scott National Historic Site` → **Kansas** | own: `Fort Scott` [input_variant] | cf: `Kansas`
- `Theodore Roosevelt Inaugural National Historic Site` → **New York** | own: `Bronx, NY` [other] | cf: `United States`
- `Fort Necessity National Battlefield` → **Pennsylvania** | own: `South Carolina` [in_pool_wrong] | cf: `Pennsylvania`
- `Channel Islands National Park` → **California** | own: `Channel Islands` [input_variant] | cf: `Channel Islands`
- `Fort Donelson National Battlefield` → **Tennessee** | own: `Gettysburg National Military Park` [other] | cf: `Tennessee`
- `Devils Postpile National Monument` → **California** | own: `Yolo County` [other] | cf: `New Mexico`
- `Wolf Trap National Park for the Performing Arts` → **Virginia** | own: `Washington, D.C` [other] | cf: `Virginia`
- `Hubbell Trading Post National Historic Site` → **Arizona** | own: `Deming` [other] | cf: `Rio Grande Village`

### natural_manmade  (counterfactual direction: pos_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 9 |
| copy_query | 9 | 2 |
| copy_demo_target | 1 | 2 |
| copy_demo_input | 1 | 1 |
| in_pool_wrong | 0 | 0 |
| input_variant | 10 | 4 |
| other | 126 | 132 |

Examples (own-ablation errors):

- `maple` → **natural** | own: `soft` [other] | cf: `natural`
- `zinnia` → **natural** | own: `girl` [other] | cf: `flower`
- `mongoose` → **natural** | own: `humquose` [other] | cf: `unmanned aerial vehicle`
- `jellyfish` → **natural** | own: `jelly` [input_variant] | cf: `octopus`
- `badger` → **natural** | own: `coarsed` [other] | cf: `animal`
- `shrike` → **natural** | own: `rubber?` [other] | cf: `stork`
- `stove` → **manmade** | own: `stoves` [input_variant] | cf: `dont use for heat,`
- `pot` → **manmade** | own: `earth` [other] | cf: `primary`
- `loon` → **natural** | own: `loonie` [input_variant] | cf: `wild animal or bird`
- `heater` → **manmade** | own: `gender` [other] | cf: `hell`

### next_item  (counterfactual direction: plural_to_singular)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 18 |
| copy_query | 16 | 20 |
| copy_demo_target | 10 | 3 |
| copy_demo_input | 2 | 7 |
| in_pool_wrong | 42 | 44 |
| input_variant | 1 | 2 |
| other | 76 | 56 |

Examples (own-ablation errors):

- `e` → **f** | own: `five` [copy_demo_target] | cf: `f`
- `20` → **21** | own: `five` [in_pool_wrong] | cf: `Five`
- `12` → **13** | own: `thursday` [in_pool_wrong] | cf: `23`
- `twelve` → **thirteen** | own: `two` [in_pool_wrong] | cf: `twenty eight`
- `YY` → **ZZ** | own: `Y` [other] | cf: `YY`
- `3` → **4** | own: `3` [copy_query] | cf: `4`
- `IX` → **X** | own: `v` [copy_demo_target] | cf: `Y`
- `J` → **K** | own: `NRF` [other] | cf: `K`
- `m` → **n** | own: `g` [in_pool_wrong] | cf: `n`
- `XIV` → **XV** | own: `PP` [in_pool_wrong] | cf: `Xabergundi`

### next_month_of_date  (counterfactual direction: verb_tense_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 26 |
| copy_query | 4 | 1 |
| copy_demo_target | 5 | 11 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 18 | 49 |
| input_variant | 12 | 6 |
| other | 108 | 56 |

Examples (own-ablation errors):

- `October 1838` → **November** | own: `32°` [other] | cf: `Surrey Downs`
- `December 1812` → **January** | own: `negro` [other] | cf: `January`
- `February 1805` → **March** | own: `Did you expect it` [other] | cf: `June 1910`
- `February 1901` → **March** | own: `Quintillions` [other] | cf: `December`
- `February 1861` → **March** | own: `I had a brother` [other] | cf: `It was in the`
- `January 1981` → **February** | own: `March` [copy_demo_target] | cf: `February`
- `November 1887` → **December** | own: `approximately November 1887` [input_variant] | cf: `July 1908`
- `January 1950` → **February** | own: `Another` [other] | cf: `March`
- `March 2008` → **April** | own: `A few` [other] | cf: `July`
- `January 1822` → **February** | own: `January 1822` [copy_query] | cf: `January 2022`

### next_number_digits  (counterfactual direction: pos_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 4 | 39 |
| copy_query | 11 | 7 |
| copy_demo_target | 10 | 4 |
| copy_demo_input | 4 | 3 |
| in_pool_wrong | 51 | 55 |
| input_variant | 0 | 0 |
| other | 70 | 42 |

Examples (own-ablation errors):

- `109` → **110** | own: `85` [in_pool_wrong] | cf: `110`
- `70` → **71** | own: `39` [in_pool_wrong] | cf: `71`
- `78` → **79** | own: `35` [in_pool_wrong] | cf: `81`
- `47` → **48** | own: `47` [copy_query] | cf: `48`
- `63` → **64** | own: `49` [in_pool_wrong] | cf: `66`
- `131` → **132** | own: `133` [in_pool_wrong] | cf: `132`
- `49` → **50** | own: `145` [in_pool_wrong] | cf: `250`
- `132` → **133** | own: `4` [copy_demo_input] | cf: `1222`
- `29` → **30** | own: `246` [other] | cf: `method(up=`
- `147` → **148** | own: `Contract` [other] | cf: `Fuel tanks`

### number_word_to_digits  (counterfactual direction: person-sport)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 41 | 69 |
| copy_query | 4 | 0 |
| copy_demo_target | 1 | 0 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 14 | 17 |
| input_variant | 8 | 0 |
| other | 82 | 64 |

Examples (own-ablation errors):

- `one hundred eighteen` → **118** | own: `1 18` [other] | cf: `118`
- `one hundred twenty-nine` → **129** | own: `120` [other] | cf: `129`
- `one thousand four hundred five` → **1405** | own: `four hundred fifty` [other] | cf: `4155`
- `one thousand one hundred seventy` → **1170** | own: `1,170` [other] | cf: `1770`
- `eighty-two` → **82** | own: `∴ 102` [other] | cf: `82`
- `one thousand three hundred fifty-five` → **1355** | own: `1,365` [other] | cf: `1,355`
- `one thousand thirteen` → **1013** | own: `13` [other] | cf: `13`
- `one thousand four hundred one` → **1401** | own: `1,400` [other] | cf: `1,400.1`
- `one thousand one hundred sixty-six` → **1166** | own: `1,166 3/` [other] | cf: `????`
- `one thousand one hundred seventy-eight` → **1178** | own: `twelve-hundred-` [other] | cf: `178`

### park-country  (counterfactual direction: german-english)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 47 |
| copy_query | 0 | 1 |
| copy_demo_target | 3 | 9 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 4 | 39 |
| input_variant | 20 | 1 |
| other | 116 | 52 |

Examples (own-ablation errors):

- `Ile-Alatau National Park` → **Kazakhstan** | own: `Yangi-Yak` [other] | cf: `Kazakhstan`
- `Chubu-Sangaku National Park` → **Japan** | own: `Machida, Tokyo` [other] | cf: `South Africa`
- `Virunga National Park` → **Congo** | own: `eastern Central African country, and Uganda` [other] | cf: `Congo`
- `Llanganates National Park` → **Ecuador** | own: `18 km Northwest of Ant` [other] | cf: `Oaxaca`
- `Cape Greco National Park` → **Cyprus** | own: `Bulgaria` [copy_demo_target] | cf: `Greece`
- `Serra do Divisor National Park` → **Brazil** | own: `Tejo` [other] | cf: `Guinea-Bissau`
- `Yankari National Park` → **Nigeria** | own: `Potosi` [other] | cf: `Mongun Lawni National`
- `Mount Elgon National Park` → **Uganda** | own: `Nyambura` [other] | cf: `Republic of Uganda`
- `Auyuittuq National Park` → **Canada** | own: `Stanley` [other] | cf: `Greenland`
- `Sevan National Park` → **Armenia** | own: `Serebri` [other] | cf: `Northern Iran`

### past_to_base  (counterfactual direction: animal_plant_object)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 12 | 32 |
| copy_query | 28 | 24 |
| copy_demo_target | 14 | 3 |
| copy_demo_input | 6 | 2 |
| in_pool_wrong | 7 | 8 |
| input_variant | 8 | 16 |
| other | 75 | 65 |

Examples (own-ablation errors):

- `sanded` → **sand** | own: `sequenceded` [other] | cf: `sanded`
- `claimed` → **claim** | own: `claimed` [copy_query] | cf: `claim`
- `minored` → **minor** | own: `mininary` [other] | cf: `little`
- `juiced` → **juice** | own: `pt1` [other] | cf: `stork.`
- `pictured` → **picture** | own: `pre/post p` [other] | cf: `carved, cut`
- `encouraged` → **encourage** | own: `encouraged` [copy_query] | cf: `encourage`
- `whited` → **white** | own: `purified` [other] | cf: `whitey`
- `beached` → **beach** | own: `effervescent` [other] | cf: `becha’`
- `solved` → **solve** | own: `mention` [copy_demo_target] | cf: `solved`
- `found` → **find** | own: `feyned` [other] | cf: `findd`

### person-instrument  (counterfactual direction: antonym)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 10 |
| copy_query | 0 | 1 |
| copy_demo_target | 2 | 11 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 19 |
| input_variant | 6 | 1 |
| other | 142 | 108 |

Examples (own-ablation errors):

- `Leonard Cohen` → **guitar** | own: `strangler` [other] | cf: `guitar`
- `Bradford Cox` → **guitar** | own: `aspen` [other] | cf: `fundraising/recognizing`
- `Jon Eardley` → **trumpet** | own: `actress` [other] | cf: `vocals`
- `Millard Powers` → **guitar** | own: `peli (play` [other] | cf: `drums`
- `Billy Taylor` → **piano** | own: `stretch` [other] | cf: `guitar, harmonica`
- `Blind Willie McTell` → **guitar** | own: `fordable` [other] | cf: `harmonica`
- `John Wesley` → **guitar** | own: `Beverage` [other] | cf: `Piano`
- `Ola Kvernberg` → **violin** | own: `nema trening` [other] | cf: `guitar, vocals`
- `Henry Litolff` → **piano** | own: `thee; that` [other] | cf: `drum kit`
- `Guy Picciotto` → **guitar** | own: `spazio` [other] | cf: `scissors`

### person-sport  (counterfactual direction: french_noun_gender)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 40 |
| copy_query | 0 | 0 |
| copy_demo_target | 1 | 4 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 15 |
| input_variant | 5 | 1 |
| other | 142 | 90 |

Examples (own-ablation errors):

- `Lionel Conacher` → **hockey** | own: `flame-thrower` [other] | cf: `hockey`
- `Gale Sayers` → **football** | own: `easily` [other] | cf: `Italy`
- `Júlio Baptista` → **soccer** | own: `I guess pass` [other] | cf: `macquarie,`
- `Teemu Sälännä` → **hockey** | own: `epic` [other] | cf: `olut`
- `Howie Morenz` → **hockey** | own: `puck-wise,` [other] | cf: `played left wing for`
- `Jermain Defoe` → **soccer** | own: `injured` [other] | cf: `a seagull`
- `Kareem Abdul-Jabbar` → **basketball** | own: `osteogenic` [other] | cf: `basketball`
- `Frank Mahovlich` → **hockey** | own: `bill jackson` [other] | cf: `hockey`
- `Dennis Rodman` → **basketball** | own: `deadpans` [other] | cf: `basketball`
- `Javier Hernández` → **soccer** | own: `«esto p` [other] | cf: `lefty slinger`

### person_place_thing  (counterfactual direction: gerund_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 6 |
| copy_query | 15 | 4 |
| copy_demo_target | 1 | 4 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 0 |
| input_variant | 4 | 10 |
| other | 127 | 125 |

Examples (own-ablation errors):

- `soldier` → **person** | own: `guerilla` [other] | cf: `person who is trained`
- `clown` → **person** | own: `traffic.` [other] | cf: `clown`
- `saxophone` → **thing** | own: `sesante` [other] | cf: `creative work[ or`
- `highchair` → **thing** | own: `food tray` [other] | cf: `please`
- `villain` → **person** | own: `villain` [copy_query] | cf: `bad person`
- `car` → **thing** | own: `autos` [other] | cf: `automobile`
- `colonel` → **person** | own: `lighthouse` [other] | cf: `man`
- `boulevard` → **place** | own: `street` [other] | cf: `linear road`
- `placemat` → **thing** | own: `tablecloth` [other] | cf: `table`
- `atoll` → **place** | own: `joke` [other] | cf: `operation`

### plural_to_singular  (counterfactual direction: english-portuguese)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 9 | 36 |
| copy_query | 24 | 20 |
| copy_demo_target | 12 | 5 |
| copy_demo_input | 5 | 0 |
| in_pool_wrong | 5 | 2 |
| input_variant | 12 | 22 |
| other | 83 | 65 |

Examples (own-ablation errors):

- `windings` → **winding** | own: `discharges` [other] | cf: `wind`
- `births` → **birth** | own: `critique` [copy_demo_target] | cf: `adjustment`
- `dioxides` → **dioxide** | own: `ozone` [other] | cf: `dioxide`
- `campuses` → **campus** | own: `campuses` [copy_query] | cf: `campus`
- `legacies` → **legacy** | own: `legacies` [copy_query] | cf: `legacy`
- `crescents` → **crescent** | own: `Virginia cassiers` [other] | cf: `renaissance`
- `miracles` → **miracle** | own: `welf` [other] | cf: `miracle. Mirror. Myth`
- `pricks` → **prick** | own: `combs` [copy_demo_input] | cf: `prick`
- `horizons` → **horizon** | own: `what horizon you want to` [other] | cf: `horizon`
- `hardships` → **hardship** | own: `challenges` [other] | cf: `hardships`

### pos_label  (counterfactual direction: larger_of_pair)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 13 |
| copy_query | 12 | 9 |
| copy_demo_target | 2 | 13 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 9 |
| input_variant | 6 | 13 |
| other | 129 | 92 |

Examples (own-ablation errors):

- `compilation` → **noun** | own: `role` [other] | cf: `dictionary`
- `endlessly` → **adverb** | own: `anarch` [other] | cf: `whiles`
- `disconnect` → **verb** | own: `kaufen. (` [other] | cf: `download download string`
- `sticker` → **noun** | own: `stencil` [other] | cf: `counsellor`
- `someday` → **adverb** | own: `yes` [other] | cf: `adverb`
- `formerly` → **adverb** | own: `weak routing` [other] | cf: `convict`
- `keeper` → **noun** | own: `kkeeper` [input_variant] | cf: `to keep (English &`
- `stuffed` → **verb** | own: `a` [other] | cf: `adjective`
- `always` → **adverb** | own: `ov` [other] | cf: `yes`
- `occasionally` → **adverb** | own: `usurpation` [other] | cf: `adjective`

### present-past  (counterfactual direction: smaller_of_pair)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 9 | 34 |
| copy_query | 49 | 28 |
| copy_demo_target | 2 | 5 |
| copy_demo_input | 16 | 2 |
| in_pool_wrong | 0 | 1 |
| input_variant | 14 | 21 |
| other | 60 | 59 |

Examples (own-ablation errors):

- `transform` → **transformed** | own: `tra*form` [other] | cf: `田太也`
- `want` → **wanted** | own: `want` [copy_query] | cf: `wanted`
- `challenge` → **challenged** | own: `see [Apple][1` [other] | cf: `colorful`
- `analyze` → **analyzed** | own: `analyze` [copy_query] | cf: `level below is the one`
- `lend` → **lent** | own: `lend` [copy_query] | cf: `lent`
- `serve` → **served** | own: `serve` [copy_query] | cf: `served`
- `use` → **used** | own: `study` [copy_demo_input] | cf: `use a.txt for`
- `reduce` → **reduced** | own: `reduce` [copy_query] | cf: `reduced`
- `remain` → **remained** | own: `persist` [other] | cf: `remain`
- `secure` → **secured** | own: `we have secured everything you` [input_variant] | cf: `defending`

### prev_number_digits  (counterfactual direction: person_place_thing)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 16 |
| copy_query | 10 | 9 |
| copy_demo_target | 6 | 5 |
| copy_demo_input | 16 | 6 |
| in_pool_wrong | 57 | 73 |
| input_variant | 0 | 0 |
| other | 60 | 41 |

Examples (own-ablation errors):

- `63` → **62** | own: `111` [in_pool_wrong] | cf: `I don't know`
- `103` → **102** | own: `149` [copy_demo_target] | cf: `Yes his majesty is`
- `191` → **190** | own: `100` [copy_demo_input] | cf: `158`
- `150` → **149** | own: `49` [copy_demo_target] | cf: `49`
- `119` → **118** | own: `112` [copy_demo_input] | cf: `115`
- `130` → **129** | own: `20` [copy_demo_target] | cf: `N?`
- `116` → **115** | own: `116` [copy_query] | cf: `15, 56,`
- `96` → **95** | own: `A is the 96` [other] | cf: `95`
- `28` → **27** | own: `028` [other] | cf: `97582`
- `172` → **171** | own: `F` [other] | cf: `146`

### product-company  (counterfactual direction: gerund_to_past)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 18 |
| copy_query | 4 | 4 |
| copy_demo_target | 4 | 5 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 4 |
| input_variant | 15 | 7 |
| other | 125 | 112 |

Examples (own-ablation errors):

- `Logic Studio` → **Apple** | own: `Intel` [copy_demo_target] | cf: `Nitro`
- `MVS` → **IBM** | own: `Subsidized` [other] | cf: `Renault`
- `Xcode` → **Apple** | own: `K2` [other] | cf: `Android Studio`
- `Report Program Generator` → **IBM** | own: `James Jenkins` [other] | cf: `Vortex`
- `Synchronized Accessible Media Interchange` → **Microsoft** | own: `Cool, the next` [other] | cf: `The property group channel`
- `Alfa Romeo MiTo` → **Fiat** | own: `Tourer` [other] | cf: `Transformer Hiro`
- `Internet Explorer 11` → **Microsoft** | own: `Internet Explorer support has` [input_variant] | cf: `Internet Explorer 11 (`
- `Audio Interchange File Format` → **Apple** | own: `r32` [other] | cf: `MP3`
- `LGM-30 Minuteman` → **Boeing** | own: `Atomic` [other] | cf: `Soviet Surface-to`
- `Symbian` → **Nokia** | own: `no longer exists` [other] | cf: `Blackberry`

### sentiment  (counterfactual direction: park-country)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 29 |
| copy_query | 0 | 0 |
| copy_demo_target | 3 | 11 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 7 |
| input_variant | 1 | 2 |
| other | 146 | 101 |

Examples (own-ablation errors):

- `Represents the depths to which the girls-behaving-badly film has fallen.` → **negative** | own: `ceremonially—pain` [other] | cf: `12`
- `Shyamalan should stop trying to please his mom.` → **negative** | own: `sad` [other] | cf: `Negative`
- `A film that's flawed and brilliant in equal measure.` → **positive** | own: `stop-motion sculpture` [other] | cf: `refused read on`
- `Fluffy neo-noir hiding behind cutesy film references.` → **negative** | own: `No.` [other] | cf: `negative`
- `...one resurrection too many.` → **negative** | own: `ex-terrestrial` [other] | cf: `negative`
- `Illiterate, often inert sci-fi action thriller.` → **negative** | own: `flick it. Lay` [other] | cf: `neutral`
- `A very charming and funny movie.` → **positive** | own: `?` [other] | cf: `formulaic, more`
- `At times, it actually hurts to watch.` → **negative** | own: `I do find it` [other] | cf: `Negative`
- `Demonstrates the unusual power of thoughtful, subjective filmmaking.` → **positive** | own: `nonobjective` [other] | cf: `negative`
- `What makes it worth watching is Quaid's performance.` → **positive** | own: `His obsessions--` [other] | cf: `A manic fring`

### singular-plural  (counterfactual direction: city-country)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 7 | 72 |
| copy_query | 52 | 15 |
| copy_demo_target | 0 | 0 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 2 | 5 |
| input_variant | 17 | 17 |
| other | 71 | 41 |

Examples (own-ablation errors):

- `soap` → **soaps** | own: `soap` [copy_query] | cf: `soap`
- `microwave` → **microwaves** | own: `electric stove` [other] | cf: `microwave`
- `blender` → **blenders** | own: `wheels` [other] | cf: `whisks`
- `boot` → **boots** | own: `boot` [copy_query] | cf: `boots`
- `hose` → **hoses** | own: `sweatier` [other] | cf: `hoses`
- `mountain` → **mountains** | own: `bicycle` [copy_demo_input] | cf: `mountains`
- `pants` → **pants** | own: `jeans` [other] | cf: `places for luggage`
- `glove` → **gloves** | own: `glove` [copy_query] | cf: `gloves`
- `river` → **rivers** | own: `bank` [other] | cf: `aries`
- `toothbrush` → **toothbrushes** | own: `living room` [other] | cf: `brushes`

### singular_or_plural  (counterfactual direction: animal_plant_object)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 2 | 11 |
| copy_query | 2 | 14 |
| copy_demo_target | 0 | 4 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 7 |
| input_variant | 15 | 20 |
| other | 131 | 94 |

Examples (own-ablation errors):

- `babies` → **plural** | own: `soft` [other] | cf: `cutlet`
- `god` → **singular** | own: `jow` [other] | cf: `nothing =?`
- `beds` → **plural** | own: `aren'ts` [other] | cf: `I went to bed`
- `fairs` → **plural** | own: `Trust` [other] | cf: `plural`
- `fishes` → **plural** | own: `gamma radiations` [other] | cf: `two fishes`
- `properties` → **plural** | own: `doesn't have properties` [input_variant] | cf: `one, more`
- `buildings` → **plural** | own: `U-shaped` [other] | cf: `plural`
- `blue` → **singular** | own: `seems-as-` [other] | cf: `bluebug`
- `balls` → **plural** | own: `isn't balls singular` [input_variant] | cf: `ball-shaped`
- `energies` → **plural** | own: `energy` [input_variant] | cf: `plural #It's`

### smaller_of_pair  (counterfactual direction: country-capital)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 14 | 33 |
| copy_query | 1 | 3 |
| copy_demo_target | 10 | 3 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 29 | 32 |
| input_variant | 3 | 8 |
| other | 92 | 71 |

Examples (own-ablation errors):

- `310 83` → **83** | own: `30 33` [other] | cf: `83 510`
- `885 424` → **424** | own: `1,068` [other] | cf: `885.424`
- `416 749` → **416** | own: `373` [other] | cf: `749`
- `880 559` → **559** | own: `16` [other] | cf: `559`
- `155 415` → **155** | own: `151 173 326 353` [other] | cf: `155 415`
- `339 245` → **245** | own: `345 435` [other] | cf: `445`
- `47 469` → **47** | own: `96 485 17` [other] | cf: `469`
- `878 549` → **549** | own: `676 899` [other] | cf: `676`
- `42 506` → **42** | own: `85` [in_pool_wrong] | cf: `42`
- `281 361` → **281** | own: `135` [copy_demo_target] | cf: `361`

### spanish-english  (counterfactual direction: plural_to_singular)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 25 | 64 |
| copy_query | 15 | 7 |
| copy_demo_target | 2 | 1 |
| copy_demo_input | 2 | 0 |
| in_pool_wrong | 4 | 6 |
| input_variant | 12 | 5 |
| other | 90 | 67 |

Examples (own-ablation errors):

- `chaqueta` → **jacket** | own: `chaqueta` [copy_query] | cf: `jacket`
- `caer` → **fall** | own: `to fall` [other] | cf: `to fall`
- `solitario` → **lonely** | own: `solitario` [copy_query] | cf: `single`
- `dinero` → **money** | own: `Dinero` [copy_query] | cf: `people`
- `nervioso` → **nervous** | own: `aliento` [other] | cf: `scared`
- `cacahuete` → **peanut** | own: `granada` [other] | cf: `palmito`
- `destruir` → **destroy** | own: `depressir` [other] | cf: `destroy`
- `leche` → **milk** | own: `Téte` [other] | cf: `milk`
- `coleccionar` → **collect** | own: `to collect` [other] | cf: `collect (for American`
- `lógica` → **logic** | own: `clock` [in_pool_wrong] | cf: `rules`

### spanish_noun_gender  (counterfactual direction: english-french)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 30 |
| copy_query | 5 | 4 |
| copy_demo_target | 0 | 7 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 9 |
| input_variant | 15 | 8 |
| other | 130 | 91 |

Examples (own-ablation errors):

- `ejército` → **masculine** | own: `Army` [other] | cf: `army, troops`
- `arco` → **masculine** | own: `colunas` [other] | cf: `masculine`
- `movimiento` → **masculine** | own: `cambio` [other] | cf: `virgular`
- `teatro` → **masculine** | own: `theater` [other] | cf: `theater`
- `caja` → **feminine** | own: `tray` [other] | cf: `box`
- `sanidad` → **feminine** | own: `health` [other] | cf: `health`
- `occidente` → **masculine** | own: `oriente` [other] | cf: `doesn't work.`
- `vigor` → **masculine** | own: `shuster` [other] | cf: `feminine, vigor`
- `tarjeta` → **feminine** | own: `wallet` [other] | cf: `negative/baro`
- `búsqueda` → **feminine** | own: `search` [other] | cf: `masculine`

### starts_with_vowel  (counterfactual direction: past_to_base)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 0 | 6 |
| copy_query | 11 | 9 |
| copy_demo_target | 1 | 8 |
| copy_demo_input | 0 | 2 |
| in_pool_wrong | 0 | 1 |
| input_variant | 12 | 4 |
| other | 126 | 120 |

Examples (own-ablation errors):

- `external` → **vowel** | own: `vertexYou: now.` [other] | cf: `entering outside`
- `apartment` → **vowel** | own: `bedroom` [other] | cf: `ADAPT`
- `ordinary` → **vowel** | own: `molar` [other] | cf: `nope`
- `unable` → **vowel** | own: `allow` [other] | cf: `to`
- `deal` → **consonant** | own: `love` [other] | cf: `card`
- `moment` → **consonant** | own: `Images` [other] | cf: `pos`
- `ice` → **vowel** | own: `rink` [other] | cf: `snow`
- `finally` → **consonant** | own: `finally` [copy_query] | cf: `possibly`
- `team` → **consonant** | own: `` [other] | cf: `consonant`
- `overall` → **vowel** | own: `improvised` [other] | cf: `-3`

### third_person_to_base  (counterfactual direction: initials_two_words)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 15 | 33 |
| copy_query | 27 | 19 |
| copy_demo_target | 16 | 10 |
| copy_demo_input | 5 | 1 |
| in_pool_wrong | 3 | 4 |
| input_variant | 7 | 15 |
| other | 77 | 68 |

Examples (own-ablation errors):

- `kids` → **kid** | own: `kids` [copy_query] | cf: `kids`
- `transitions` → **transition** | own: `transitions` [copy_query] | cf: `it's transitioned`
- `gardens` → **garden** | own: `galapogues` [other] | cf: `garden`
- `mounts` → **mount** | own: `sheep` [other] | cf: `mount`
- `adds` → **add** | own: `one` [other] | cf: `raises`
- `arrests` → **arrest** | own: `arrestee` [input_variant] | cf: `arrest`
- `lows` → **low** | own: `lows` [copy_query] | cf: `low`
- `owns` → **own** | own: `his, her,` [other] | cf: `**Awne`
- `grounds` → **ground** | own: `grounds` [copy_query] | cf: `gounts`
- `lasts` → **last** | own: `day` [other] | cf: `point`

### titlecase_phrase  (counterfactual direction: verb_tense_label)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 25 | 44 |
| copy_query | 0 | 0 |
| copy_demo_target | 6 | 5 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 0 | 0 |
| input_variant | 28 | 27 |
| other | 91 | 74 |

Examples (own-ablation errors):

- `empty protest` → **Empty Protest** | own: `Unresolved conventions or conflicts` [other] | cf: `empty protest`
- `immense custom` → **Immense Custom** | own: `Impressive customer` [other] | cf: `imence 2 virus`
- `harmless spirit` → **Harmless Spirit** | own: `breath of Satan` [other] | cf: `Harmless  Centaurs`
- `complex feed` → **Complex Feed** | own: `Complex feed is basically a series of` [input_variant] | cf: `No Clear Idea`
- `fleet counter` → **Fleet Counter** | own: `satellite system` [other] | cf: `Fleet counter`
- `famous cap` → **Famous Cap** | own: `capitalized` [other] | cf: `Selfish Ending`
- `vacant trouble` → **Vacant Trouble** | own: `vacate` [input_variant] | cf: `Great Vacant Here`
- `bum private` → **Bum Private** | own: `cush a couch` [other] | cf: `bum private`
- `worthy whose` → **Worthy Whose** | own: `Worthy company` [input_variant] | cf: `Worthy`
- `fatty royal` → **Fatty Royal** | own: `Fatty Worth` [input_variant] | cf: `Fatty Royal`

### uppercase_word  (counterfactual direction: english-spanish)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 30 | 65 |
| copy_query | 0 | 0 |
| copy_demo_target | 20 | 8 |
| copy_demo_input | 0 | 0 |
| in_pool_wrong | 13 | 8 |
| input_variant | 21 | 9 |
| other | 66 | 60 |

Examples (own-ablation errors):

- `private` → **PRIVATE** | own: `public` [in_pool_wrong] | cf: `PRIVATE`
- `special` → **SPECIAL** | own: `speciality` [input_variant] | cf: `SEGR`
- `where` → **WHERE** | own: `when` [in_pool_wrong] | cf: `IN-FUSER`
- `not` → **NOT** | own: `not know` [other] | cf: `Not`
- `sports` → **SPORTS** | own: `football` [in_pool_wrong] | cf: `GYMNASTICS`
- `actually` → **ACTUALLY** | own: `actually he should so that he` [input_variant] | cf: `Wassa`
- `welcome` → **WELCOME** | own: `THANK YOU!!!` [other] | cf: `WARRANT!`
- `said` → **SAID** | own: `Which is why all multiples of` [other] | cf: `VERY`
- `front` → **FRONT** | own: `ABX` [other] | cf: `FRONT`
- `clean` → **CLEAN** | own: `CLEAN besht` [input_variant] | cf: `CLEAN`

### us-city-state  (counterfactual direction: starts_with_vowel)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 3 | 54 |
| copy_query | 15 | 2 |
| copy_demo_target | 3 | 2 |
| copy_demo_input | 1 | 0 |
| in_pool_wrong | 3 | 38 |
| input_variant | 12 | 1 |
| other | 113 | 53 |

Examples (own-ablation errors):

- `Fargo` → **North Dakota** | own: `Bestiality` [other] | cf: `Some other state`
- `Scottsboro` → **Alabama** | own: `Scottsboro, Ala` [input_variant] | cf: `Birmingham, Alabama`
- `Trenton` → **New Jersey** | own: `Pigeon Forge` [copy_demo_input] | cf: `Pennsylvania`
- `Clayton` → **Delaware** | own: `Mets` [other] | cf: `Aperture`
- `Norway` → **Maine** | own: `Oslo` [other] | cf: `Dale (don't laugh`
- `Oak Hill` → **West Virginia** | own: `162` [other] | cf: `Rockland County`
- `Reading` → **Pennsylvania** | own: `Cisma` [other] | cf: `English`
- `Rockland` → **Maine** | own: `Orange` [other] | cf: `Maine`
- `Macon` → **Georgia** | own: `Macon` [copy_query] | cf: `North Alabama`
- `Bakersfield` → **California** | own: `Pasadena` [other] | cf: `California`

### verb_tense_label  (counterfactual direction: word_polarity)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 7 |
| copy_query | 2 | 3 |
| copy_demo_target | 2 | 13 |
| copy_demo_input | 0 | 1 |
| in_pool_wrong | 0 | 0 |
| input_variant | 19 | 11 |
| other | 126 | 115 |

Examples (own-ablation errors):

- `ratchetted` → **past** | own: `racheting` [other] | cf: `verb2 -2--`
- `riffed` → **past** | own: `blued` [other] | cf: `cocked`
- `shapes` → **present** | own: `Ball faces, Spheres` [other] | cf: `shapes`
- `hulls` → **present** | own: `hulless` [input_variant] | cf: `hull`
- `hooked` → **past** | own: `caught` [other] | cf: `habituated`
- `valuing` → **gerund** | own: `worths` [other] | cf: `the following blank should be`
- `shines` → **present** | own: `ophire` [other] | cf: `moonlight`
- `operated` → **past** | own: `stabilised` [other] | cf: `risen`
- `skilled` → **past** | own: `art.` [other] | cf: `tailored`
- `previews` → **present** | own: `premiers` [other] | cf: `attended`

### verb_to_third_person  (counterfactual direction: next_month_of_date)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 8 | 35 |
| copy_query | 36 | 26 |
| copy_demo_target | 4 | 3 |
| copy_demo_input | 9 | 2 |
| in_pool_wrong | 0 | 4 |
| input_variant | 15 | 23 |
| other | 78 | 57 |

Examples (own-ablation errors):

- `run` → **runs** | own: `run` [copy_query] | cf: `running`
- `explain` → **explains** | own: `explain boats are what msw` [input_variant] | cf: `"boats"`
- `rain` → **rains** | own: `pieces falling off things` [other] | cf: `rainfall … often observed as surface`
- `purchase` → **purchases** | own: `purchase` [copy_query] | cf: `purchases`
- `network` → **networks** | own: `IP` [other] | cf: `networks`
- `stone` → **stones** | own: `stone` [copy_query] | cf: `as`
- `mirror` → **mirrors** | own: `mirror` [copy_query] | cf: `mirror`
- `tie` → **ties** | own: `no time limit` [other] | cf: `tie`
- `throw` → **throws** | own: `was` [other] | cf: `Throw`
- `solve` → **solves** | own: `` [other] | cf: `DSTS`

### word_polarity  (counterfactual direction: capitalize_first_letter)

| bucket | own $\hat u_A$ ablated | counterfactual ablated |
|---|---:|---:|
| correct | 1 | 34 |
| copy_query | 13 | 4 |
| copy_demo_target | 1 | 7 |
| copy_demo_input | 2 | 0 |
| in_pool_wrong | 0 | 5 |
| input_variant | 13 | 3 |
| other | 120 | 97 |

Examples (own-ablation errors):

- `harassed` → **negative** | own: `harassed` [copy_query] | cf: `needless`
- `talented` → **positive** | own: `aspiring` [other] | cf: `great contribution to society`
- `qualified` → **positive** | own: `prof.` [other] | cf: `good`
- `victorious` → **positive** | own: `victorious` [copy_query] | cf: `sad`
- `renowned` → **positive** | own: `verdant` [other] | cf: `important`
- `gloomy` → **negative** | own: `day` [other] | cf: `intense`
- `grimy` → **negative** | own: `volatile` [other] | cf: `dirty`
- `dismissive` → **negative** | own: `moody` [other] | cf: `cold`
- `friendly` → **positive** | own: `infectious` [other] | cf: `positive`
- `joyful` → **positive** | own: `bizarre` [other] | cf: `cheerful`
