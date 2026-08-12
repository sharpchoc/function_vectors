"""Shared curated data for the compound_split family of tasks
(compound_head, compound_first, compound_join; spec indices 97-99).

This module is hand-curated (LLM-curated per the task recipe) list of genuine
English CLOSED compound nouns, each stored as (part1, part2) where
part1 + part2 == the attested closed-spelled compound, and part2 is the
semantic head. All entries were chosen because they are transparent,
compositional, well-known compounds (not pseudo-compounds like "carpet" or
"understand" that happen to contain two dictionary words).

Programmatic verification (done by build_compound_pairs()):
  - compound = part1 + part2 is a single alphabetic string
  - part1, part2, and the compound are each looked up in wordfreq
    (zipf_frequency, 'en') and must clear minimum-frequency thresholds so a
    GPT-J-plausible reader would know all three forms
  - part1/part2 length >= 3
  - dedup by compound (each compound kept once)
  - drop any compound whose part2 falls in a suffix-like closed class
    that is not a genuine independent noun/verb head (ing, er, ed, ion,
    ness, ment, able, age, ...) -- guards against accidental pseudo-splits
"""

from wordfreq import zipf_frequency

# (part1, part2) curated pairs; part1+part2 must equal the real compound.
RAW_PAIRS = [
    # --- body ---
    ("tooth", "brush"), ("tooth", "pick"), ("tooth", "ache"), ("tooth", "paste"),
    ("finger", "print"), ("finger", "nail"), ("finger", "tip"),
    ("thumb", "nail"), ("thumb", "tack"),
    ("hair", "cut"), ("hair", "line"), ("hair", "brush"), ("hair", "pin"), ("hair", "style"),
    ("eye", "lash"), ("eye", "lid"), ("eye", "ball"), ("eye", "sight"), ("eye", "brow"), ("eye", "witness"),
    ("hand", "shake"), ("hand", "writing"), ("hand", "cuff"), ("hand", "bag"), ("hand", "book"),
    ("hand", "out"), ("hand", "made"), ("hand", "rail"), ("hand", "gun"), ("hand", "ball"),
    ("head", "ache"), ("head", "band"), ("head", "light"), ("head", "line"), ("head", "phone"),
    ("head", "quarter"), ("head", "master"), ("head", "way"), ("head", "gear"), ("head", "set"),
    ("heart", "beat"), ("heart", "break"), ("heart", "burn"),
    ("back", "bone"), ("back", "pack"), ("back", "ground"), ("back", "yard"), ("back", "stage"),
    ("back", "log"), ("back", "fire"), ("back", "drop"), ("back", "hand"),
    ("arm", "chair"), ("arm", "pit"), ("arm", "band"),
    ("foot", "print"), ("foot", "ball"), ("foot", "note"), ("foot", "path"), ("foot", "step"),
    ("foot", "hold"), ("foot", "hill"), ("foot", "wear"),
    ("ear", "drum"), ("ear", "ring"), ("ear", "lobe"),
    ("horse", "back"),
    ("sweet", "heart"),
    ("wave", "length"),
    # --- family with "step/grand" ---
    ("grand", "child"), ("grand", "father"), ("grand", "mother"), ("grand", "parent"),
    ("grand", "son"), ("grand", "daughter"), ("grand", "stand"),
    ("step", "mother"), ("step", "father"), ("step", "son"), ("step", "daughter"),
    ("step", "child"), ("step", "brother"), ("step", "sister"),
    # --- house / furniture / rooms ---
    ("bed", "room"), ("bed", "time"), ("bed", "spread"), ("bed", "side"), ("bed", "bug"), ("bed", "rock"),
    ("bath", "room"), ("bath", "tub"), ("bath", "robe"),
    ("class", "room"), ("class", "mate"),
    ("book", "shelf"), ("book", "store"), ("book", "case"), ("book", "mark"), ("book", "worm"),
    ("book", "end"), ("note", "book"), ("text", "book"), ("work", "book"), ("cook", "book"), ("year", "book"),
    ("door", "way"), ("door", "bell"), ("door", "step"), ("door", "knob"), ("door", "mat"),
    ("key", "board"), ("key", "hole"), ("key", "note"), ("key", "stone"), ("key", "word"),
    ("table", "cloth"), ("table", "spoon"),
    ("cup", "board"), ("cup", "cake"), ("tea", "cup"), ("tea", "pot"), ("tea", "spoon"),
    ("lamp", "shade"),
    ("wall", "paper"),
    ("floor", "board"),
    ("roof", "top"),
    ("fire", "place"), ("fire", "wood"), ("fire", "fly"), ("fire", "work"), ("fire", "arm"),
    ("fire", "cracker"), ("fire", "fighter"), ("fire", "house"), ("fire", "proof"), ("fire", "side"), ("fire", "storm"),
    ("candle", "stick"),
    ("stair", "case"), ("stair", "way"),
    ("court", "yard"), ("house", "hold"), ("house", "wife"), ("house", "work"), ("house", "boat"),
    ("ware", "house"), ("green", "house"), ("light", "house"), ("farm", "house"), ("court", "house"),
    ("dog", "house"), ("bird", "house"), ("barn", "yard"), ("farm", "yard"), ("farm", "land"),
    # --- nature / weather ---
    ("sun", "flower"), ("sun", "light"), ("sun", "rise"), ("sun", "set"), ("sun", "shine"),
    ("sun", "burn"), ("sun", "tan"), ("sun", "beam"), ("sun", "glasses"),
    ("moon", "light"), ("moon", "shine"), ("moon", "beam"),
    ("star", "fish"), ("star", "light"), ("star", "dust"),
    ("rain", "bow"), ("rain", "coat"), ("rain", "drop"), ("rain", "fall"), ("rain", "storm"), ("rain", "water"),
    ("snow", "flake"), ("snow", "man"), ("snow", "ball"), ("snow", "fall"), ("snow", "storm"), ("snow", "plow"),
    ("wind", "mill"), ("wind", "pipe"), ("wind", "shield"),
    ("thunder", "storm"), ("thunder", "bolt"), ("thunder", "clap"),
    ("water", "fall"), ("water", "mark"), ("water", "melon"), ("water", "proof"), ("water", "way"),
    ("water", "shed"), ("water", "front"), ("water", "color"), ("water", "line"),
    ("earth", "quake"), ("earth", "worm"),
    ("land", "slide"), ("land", "mark"), ("land", "lord"), ("land", "owner"), ("land", "fill"),
    ("sea", "shore"), ("sea", "shell"), ("sea", "side"), ("sea", "weed"), ("sea", "food"),
    ("sea", "horse"), ("sea", "port"), ("sea", "sick"),
    ("river", "bank"), ("river", "side"),
    ("lake", "side"),
    ("mountain", "side"), ("hill", "side"), ("hill", "top"),
    ("tree", "top"),
    ("wood", "land"), ("wood", "cutter"), ("wood", "pecker"), ("wood", "work"), ("wood", "shed"),
    ("drift", "wood"), ("red", "wood"),
    ("wild", "flower"), ("wild", "life"), ("wild", "cat"),
    ("sand", "storm"), ("sand", "paper"), ("sand", "box"), ("sand", "castle"),
    ("north", "east"), ("north", "west"), ("south", "east"), ("south", "west"),
    # --- fruit / plants / food ---
    ("straw", "berry"), ("blue", "berry"), ("black", "berry"), ("goose", "berry"),
    ("pine", "apple"), ("grape", "fruit"),
    ("pea", "nut"), ("pea", "cock"),
    ("butter", "milk"), ("butter", "cup"), ("butter", "fly"),
    ("pop", "corn"), ("pan", "cake"), ("short", "cake"), ("cup", "cake"), ("milk", "shake"),
    ("egg", "plant"), ("egg", "shell"),
    ("corn", "bread"), ("corn", "field"), ("corn", "meal"),
    ("break", "fast"), ("lunch", "box"),
    ("honey", "bee"),
    # --- animals ---
    ("lady", "bug"), ("fire", "fly"), ("horse", "fly"), ("horse", "shoe"),
    ("cow", "boy"), ("cow", "girl"), ("cow", "hide"),
    ("sheep", "dog"), ("watch", "dog"), ("bull", "dog"), ("bull", "frog"),
    ("jelly", "fish"), ("cat", "fish"), ("sword", "fish"), ("gold", "fish"), ("rattle", "snake"),
    ("race", "horse"), ("work", "horse"),
    ("grass", "hopper"),
    ("spider", "web"),
    ("pig", "tail"), ("pony", "tail"), ("dog", "sled"), ("bird", "bath"), ("bird", "cage"),
    # --- time ---
    ("week", "end"), ("week", "day"),
    ("day", "light"), ("day", "time"), ("day", "dream"), ("day", "break"),
    ("night", "time"), ("night", "mare"), ("night", "fall"), ("night", "club"),
    ("mid", "night"), ("mid", "day"), ("mid", "way"), ("mid", "summer"), ("mid", "term"),
    ("after", "noon"), ("after", "thought"),
    ("year", "book"),
    ("time", "table"), ("life", "time"),
    ("birth", "day"),
    ("over", "time"), ("over", "night"),
    ("war", "time"),
    # --- clothes ---
    ("over", "coat"), ("night", "gown"), ("under", "wear"), ("under", "shirt"), ("under", "pants"),
    ("sweat", "shirt"), ("sweat", "pants"), ("sweat", "band"), ("night", "shirt"),
    ("wrist", "watch"), ("neck", "tie"), ("neck", "lace"), ("head", "band"),
    # --- vehicles / transport ---
    ("air", "plane"), ("air", "port"), ("air", "craft"), ("air", "line"), ("air", "way"), ("air", "field"),
    ("space", "ship"), ("space", "craft"), ("space", "walk"), ("space", "man"),
    ("sail", "boat"), ("row", "boat"), ("speed", "boat"), ("tug", "boat"), ("house", "boat"),
    ("steam", "boat"), ("steam", "ship"), ("war", "ship"), ("battle", "ship"), ("ship", "yard"), ("ship", "wreck"),
    ("motor", "cycle"), ("rail", "road"), ("rail", "way"), ("high", "way"), ("free", "way"),
    ("drive", "way"), ("run", "way"), ("side", "walk"), ("cross", "walk"), ("board", "walk"),
    ("road", "way"), ("road", "side"), ("speed", "way"),
    # --- tools / work ---
    ("screw", "driver"), ("black", "smith"), ("gold", "smith"), ("lock", "smith"),
    ("work", "shop"), ("work", "room"), ("work", "place"), ("work", "out"), ("work", "force"),
    ("work", "load"), ("work", "sheet"), ("work", "day"), ("work", "horse"), ("work", "man"),
    ("tool", "box"), ("tool", "kit"), ("brief", "case"), ("suit", "case"), ("back", "pack"),
    ("pass", "word"), ("pass", "port"),
    ("type", "writer"), ("song", "writer"),
    ("sand", "paper"), ("soft", "ware"),
    # --- occupations with -man/-woman ---
    ("police", "man"), ("fire", "man"), ("mail", "man"), ("post", "man"), ("sales", "man"),
    ("chair", "man"), ("door", "man"), ("watch", "man"), ("sports", "man"), ("business", "man"),
    ("shoe", "maker"),
    # --- school / play ---
    ("home", "work"), ("school", "yard"), ("school", "house"),
    ("team", "mate"), ("room", "mate"), ("play", "mate"), ("play", "ground"), ("play", "house"),
    ("play", "time"),
    # --- sports ---
    ("base", "ball"), ("basket", "ball"), ("volley", "ball"), ("soft", "ball"), ("snow", "ball"),
    ("touch", "down"), ("score", "board"), ("quarter", "back"), ("side", "line"),
    ("out", "field"), ("in", "field"), ("grand", "stand"),
    # --- home & building ---
    ("sky", "scraper"), ("sky", "line"), ("sky", "light"), ("sky", "dive"),
    ("court", "yard"), ("light", "house"),
    # --- colors / objects ---
    ("black", "board"), ("black", "bird"), ("black", "out"), ("black", "berry"),
    ("white", "wash"), ("white", "board"),
    ("red", "wood"), ("silver", "ware"),
    # --- direction / position ---
    ("up", "stairs"), ("down", "stairs"), ("up", "town"), ("down", "town"),
    ("in", "side"), ("out", "side"), ("up", "hill"), ("down", "hill"),
    ("under", "ground"), ("under", "line"), ("under", "water"), ("under", "dog"), ("under", "pass"),
    ("over", "board"), ("over", "head"), ("over", "seas"), ("over", "view"),
    ("up", "load"), ("out", "put"), ("out", "line"), ("out", "break"), ("out", "cry"),
    ("out", "doors"), ("out", "post"), ("out", "skirts"), ("set", "up"),
    # --- other common ---
    ("news", "paper"), ("news", "stand"),
    ("gun", "powder"), ("gun", "shot"), ("gun", "man"),
    ("arrow", "head"),
    ("drum", "stick"), ("drum", "beat"),
    ("cross", "word"), ("cross", "fire"), ("cross", "road"),
    ("down", "fall"), ("down", "pour"), ("draw", "back"), ("fall", "out"), ("feed", "back"),
    ("flash", "light"),
    ("home", "land"), ("home", "made"), ("home", "sick"), ("home", "town"),
    ("life", "boat"), ("life", "line"), ("life", "span"), ("life", "guard"), ("life", "style"), ("life", "long"),
    ("trade", "mark"),
    ("wheel", "chair"),
    ("wild", "flower"),
    ("note", "worthy"),
    ("tail", "light"),
    ("spot", "light"),
    # --- birds / fish / insects extra ---
    ("song", "bird"), ("sea", "bird"), ("blue", "bird"), ("lady", "bird"),
    ("mocking", "bird"), ("humming", "bird"),
    ("dog", "fish"), ("sun", "fish"), ("silver", "fish"),
    ("dragon", "fly"), ("may", "fly"),
    ("ground", "hog"),
    # --- plants / trees extra ---
    ("dog", "wood"), ("pine", "wood"), ("rose", "wood"), ("sandal", "wood"), ("hard", "wood"),
    ("blue", "bell"), ("snow", "drop"), ("fox", "glove"),
    ("hay", "stack"),
    ("silk", "worm"), ("tape", "worm"), ("ring", "worm"),
    ("hazel", "nut"), ("dough", "nut"),
    ("flower", "pot"), ("jack", "pot"), ("crack", "pot"),
    ("fire", "ball"), ("meat", "ball"), ("odd", "ball"), ("screw", "ball"), ("hair", "ball"),
    ("pin", "ball"), ("spit", "ball"),
    # --- light / sky ---
    ("flood", "light"), ("search", "light"), ("night", "light"), ("street", "light"),
    ("gas", "light"), ("candle", "light"), ("lime", "light"),
    # --- weather extra ---
    ("hail", "storm"), ("wind", "storm"), ("snow", "drift"),
    # --- containers / household extra ---
    ("match", "box"), ("mail", "box"), ("shoe", "box"), ("in", "box"), ("out", "box"),
    ("pill", "box"), ("ice", "box"),
    ("dish", "washer"), ("waste", "basket"), ("trash", "can"), ("ash", "tray"),
    ("night", "stand"), ("foot", "stool"), ("card", "board"), ("surf", "board"),
    ("skate", "board"), ("spring", "board"), ("snow", "board"), ("bill", "board"),
    ("switch", "board"), ("head", "board"),
    # --- tools / metal extra ---
    ("crow", "bar"), ("sledge", "hammer"), ("jack", "hammer"), ("wheel", "barrow"),
    ("pitch", "fork"), ("clothes", "pin"), ("paper", "clip"), ("yard", "stick"),
    ("broom", "stick"), ("match", "stick"), ("lip", "stick"), ("chop", "stick"),
    # --- clothing extra ---
    ("wrist", "band"), ("head", "scarf"), ("ear", "muff"), ("loin", "cloth"), ("waist", "band"),
    # --- transport extra ---
    ("sea", "plane"), ("war", "plane"), ("hover", "craft"), ("water", "craft"),
    ("hand", "cart"), ("push", "cart"), ("street", "car"), ("box", "car"), ("race", "car"),
    ("side", "car"), ("taxi", "cab"),
    # --- buildings / places extra ---
    ("club", "house"), ("tree", "house"), ("store", "house"), ("boat", "house"),
    ("jail", "house"), ("out", "house"), ("power", "house"), ("guest", "house"), ("bath", "house"),
    # --- workplace extra ---
    ("sweat", "shop"), ("book", "shop"), ("work", "space"),
    # --- sports extra ---
    ("goal", "keeper"), ("lines", "man"), ("wing", "man"), ("team", "work"), ("ball", "park"),
    # --- water / sea extra ---
    ("tide", "water"), ("flood", "water"), ("fresh", "water"), ("salt", "water"), ("waste", "water"),
    # --- body extra ---
    ("cheek", "bone"), ("jaw", "bone"), ("wish", "bone"), ("breast", "bone"), ("hip", "bone"),
    ("knee", "cap"), ("skull", "cap"),
    ("ear", "ache"), ("heart", "ache"), ("stomach", "ache"), ("back", "ache"), ("belly", "ache"),
    # --- color extra ---
    ("black", "list"),
    # --- direction extra ---
    ("in", "land"), ("in", "bound"), ("out", "bound"), ("off", "shore"), ("on", "shore"),
    ("in", "doors"),
    # --- writing / lines extra ---
    ("by", "line"), ("dead", "line"), ("guide", "line"), ("time", "line"), ("shore", "line"),
    ("border", "line"), ("hot", "line"), ("main", "line"),
    # --- military extra ---
    ("war", "head"), ("cease", "fire"), ("war", "path"), ("gun", "fire"), ("gun", "boat"),
    # --- music extra ---
    ("sound", "track"), ("song", "book"), ("band", "stand"),
    # --- air / mainland extra ---
    ("air", "tight"), ("air", "space"), ("air", "bag"), ("air", "base"), ("air", "flow"), ("air", "fare"),
    ("main", "land"), ("main", "stream"), ("main", "frame"),
    ("grass", "land"), ("wet", "land"),
    ("life", "blood"), ("blood", "hound"), ("blood", "stream"), ("blood", "shed"),
    ("nose", "bleed"), ("heat", "wave"),
    ("moon", "walk"), ("cat", "walk"),
    ("pit", "fall"), ("short", "fall"), ("free", "fall"), ("land", "fall"),
    # --- ground / yard / field / mill / stone extra ---
    ("fair", "ground"), ("battle", "ground"), ("camp", "ground"), ("fore", "ground"),
    ("grave", "yard"), ("vine", "yard"), ("junk", "yard"), ("scrap", "yard"),
    ("battle", "field"), ("mine", "field"),
    ("saw", "mill"), ("tread", "mill"),
    ("mile", "stone"), ("lime", "stone"), ("corner", "stone"), ("gem", "stone"), ("grave", "stone"),
    ("birth", "stone"), ("tomb", "stone"), ("flag", "stone"), ("sand", "stone"), ("brown", "stone"),
    ("cap", "stone"),
    # --- -man extra ---
    ("milk", "man"), ("weather", "man"), ("fresh", "man"), ("crafts", "man"), ("states", "man"),
    ("handy", "man"), ("fore", "man"), ("horse", "man"),
    # --- -work extra ---
    ("art", "work"), ("net", "work"), ("frame", "work"), ("patch", "work"), ("guess", "work"),
    ("course", "work"), ("ground", "work"), ("body", "work"),
    # --- -time extra ---
    ("meal", "time"), ("tea", "time"), ("half", "time"), ("rag", "time"), ("spring", "time"),
    ("summer", "time"), ("winter", "time"), ("peace", "time"),
    # --- -place extra ---
    ("birth", "place"), ("market", "place"),
    # --- -out extra ---
    ("drop", "out"), ("check", "out"), ("lay", "out"), ("print", "out"), ("burn", "out"),
    ("time", "out"), ("walk", "out"), ("break", "out"), ("look", "out"), ("hide", "out"),
    ("shoot", "out"), ("knock", "out"), ("stand", "out"), ("blow", "out"), ("cook", "out"),
    # --- watchtower / -hold ---
    ("watch", "tower"), ("strong", "hold"),
    # --- weapons/military extra ---
    ("cross", "bow"), ("long", "bow"), ("gun", "smith"), ("swords", "man"), ("marks", "man"),
    # --- -mine / -sand / -stone extra ---
    ("oil", "field"), ("mid", "field"), ("gold", "mine"), ("coal", "mine"), ("land", "mine"),
    ("quick", "sand"), ("mill", "stone"), ("cobble", "stone"), ("hail", "stone"), ("moon", "stone"),
    # --- kitchen / -ware extra ---
    ("blow", "torch"), ("cook", "ware"), ("table", "ware"), ("stone", "ware"), ("hard", "ware"),
    ("kitchen", "ware"), ("glass", "ware"), ("outer", "wear"), ("sports", "wear"), ("swim", "wear"),
    # --- food extra ---
    ("cock", "tail"), ("fish", "tail"), ("ox", "tail"),
    ("oat", "meal"), ("corn", "starch"), ("apple", "sauce"),
    ("cheese", "cake"), ("fruit", "cake"), ("beef", "steak"),
    ("ginger", "bread"), ("short", "bread"), ("corn", "flake"),
    ("pepper", "mint"), ("spear", "mint"), ("grape", "vine"),
    # --- animal extra ---
    ("cat", "nap"), ("copy", "cat"), ("lap", "dog"), ("hot", "dog"), ("corn", "dog"),
    ("sheep", "skin"), ("pig", "skin"), ("snake", "skin"), ("buck", "skin"),
    ("horse", "power"), ("war", "horse"), ("house", "fly"), ("wood", "worm"), ("glow", "worm"),
    # --- head compounds extra ---
    ("red", "head"), ("bone", "head"), ("hot", "head"), ("egg", "head"), ("air", "head"),
    ("bulk", "head"), ("fore", "head"), ("spear", "head"), ("letter", "head"), ("figure", "head"),
    # --- cap extra ---
    ("night", "cap"), ("hub", "cap"), ("ice", "cap"),
    # --- power extra ---
    ("man", "power"), ("will", "power"), ("fire", "power"), ("super", "power"),
    # --- hole extra ---
    ("peep", "hole"), ("pot", "hole"), ("man", "hole"), ("loop", "hole"), ("fox", "hole"),
    ("button", "hole"), ("blow", "hole"), ("pin", "hole"), ("worm", "hole"),
    # --- weather / wind extra ---
    ("snow", "suit"), ("rain", "check"), ("rain", "forest"),
    ("cross", "wind"), ("head", "wind"), ("tail", "wind"), ("whirl", "wind"), ("down", "wind"),
    # --- top extra ---
    ("mountain", "top"), ("table", "top"), ("desk", "top"), ("lap", "top"),
    # --- computer / office extra ---
    ("data", "base"), ("web", "page"), ("web", "site"), ("home", "page"), ("user", "name"),
    ("screen", "shot"), ("down", "load"), ("fire", "wall"), ("band", "width"), ("broad", "band"),
    ("tool", "bar"), ("side", "bar"), ("cross", "bar"),
    # --- coat extra ---
    ("waist", "coat"), ("top", "coat"),
    # --- cufflink / bootstrap / outback / bootcamp / campfire / campsite ---
    ("cuff", "link"), ("boot", "strap"), ("out", "back"), ("boot", "camp"),
    ("camp", "fire"), ("camp", "site"),
    # --- land extra ---
    ("marsh", "land"), ("high", "land"), ("low", "land"),
    # --- work-material extra ---
    ("brick", "work"), ("stone", "work"), ("iron", "work"),
    # --- -off extra ---
    ("kick", "off"), ("play", "off"), ("stand", "off"), ("take", "off"), ("lay", "off"),
    ("cut", "off"), ("face", "off"),
    # --- -down extra ---
    ("show", "down"), ("break", "down"), ("melt", "down"), ("shut", "down"), ("count", "down"),
    ("let", "down"), ("crack", "down"),
    # --- -back extra ---
    ("come", "back"), ("set", "back"), ("kick", "back"), ("throw", "back"), ("pay", "back"),
    ("flash", "back"), ("call", "back"), ("hump", "back"), ("hunch", "back"), ("green", "back"),
    ("full", "back"), ("half", "back"),
    # --- -pad extra ---
    ("key", "pad"), ("touch", "pad"), ("mouse", "pad"),
    # --- bone extra ---
    ("shin", "bone"), ("collar", "bone"), ("thigh", "bone"),
    # --- gunpoint/gunfight extra ---
    ("gun", "point"), ("gun", "fight"),
    # --- front extra ---
    ("battle", "front"), ("front", "line"), ("home", "front"), ("store", "front"), ("lake", "front"),
    # --- cross extra ---
    ("cross", "check"), ("cross", "cut"),
    # --- bound extra ---
    ("north", "bound"), ("south", "bound"), ("east", "bound"), ("west", "bound"), ("home", "bound"),
    # --- water extra ---
    ("back", "water"), ("sea", "water"), ("ground", "water"),
    # --- board extra ---
    ("clip", "board"), ("dash", "board"), ("wash", "board"), ("chess", "board"), ("checker", "board"),
    # --- point / pit extra ---
    ("pin", "point"), ("view", "point"), ("stand", "point"), ("check", "point"), ("mid", "point"),
    ("end", "point"), ("cock", "pit"), ("fire", "pit"), ("wind", "fall"),
    # --- post extra ---
    ("lamp", "post"), ("sign", "post"),
    # --- clothes / shoe extra ---
    ("house", "coat"), ("snow", "shoe"),
    # --- room extra ---
    ("play", "room"), ("store", "room"), ("stock", "room"), ("show", "room"), ("rest", "room"),
    ("dark", "room"), ("court", "room"), ("ball", "room"),
    # --- mate extra ---
    ("ship", "mate"), ("house", "mate"), ("soul", "mate"), ("check", "mate"),
    # --- sports/game extra ---
    ("back", "stroke"), ("breast", "stroke"), ("ball", "game"), ("end", "game"),
    # --- cloud/burst extra ---
    ("rain", "cloud"), ("thunder", "cloud"), ("cloud", "burst"), ("sun", "burst"), ("out", "burst"),
    ("mid", "air"),
    # --- line extra ---
    ("punch", "line"), ("waist", "line"), ("stream", "line"), ("coast", "line"), ("pipe", "line"),
    ("clothes", "line"),
    # --- misc small tools ---
    ("nut", "cracker"), ("pick", "pocket"),
    # --- piece extra ---
    ("mouth", "piece"), ("center", "piece"), ("master", "piece"), ("time", "piece"),
    ("ear", "piece"), ("head", "piece"), ("show", "piece"),
    # --- cast extra ---
    ("broad", "cast"), ("fore", "cast"), ("over", "cast"),
    # --- way extra ---
    ("hall", "way"), ("gate", "way"), ("path", "way"), ("walk", "way"), ("cause", "way"),
    ("by", "way"), ("park", "way"),
    # --- print extra ---
    ("blue", "print"), ("news", "print"),
    # --- shell/fish extra ---
    ("bomb", "shell"), ("shell", "fish"),
    # --- land extra ---
    ("head", "land"), ("heart", "land"),
    # --- ball extra ---
    ("paint", "ball"), ("moth", "ball"), ("cannon", "ball"), ("curve", "ball"),
    # --- shed / watch / man / wheel extra ---
    ("tool", "shed"), ("stop", "watch"), ("straw", "man"), ("scare", "crow"),
    ("pin", "wheel"), ("cart", "wheel"), ("fly", "wheel"), ("gear", "box"),
    # --- football extra ---
    ("corner", "back"), ("line", "backer"),
    # --- track / stream / grade / coach / work extra ---
    ("race", "track"), ("back", "track"), ("down", "stream"), ("up", "stream"),
    ("down", "grade"), ("stage", "coach"), ("clock", "work"),
    # --- hand extra ---
    ("fore", "hand"), ("short", "hand"), ("long", "hand"), ("off", "hand"),
    ("second", "hand"), ("under", "hand"),
    # --- foot/tooth extra ---
    ("bare", "foot"), ("saw", "tooth"),
    # --- under/over extra ---
    ("under", "tone"), ("under", "arm"), ("under", "tow"),
    ("over", "tone"), ("over", "hang"), ("over", "pass"), ("over", "flow"),
    ("over", "dose"), ("over", "load"), ("over", "weight"),
    # --- hound/bug/fly extra ---
    ("fox", "hound"), ("grey", "hound"), ("june", "bug"), ("fire", "bug"),
    ("litter", "bug"), ("shutter", "bug"),
    # --- misc food/nature extra ---
    ("frost", "bite"), ("bread", "box"), ("hot", "pot"), ("pen", "knife"), ("jack", "knife"),
    ("pepper", "corn"), ("crab", "grass"), ("rag", "weed"), ("milk", "weed"), ("tumble", "weed"),
    ("may", "flower"), ("clover", "leaf"),
    # --- side extra ---
    ("back", "side"), ("under", "side"), ("top", "side"), ("blind", "side"), ("curb", "side"),
    ("country", "side"), ("pool", "side"), ("ring", "side"), ("way", "side"), ("off", "side"),
    # --- away extra ---
    ("get", "away"), ("run", "away"), ("cast", "away"), ("stow", "away"), ("break", "away"),
    ("give", "away"), ("take", "away"), ("lay", "away"),
    # --- case extra ---
    ("pillow", "case"), ("show", "case"), ("lower", "case"), ("upper", "case"),
    # --- glass extra ---
    ("hour", "glass"), ("eye", "glass"), ("spy", "glass"),
]

# closed suffix-like class that must never appear as a curated part2
# (defensive; none of RAW_PAIRS should hit this, but keep as a filter)
SUFFIX_LIKE = {"ing", "er", "ed", "ion", "ness", "ment", "able", "age", "ally", "ance", "self"}

MIN_PART_ZIPF = 2.8
MIN_COMPOUND_ZIPF = 1.4


def build_compound_pairs():
    """Verify RAW_PAIRS and return a deduplicated, frequency-filtered list
    of (compound, part1, part2) triples."""
    seen_compounds = set()
    seen_part1_only = {}  # part1 -> set of part2s used (to allow reuse check)
    triples = []
    for part1, part2 in RAW_PAIRS:
        compound = part1 + part2
        if not compound.isalpha():
            continue
        if compound in seen_compounds:
            continue
        if len(part1) < 3 or len(part2) < 3:
            continue
        if part2 in SUFFIX_LIKE:
            continue
        if zipf_frequency(part1, "en") < MIN_PART_ZIPF:
            continue
        if zipf_frequency(part2, "en") < MIN_PART_ZIPF:
            continue
        if zipf_frequency(compound, "en") < MIN_COMPOUND_ZIPF:
            continue
        seen_compounds.add(compound)
        triples.append((compound, part1, part2))
    return triples


if __name__ == "__main__":
    triples = build_compound_pairs()
    print(f"Raw pairs: {len(RAW_PAIRS)}")
    print(f"Verified unique triples: {len(triples)}")
    dropped = len(RAW_PAIRS) - len(triples)
    print(f"Dropped: {dropped}")
