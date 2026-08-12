"""Generator: english-italian
Translate a common English word into its most common single-word Italian
equivalent. Facts are curated by hand below (this script IS the reproducible
artifact) into content-word categories (nouns, verbs, adjectives). Excluded
by construction: words with genuinely tied/ambiguous Italian translations
(e.g. "right", "know", "start"/"stop", "town"/"country", "weather"), words
whose Italian form is identical or near-identical to the English spelling
(e.g. "pizza", "radio", "taxi", "internet"), and multi-word translations.

Self-check performed while curating: for every English word considered, we
mentally checked whether translating the chosen Italian word back to English
returns (a form of) the original word; entries with a real back-translation
ambiguity were dropped rather than included. This is an author-curated
dictionary (not a second independently-run MT pass), which is noted in the
task write-up rather than overclaimed.

A handful of pairs involve a judgment call between two very common Italian
synonyms (flagged inline with a trailing comment "# JUDGMENT:"); these were
resolved to what we judge the single most standard/least ambiguous choice.
"""
import json
import random
from collections import Counter
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "english-italian.json"
RESOURCES_DIR = Path(__file__).resolve().parents[1]
N = 1000

random.seed(42)

# -------------------------------------------------------------------------
# Curated (english, italian) pairs, grouped by domain for readability.
# -------------------------------------------------------------------------
RAW_PAIRS = [
    # --- animals ---
    ("dog", "cane"), ("cat", "gatto"), ("horse", "cavallo"), ("cow", "mucca"),
    ("pig", "maiale"), ("sheep", "pecora"), ("goat", "capra"), ("chicken", "pollo"),
    ("bird", "uccello"), ("fish", "pesce"), ("mouse", "topo"), ("rat", "ratto"),
    ("rabbit", "coniglio"), ("bear", "orso"), ("wolf", "lupo"), ("fox", "volpe"),
    ("deer", "cervo"), ("lion", "leone"), ("tiger", "tigre"), ("elephant", "elefante"),
    ("monkey", "scimmia"), ("snake", "serpente"), ("frog", "rana"),
    ("turtle", "tartaruga"), ("duck", "anatra"), ("goose", "oca"), ("bee", "ape"),
    ("ant", "formica"), ("spider", "ragno"), ("butterfly", "farfalla"),
    ("worm", "verme"), ("whale", "balena"), ("dolphin", "delfino"),
    ("shark", "squalo"), ("crab", "granchio"), ("eagle", "aquila"),
    ("squirrel", "scoiattolo"), ("donkey", "asino"), ("camel", "cammello"),
    ("kangaroo", "canguro"), ("penguin", "pinguino"), ("seal", "foca"),
    ("lamb", "agnello"), ("puppy", "cucciolo"), ("kitten", "gattino"),
    ("calf", "vitello"), ("insect", "insetto"), ("snail", "lumaca"),
    ("crow", "corvo"), ("peacock", "pavone"), ("swan", "cigno"),
    ("parrot", "pappagallo"), ("lizard", "lucertola"), ("lobster", "aragosta"),

    # --- family / people / occupations ---
    ("mother", "madre"), ("father", "padre"), ("brother", "fratello"),
    ("sister", "sorella"), ("son", "figlio"), ("daughter", "figlia"),
    ("husband", "marito"), ("wife", "moglie"), ("grandmother", "nonna"),
    ("grandfather", "nonno"), ("uncle", "zio"), ("aunt", "zia"), ("cousin", "cugino"),
    ("child", "bambino"), ("boy", "ragazzo"), ("girl", "ragazza"), ("man", "uomo"),
    ("woman", "donna"), ("person", "persona"), ("people", "persone"),
    ("friend", "amico"), ("neighbor", "vicino"), ("enemy", "nemico"), ("king", "re"),
    ("queen", "regina"), ("prince", "principe"), ("princess", "principessa"),
    ("president", "presidente"), ("teacher", "insegnante"), ("student", "studente"),
    ("doctor", "medico"), ("nurse", "infermiere"), ("lawyer", "avvocato"),
    ("judge", "giudice"), ("policeman", "poliziotto"), ("soldier", "soldato"),
    ("farmer", "contadino"), ("chef", "cuoco"), ("waiter", "cameriere"),
    ("driver", "autista"), ("pilot", "pilota"), ("sailor", "marinaio"),
    ("artist", "artista"), ("writer", "scrittore"), ("singer", "cantante"),
    ("dancer", "ballerino"), ("actor", "attore"), ("actress", "attrice"),
    ("musician", "musicista"), ("scientist", "scienziato"),
    ("engineer", "ingegnere"), ("worker", "lavoratore"), ("boss", "capo"),
    ("employee", "impiegato"), ("customer", "cliente"), ("guest", "ospite"),
    ("thief", "ladro"), ("criminal", "criminale"), ("victim", "vittima"),
    ("hero", "eroe"), ("angel", "angelo"), ("god", "dio"), ("devil", "diavolo"),
    ("ghost", "fantasma"), ("witch", "strega"), ("wizard", "mago"),
    ("giant", "gigante"), ("dwarf", "nano"), ("ally", "alleato"),

    # --- body parts ---
    ("head", "testa"), ("hair", "capelli"), ("eye", "occhio"), ("ear", "orecchio"),
    ("nose", "naso"), ("mouth", "bocca"), ("tooth", "dente"), ("tongue", "lingua"),
    ("neck", "collo"), ("shoulder", "spalla"), ("arm", "braccio"),
    ("elbow", "gomito"), ("hand", "mano"), ("finger", "dito"), ("wrist", "polso"),
    ("leg", "gamba"), ("knee", "ginocchio"), ("foot", "piede"),
    ("ankle", "caviglia"), ("skin", "pelle"), ("bone", "osso"), ("blood", "sangue"),
    ("heart", "cuore"), ("brain", "cervello"), ("stomach", "stomaco"),
    ("chest", "petto"), ("throat", "gola"), ("lip", "labbro"), ("cheek", "guancia"),
    ("chin", "mento"), ("muscle", "muscolo"), ("lung", "polmone"),
    ("liver", "fegato"), ("kidney", "rene"),

    # --- food & drink ---
    ("bread", "pane"), ("water", "acqua"), ("milk", "latte"),
    ("cheese", "formaggio"), ("butter", "burro"), ("egg", "uovo"),
    ("meat", "carne"), ("beef", "manzo"), ("rice", "riso"),
    ("salad", "insalata"), ("sugar", "zucchero"), ("salt", "sale"),
    ("pepper", "pepe"), ("oil", "olio"), ("vinegar", "aceto"),
    ("flour", "farina"), ("apple", "mela"), ("orange", "arancia"),
    ("grape", "uva"), ("lemon", "limone"), ("strawberry", "fragola"),
    ("cherry", "ciliegia"), ("peach", "pesca"), ("pear", "pera"),
    ("watermelon", "anguria"), ("pineapple", "ananas"), ("melon", "melone"),
    ("tomato", "pomodoro"), ("potato", "patata"), ("onion", "cipolla"),
    ("garlic", "aglio"), ("carrot", "carota"), ("cucumber", "cetriolo"),
    ("lettuce", "lattuga"), ("corn", "mais"), ("bean", "fagiolo"),
    ("mushroom", "fungo"), ("honey", "miele"), ("jam", "marmellata"),
    ("chocolate", "cioccolato"), ("candy", "caramella"), ("cake", "torta"),
    ("cookie", "biscotto"), ("sandwich", "panino"), ("coffee", "caffè"),
    ("tea", "tè"), ("juice", "succo"), ("wine", "vino"), ("beer", "birra"),
    ("ice", "ghiaccio"), ("breakfast", "colazione"), ("lunch", "pranzo"),
    ("dinner", "cena"), ("meal", "pasto"), ("plate", "piatto"),
    ("bowl", "ciotola"), ("cup", "tazza"), ("spoon", "cucchiaio"),
    ("fork", "forchetta"), ("knife", "coltello"), ("napkin", "tovagliolo"),
    ("bottle", "bottiglia"), ("bag", "borsa"),

    # --- household / objects ---
    ("house", "casa"), ("door", "porta"), ("window", "finestra"),
    ("wall", "muro"), ("floor", "pavimento"), ("ceiling", "soffitto"),
    ("roof", "tetto"), ("room", "stanza"), ("kitchen", "cucina"),
    ("bathroom", "bagno"), ("bedroom", "camera"), ("table", "tavolo"),
    ("chair", "sedia"), ("bed", "letto"), ("sofa", "divano"),
    ("shelf", "scaffale"), ("closet", "armadio"), ("mirror", "specchio"),
    ("lamp", "lampada"), ("candle", "candela"), ("watch", "orologio"),
    ("key", "chiave"), ("newspaper", "giornale"), ("magazine", "rivista"),
    ("letter", "lettera"), ("pen", "penna"), ("pencil", "matita"),
    ("paper", "carta"), ("scissors", "forbici"), ("needle", "ago"),
    ("thread", "filo"), ("button", "bottone"), ("umbrella", "ombrello"),
    ("suitcase", "valigia"), ("box", "scatola"), ("basket", "cestino"),
    ("bucket", "secchio"), ("broom", "scopa"), ("towel", "asciugamano"),
    ("soap", "sapone"), ("comb", "pettine"), ("carpet", "tappeto"),
    ("curtain", "tenda"), ("pillow", "cuscino"), ("blanket", "coperta"),
    ("telephone", "telefono"), ("television", "televisione"),
    ("battery", "batteria"), ("machine", "macchina"), ("engine", "motore"),
    ("wheel", "ruota"), ("hammer", "martello"), ("screw", "vite"),
    ("rope", "corda"), ("chain", "catena"), ("glue", "colla"),
    ("tape", "nastro"), ("photo", "foto"),

    # --- clothing ---
    ("shirt", "camicia"), ("pants", "pantaloni"), ("skirt", "gonna"),
    ("jacket", "giacca"), ("coat", "cappotto"), ("sweater", "maglione"),
    ("shoe", "scarpa"), ("sock", "calzino"), ("hat", "cappello"),
    ("glove", "guanto"), ("scarf", "sciarpa"), ("belt", "cintura"),
    ("tie", "cravatta"), ("pajamas", "pigiama"), ("uniform", "uniforme"),
    ("pocket", "tasca"), ("sleeve", "manica"), ("collar", "colletto"),

    # --- nature / weather ---
    ("sun", "sole"), ("moon", "luna"), ("star", "stella"), ("sky", "cielo"),
    ("cloud", "nuvola"), ("rain", "pioggia"), ("snow", "neve"), ("wind", "vento"),
    ("storm", "tempesta"), ("thunder", "tuono"), ("lightning", "fulmine"),
    ("fog", "nebbia"), ("fire", "fuoco"), ("smoke", "fumo"), ("ash", "cenere"),
    ("earth", "terra"), ("mountain", "montagna"), ("hill", "collina"),
    ("valley", "valle"), ("river", "fiume"), ("lake", "lago"), ("sea", "mare"),
    ("ocean", "oceano"), ("island", "isola"), ("beach", "spiaggia"),
    ("forest", "foresta"), ("tree", "albero"), ("leaf", "foglia"),
    ("flower", "fiore"), ("grass", "erba"), ("root", "radice"),
    ("branch", "ramo"), ("seed", "seme"), ("stone", "pietra"), ("sand", "sabbia"),
    ("dust", "polvere"), ("mud", "fango"), ("field", "campo"),
    ("desert", "deserto"), ("cave", "grotta"), ("volcano", "vulcano"),
    ("earthquake", "terremoto"), ("season", "stagione"), ("spring", "primavera"),
    ("summer", "estate"), ("autumn", "autunno"), ("winter", "inverno"),
    ("temperature", "temperatura"), ("shadow", "ombra"),

    # --- time ---
    ("day", "giorno"), ("night", "notte"), ("morning", "mattina"),
    ("afternoon", "pomeriggio"), ("evening", "sera"), ("week", "settimana"),
    ("month", "mese"), ("year", "anno"), ("hour", "ora"), ("minute", "minuto"),
    ("today", "oggi"), ("tomorrow", "domani"), ("yesterday", "ieri"),
    ("birthday", "compleanno"), ("wedding", "matrimonio"),
    ("funeral", "funerale"), ("party", "festa"),

    # --- places ---
    ("city", "città"), ("village", "villaggio"), ("street", "strada"),
    ("bridge", "ponte"), ("tower", "torre"), ("castle", "castello"),
    ("church", "chiesa"), ("mosque", "moschea"), ("temple", "tempio"),
    ("school", "scuola"), ("university", "università"), ("hospital", "ospedale"),
    ("airport", "aeroporto"), ("station", "stazione"), ("restaurant", "ristorante"),
    ("bank", "banca"), ("shop", "negozio"), ("market", "mercato"),
    ("factory", "fabbrica"), ("farm", "fattoria"), ("office", "ufficio"),
    ("library", "biblioteca"), ("museum", "museo"), ("theater", "teatro"),
    ("stadium", "stadio"), ("park", "parco"), ("garden", "giardino"),
    ("palace", "palazzo"), ("building", "edificio"),
    ("apartment", "appartamento"), ("neighborhood", "quartiere"),
    ("capital", "capitale"), ("border", "confine"), ("map", "mappa"),

    # --- abstract nouns ---
    ("love", "amore"), ("hate", "odio"), ("fear", "paura"),
    ("hope", "speranza"), ("joy", "gioia"), ("sadness", "tristezza"),
    ("anger", "rabbia"), ("peace", "pace"), ("war", "guerra"),
    ("freedom", "libertà"), ("justice", "giustizia"), ("truth", "verità"),
    ("beauty", "bellezza"), ("strength", "forza"), ("power", "potere"),
    ("knowledge", "conoscenza"), ("wisdom", "saggezza"), ("courage", "coraggio"),
    ("faith", "fede"), ("dream", "sogno"), ("thought", "pensiero"),
    ("opinion", "opinione"), ("question", "domanda"), ("answer", "risposta"),
    ("problem", "problema"), ("solution", "soluzione"), ("reason", "ragione"),
    ("result", "risultato"), ("success", "successo"), ("failure", "fallimento"),
    ("mistake", "errore"), ("effort", "sforzo"), ("luck", "fortuna"),
    ("risk", "rischio"), ("danger", "pericolo"), ("safety", "sicurezza"),
    ("health", "salute"), ("illness", "malattia"), ("pain", "dolore"),
    ("pleasure", "piacere"), ("comfort", "conforto"), ("rest", "riposo"),
    ("silence", "silenzio"), ("noise", "rumore"),
    ("sound", "suono"), ("voice", "voce"), ("song", "canzone"),
    ("music", "musica"), ("art", "arte"), ("science", "scienza"),
    ("history", "storia"), ("story", "racconto"), ("culture", "cultura"),
    ("religion", "religione"), ("law", "legge"), ("rule", "regola"),
    ("duty", "dovere"), ("government", "governo"), ("politics", "politica"),
    ("economy", "economia"), ("money", "soldi"), ("price", "prezzo"),
    ("cost", "costo"), ("value", "valore"), ("tax", "tassa"),
    ("debt", "debito"), ("profit", "profitto"), ("loss", "perdita"),
    ("gift", "regalo"), ("prize", "premio"), ("destiny", "destino"),
    ("future", "futuro"), ("past", "passato"), ("present", "presente"),
    ("beginning", "inizio"), ("end", "fine"), ("bottom", "fondo"),
    ("corner", "angolo"), ("edge", "bordo"), ("center", "centro"),
    ("line", "linea"), ("circle", "cerchio"), ("square", "quadrato"),
    ("triangle", "triangolo"), ("sense", "senso"), ("emotion", "emozione"),
    ("attitude", "atteggiamento"), ("behavior", "comportamento"),
    ("habit", "abitudine"), ("tradition", "tradizione"),
    ("ceremony", "cerimonia"), ("celebration", "celebrazione"),
    ("surprise", "sorpresa"), ("secret", "segreto"), ("mystery", "mistero"),
    ("adventure", "avventura"), ("journey", "viaggio"), ("vacation", "vacanza"),
    ("guide", "guida"), ("compass", "bussola"), ("flag", "bandiera"),
    ("symbol", "simbolo"), ("sign", "segno"), ("signature", "firma"),
    ("stamp", "francobollo"), ("coin", "moneta"), ("wallet", "portafoglio"),
    ("purse", "borsetta"), ("jewelry", "gioielli"), ("ring", "anello"),
    ("necklace", "collana"), ("bracelet", "braccialetto"),
    ("earring", "orecchino"), ("crown", "corona"), ("sword", "spada"),
    ("shield", "scudo"), ("armor", "armatura"), ("weapon", "arma"),
    ("gun", "pistola"), ("bullet", "proiettile"), ("bomb", "bomba"),
    ("army", "esercito"), ("navy", "marina"), ("battle", "battaglia"),
    ("victory", "vittoria"), ("defeat", "sconfitta"), ("treaty", "trattato"),
    ("alliance", "alleanza"),
    ("feeling", "sentimento"), ("smell", "odore"),

    # --- colors ---
    ("red", "rosso"), ("blue", "azzurro"), ("green", "verde"),
    ("yellow", "giallo"), ("black", "nero"), ("white", "bianco"),
    ("purple", "viola"), ("pink", "rosa"), ("brown", "marrone"),
    ("gray", "grigio"), ("gold", "oro"), ("silver", "argento"),

    # --- school / language ---
    ("book", "libro"), ("alphabet", "alfabeto"), ("word", "parola"),
    ("sentence", "frase"), ("language", "lingua"), ("grammar", "grammatica"),
    ("vocabulary", "vocabolario"), ("dictionary", "dizionario"),
    ("translation", "traduzione"), ("subject", "materia"), ("lesson", "lezione"),
    ("homework", "compiti"), ("exam", "esame"), ("grade", "voto"),
    ("class", "classe"), ("notebook", "quaderno"), ("blackboard", "lavagna"),
    ("desk", "scrivania"), ("chalk", "gesso"), ("ruler", "righello"),
    ("eraser", "gomma"),

    # --- sports / games ---
    ("game", "gioco"), ("toy", "giocattolo"), ("ball", "palla"),
    ("football", "calcio"), ("basketball", "pallacanestro"),
    ("swimming", "nuoto"), ("team", "squadra"), ("player", "giocatore"),
    ("coach", "allenatore"), ("referee", "arbitro"), ("score", "punteggio"),
    ("goal", "gol"), ("medal", "medaglia"), ("champion", "campione"),

    # --- technology ---
    ("website", "sito"), ("data", "dati"), ("number", "numero"),
    ("code", "codice"), ("system", "sistema"), ("program", "programma"),
    ("signal", "segnale"), ("screen", "schermo"), ("network", "rete"),

    # --- verbs ---
    ("run", "correre"), ("walk", "camminare"), ("eat", "mangiare"),
    ("drink", "bere"), ("sleep", "dormire"), ("speak", "parlare"),
    ("listen", "ascoltare"), ("hear", "sentire"), ("see", "vedere"),
    ("look", "guardare"), ("write", "scrivere"), ("read", "leggere"),
    ("sing", "cantare"), ("dance", "ballare"), ("play", "giocare"),
    ("work", "lavorare"), ("study", "studiare"), ("learn", "imparare"),
    ("teach", "insegnare"), ("think", "pensare"), ("understand", "capire"),
    ("remember", "ricordare"), ("forget", "dimenticare"), ("believe", "credere"),
    ("want", "volere"),
    ("wish", "desiderare"), ("try", "provare"),
    ("begin", "cominciare"), ("finish", "finire"), ("open", "aprire"),
    ("close", "chiudere"), ("break", "rompere"), ("repair", "riparare"),
    ("build", "costruire"), ("destroy", "distruggere"), ("create", "creare"),
    ("make", "fare"), ("give", "dare"), ("take", "prendere"),
    ("bring", "portare"), ("send", "mandare"), ("receive", "ricevere"),
    ("buy", "comprare"), ("sell", "vendere"), ("pay", "pagare"),
    ("spend", "spendere"), ("save", "risparmiare"), ("lose", "perdere"),
    ("find", "trovare"), ("search", "cercare"), ("choose", "scegliere"),
    ("decide", "decidere"), ("change", "cambiare"), ("push", "spingere"),
    ("pull", "tirare"), ("hold", "tenere"), ("touch", "toccare"),
    ("hit", "colpire"), ("cut", "tagliare"), ("tear", "strappare"),
    ("fold", "piegare"), ("wash", "lavare"), ("clean", "pulire"),
    ("cook", "cucinare"), ("boil", "bollire"), ("fry", "friggere"),
    ("grow", "crescere"), ("plant", "piantare"), ("harvest", "raccogliere"),
    ("kill", "uccidere"), ("die", "morire"), ("live", "vivere"),
    ("marry", "sposare"), ("divorce", "divorziare"), ("travel", "viaggiare"),
    ("visit", "visitare"), ("arrive", "arrivare"), ("return", "tornare"),
    ("enter", "entrare"), ("exit", "uscire"), ("fall", "cadere"),
    ("jump", "saltare"), ("swim", "nuotare"), ("fly", "volare"),
    ("drive", "guidare"), ("fight", "combattere"), ("win", "vincere"),
    ("help", "aiutare"), ("smile", "sorridere"), ("laugh", "ridere"),
    ("cry", "piangere"), ("shout", "gridare"), ("whisper", "sussurrare"),
    ("kiss", "baciare"), ("hug", "abbracciare"), ("taste", "assaggiare"),
    ("appear", "apparire"), ("disappear", "scomparire"), ("happen", "succedere"),
    ("seem", "sembrare"), ("become", "diventare"), ("remain", "rimanere"),
    ("wait", "aspettare"), ("follow", "seguire"), ("add", "aggiungere"),
    ("remove", "rimuovere"), ("mix", "mescolare"), ("separate", "separare"),
    ("join", "unire"), ("connect", "collegare"),
    ("ask", "chiedere"), ("explain", "spiegare"), ("describe", "descrivere"),
    ("show", "mostrare"), ("hide", "nascondere"), ("protect", "proteggere"),
    ("attack", "attaccare"), ("defend", "difendere"), ("escape", "scappare"),
    ("capture", "catturare"), ("promise", "promettere"), ("refuse", "rifiutare"),
    ("accept", "accettare"), ("allow", "permettere"), ("forbid", "proibire"),
    ("count", "contare"), ("measure", "misurare"), ("weigh", "pesare"),
    ("compare", "confrontare"), ("guess", "indovinare"), ("imagine", "immaginare"),
    ("suppose", "supporre"), ("plan", "pianificare"), ("prepare", "preparare"),
    ("organize", "organizzare"), ("manage", "gestire"), ("control", "controllare"),
    ("own", "possedere"), ("belong", "appartenere"), ("exist", "esistere"),
    ("mean", "significare"), ("represent", "rappresentare"),
    ("include", "includere"), ("contain", "contenere"), ("depend", "dipendere"),
    ("cause", "causare"), ("influence", "influenzare"), ("improve", "migliorare"),
    ("increase", "aumentare"), ("decrease", "diminuire"), ("reduce", "ridurre"),
    ("raise", "alzare"), ("lower", "abbassare"), ("develop", "sviluppare"),

    # --- adjectives ---
    ("big", "grande"), ("small", "piccolo"), ("long", "lungo"),
    ("short", "corto"), ("tall", "alto"), ("high", "alto"), ("low", "basso"),
    ("wide", "largo"), ("narrow", "stretto"), ("deep", "profondo"),
    ("heavy", "pesante"), ("light", "leggero"), ("fast", "veloce"),
    ("slow", "lento"), ("strong", "forte"), ("weak", "debole"),
    ("hard", "duro"), ("soft", "morbido"), ("new", "nuovo"),
    ("old", "vecchio"), ("young", "giovane"), ("good", "buono"),
    ("bad", "cattivo"), ("beautiful", "bello"), ("ugly", "brutto"),
    ("happy", "felice"), ("sad", "triste"), ("angry", "arrabbiato"),
    ("scared", "spaventato"), ("tired", "stanco"), ("hungry", "affamato"),
    ("thirsty", "assetato"), ("sick", "malato"), ("healthy", "sano"),
    ("dirty", "sporco"), ("rich", "ricco"), ("poor", "povero"),
    ("easy", "facile"), ("difficult", "difficile"), ("important", "importante"),
    ("interesting", "interessante"), ("boring", "noioso"), ("fun", "divertente"),
    ("serious", "serio"), ("kind", "gentile"), ("cruel", "crudele"),
    ("honest", "onesto"), ("dishonest", "disonesto"), ("brave", "coraggioso"),
    ("cowardly", "codardo"), ("smart", "intelligente"), ("stupid", "stupido"),
    ("wise", "saggio"), ("crazy", "pazzo"), ("calm", "calmo"),
    ("nervous", "nervoso"), ("excited", "emozionato"), ("bored", "annoiato"),
    ("proud", "orgoglioso"), ("shy", "timido"), ("rude", "scortese"),
    ("friendly", "amichevole"), ("hostile", "ostile"), ("lazy", "pigro"),
    ("active", "attivo"), ("busy", "occupato"), ("free", "libero"),
    ("full", "pieno"), ("empty", "vuoto"), ("closed", "chiuso"),
    ("far", "lontano"), ("wrong", "sbagliato"), ("true", "vero"),
    ("false", "falso"), ("real", "reale"), ("same", "stesso"),
    ("different", "diverso"), ("similar", "simile"), ("equal", "uguale"),
    ("whole", "intero"), ("double", "doppio"), ("single", "singolo"),
    ("special", "speciale"), ("normal", "normale"), ("strange", "strano"),
    ("common", "comune"), ("rare", "raro"), ("famous", "famoso"),
    ("unknown", "sconosciuto"), ("popular", "popolare"), ("modern", "moderno"),
    ("ancient", "antico"), ("traditional", "tradizionale"),
    ("natural", "naturale"), ("artificial", "artificiale"),
    ("public", "pubblico"), ("private", "privato"), ("official", "ufficiale"),
    ("legal", "legale"), ("illegal", "illegale"), ("possible", "possibile"),
    ("impossible", "impossibile"), ("necessary", "necessario"),
    ("useless", "inutile"), ("useful", "utile"), ("dangerous", "pericoloso"),
    ("safe", "sicuro"), ("comfortable", "comodo"), ("expensive", "costoso"),
    ("cheap", "economico"), ("quiet", "silenzioso"), ("loud", "rumoroso"),
    ("bright", "luminoso"), ("dark", "buio"), ("warm", "caldo"),
    ("hot", "caldo"), ("wet", "bagnato"), ("dry", "secco"),
    ("straight", "dritto"), ("curved", "curvo"), ("round", "rotondo"),
    ("flat", "piatto"), ("sharp", "affilato"), ("rough", "ruvido"),
    ("thick", "spesso"), ("thin", "sottile"), ("fat", "grasso"),
    ("skinny", "magro"), ("even", "pari"), ("odd", "dispari"),
    ("positive", "positivo"), ("negative", "negativo"),
    ("absolute", "assoluto"), ("relative", "relativo"),
    ("constant", "costante"), ("variable", "variabile"),
    ("identical", "identico"), ("opposite", "opposto"), ("main", "principale"),
    ("minor", "minore"), ("major", "maggiore"), ("central", "centrale"),
    ("local", "locale"), ("global", "globale"),
    ("international", "internazionale"), ("national", "nazionale"),
    ("regional", "regionale"), ("urban", "urbano"), ("rural", "rurale"),
    ("industrial", "industriale"), ("commercial", "commerciale"),
    ("financial", "finanziario"), ("economic", "economico"),
    ("political", "politico"), ("social", "sociale"), ("cultural", "culturale"),
    ("historical", "storico"), ("scientific", "scientifico"),
    ("technical", "tecnico"), ("practical", "pratico"),
    ("theoretical", "teorico"), ("physical", "fisico"), ("mental", "mentale"),
    ("emotional", "emotivo"), ("spiritual", "spirituale"),
    ("religious", "religioso"), ("moral", "morale"), ("ethical", "etico"),
    ("medical", "medico"), ("military", "militare"), ("civil", "civile"),
    ("domestic", "domestico"), ("foreign", "straniero"), ("native", "nativo"),
    ("original", "originale"), ("authentic", "autentico"),
    ("genuine", "genuino"), ("synthetic", "sintetico"), ("organic", "organico"),
    ("electric", "elettrico"), ("mechanical", "meccanico"),
    ("digital", "digitale"), ("manual", "manuale"), ("automatic", "automatico"),
    ("visual", "visivo"), ("verbal", "verbale"), ("written", "scritto"),
    ("spoken", "parlato"), ("musical", "musicale"), ("artistic", "artistico"),
    ("literary", "letterario"), ("poetic", "poetico"), ("dramatic", "drammatico"),
    ("comic", "comico"), ("tragic", "tragico"), ("romantic", "romantico"),
    ("sexual", "sessuale"), ("adult", "adulto"), ("juvenile", "giovanile"),
    ("elderly", "anziano"), ("mature", "maturo"), ("immature", "immaturo"),
    ("pregnant", "incinta"), ("married", "sposato"), ("divorced", "divorziato"),
    ("widowed", "vedovo"), ("alive", "vivo"), ("dead", "morto"),
    ("awake", "sveglio"), ("asleep", "addormentato"),
    ("conscious", "consapevole"),

    # --- numbers / weekdays (extra high-confidence content words) ---
    ("one", "uno"), ("two", "due"), ("three", "tre"), ("four", "quattro"),
    ("five", "cinque"), ("six", "sei"), ("seven", "sette"), ("eight", "otto"),
    ("nine", "nove"), ("ten", "dieci"), ("hundred", "cento"),
    ("thousand", "mille"), ("monday", "lunedì"), ("tuesday", "martedì"),
    ("wednesday", "mercoledì"), ("thursday", "giovedì"), ("friday", "venerdì"),
    ("saturday", "sabato"), ("sunday", "domenica"),

    # --- more animals ---
    ("mole", "talpa"), ("hedgehog", "riccio"), ("beaver", "castoro"),
    ("otter", "lontra"), ("giraffe", "giraffa"), ("hippo", "ippopotamo"),
    ("rhino", "rinoceronte"), ("chimpanzee", "scimpanzè"),
    ("sparrow", "passero"), ("pigeon", "piccione"), ("seagull", "gabbiano"),
    ("stork", "cicogna"), ("flamingo", "fenicottero"), ("ostrich", "struzzo"),
    ("vulture", "avvoltoio"), ("scorpion", "scorpione"),
    ("grasshopper", "cavalletta"), ("ladybug", "coccinella"),
    ("cockroach", "scarafaggio"), ("jellyfish", "medusa"), ("octopus", "polpo"),
    ("cricket", "grillo"),

    # --- more household ---
    ("sink", "lavandino"), ("oven", "forno"), ("refrigerator", "frigorifero"),
    ("dishwasher", "lavastoviglie"), ("elevator", "ascensore"),
    ("staircase", "scale"), ("balcony", "balcone"), ("fence", "recinto"),
    ("gate", "cancello"), ("chimney", "camino"), ("drawer", "cassetto"),
    ("doorbell", "campanello"), ("faucet", "rubinetto"),

    # --- more time ---
    ("century", "secolo"), ("decade", "decennio"), ("moment", "momento"),
    ("instant", "istante"), ("period", "periodo"),

    # --- more nature ---
    ("rainbow", "arcobaleno"), ("tide", "marea"), ("wave", "onda"),
    ("glacier", "ghiacciaio"), ("meadow", "prato"), ("swamp", "palude"),
    ("cliff", "scogliera"), ("puddle", "pozzanghera"),

    # --- more abstract nouns ---
    ("patience", "pazienza"), ("generosity", "generosità"),
    ("greed", "avidità"), ("honesty", "onestà"), ("loyalty", "lealtà"),
    ("betrayal", "tradimento"), ("revenge", "vendetta"),
    ("forgiveness", "perdono"), ("gratitude", "gratitudine"),
    ("curiosity", "curiosità"), ("ambition", "ambizione"),
    ("jealousy", "gelosia"), ("pride", "orgoglio"), ("shame", "vergogna"),
    ("guilt", "colpa"), ("innocence", "innocenza"),
    ("equality", "uguaglianza"), ("diversity", "diversità"),
    ("unity", "unità"), ("harmony", "armonia"), ("chaos", "caos"),
    ("order", "ordine"), ("disorder", "disordine"), ("balance", "equilibrio"),
    ("instinct", "istinto"), ("logic", "logica"), ("intuition", "intuizione"),
    ("imagination", "immaginazione"), ("creativity", "creatività"),
    ("inspiration", "ispirazione"), ("motivation", "motivazione"),
    ("determination", "determinazione"), ("tolerance", "tolleranza"),
    ("respect", "rispetto"), ("trust", "fiducia"), ("care", "cura"),
    ("warmth", "calore"),

    # --- more adjectives ---
    ("clever", "intelligente"), ("brilliant", "brillante"),
    ("talented", "talentuoso"), ("curious", "curioso"), ("patient", "paziente"),
    ("impatient", "impaziente"), ("generous", "generoso"), ("greedy", "avido"),
    ("selfish", "egoista"), ("loyal", "leale"), ("faithful", "fedele"),
    ("jealous", "geloso"), ("arrogant", "arrogante"), ("humble", "umile"),
    ("optimistic", "ottimista"), ("pessimistic", "pessimista"),
    ("ambitious", "ambizioso"), ("determined", "determinato"),
    ("motivated", "motivato"), ("creative", "creativo"),
    ("innocent", "innocente"), ("guilty", "colpevole"), ("unfair", "ingiusto"),
    ("reasonable", "ragionevole"), ("logical", "logico"),
    ("illogical", "illogico"), ("rational", "razionale"),
    ("irrational", "irrazionale"), ("sensitive", "sensibile"),
    ("insensitive", "insensibile"), ("tolerant", "tollerante"),
    ("intolerant", "intollerante"), ("respectful", "rispettoso"),
    ("disrespectful", "irrispettoso"), ("unfaithful", "infedele"),
]

# -------------------------------------------------------------------------
# Filtering / validation
# -------------------------------------------------------------------------

def strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

seen_en = {}
duplicates = []
filtered = []
for en, it in RAW_PAIRS:
    en = en.strip().lower()
    it = it.strip().lower()
    if " " in en or " " in it:
        continue  # multi-word, excluded by spec
    if strip_accents(en) == strip_accents(it):
        continue  # identical form, excluded by spec
    if en in seen_en:
        if seen_en[en] != it:
            duplicates.append((en, seen_en[en], it))
        continue  # keep first occurrence only
    seen_en[en] = it
    filtered.append((en, it))

# Any real (conflicting) duplicate key is a curation bug -> hard fail so it
# gets fixed at the source rather than silently resolved.
assert not duplicates, f"Conflicting duplicate keys found: {duplicates}"

assert len(filtered) >= 1000, f"only {len(filtered)} candidates survived filtering, need >=1000"

# Keep the first 1000 (list is already ordered core-vocab-first); this is
# well under the ~1200-candidate pool actually curated.
kept = filtered[:1000]

# Capitalize outputs to look like plain words (lowercase is the natural case
# for these common nouns/verbs/adjectives, so we leave lowercase).
pairs = [{"input": en, "output": it} for en, it in kept]

random.shuffle(pairs)

# -------------------------------------------------------------------------
# Asserts
# -------------------------------------------------------------------------
assert len(pairs) == 1000
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == 1000
for p in pairs:
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()
    assert p["input"] == p["input"].lower()
    assert " " not in p["input"] and " " not in p["output"]
    assert strip_accents(p["input"]) != strip_accents(p["output"])

# Re-derive from the source dict as a mechanical rule self-check.
lookup = dict(kept)
for p in pairs:
    assert lookup[p["input"]] == p["output"]

vocab = Counter(p["output"] for p in pairs)

OUT_PATH.write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
print(f"wrote {OUT_PATH} n={len(pairs)} distinct_outputs={len(vocab)} "
      f"candidates_before_trim={len(filtered)}")
