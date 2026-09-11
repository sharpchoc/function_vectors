"""Per-family specification for the Spanish source corpus of the style-translation study.

For each of the 17 style families (registry: src/sandbox/ext_styleprops/properties.py) this
module says WHAT THE SPANISH MUST CONTAIN so that an English translation is forced to choose
between the family's two styles at >= 5 places ("opportunities", aim 5-7), how to instruct the
generator, how to pre-check the Spanish with a regex, and the rubric a verification agent uses.

The Spanish itself is ALWAYS in one fixed natural convention (user decision 2026-09-07, RAE
standard with all numbers as digits):
  cardinals as digits (3 dias), ordinals as digits with the RAE marker (3.o / 3.a), percentages
  "15 %", quotations << >>, asides with rayas (attached: palabra --inciso-- palabra), ellipsis "...",
  single space after periods, normal capitalisation. See CONVENTIONS below (rendered in the prompt).
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.ext_styleprops.properties import PROPS  # the 17-style registry (labels only)

CONVENTIONS = """Spanish conventions (apply ALL of them, every time, no exceptions):
- Standard European Spanish, normal sentence capitalisation, one space after every period.
- Every number as digits: cardinals as digits (3 días, 12 personas — never «tres», «doce»);
  ordinals as digits with the RAE marker (1.º, 2.ª, 3.º, 10.º — never «primero», «tercer»);
  percentages with the sign and a space (15 %) — never «por ciento».
- Quotations in angular quotes « » only. Never straight " or curly “ ” double quotes.
- Asides with rayas attached to the aside: palabra —inciso breve— palabra. Never a spaced hyphen ( - ) or en dash (–).
- Ellipsis as three dots (...). Never the single-glyph ellipsis (…).
- One paragraph of 6 to 8 sentences, about 120 to 180 words, coherent and natural, on the given topic. Return ONLY the paragraph."""

# ---- Spanish-side lexicons ------------------------------------------------------------------
# us_uk: Spanish words whose standard English translation is a US/UK-variant word
ES_USUK = [
    "color", "colores", "colorido", "colorida", "sabor", "sabores", "favorito", "favorita",
    "favoritos", "favoritas", "honor", "humor", "labor", "vecino", "vecina", "vecinos", "vecinas",
    "vecindario", "rumor", "rumores", "vapor", "comportamiento", "comportamientos", "olor",
    "olores", "vigor", "centro", "centros", "teatro", "teatros", "litro", "litros", "metro",
    "metros", "fibra", "fibras", "catálogo", "catálogos", "diálogo", "diálogos", "defensa",
    "ofensa", "viajero", "viajera", "viajeros", "viajó", "viajaron", "viajando", "etiquetado",
    "etiquetada", "etiquetar", "cancelado", "cancelada", "canceló", "cancelaron", "joyería", "gris",
    "grises", "aluminio", "pijama", "pijamas", "artefacto", "artefactos", "puerto",
]
ES_ISEIZE_STEMS = [
    "organiz", "reconoc", "reconoz", "disculp", "critic", "enfatiz", "resum", "minimiz", "maximiz",
    "prioriz", "especializ", "caracteriz", "categoriz", "estandariz", "utiliz", "memoriz",
    "finaliz", "generaliz", "analiz", "paraliz",
]
ES_ISEIZE_PHRASES = ["darse cuenta", "se dio cuenta", "me di cuenta", "nos dimos cuenta", "te das cuenta",
                     "se dan cuenta", "me doy cuenta", "se da cuenta", "dándose cuenta", "darme cuenta"]
ES_TPAST = [  # PRETERITE forms of aprender, deletrear, quemar, soñar, saltar, apoyarse/inclinarse, derramar, estropearse/echar a perder
    r"aprend(ió|ieron|í|imos|iste|isteis)", r"deletre(ó|aron|é|amos|aste)", r"(se |me |te |nos )?quem(ó|aron|é|amos|aste)",
    r"soñ(ó|aron|é|amos|aste)", r"salt(ó|aron|é|amos|aste)", r"(se |me |te |nos )?(apoy|inclin)(ó|aron|é|amos|aste)",
    r"(se |me |te |nos )?derram(ó|aron|é|amos|aste)", r"(se |me |te |nos )?estrope(ó|aron|é|amos)", r"(se )?ech(ó|aron|é|amos) a perder",
]
ES_WHILST = [r"\bmientras\b", r"\bentre\b", r"\ben medio de\b"]

# ---- regex pre-checks (count of opportunities in the Spanish) ------------------------------
_SENT_END = re.compile(r"[.!?»)]+(?=\s+[A-ZÁÉÍÓÚÑ¿¡«—]|\s*$)")


def count_sentences(t):
    return len(_SENT_END.findall(t.strip()))


def count_lexicon(words):
    rx = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
    return lambda t: len(rx.findall(t))


def count_ise_ize(t):
    n = len(re.findall(r"\b(" + "|".join(ES_ISEIZE_STEMS) + r")\w*", t, re.I))
    n += sum(len(re.findall(re.escape(p), t, re.I)) for p in ES_ISEIZE_PHRASES)
    return n


def count_tpast(t):
    return sum(len(re.findall(r"\b" + p + r"\b", t, re.I)) for p in ES_TPAST)


def count_whilst(t):
    return sum(len(re.findall(p, t, re.I)) for p in ES_WHILST)


def count_contractible(t):
    """Loose: negated verb phrases + pronoun/ser copulas that English renders contractibly."""
    n = len(re.findall(r"\b(?:no|tampoco|nunca)\s+(?:es|está|están|son|puede|pueden|hay|tiene|tienen|quiere|quieren|debe|deben|va|van|había|será|serán|sería|serían|podría|podrían|fue|estaba|estaban|era|eran|sabe|saben|importa|funciona|funcionan|hace|hacen|necesita|necesitan|deberías|debería|deberíamos|puedes|podemos|tienes|tenemos|quieres|queremos|vas|vamos|has|hemos|he|habrá|existe|existen|significa|basta|sirve|sirven|cuesta|cuestan)\b", t, re.I))
    n += len(re.findall(r"\b(?:eso|esto|ello|aquello|todo|lo que importa)\s+es\b", t, re.I))
    n += len(re.findall(r"\b(?:estoy|estás|estamos|soy|eres|somos|habrá que|hay que)\b", t, re.I))
    return n


def count_and_pairs(t):
    return len(re.findall(r"\b[\wáéíóúñ]+ y [\wáéíóúñ]+\b", t, re.I))


def count_lists(t):
    """Lists of >=3 items ending in y/o; items may be short phrases (up to ~45 chars, no punctuation)."""
    return len(re.findall(r"[^,.;:«»—]{2,45}, [^,.;:«»—]{2,45}(?:, [^,.;:«»—]{2,45})*,? (?:y|o|e|u) [^,.;:«»—]{2,45}", t, re.I))


def count_quotes(t):
    return len(re.findall(r"«[^»]{2,}»", t))


def count_quote_punct(t):
    """Quotations that end at a clause/sentence boundary (punctuation right after »)."""
    return len(re.findall(r"«[^»]{2,}»\s*[.,;]", t)) + len(re.findall(r"[.,;]\s*»", t))


def count_rayas(t):
    return len(re.findall(r"—[^—]{2,}—", t)) + len(re.findall(r"—[^—]{2,}[.!?](?=\s|$)", t))


def count_ellipsis(t):
    return len(re.findall(r"\.\.\.(?=\s*\S)", t))   # followed by more text


_NUM_EXCL = re.compile(r"(?<![\d.,:])\b([2-9]|1\d|20)\b(?![\d.,:]|\s?%|\.[ºª]|\s?(?:h\b|:\d))")


def count_cardinals(t):
    return len(_NUM_EXCL.findall(t))


def count_percent(t):
    return len(re.findall(r"\d+\s?%", t))


def count_ordinals(t):
    return len(re.findall(r"\b(?:[1-9]|10)\.(?:[ºª]|er|o|a)\b", t))


# ---- lexically diverse families (2026-09-11) ------------------------------------------------
ES_UKVOCAB = ["camión", "camiones", "vacaciones", "basura", "cubo de basura", "gasolina", "linterna", "linternas",
    "teléfono móvil", "móvil", "celular", "acera", "aceras", "galleta", "galletas", "jersey", "suéter", "pantalones",
    "pañal", "pañales", "armario", "armarios", "berenjena", "berenjenas", "calabacín", "calabacines", "cilantro",
    "matemáticas", "autopista", "autopistas", "aparcamiento", "estacionamiento", "gasolinera", "gasolineras",
    "carrito de la compra", "carrito", "cochecito", "carriola", "cuna", "parabrisas", "llave inglesa", "guardabarros",
    "sentido antihorario", "contrario a las agujas del reloj", "gachas", "avena", "comida para llevar", "código postal",
    "ferrocarril", "matrícula", "buzón", "cartero", "chaleco", "grifo", "farmacia", "piso", "apartamento", "ascensor"]
ES_REGISTER = ["empezar", "empezó", "empezamos", "empecé", "comenzar", "comenzó", "comencé", "comprar", "compró", "compré",
    "compramos", "intentar", "intenté", "intentó", "intentamos", "vivir", "vivo", "vive", "vivimos", "vivía", "revisar", "revisé",
    "revisó", "comprobar", "comprobé", "comprobó", "arreglar", "arreglé", "arregló", "reparar", "reparé", "niños", "niñas", "niño",
    "mostrar", "mostró", "mostré", "enseñar", "enseñó", "más tarde", "luego", "también", "quizá", "quizás", "tal vez", "grande",
    "grandes", "averiguar", "averigüé", "averiguó", "descubrir", "descubrí", "descubrió", "investigar", "investigué", "tirar",
    "tiré", "tiró", "desechar", "resolver", "resolví", "determinar", "participar", "participé", "participó", "deshacerse de",
    "deshacerme de", "eliminar", "eliminé", "buscar", "busqué", "buscó", "buscando"]
ES_DIACRITICS = ["café", "cafetería", "ingenuo", "ingenua", "cliché", "tópico", "prometido", "prometida", "fachada", "jalapeño",
    "jalapeños", "decoración", "saltear", "salteado", "salteada", "puré", "crema", "plato principal", "matiné", "debut", "estreno",
    "élite", "señor", "señora", "piñata", "pupilo", "protegido", "déjà vu", "a la carta", "velada", "canapé", "canapés", "paté",
    "macramé", "aplique", "guardería", "papel maché", "ingenuidad"]
ES_LATINABBR = [r"por ejemplo", r"es decir", r"etcétera", r"y así sucesivamente", r"frente a", r"\bcontra\b", r"\bversus\b",
    r"aproximadamente", r"\bunos? \d", r"\bunas \d"]
ES_HYPHEN = ["correo electrónico", "correos electrónicos", "en línea", "sitio web", "página web", "bienestar", "cooperar",
    "cooperación", "reutilizar", "reutilizable", "reutilizables", "casero", "casera", "caseros", "sin parar", "estilo de vida",
    "entrenamiento", "entrenamientos", "revisión médica", "chequeo", "maquillaje", "configuración", "copia de seguridad",
    "empresa emergente", "coordinar", "coordinación", "precalentar", "precalienta", "multitarea", "teléfono inteligente",
    "trabajo en equipo", "durante el día", "durante la noche", "fin de semana", "fines de semana", "protector solar",
    "cepillo de dientes", "hecho a mano", "hecha a mano", "cronología", "nombre de usuario", "inicio de sesión", "sin conexión",
    "microondas", "agua de lluvia", "leña", "invernadero", "cortacésped", "carretilla", "manillar", "pata de cabra",
    "faro delantero", "faros", "luz trasera", "tija", "mochila", "mochilas"]
ES_FLATADV = [r"despacio", r"lentamente", r"con lentitud", r"\brápido\b", r"rápidamente", r"con rapidez", r"deprisa", r"con suavidad",
    r"suavemente", r"con fuerza", r"firmemente", r"con firmeza", r"\bfuerte\b", r"en voz alta", r"con seguridad", r"de forma segura",
    r"profundamente", r"\bhondo\b", r"a fondo", r"barato", r"directamente", r"\bdirecto\b", r"de otra manera", r"de forma distinta",
    r"\bmal\b", r"\bfirme\b", r"bajito", r"en voz baja", r"uniformemente", r"de manera uniforme", r"con cuidado", r"cuidadosamente",
    r"en silencio", r"calladamente", r"de cerca", r"con constancia", r"de forma constante", r"claramente", r"\bclaro\b", r"fácilmente",
    r"con facilidad", r"brillante", r"ligeramente", r"en rodajas finas", r"\bfino\b", r"finamente", r"recién", r"correctamente", r"como es debido"]
ES_IRREGPAST = [
    r"(se |me |nos |te )?(zambull|sumerg)(ió|ieron|í|imos|iste)", r"buce(ó|aron|é|amos|aste)",
    r"(se |me |nos |te )?(lanz|tir|arroj)(ó|aron|é|amos|aste) al (agua|lago|río|mar|piscina|estanque)", r"salt(ó|aron|é|amos) al (agua|lago|río|mar|piscina)",
    r"(se |me |nos |te )?(col|escabull)(ó|aron|é|amos|aste|ió|eron|í|imos|iste)", r"entr(ó|aron|é|amos|aste) (a hurtadillas|sigilosamente|sin hacer ruido|de puntillas)",
    r"(encend|prend)(ió|ieron|í|imos|iste)", r"ilumin(ó|aron|é|amos)",
    r"(suplic|rog|implor)(ó|aron|é|amos|aste|ué|uemos)",
    r"aceler(ó|aron|é|amos|aste)", r"(fue|fueron|fui|fuimos|corri(ó|eron|í|imos)|pas(ó|aron|é|amos)|sali(ó|eron|í|imos)) (a toda velocidad|a toda prisa|disparad[oa]s?)",
    r"tej(ió|ieron|í|imos|iste)", r"entretej(ió|ieron|í|imos)",
    r"(brill|resplandec|reluci)(ó|aron|é|amos|ió|ieron)",
    r"(se |me |nos |te )?esforz(ó|aron|amos|aste)", r"me esforcé", r"luch(ó|aron|é|amos) por",
    r"(se |me |nos |te )?(arrodill|hinc)(ó|aron|é|amos|aste)",
]

ES_LATINPLURAL = ["índices", "fórmulas", "cactus", "apéndices", "planes de estudio", "currículos", "estadios", "foros", "antenas",
    "hongos", "temarios", "programas de estudio", "pulpos", "radios", "núcleos", "memorandos", "referéndums", "referendos",
    "milenios", "acuarios", "terrarios", "vértices", "matrices", "larvas", "nebulosas", "vértebras", "tesauros", "gimnasios",
    "simposios", "hipopótamos", "antiguos alumnos", "exalumnos"]


def count_unit_symbols(t):
    return len(re.findall(r"\d+(?:[.,]\d+)? ?(?:km|cm|mm|kg|g|ml|°C)\b", t))


def count_titles(t):
    return len(re.findall(r"\b(?:doctor|doctora|profesor|profesora|señor|monte|san|santa|calle|avenida|carretera|camino|bulevar)\s+[A-ZÁÉÍÓÚÑ]", t, re.I))


def count_patterns(pats):
    return lambda t: sum(len(re.findall(p, t, re.I)) for p in pats)


@dataclass
class Family:
    name: str
    nat: str
    alt: str
    opportunity: str            # what the English translator must decide (for the verifier)
    instruction: str            # what the Spanish must contain (for the generator)
    regex_k: Callable[[str], int]
    anchors_hint: str = ""      # for the verifier: what to list as anchors
    extra_conventions: str = "" # family-specific reminders


def _f(name, opportunity, instruction, regex_k, anchors_hint, extra=""):
    p = PROPS[name]
    return Family(name, p.nat_label, p.alt_label, opportunity, instruction, regex_k, anchors_hint, extra)


FAMILIES: List[Family] = [
    _f("sentence_caps",
       "how each sentence begins in the English (standard capital vs lowercase start)",
       "Write 6 to 8 full sentences; every sentence boundary is an opportunity, so make sure there are at least 6 clearly separate sentences.",
       count_sentences, "the first word of each sentence"),
    _f("all_caps",
       "whether each English sentence is written in standard case or entirely in capitals",
       "Write 6 to 8 full sentences; each sentence is an opportunity, so make sure there are at least 6 clearly separate sentences.",
       count_sentences, "each sentence"),
    _f("double_space",
       "one vs two spaces after each sentence-final period in the English",
       "Write 6 to 8 full sentences ending in periods; every sentence boundary followed by another sentence is an opportunity, so use at least 6 sentences.",
       count_sentences, "each sentence boundary"),
    _f("us_uk",
       "American vs British spelling of a word (color/colour, center/centre, favorite/favourite, neighbor/neighbour, traveled/travelled, gray/grey, catalog/catalogue, defense/defence, theater/theatre, liter/litre, meter/metre, behavior/behaviour, flavor/flavour, honor/honour, humor/humour, labor/labour, harbor/harbour, jewelry/jewellery, aluminum/aluminium, pajamas/pyjamas, artifact/artefact, labeled/labelled, canceled/cancelled)",
       "Use at least 6 (ideally 6 to 7) DIFFERENT words from this list, spread through the paragraph: color, sabor, favorito/favorita, honor, humor, labor, vecino/vecina/vecindario, rumor, vapor, comportamiento, olor, vigor, centro, teatro, litro, metro, fibra, catálogo, diálogo, defensa, viajero/viajó, etiquetado, cancelado/canceló, joyería, gris, aluminio, pijama, artefacto, puerto (harbour).",
       count_lexicon(ES_USUK), "each Spanish word whose English equivalent has a US/UK spelling variant"),
    _f("ise_ize",
       "-ize vs -ise spelling of a verb or noun (organize/organise, realize/realise, recognize/recognise, apologize/apologise, criticize/criticise, emphasize/emphasise, summarize/summarise, minimize/minimise, prioritize/prioritise, specialize/specialise, characterize/characterise, categorize/categorise, standardize/standardise, utilize/utilise, memorize/memorise, finalize/finalise, generalize/generalise, analyze/analyse, paralyze/paralyse)",
       "Use at least 6 (ideally 6 to 7) DIFFERENT items from this list, spread through the paragraph: organizar/organización, reconocer, darse cuenta, disculparse, criticar, enfatizar, resumir, minimizar, maximizar, priorizar, especializarse, caracterizar, categorizar, estandarizar, utilizar, memorizar, finalizar, generalizar, analizar, paralizar.",
       count_ise_ize, "each Spanish verb/noun whose English equivalent ends in -ize/-ise (or -yze/-yse, -ization/-isation)"),
    _f("brit_t_past",
       "the -ed vs -t past form of a verb (learned/learnt, spelled/spelt, burned/burnt, dreamed/dreamt, leaped/leapt, leaned/leant, spilled/spilt, spoiled/spoilt)",
       "Tell an anecdote in the past tense (first or third person) in which these things happen NATURALLY, each in a situation where the verb is the obvious word: someone learned something (aprendió / aprendí), spelled a word out loud (deletreó), burned food or a finger (quemó / se quemó), dreamed something (soñó / soñé), leaped or jumped (saltó), leaned on something (se apoyó / se inclinó), spilled a drink (derramó / se derramó), and food or a plan spoiled (se estropeó / se echó a perder). Use at least 6 of these past-tense forms in total (aim 6 to 7) with at least 4 different verbs. Stay on the given topic but let the story visit a kitchen, a lesson or a night's sleep if that is what makes the verbs natural; never force a verb onto an object that cannot do it (tickets do not spill, trucks do not burn parts).",
       count_tpast, "each past-tense verb whose English past has a -t alternative"),
    _f("whilst",
       "while vs whilst, among vs amongst, amid vs amidst",
       "Use at least 6 (ideally 6 to 7) occurrences, spread through the paragraph, of: «mientras» as a conjunction (mientras + clause), «entre» meaning among (entre los vecinos), and «en medio de» meaning amid. Use at least 2 of the 3 forms.",
       count_whilst, "each mientras / entre (among) / en medio de"),
    _f("contractions",
       "contracted vs expanded English forms (isn't/is not, don't/do not, can't/cannot, it's/it is, there's/there is, I'm/I am, they're/they are, won't/will not)",
       "Write in a conversational register with at least 6 (ideally 6 to 7) clauses whose natural English translation contains an auxiliary or copula that can be contracted: negations like «no es», «no está», «no puede», «no hay», «no tiene», «no quiere», «no funciona»; pronoun + ser like «eso es», «esto es»; and first/second person «estoy», «estás», «soy». Spread them through the paragraph.",
       count_contractible, "each clause whose English rendering contains a contractible auxiliary/copula (is not, do not, cannot, it is, there is, I am ...)"),
    _f("ampersand",
       "the word and vs an ampersand (&) between two nouns",
       "Include at least 6 (ideally 6 to 7) pairs of two nouns joined by «y» (e.g. «pan y queso», «padres y profesores», «tiempo y dinero»), spread through the paragraph, each pair as a simple two-noun pair (not a longer list).",
       count_and_pairs, "each two-noun pair joined by y"),
    _f("oxford_comma",
       "serial (Oxford) comma vs none before the final and/or in a list of three or more items",
       "Include at least 6 (ideally 6 to 7) lists of exactly three single-word or short items joined as «A, B y C» (or «A, B o C»), spread through the paragraph.",
       count_lists, "each list of three or more items ending in y/o"),
    _f("curly_quotes",
       "straight (\") vs curly (“ ”) quotation marks around a quotation",
       "Include at least 6 (ideally 6 to 7) short direct quotations or quoted terms in angular quotes « », spread through the paragraph, e.g. things people said or names of things.",
       count_quotes, "each quoted span"),
    _f("quote_punct",
       "whether the period or comma goes inside (US) or outside (UK) the closing quotation mark",
       "Include at least 6 (ideally 6 to 7) short direct quotations in angular quotes « » where the quotation ends a sentence or is followed by a comma (e.g. «..., dijo ella.» patterns: the quote is immediately followed by a period or comma), spread through the paragraph.",
       count_quote_punct, "each quotation that ends at a period or comma"),
    _f("em_dash",
       "an attached em dash (word—word) vs a spaced hyphen (word - word) around an aside",
       "Include at least 6 (ideally 6 to 7) parenthetical asides set off with rayas attached to the aside (palabra —inciso breve— palabra), spread through the paragraph.",
       count_rayas, "each aside set off with rayas"),
    _f("ellipsis",
       "a three-dot ellipsis (...) vs the single-glyph ellipsis (…)",
       "Include at least 6 (ideally 6 to 7) pauses or trailing-offs written with three dots (...), each followed by more text in the same paragraph; the paragraph must not end with an ellipsis.",
       count_ellipsis, "each ... followed by more text"),
    _f("num_words",
       "a cardinal number 2-20 written as digits (7) vs spelled out (seven)",
       "Include at least 6 (ideally 6 to 7) DIFFERENT small whole numbers between 2 and 20, each as digits and used as a plain count or quantity (3 días, 12 personas, 7 minutos), spread through the paragraph. Do not use percentages, ordinals, times of day, decimals, or any number above 20.",
       count_cardinals, "each cardinal 2-20"),
    _f("percent_sign",
       "N% with the sign vs N percent spelled out",
       "Include at least 6 (ideally 6 to 7) different percentages written as digits followed by a space and the sign (15 %), spread through the paragraph. Do not use other numbers.",
       count_percent, "each percentage"),
    _f("ordinal_words",
       "a digit ordinal (3rd) vs a spelled-out ordinal (third)",
       "Include at least 6 (ideally 6 to 7) DIFFERENT ordinals from 1.º to 10.º, written as digits with the marker (1.º, 2.ª, 3.º, 4.º, ...), spread through the paragraph (steps, attempts, floors, rounds, anniversaries). Do not use other numbers.",
       count_ordinals, "each ordinal 1st-10th"),
    # ---- lexically diverse families (2026-09-11) ----
    _f("uk_vocab",
       "American vs British vocabulary for the same object (truck/lorry, vacation/holiday, trash/rubbish, gasoline/petrol, flashlight/torch, cell phone/mobile phone, sidewalk/pavement, cookie/biscuit, sweater/jumper, pants/trousers, diaper/nappy, closet/wardrobe, eggplant/aubergine, zucchini/courgette, cilantro/coriander, math/maths, highway/motorway, parking lot/car park, gas station/petrol station, shopping cart/trolley, stroller/pushchair, crib/cot, windshield/windscreen, wrench/spanner, fender/mudguard, counterclockwise/anticlockwise, oatmeal/porridge, takeout/takeaway, zip code/postcode, railroad/railway, license plate/number plate, mailbox/postbox, mailman/postman, vest/waistcoat, faucet/tap, drugstore/chemist's, apartment/flat, elevator/lift)",
       "Use at least 6 (ideally 6 to 7) DIFFERENT objects from this list, spread through the paragraph, each named with the everyday Spanish word: camión, vacaciones, basura / cubo de basura, gasolina, linterna, teléfono móvil, acera, galletas, jersey, pantalones, pañal, armario, berenjena, calabacín, cilantro, matemáticas, autopista, aparcamiento, gasolinera, carrito de la compra, cochecito (de bebé), cuna, parabrisas, llave inglesa, guardabarros, sentido antihorario, gachas de avena, comida para llevar, código postal, ferrocarril, matrícula (del coche), buzón, cartero, chaleco, grifo, farmacia, piso, ascensor.",
       count_lexicon(ES_UKVOCAB), "each Spanish word whose English equivalent differs between American and British vocabulary"),
    _f("register",
       "plain everyday wording vs formal/Latinate wording with the same meaning (begin/commence, buy/purchase, try/attempt, live/reside, check/verify, fix/repair, kids/children, show/demonstrate, later/subsequently, also/additionally, maybe/perhaps, big/large, find out/discover, look into/investigate, throw away/discard, figure out/determine, take part/participate, get rid of/eliminate, look for/seek)",
       "Use at least 6 (ideally 6 to 7) DIFFERENT items from this list, spread through the paragraph: empezar/comenzar, comprar, intentar, vivir, revisar/comprobar, arreglar/reparar, niños, mostrar/enseñar, más tarde/luego, también, quizá/tal vez, grande, averiguar/descubrir, investigar, tirar/desechar, resolver/determinar, participar, deshacerse de/eliminar, buscar.",
       count_lexicon(ES_REGISTER), "each Spanish word whose English equivalent has a plain and a formal variant"),
    _f("unit_abbr",
       "unit symbols vs spelled-out units after a number (5 km/5 kilometers, 2 kg/2 kilograms, 30 ml/30 milliliters, 20 cm/20 centimeters, 3 mm/3 millimeters, 250 g/250 grams, 15 °C/15 degrees Celsius)",
       "Include at least 7 measurements, each a number followed by a space and ONE of these seven unit symbols only: km, cm, mm, kg, g, ml, °C (e.g. «5 km», «30 cm», «2 kg», «250 g», «30 ml», «15 °C»). Use at least 4 different symbols from that list. Do NOT use any other unit (no m, l, L, min, h, bar, %, MB, Nm) and never write a unit as a word (no «kilómetros», «gramos»).",
       count_unit_symbols, "each number followed by a metric unit symbol",
       "- Units of measurement always as symbols after a space: 5 km, 2 kg, 30 ml, 20 cm, 15 °C (never «kilómetros», «kilos», «grados»)."),
    _f("diacritics",
       "whether a loanword keeps its accent in English (cafe/café, naive/naïve, cliche/cliché, fiance/fiancé, fiancee/fiancée, facade/façade, jalapeno/jalapeño, decor/décor, saute/sauté, puree/purée, creme/crème, entree/entrée, matinee/matinée, debut/début, elite/élite, senor/señor, pinata/piñata, protege/protégé, deja vu/déjà vu, a la carte/à la carte, soiree/soirée, canape/canapé, pate/pâté, macrame/macramé, applique/appliqué, fete/fête, creche/crèche, papier-mache/papier-mâché, naivete/naïveté, vis-a-vis/vis-à-vis, blase/blasé)",
       "Use at least 7 DIFFERENT items from this list, each in a sentence where it makes literal sense (food, a party, decoration, a couple, the theatre): café/cafetería, cliché/tópico, prometido/prometida, fachada, jalapeño, decoración, saltear/salteado, puré, plato principal (entrée), matiné, debut/estreno, élite, piñata, protegido/pupilo (protégé), déjà vu, a la carta, velada (soirée), canapé, paté, macramé, guardería (crèche), papel maché, ingenuo/ingenua, ingenuidad, aplique. Keep the paragraph natural — no forced insertions, no lists of unrelated things.",
       count_lexicon(ES_DIACRITICS), "each Spanish word whose English equivalent is a loanword that can carry an accent"),
    _f("latin_abbr",
       "English phrase vs Latin abbreviation (for example/e.g., that is/i.e., and so on/etc., versus/vs., approximately/approx.)",
       "Use at least 7 of these connectives, spread through the paragraph and written IN FULL in Spanish, with at least 4 different ones: «por ejemplo», «es decir», «etcétera» or «y así sucesivamente» (after a short list), «frente a» or «contra» (as versus, comparing two options), «aproximadamente» or «unos/unas» directly before a number. Every one must be used in its normal sense inside a fluent sentence.",
       count_patterns(ES_LATINABBR), "each connective (por ejemplo, es decir, etcétera, frente a/contra, aproximadamente/unos)",
       "- Write connectives in full: «por ejemplo», «es decir», «etcétera» (never «p. ej.», «etc.», «aprox.»)."),
    _f("hyphen_compound",
       "closed vs hyphenated compound word (email/e-mail, online/on-line, website/web-site, wellbeing/well-being, cooperate/co-operate, reuse/re-use, homemade/home-made, nonstop/non-stop, lifestyle/life-style, workout/work-out, checkup/check-up, makeup/make-up, setup/set-up, backup/back-up, startup/start-up, coordinate/co-ordinate, preheat/pre-heat, multitask/multi-task, smartphone/smart-phone, teamwork/team-work, daytime/day-time, nighttime/night-time, weekend/week-end, sunscreen/sun-screen, toothbrush/tooth-brush, handmade/hand-made, overnight/over-night, timeline/time-line, username/user-name, login/log-in, offline/off-line, microwave/micro-wave, rainwater/rain-water, firewood/fire-wood, greenhouse/green-house, lawnmower/lawn-mower, wheelbarrow/wheel-barrow, handlebars/handle-bars, kickstand/kick-stand, headlight/head-light, taillight/tail-light, seatpost/seat-post, backpack/back-pack)",
       "Use at least 6 (ideally 6 to 7) DIFFERENT items from this list, spread through the paragraph: correo electrónico, en línea, sitio web/página web, bienestar, cooperar, reutilizar/reutilizable, casero/casera, sin parar, estilo de vida, entrenamiento, revisión médica/chequeo, maquillaje, configuración, copia de seguridad, empresa emergente, coordinar/coordinación, precalentar, multitarea, teléfono inteligente, trabajo en equipo, durante el día, durante la noche, fin de semana, protector solar, cepillo de dientes, hecho a mano, cronología, nombre de usuario, inicio de sesión, sin conexión, microondas, agua de lluvia, leña, invernadero, cortacésped, carretilla, manillar, pata de cabra (de la bici), faro delantero, luz trasera, tija (del sillín), mochila.",
       count_lexicon(ES_HYPHEN), "each Spanish term whose English equivalent is a compound that can be closed or hyphenated"),
    _f("flat_adverb",
       "-ly adverb vs flat adverb after a verb (drive slowly/drive slow, hold tightly/hold tight, brake gently/brake gentle, breathe deeply/breathe deep, speak quietly/speak quiet, cut thinly/cut thin, drive safely/drive safe, turn the screw firmly/firm)",
       "Use at least 6 (ideally 6 to 7) DIFFERENT manner adverbs from this list, each placed RIGHT AFTER its verb (or after a short object) to say how the action is done: despacio, rápido, con fuerza, suavemente, hondo, en voz alta, bajito, con cuidado, firmemente, directamente, barato, mal, fino, recién, claramente, fácilmente, de cerca, en silencio, uniformemente, con seguridad, ligeramente, brillante (e.g. «pedalea despacio», «sujeta el manillar con fuerza», «frena suavemente», «respira hondo», «habla bajito», «corta fino»). Do NOT rely on other -mente adverbs (regularmente, periódicamente, adecuadamente do not count).",
       count_patterns(ES_FLATADV), "each verb followed by one of the listed manner adverbs"),
    _f("irreg_past",
       "irregular vs regular English past form of a verb (dove/dived, snuck/sneaked, lit/lighted, pled/pleaded, sped/speeded, wove/weaved, shone/shined, strove/strived, knelt/kneeled)",
       "Write the paragraph as a FIRST-PERSON NARRATIVE of one past episode (all verbs in the pretérito indefinido, never the imperfecto for these actions), loosely connected to the topic, in which AT LEAST 6 of these 9 actions happen, each as a finished past event: (1) zambullirse/lanzarse al agua/bucear — «me zambullí», «se lanzó al agua»; (2) colarse/escabullirse/entrar a hurtadillas — «me colé», «se escabulló»; (3) encender una vela/una hoguera/una linterna — «encendí», «encendió»; (4) suplicar/rogar — «le rogué», «suplicó»; (5) acelerar/ir a toda velocidad — «aceleré», «salió a toda velocidad»; (6) tejer — «tejió», «tejimos»; (7) brillar (el sol, una luz) — «brilló», «brillaron»; (8) esforzarse — «me esforcé», «se esforzaron»; (9) arrodillarse — «me arrodillé», «se arrodilló». Make the story natural (a camping trip, a night hike, a rescue, a festival, a childhood memory that fits the topic).",
       count_patterns(ES_IRREGPAST), "each preterite verb from the list of 9 actions"),
    _f("latin_plural",
       "anglicised vs classical plural (indexes/indices, formulas/formulae, cactuses/cacti, appendixes/appendices, curriculums/curricula, stadiums/stadia, forums/fora, antennas/antennae, funguses/fungi, syllabuses/syllabi, octopuses/octopi, radiuses/radii, nucleuses/nuclei, memorandums/memoranda, referendums/referenda, millenniums/millennia, aquariums/aquaria, terrariums/terraria, vertexes/vertices, matrixes/matrices, larvas/larvae, nebulas/nebulae, vertebras/vertebrae, thesauruses/thesauri, gymnasiums/gymnasia, symposiums/symposia, hippopotamuses/hippopotami, alumnuses/alumni)",
       "Use at least 6 (ideally 7) DIFFERENT plural nouns from this list, always in the PLURAL and each used ONLY where it makes literal sense (never as a metaphor, joke or forced comparison): hongos, larvas and cactus (garden and plant care), acuarios and terrarios (pets), gimnasios and estadios (sport), fórmulas, índices, matrices and radios (a spreadsheet, a budget, geometry), foros (online discussion boards), temarios, planes de estudio and antiguos alumnos (a course or school), memorandos and simposios (an office or conference), antenas and núcleos (technical), vértebras (posture, back pain), nebulosas (stargazing), milenios (history), apéndices (documents). Set the topic in a context where 6–7 of these fit naturally (a school science club, a community garden, a gym, an evening course, a spreadsheet at work) and keep every sentence plausible.",
       count_lexicon(ES_LATINPLURAL), "each plural noun whose English plural has an anglicised and a classical form"),
    _f("title_abbr",
       "title or street word in full vs abbreviated before/after a name (Doctor Smith/Dr. Smith, Professor Lee/Prof. Lee, Mister Brown/Mr. Brown, Mount Fuji/Mt. Fuji, Saint Paul/St. Paul, Main Street/Main St., Fifth Avenue/Fifth Ave., Oak Road/Oak Rd., Sunset Boulevard/Sunset Blvd.)",
       "Mention at least 7 DIFFERENT named people or places with a title or street word written IN FULL in Spanish and immediately followed by a capitalised name: «el doctor García», «la doctora Ruiz», «el profesor Ortega», «el señor Pérez», «el monte Perdido», «san Isidro» / «santa Clara» (a church, square or town named after a saint), «la calle Mayor», «la avenida Libertad», «la carretera Nacional», «el bulevar Central». Use invented but plausible names, at least 4 different kinds of title/street word, and never abbreviate (no Dr., Sr., Avda.).",
       count_titles, "each title or street word followed by a proper name",
       "- Titles and street words always in full: «doctor», «profesor», «señor», «monte», «san»/«santa», «calle», «avenida» (never «Dr.», «Sr.», «c/», «avda.»)."),
]
FAMILY = {f.name: f for f in FAMILIES}
assert len(FAMILIES) == 27 and set(FAMILY) == set(PROPS)
