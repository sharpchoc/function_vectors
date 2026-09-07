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
]
FAMILY = {f.name: f for f in FAMILIES}
assert len(FAMILIES) == 17 and set(FAMILY) == set(PROPS)
