"""Property registry for the free-form style-property read/write study.

Each *property* is a binary stylistic convention with two polarities:
  * nat — the US-standard / typographically plain pole (the base-corpus convention)
  * alt — the toggled pole (all-lowercase sentence starts, UK spelling, double space, ...)

A property provides:
  * find_opps(base_text) -> [Opp]: character-span *opportunity sites* in the base text,
    each with its two renderings (nat/alt). Detectors match BOTH surface forms, so a base
    doc is fully polarity-consistent after rendering regardless of the form it was
    generated with.
  * classify(tail) -> "nat" | "alt" | None: property-level classification of a sampled
    continuation at a cue token, applied AFTER the strict expected-continuation
    prefix match fails (loose fallback; None = unscorable). Properties without a
    meaningful loose rule return None.

Rendering (render()) applies the opp substitutions and returns the text plus the
rendered char span of every opp, so downstream code can locate cue/evidence tokens
via fast-tokenizer offset mappings (tokenizer-verified positions — DECISIONS 2026-07-13).

Terminology mapping to the 69-task study: opportunity sites where the property has
manifested = evidence (read) sites; the last token before a site's nat/alt divergence =
the cue (write-site) token, identity-matched across the twin pair by construction.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Opp:
    start: int          # char span in the BASE text
    end: int
    nat: str            # rendering under the nat polarity
    alt: str            # rendering under the alt polarity

    @property
    def div(self) -> int:
        """Chars of common prefix between the two renderings (divergence offset)."""
        n = 0
        for a, b in zip(self.nat, self.alt):
            if a != b:
                break
            n += 1
        return n


def _dedup(opps):
    """Sort and drop overlapping opportunities (keep the earlier one)."""
    out, last_end = [], -1
    for o in sorted(opps, key=lambda o: (o.start, o.end)):
        if o.start >= last_end and o.nat != o.alt:
            out.append(o)
            last_end = o.end
    return out


def render(base_text: str, opps, polarity: str):
    """Apply the opp substitutions for one polarity.
    Returns (text, spans) where spans[i] = (char_start, char_end) of opp i in the
    rendered text, plus base->rendered char offset shifts handled internally."""
    assert polarity in ("nat", "alt")
    parts, spans, pos, shift = [], [], 0, 0
    for o in opps:
        rep = o.nat if polarity == "nat" else o.alt
        parts.append(base_text[pos:o.start])
        spans.append((o.start + shift, o.start + shift + len(rep)))
        parts.append(rep)
        shift += len(rep) - (o.end - o.start)
        pos = o.end
    parts.append(base_text[pos:])
    return "".join(parts), spans


class Property:
    name = ""
    family = ""
    nat_label = ""      # human description of the nat pole
    alt_label = ""
    confound = "low"    # register/persona leakage rating (Stage-0 spec sheet)
    max_new_cap = 16

    def find_opps(self, text):
        raise NotImplementedError

    def classify(self, tail):
        """Loose property-level classification of the continuation past the shared
        prefix; None = unscorable / no loose rule."""
        return None

    def resample(self, opp, rng):
        """Optionally replace the lexical ITEM at an opportunity (same item in both
        twins) to decorrelate item identity from document position / k. Default:
        identity. Number properties override (user decision 2026-09-02: at every k the
        items must be equally distributed — 'first' otherwise dominates k=0 and the high
        ordinals only appear deep in ordinal-rich documents)."""
        return opp


# --------------------------------------------------------------------------- case
class SentenceCaps(Property):
    name = "sentence_caps"
    family = "case"
    nat_label = "standard sentence-initial capitalization"
    alt_label = "lowercase sentence starts"
    # first letter of each sentence (incl. document start); skip acronym-initial words
    _re = re.compile(r"(?:^|(?<=[.!?] )|(?<=[.!?]\n)|(?<=\n\n))([A-Za-z])(?![A-Z])")
    _abbrev = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|St|vs|etc|e\.g|i\.e|Jr|Sr|No|Fig)\.\s$")

    def find_opps(self, text):
        return _dedup([Opp(m.start(1), m.end(1), m.group(1).upper(), m.group(1).lower())
                       for m in self._re.finditer(text)
                       if not self._abbrev.search(text[:m.start(1)])])

    def classify(self, tail):
        for c in tail:
            if c.isalpha():
                return "nat" if c.isupper() else "alt"
            if not c.isspace() and c not in "\"'“”(":
                return None
        return None


class AllCaps(Property):
    name = "all_caps"
    family = "case"
    nat_label = "standard case"
    alt_label = "ENTIRE TEXT IN CAPITALS"
    confound = "high-tokenizer-divergence"
    max_new_cap = 10
    _re = re.compile(r"[^.!?\n]+[.!?]?")

    def find_opps(self, text):
        opps = []
        for m in self._re.finditer(text):
            seg = m.group(0)
            low = seg.lower()
            # canonical nat rendering: the segment as written (assumed standard case)
            if seg.strip() and seg != seg.upper():
                opps.append(Opp(m.start(), m.end(), seg, seg.upper()))
            elif seg.strip() and seg == seg.upper() and low != seg:
                # base sentence already all-caps: nat = capitalized-normal is unknowable;
                # skip (corpus is generated in standard case, so this is rare noise).
                continue
        return _dedup(opps)

    def classify(self, tail):
        alpha = [c for c in tail if c.isalpha()]
        if len(alpha) < 3:
            return None
        frac = sum(c.isupper() for c in alpha) / len(alpha)
        if frac >= 0.8:
            return "alt"
        if frac <= 0.4:
            return "nat"
        return None


# ----------------------------------------------------------------- spelling lexicons
def _capvariants(us, uk):
    yield us, uk
    yield us.capitalize(), uk.capitalize()


def _lexicon_property(cls_name, prop_name, pairs, nat_lab, alt_lab, confound="low"):
    """Build a Property subclass toggling word pairs (nat=first, alt=second)."""
    table = {}
    for us, uk in pairs:
        for a, b in _capvariants(us, uk):
            table[a] = (a, b)
            table[b] = (a, b)
    rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, table), key=len, reverse=True)) + r")\b")

    class L(Property):
        name = prop_name
        family = "spelling"

        def find_opps(self, text):
            opps = []
            for m in rx.finditer(text):
                nat, alt = table[m.group(1)]
                opps.append(Opp(m.start(1), m.end(1), nat, alt))
            return _dedup(opps)

        def classify(self, tail):
            # loose: ANY lexicon form in the continuation reveals the polarity
            m = rx.search(tail)
            if m is None:
                return None
            nat, alt = table[m.group(1)]
            return "nat" if m.group(1) in (nat, nat.capitalize()) else "alt"

    L.__name__ = cls_name
    L.nat_label, L.alt_label, L.confound = nat_lab, alt_lab, confound
    return L


# US/UK pairs (harvested from create_ambiguous_datasets.US_UK_PAIRS, curated:
# sense-ambiguous pairs removed — check/cheque, tire/tyre, curb/kerb, mold/mould,
# program/programme, mom/mum, donut, cozy; the z/s family moved to ise_ize).
_US_UK = [
    ("color", "colour"), ("colors", "colours"), ("colored", "coloured"),
    ("colorful", "colourful"), ("flavor", "flavour"), ("flavors", "flavours"),
    ("flavored", "flavoured"), ("favor", "favour"), ("favors", "favours"),
    ("favorite", "favourite"), ("favorites", "favourites"), ("honor", "honour"),
    ("honored", "honoured"), ("humor", "humour"), ("labor", "labour"),
    ("neighbor", "neighbour"), ("neighbors", "neighbours"),
    ("neighborhood", "neighbourhood"), ("neighboring", "neighbouring"),
    ("rumor", "rumour"), ("rumors", "rumours"), ("vapor", "vapour"),
    ("behavior", "behaviour"), ("behaviors", "behaviours"), ("harbor", "harbour"),
    ("odor", "odour"), ("odors", "odours"), ("vigor", "vigour"),
    ("center", "centre"), ("centers", "centres"), ("theater", "theatre"),
    ("theaters", "theatres"), ("liter", "litre"), ("liters", "litres"),
    ("meter", "metre"), ("meters", "metres"), ("fiber", "fibre"),
    ("fibers", "fibres"), ("catalog", "catalogue"), ("catalogs", "catalogues"),
    ("dialog", "dialogue"), ("defense", "defence"), ("offense", "offence"),
    ("traveler", "traveller"), ("travelers", "travellers"),
    ("traveling", "travelling"), ("traveled", "travelled"),
    ("labeled", "labelled"), ("labeling", "labelling"),
    ("modeled", "modelled"), ("modeling", "modelling"),
    ("canceled", "cancelled"), ("jewelry", "jewellery"), ("gray", "grey"),
    ("aluminum", "aluminium"), ("pajamas", "pyjamas"), ("artifact", "artefact"),
    ("artifacts", "artefacts"),
]

# -ize/-ise family (incl. -yze/-yse and -ization/-isation), generated from US forms.
_IZE_STEMS = [
    "organize", "organizes", "organized", "organizing", "organization", "organizations",
    "recognize", "recognizes", "recognized", "recognizing",
    "realize", "realizes", "realized", "realizing",
    "apologize", "apologized", "apologizing",
    "criticize", "criticized", "emphasize", "emphasizes", "emphasized",
    "summarize", "summarized", "minimize", "minimized", "maximize", "maximized",
    "prioritize", "prioritized", "specialize", "specializes", "specialized",
    "characterize", "characterized", "categorize", "categorized",
    "standardize", "standardized", "utilize", "utilized", "utilizing",
    "memorize", "memorized", "finalize", "finalized", "generalize", "generalized",
    "analyze", "analyzes", "analyzed", "analyzing", "paralyze", "paralyzed",
]
_IZE = [(w, w.replace("iz", "is") if "iz" in w else w.replace("yz", "ys"))
        for w in _IZE_STEMS]

_T_PAST = [
    ("learned", "learnt"), ("spelled", "spelt"), ("burned", "burnt"),
    ("dreamed", "dreamt"), ("leaped", "leapt"), ("leaned", "leant"),
    ("spilled", "spilt"), ("spoiled", "spoilt"),
]

_WHILST = [("while", "whilst"), ("among", "amongst"), ("amid", "amidst")]

UsUk = _lexicon_property("UsUk", "us_uk", _US_UK,
                         "American spelling", "British spelling")
IseIze = _lexicon_property("IseIze", "ise_ize", _IZE,
                           "-ize/-yze spellings", "-ise/-yse spellings")
TPast = _lexicon_property("TPast", "brit_t_past", _T_PAST,
                          "-ed past forms", "-t past forms (learnt, spelt)")


class Whilst(Property):
    name = "whilst"
    family = "spelling"
    nat_label = "while/among/amid"
    alt_label = "whilst/amongst/amidst"
    confound = "medium-register"
    # exclude noun uses "a while", "the while", "worth while"
    _rx = re.compile(r"(?<!\ba )(?<!\bthe )\b(while|whilst|among|amongst|amid|amidst"
                     r"|While|Whilst|Among|Amongst|Amid|Amidst)\b")
    _map = {}
    for a, b in _WHILST:
        for x, y in _capvariants(a, b):
            _map[x] = (x, y)
            _map[y] = (x, y)

    def find_opps(self, text):
        return _dedup([Opp(m.start(1), m.end(1), *self._map[m.group(1)])
                       for m in self._rx.finditer(text)])

    def classify(self, tail):
        m = self._rx.search(tail)
        if m is None:
            return None
        nat, _ = self._map[m.group(1)]
        return "nat" if m.group(1) in (nat, nat.capitalize()) else "alt"


# --------------------------------------------------------- punctuation / typography
class DoubleSpace(Property):
    name = "double_space"
    family = "typography"
    nat_label = "one space after sentence-final period"
    alt_label = "two spaces after sentence-final period"
    max_new_cap = 8
    _rx = re.compile(r"(?<=[.!?])( {1,2})(?=[A-Z\"“])")

    def find_opps(self, text):
        return _dedup([Opp(m.start(1), m.end(1), " ", "  ") for m in self._rx.finditer(text)])

    def classify(self, tail):
        if tail.startswith("  "):
            return "alt"
        if tail.startswith(" ") and len(tail) > 1 and not tail[1].isspace():
            return "nat"
        return None


class OxfordComma(Property):
    name = "oxford_comma"
    family = "typography"
    nat_label = "serial (Oxford) comma before and/or"
    alt_label = "no serial comma"
    # X, Y(,) and Z  — require a preceding comma-separated item so it's a real 3-list
    _rx = re.compile(r"\b[\w'-]+, [\w'-]+(,?) (and|or)\b")

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            conj = m.group(2)
            opps.append(Opp(m.start(1), m.end(2), f", {conj}", f" {conj}"))
        return _dedup(opps)

    def classify(self, tail):
        for pre, lab in ((", and", "nat"), (", or", "nat"), (" and", "alt"), (" or", "alt")):
            if tail.startswith(pre):
                return lab
        return None


class CurlyQuotes(Property):
    name = "curly_quotes"
    family = "typography"
    nat_label = 'straight quotation marks (")'
    alt_label = "curly quotation marks (“ ”)"

    def find_opps(self, text):
        opps, open_ = [], True
        for i, c in enumerate(text):
            if c == '"':
                # decide open/close from context: opening if preceded by space/start
                is_open = i == 0 or text[i - 1].isspace() or text[i - 1] in "(—-"
                opps.append(Opp(i, i + 1, '"', "“" if is_open else "”"))
            elif c == "“":
                opps.append(Opp(i, i + 1, '"', "“"))
            elif c == "”":
                opps.append(Opp(i, i + 1, '"', "”"))
        return _dedup(opps)

    def classify(self, tail):
        for c in tail:
            if c == '"':
                return "nat"
            if c in "“”":
                return "alt"
        return None


class EmDash(Property):
    name = "em_dash"
    family = "typography"
    nat_label = "attached em dash (word—word)"
    alt_label = "spaced hyphen (word - word)"
    _rx = re.compile(r"\s?[—–]\s?| - ")

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            # skip numeric ranges like "3 - 5" / "3–5"
            l = text[max(0, m.start() - 1):m.start()]
            r = text[m.end():m.end() + 1]
            if l.isdigit() and r.isdigit():
                continue
            opps.append(Opp(m.start(), m.end(), "—", " - "))
        return _dedup(opps)

    def classify(self, tail):
        i_em, i_hy = tail.find("—"), tail.find(" - ")
        if i_em == -1 and i_hy == -1:
            return None
        if i_hy == -1 or (i_em != -1 and i_em < i_hy):
            return "nat"
        return "alt"


class Ellipsis3(Property):
    name = "ellipsis"
    family = "typography"
    nat_label = "three-dot ellipsis (...)"
    alt_label = "single-glyph ellipsis (…)"
    _rx = re.compile(r"\.\.\.|…")

    def find_opps(self, text):
        return _dedup([Opp(m.start(), m.end(), "...", "…") for m in self._rx.finditer(text)])

    def classify(self, tail):
        i3, i1 = tail.find("..."), tail.find("…")
        if i3 == -1 and i1 == -1:
            return None
        if i1 == -1 or (i3 != -1 and i3 < i1):
            return "nat"
        return "alt"


class QuotePunct(Property):
    name = "quote_punct"
    family = "typography"
    nat_label = "period/comma inside closing quote (US)"
    alt_label = "period/comma outside closing quote (UK)"
    _rx = re.compile(r'([,.])(["”])|(["”])([,.])')

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            if m.group(1) is not None:
                p = m.group(1)
            else:
                p = m.group(4)
            opps.append(Opp(m.start(), m.end(), f'{p}"', f'"{p}'))
        return _dedup(opps)

    def classify(self, tail):
        m = re.search(r'[,.]["”]|["”][,.]', tail)
        if m is None:
            return None
        return "nat" if m.group(0)[0] in ",." else "alt"


# ------------------------------------------------------------------ number rendering
_NUM_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
              "seventeen", "eighteen", "nineteen", "twenty"]
_W2D = {w: str(i) for i, w in enumerate(_NUM_WORDS, start=2)}
_D2W = {v: k for k, v in _W2D.items()}


class NumWords(Property):
    name = "num_words"
    family = "number"
    nat_label = "cardinals 2-20 as digits"
    alt_label = "cardinals 2-20 spelled out"
    _rx = re.compile(r"(?<![\d.$:/-])\b(" + "|".join(list(_D2W) + _NUM_WORDS + [w.capitalize() for w in _NUM_WORDS])
                     + r")\b(?![\d.:%/-])(?! ?(?:%|percent|st\b|nd\b|rd\b|th\b|o'clock))")

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            g = m.group(1)
            if g in _D2W:
                nat, alt = g, _D2W[g]
            else:
                nat, alt = _W2D[g.lower()], g.lower()
            opps.append(Opp(m.start(1), m.end(1), nat, alt))
        return _dedup(opps)

    def resample(self, opp, rng):
        """Uniform cardinal in 2..20 (item balanced across k); keeps capitalization."""
        n = int(rng.integers(2, 21))
        word = _D2W[str(n)]
        if opp.alt[:1].isupper():
            word = word.capitalize()
        return Opp(opp.start, opp.end, str(n), word)

    # classifier mirrors the opportunity detector's scope: digits count only if the value
    # is 2-20 AND not in an excluded context (times 5:30, percentages, decimals, ranges,
    # ordinals, AM/PM). Before this fix any digit scored "nat", so alt-context docs were
    # penalised for "1,500 books" / "5:30" — numbers the convention doesn't govern — and the
    # penalty grew with k in number-dense documents (2026-09-01 audit).
    _cls_rx = re.compile(r"^\s*(\d+|[A-Za-z]+)(.{0,6})", re.S)
    _excl_rx = re.compile(r"\s*(:|%|\.\d|/|-|,\d|\d|st\b|nd\b|rd\b|th\b|o'clock|\s?[AaPp]\.?[Mm]\b)")

    def classify(self, tail):
        m = self._cls_rx.match(tail)
        if not m:
            return None
        tok, after = m.group(1), m.group(2)
        if tok.isdigit():
            if not (2 <= int(tok) <= 20) or self._excl_rx.match(after):
                return None
            return "nat"
        if tok.lower() in _W2D:
            return "alt"
        return None


class PercentSign(Property):
    name = "percent_sign"
    family = "number"
    nat_label = "N% with the sign"
    alt_label = "N percent spelled out"
    _rx = re.compile(r"\d(%| ?percent)\b|\d(%)")

    def find_opps(self, text):
        opps = []
        for m in re.finditer(r"\d(%| percent\b)", text):
            opps.append(Opp(m.start(1), m.end(1), "%", " percent"))
        return _dedup(opps)

    def classify(self, tail):
        if tail.startswith("%"):
            return "nat"
        if tail.startswith(" percent"):
            return "alt"
        return None


_ORD_D = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
_ORD_W = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
          "eighth", "ninth", "tenth"]


class OrdinalWords(Property):
    name = "ordinal_words"
    family = "number"
    nat_label = "digit ordinals (3rd)"
    alt_label = "spelled ordinals (third)"
    _map = {}
    for d, w in zip(_ORD_D, _ORD_W):
        _map[d] = (d, w)
        _map[w] = (d, w)
        _map[w.capitalize()] = (d, w)
    _rx = re.compile(r"\b(" + "|".join(_ORD_D + _ORD_W + [w.capitalize() for w in _ORD_W]) + r")\b")

    def resample(self, opp, rng):
        """Uniform ordinal in 1st..10th (item balanced across k); keeps capitalization."""
        i = int(rng.integers(0, len(_ORD_D)))
        word = _ORD_W[i]
        if opp.alt[:1].isupper():
            word = word.capitalize()
        return Opp(opp.start, opp.end, _ORD_D[i], word)

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            nat, alt = self._map[m.group(1)]
            opps.append(Opp(m.start(1), m.end(1), nat, alt))
        return _dedup(opps)

    def classify(self, tail):
        m = self._rx.search(tail)
        if m is None:
            return None
        nat, _ = self._map[m.group(1)]
        return "nat" if m.group(1) == nat else "alt"


# ------------------------------------------------------------------------- lexical
_CONTRACTIONS = [
    ("don't", "do not"), ("doesn't", "does not"), ("didn't", "did not"),
    ("isn't", "is not"), ("aren't", "are not"), ("wasn't", "was not"),
    ("weren't", "were not"), ("can't", "cannot"), ("couldn't", "could not"),
    ("wouldn't", "would not"), ("shouldn't", "should not"), ("won't", "will not"),
    ("hasn't", "has not"), ("haven't", "have not"), ("hadn't", "had not"),
    ("it's", "it is"), ("that's", "that is"), ("there's", "there is"),
    ("they're", "they are"), ("we're", "we are"), ("you're", "you are"),
    ("I'm", "I am"), ("I've", "I have"), ("we've", "we have"),
    ("you've", "you have"), ("they've", "they have"),
    ("I'll", "I will"), ("we'll", "we will"), ("you'll", "you will"),
    ("they'll", "they will"), ("isn't", "is not"),
]


class Contractions(Property):
    name = "contractions"
    family = "lexical"
    nat_label = "contracted forms (don't)"
    alt_label = "expanded forms (do not)"
    confound = "medium-register"
    _map = {}
    for c, e in _CONTRACTIONS:
        for a, b in {(c, e), (c.capitalize(), e.capitalize())}:
            _map[a] = (a, b)
            _map[b] = (a, b)
    _rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _map), key=len, reverse=True)) + r")\b")

    def find_opps(self, text):
        return _dedup([Opp(m.start(1), m.end(1), *self._map[m.group(1)])
                       for m in self._rx.finditer(text)])

    def classify(self, tail):
        m = self._rx.search(tail)
        if m is None:
            return None
        nat, _ = self._map[m.group(1)]
        return "nat" if m.group(1) in (nat, nat.capitalize()) else "alt"


class Ampersand(Property):
    name = "ampersand"
    family = "lexical"
    nat_label = "the word and"
    alt_label = "an ampersand (&)"
    confound = "medium-register"
    _rx = re.compile(r"(?<=[a-zA-Z0-9])( (?:and|&) )(?=[a-zA-Z0-9])")

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            if text[m.start(1) - 1] == ",":   # leave serial-comma lists to oxford_comma
                continue
            opps.append(Opp(m.start(1), m.end(1), " and ", " & "))
        return _dedup(opps)

    def classify(self, tail):
        if tail.startswith(" and"):
            return "nat"
        if tail.startswith(" &"):
            return "alt"
        return None


ALL_PROPERTIES = [
    SentenceCaps(), AllCaps(),
    UsUk(), IseIze(), TPast(), Whilst(),
    DoubleSpace(), OxfordComma(), CurlyQuotes(), EmDash(), Ellipsis3(), QuotePunct(),
    NumWords(), PercentSign(), OrdinalWords(),
    Contractions(), Ampersand(),
]
PROPS = {p.name: p for p in ALL_PROPERTIES}
assert len(PROPS) == len(ALL_PROPERTIES)
