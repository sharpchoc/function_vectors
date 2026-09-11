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




def _at_sentence_start(text, pos):
    """True if a token at char `pos` opens a sentence (start of text, or after . ! ? possibly
    followed by a closing quote). Used so spelled-out numbers/ordinals keep sentence caps."""
    prev = text[:pos].rstrip()
    if not prev:
        return True
    if prev[-1] in ".!?":
        return True
    return prev[-1] in '"\u201d' and len(prev) > 1 and prev[-2] in ".!?"

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
            if _at_sentence_start(text, m.start(1)):      # "3 days." -> "Three days." (2026-09-07)
                alt = alt[:1].upper() + alt[1:]
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
        # USER DECISION 2026-09-07: a space before the sign ("50 %", copied from the Spanish
        # typography) still counts as the sign convention; "per cent" counts as spelled out.
        if re.match(r"\s?%", tail):
            return "nat"
        if re.match(r"\s?per\s?cent\b", tail):
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
            if _at_sentence_start(text, m.start(1)):      # "1st, ..." -> "First, ..." (2026-09-07)
                alt = alt[:1].upper() + alt[1:]
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
        opps = []
        for m in self._rx.finditer(text):
            nat, alt = self._map[m.group(1)]
            # the modal "have to" (you have to / you've to) is not a contraction site:
            # contracting it is ungrammatical (2026-09-07, found by the English verifier)
            if alt.lower().endswith(" have") and re.match(r"\s+to\b", text[m.end(1):]):
                continue
            # the connective ", that is," (= i.e.) is never contracted (latin_abbr family, 2026-09-11)
            if alt.lower() == "that is" and re.match(r",", text[m.end(1):]) \
                    and (m.start(1) == 0 or re.search(r"[,;(.!?]\s*$", text[max(0, m.start(1) - 3):m.start(1)])):
                continue
            opps.append(Opp(m.start(1), m.end(1), nat, alt))
        return _dedup(opps)

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



# ================================================================ lexically diverse families (2026-09-11)
# User decision: ten more families where the convention applies across many different words.
# nat = house style (standard American English), alt = the flipped convention. Only sense-unambiguous
# items are kept so that both detection directions are safe.

_UK_VOCAB = [   # (US, UK)
    ("truck", "lorry"), ("trucks", "lorries"), ("vacation", "holiday"), ("vacations", "holidays"),
    ("trash", "rubbish"), ("trash can", "rubbish bin"), ("gasoline", "petrol"),
    ("flashlight", "torch"), ("flashlights", "torches"), ("cell phone", "mobile phone"),
    ("cell phones", "mobile phones"), ("sidewalk", "pavement"), ("sidewalks", "pavements"),
    ("cookie", "biscuit"), ("cookies", "biscuits"), ("sweater", "jumper"), ("sweaters", "jumpers"),
    ("pants", "trousers"), ("diaper", "nappy"), ("diapers", "nappies"), ("closet", "wardrobe"),
    ("closets", "wardrobes"), ("eggplant", "aubergine"), ("eggplants", "aubergines"),
    ("zucchini", "courgette"), ("zucchinis", "courgettes"), ("cilantro", "coriander"),
    ("math", "maths"), ("highway", "motorway"), ("highways", "motorways"),
    ("parking lot", "car park"), ("parking lots", "car parks"), ("gas station", "petrol station"),
    ("gas stations", "petrol stations"), ("shopping cart", "trolley"), ("shopping carts", "trolleys"),
    ("stroller", "pushchair"), ("strollers", "pushchairs"), ("crib", "cot"), ("cribs", "cots"),
    ("windshield", "windscreen"), ("wrench", "spanner"), ("wrenches", "spanners"),
    ("fender", "mudguard"), ("fenders", "mudguards"), ("counterclockwise", "anticlockwise"),
    ("oatmeal", "porridge"), ("takeout", "takeaway"), ("zip code", "postcode"), ("zip codes", "postcodes"),
    ("subway", "underground"), ("railroad", "railway"), ("license plate", "number plate"),
    ("mailbox", "postbox"), ("mailboxes", "postboxes"), ("mailman", "postman"), ("vest", "waistcoat"),
    ("vests", "waistcoats"), ("faucet", "tap"), ("faucets", "taps"), ("drugstore", "chemist's"),
    ("apartment", "flat"), ("apartments", "flats"), ("elevator", "lift"), ("elevators", "lifts"),
]
# NOTE: "flat", "lift", "tap" are ambiguous as ALT forms (adjective / verb); the classifier only sees the
# continuation after a cue whose Spanish source names the object, so the risk is accepted (audited).

_REGISTER = [   # (plain, formal) incl. inflections; phrasal verb <-> single verb
    ("begin", "commence"), ("begins", "commences"), ("began", "commenced"), ("beginning", "commencing"),
("buy", "purchase"), ("buys", "purchases"), ("bought", "purchased"),
    ("buying", "purchasing"), ("try", "attempt"), ("tries", "attempts"), ("tried", "attempted"),
    ("trying", "attempting"), ("live", "reside"), ("lives", "resides"), ("lived", "resided"),
    ("living", "residing"), ("check", "verify"), ("checks", "verifies"), ("checked", "verified"),
    ("checking", "verifying"), ("fix", "repair"), ("fixes", "repairs"), ("fixed", "repaired"),
    ("fixing", "repairing"), ("kids", "children"), ("kid", "child"), ("show", "demonstrate"),
    ("shows", "demonstrates"), ("showed", "demonstrated"), ("showing", "demonstrating"),
    ("later", "subsequently"), ("also", "additionally"), ("maybe", "perhaps"), ("big", "large"),
    ("bigger", "larger"), ("biggest", "largest"), ("find out", "discover"), ("finds out", "discovers"),
    ("found out", "discovered"), ("finding out", "discovering"),     ("look into", "investigate"), ("looks into", "investigates"), ("looked into", "investigated"),
    ("looking into", "investigating"), ("throw away", "discard"),
    ("throws away", "discards"), ("threw away", "discarded"), ("throwing away", "discarding"),
    ("figure out", "determine"), ("figures out", "determines"), ("figured out", "determined"),
    ("figuring out", "determining"), ("take part", "participate"), ("takes part", "participates"),
    ("took part", "participated"), ("taking part", "participating"), ("get rid of", "eliminate"),
    ("gets rid of", "eliminates"), ("got rid of", "eliminated"), ("getting rid of", "eliminating"),
    ("look for", "seek"), ("looks for", "seeks"), ("looked for", "sought"), ("looking for", "seeking"),
]

_DIACRITICS = [   # (plain ASCII, accented)
    ("cafe", "café"), ("cafes", "cafés"), ("naive", "naïve"), ("cliche", "cliché"), ("cliches", "clichés"),
    ("fiance", "fiancé"), ("fiancee", "fiancée"), ("facade", "façade"), ("facades", "façades"),
    ("jalapeno", "jalapeño"), ("jalapenos", "jalapeños"), ("decor", "décor"), ("saute", "sauté"),
    ("sauteed", "sautéed"), ("sauteing", "sautéing"), ("puree", "purée"), ("pureed", "puréed"),
    ("creme", "crème"), ("entree", "entrée"), ("entrees", "entrées"), ("matinee", "matinée"),
    ("debut", "début"), ("elite", "élite"), ("senor", "señor"), ("senora", "señora"), ("pinata", "piñata"),
    ("pinatas", "piñatas"), ("protege", "protégé"), ("deja vu", "déjà vu"), ("a la carte", "à la carte"),
    ("soiree", "soirée"), ("canape", "canapé"), ("canapes", "canapés"), ("pate", "pâté"),
    ("macrame", "macramé"), ("applique", "appliqué"), ("fete", "fête"), ("creche", "crèche"),
    ("papier-mache", "papier-mâché"), ("naivete", "naïveté"), ("vis-a-vis", "vis-à-vis"), ("blase", "blasé"),
]

_HYPHEN = [   # (closed compound, hyphenated)
    ("email", "e-mail"), ("emails", "e-mails"), ("emailed", "e-mailed"), ("online", "on-line"),
    ("website", "web-site"), ("websites", "web-sites"), ("wellbeing", "well-being"),
    ("cooperate", "co-operate"), ("cooperation", "co-operation"), ("cooperative", "co-operative"),
    ("reuse", "re-use"), ("reused", "re-used"), ("reusable", "re-usable"), ("reusing", "re-using"),
    ("homemade", "home-made"), ("nonstop", "non-stop"), ("lifestyle", "life-style"), ("workout", "work-out"),
    ("workouts", "work-outs"), ("checkup", "check-up"), ("checkups", "check-ups"), ("makeup", "make-up"),
    ("setup", "set-up"), ("backup", "back-up"), ("backups", "back-ups"), ("startup", "start-up"),
    ("startups", "start-ups"), ("coordinate", "co-ordinate"), ("coordinates", "co-ordinates"),
    ("coordinated", "co-ordinated"), ("coordination", "co-ordination"), ("preheat", "pre-heat"),
    ("preheated", "pre-heated"), ("multitask", "multi-task"), ("multitasking", "multi-tasking"),
    ("smartphone", "smart-phone"), ("smartphones", "smart-phones"), ("teamwork", "team-work"),
    ("daytime", "day-time"), ("nighttime", "night-time"), ("weekend", "week-end"), ("weekends", "week-ends"),
    ("sunscreen", "sun-screen"), ("toothbrush", "tooth-brush"), ("handmade", "hand-made"),
    ("overnight", "over-night"), ("timeline", "time-line"), ("username", "user-name"), ("login", "log-in"),
    ("offline", "off-line"), ("microwave", "micro-wave"), ("rainwater", "rain-water"),
    ("firewood", "fire-wood"), ("greenhouse", "green-house"), ("lawnmower", "lawn-mower"),
    ("wheelbarrow", "wheel-barrow"), ("handlebars", "handle-bars"), ("kickstand", "kick-stand"),
    ("headlight", "head-light"), ("headlights", "head-lights"), ("taillight", "tail-light"),
    ("seatpost", "seat-post"), ("chainring", "chain-ring"), ("backpack", "back-pack"), ("backpacks", "back-packs"),
]

_IRREG_PAST = [   # (irregular = standard American, regular)
    ("dove", "dived"), ("snuck", "sneaked"), ("lit", "lighted"), ("pled", "pleaded"), ("sped", "speeded"),
    ("wove", "weaved"), ("shone", "shined"), ("strove", "strived"), ("knelt", "kneeled"),
]

_LATIN_PLURAL = [   # (anglicised, classical)
    ("indexes", "indices"), ("formulas", "formulae"), ("cactuses", "cacti"), ("appendixes", "appendices"),
    ("curriculums", "curricula"), ("stadiums", "stadia"), ("forums", "fora"), ("antennas", "antennae"),
    ("syllabuses", "syllabi"), ("octopuses", "octopi"), ("radiuses", "radii"),
    ("memorandums", "memoranda"), ("referendums", "referenda"),
    ("millenniums", "millennia"), ("aquariums", "aquaria"), ("terrariums", "terraria"),
    ("vertexes", "vertices"), ("matrixes", "matrices"), ("nebulas", "nebulae"),
    ("thesauruses", "thesauri"), ("gymnasiums", "gymnasia"),
    ("symposiums", "symposia"), ("hippopotamuses", "hippopotami")]

UkVocab = _lexicon_property("UkVocab", "uk_vocab", _UK_VOCAB, "American vocabulary (truck, vacation, trash)",
                            "British vocabulary (lorry, holiday, rubbish)", confound="medium")
Register = _lexicon_property("Register", "register", _REGISTER, "plain everyday words (begin, buy, kids, find out)",
                             "formal/Latinate words (commence, purchase, children, discover)", confound="medium-register")
Diacritics = _lexicon_property("Diacritics", "diacritics", _DIACRITICS, "loanwords without accents (cafe, naive)",
                               "loanwords with accents (café, naïve)")
HyphenCompound = _lexicon_property("HyphenCompound", "hyphen_compound", _HYPHEN, "closed compounds (email, online)",
                                   "hyphenated compounds (e-mail, on-line)")
IrregPast = _lexicon_property("IrregPast", "irreg_past", _IRREG_PAST, "irregular past forms (dove, snuck, lit)",
                              "regular past forms (dived, sneaked, lighted)")
LatinPlural = _lexicon_property("LatinPlural", "latin_plural", _LATIN_PLURAL, "anglicised plurals (indexes, formulas)",
                                "classical plurals (indices, formulae)")
for _c in (UkVocab, Register, Diacritics, HyphenCompound, IrregPast, LatinPlural):
    _c.family = "lexical"


class UnitAbbr(Property):
    """5 km / 2 kg / 30 ml / 15 °C (nat, symbol with a space) vs 5 kilometers / 2 kilograms (alt)."""
    name = "unit_abbr"
    family = "lexical"
    nat_label = "unit symbols after the number (5 km, 2 kg, 15 °C)"
    alt_label = "units spelled out (5 kilometers, 2 kilograms, 15 degrees Celsius)"
    _units = {"km": "kilometer", "cm": "centimeter", "mm": "millimeter", "kg": "kilogram", "g": "gram",
              "ml": "milliliter", "°C": "degrees Celsius"}
    _words = {v: k for k, v in _units.items()}
    _rx_nat = re.compile(r"(\d+(?:[.,]\d+)?)( ?)(km|cm|mm|kg|g|ml|°C)(?![\w°])")
    _rx_alt = re.compile(r"(\d+(?:[.,]\d+)?) (kilometers?|centimeters?|millimeters?|kilograms?|grams?|milliliters?|degrees Celsius)\b")

    def _alt_word(self, num, unit):
        base = self._units[unit]
        if base == "degrees Celsius":
            return base
        return base if num.replace(",", ".") in ("1", "1.0") else base + "s"

    def find_opps(self, text):
        opps = []
        for m in self._rx_nat.finditer(text):
            num, unit = m.group(1), m.group(3)
            opps.append(Opp(m.end(1), m.end(3), " " + unit, " " + self._alt_word(num, unit)))
        for m in self._rx_alt.finditer(text):
            num, word = m.group(1), m.group(2)
            base = word if word == "degrees Celsius" else word.rstrip("s") if word != "degrees Celsius" else word
            base = "degrees Celsius" if word == "degrees Celsius" else word[:-1] if word.endswith("s") else word
            unit = self._words.get(base)
            if unit:
                opps.append(Opp(m.end(1), m.end(2), " " + unit, " " + word))
        return _dedup(opps)

    _lead_nat = re.compile(r"^\s?(km|cm|mm|kg|g|ml|°C)(?![\w°])")
    _lead_alt = re.compile(r"^\s?(kilometers?|centimeters?|millimeters?|kilograms?|grams?|milliliters?|degrees Celsius)\b")

    def classify(self, tail):
        if self._lead_nat.match(tail):
            return "nat"
        if self._lead_alt.match(tail):
            return "alt"
        a, b = self._rx_nat.search(tail), self._rx_alt.search(tail)
        if a and (not b or a.start() <= b.start()):
            return "nat"
        if b:
            return "alt"
        return None


class LatinAbbr(Property):
    """for example / that is / and so on / versus / approximately (nat) vs e.g. / i.e. / etc. / vs. / approx. (alt)."""
    name = "latin_abbr"
    family = "lexical"
    nat_label = "English phrases (for example, that is, and so on, versus, approximately)"
    alt_label = "Latin abbreviations (e.g., i.e., etc., vs., approx.)"
    _pairs = [("for example", "e.g."), ("For example", "E.g."), ("that is", "i.e."), ("That is", "I.e."), ("and so on", "etc."),
              ("versus", "vs."), ("Versus", "Vs."), ("approximately", "approx."), ("Approximately", "Approx.")]
    _nat_rx = re.compile(r"\b(For example|for example|that is|That is|and so on|versus|Versus|approximately|Approximately)\b(?=[,;:.\s)]|$)")
    _alt_rx = re.compile(r"(?<![\w.])(E\.g\.|e\.g\.|i\.e\.|I\.e\.|etc\.|vs\.|Vs\.|approx\.|Approx\.)")
    _alt_of = dict(_pairs)
    _nat_of = {b: a for a, b in _pairs}

    def find_opps(self, text):
        opps = []
        for m in self._nat_rx.finditer(text):
            w = m.group(1)
            if w == "that is" and not re.search(r"[,(]\s*$", text[max(0, m.start() - 3):m.start()]):
                continue                                   # "a tool that is useful" is not the connective
            if w == "That is" and not re.match(r",", text[m.end():]):
                continue                                   # "That is why..." is not the connective
            alt = self._alt_of[w]
            if text[m.end():m.end() + 1] == "." and alt.endswith("."):
                alt = alt[:-1]                             # avoid "etc.." at a sentence end
            opps.append(Opp(m.start(1), m.end(1), w, alt))
        for m in self._alt_rx.finditer(text):
            a = m.group(1); nat = self._nat_of[a]; end = m.end(1)
            if re.match(r"\s+[A-Z¿¡«\"]|\s*$", text[end:end + 3]):        # sentence-final: the period is the sentence's
                end -= 1; a = a[:-1]
            opps.append(Opp(m.start(1), end, nat, a))
        return _dedup(opps)

    def classify(self, tail):
        a, b = self._nat_rx.search(tail), self._alt_rx.search(tail)
        if a and a.group(1) == "that is" and a.start() > 1 and not re.search(r"[,(]\s*$", tail[max(0, a.start() - 3):a.start()]):
            a = None
        if a and a.group(1) == "That is" and not re.match(r",", tail[a.end():]):
            a = None
        if a and (not b or a.start() <= b.start()):
            return "nat"
        if b:
            return "alt"
        return None


class FlatAdverb(Property):
    """-ly adverb after a verb (drive slowly, hold it tightly) vs the colloquial flat form (drive slow, hold it tight)."""
    name = "flat_adverb"
    family = "lexical"
    nat_label = "-ly adverbs after verbs (drive slowly, hold tightly)"
    alt_label = "flat adverbs (drive slow, hold tight)"
    confound = "medium-register"
    _pairs = [("slowly", "slow"), ("quickly", "quick"), ("gently", "gentle"), ("tightly", "tight"), ("loudly", "loud"),
              ("safely", "safe"), ("deeply", "deep"), ("cheaply", "cheap"), ("directly", "direct"),
              ("differently", "different"), ("wrongly", "wrong"), ("firmly", "firm"), ("softly", "soft"),
              ("evenly", "even"), ("smoothly", "smooth"), ("carefully", "careful"), ("quietly", "quiet"),
              ("closely", "close"), ("steadily", "steady"), ("clearly", "clear"), ("easily", "easy"),
              ("brightly", "bright"), ("thinly", "thin"), ("finely", "fine"),
              ("freshly", "fresh"), ("properly", "proper"), ("slower", "slower")]
    _pairs = [p for p in _pairs if p[0] != p[1]]
    _map = {}
    for a, b in _pairs:
        _map[a] = (a, b); _map[b] = (a, b)
    # pre-verbal / non-verb context words: an adverb right after one of these is NOT in post-verbal position
    _STOP = ("to|very|so|too|quite|really|pretty|is|are|was|were|be|been|being|am|can|could|will|would|"
             "should|may|might|must|has|have|had|do|does|did|not|and|or|but|then|also|more|most|less|a|an|the|this|that|these|"
             "those|how|as|by|of|in|on|at|for|with|from|into|than|something|anything|nothing|who|which|when|while|if|because|"
             "still|just|only|even|always|never|often|usually|already|1st|first|"
             "seem|seems|seemed|look|looks|looked|feel|feels|felt|sound|sounds|sounded|taste|tastes|tasted|smell|smells|"
             "become|becomes|became|get|gets|got|getting|stay|stays|stayed|remain|remains|remained|grow|grows|grew|turn|turns|turned")
    # "made it clear", "keep it tight": causative verb + object + ADJECTIVE — not an adverb site
    _CAUS = re.compile(r"\b(?:make|makes|made|making|keep|keeps|kept|keeping|find|finds|found|consider|considers|considered|leave|leaves|left|"
                       r"get|gets|got|call|calls|called|deem|deemed|render|rendered)\s+(?:it|them|him|her|us|me|this|that|everything|things|"
                       r"the [\w'’]+|this [\w'’]+|that [\w'’]+|your [\w'’]+|my [\w'’]+|his [\w'’]+|her [\w'’]+|our [\w'’]+|their [\w'’]+)\s+$", re.IGNORECASE)
    # a bare (flat) form is ambiguous with an adjective ("select firm apples", "good light"): in running text it only counts
    # after a clearly verbal word (-ing/-ed form, a listed verb) or an object pronoun
    _VERBISH = re.compile(r"(?:\w{3,}(?:ing|ed)|it|them|him|her|us|me|go|goes|went|gone|drive|drives|drove|driven|ride|rides|rode|ridden|"
                          r"run|runs|ran|walk|walks|breathe|breathes|hold|holds|held|grip|grips|speak|speaks|spoke|spoken|talk|talks|"
                          r"sing|sings|sang|sung|think|thinks|thought|cut|cuts|hit|hits|do|does|did|done|buy|buys|bought|sit|sits|sat|"
                          r"stand|stands|stood|sleep|sleeps|slept|eat|eats|ate|eaten|dig|digs|dug|shine|shines|shone|play|plays|"
                          r"work|works|move|moves|step|steps|turn|turns|brake|brakes|pedal|pedals|steer|steers|land|lands|fly|flies|flew|"
                          r"swim|swims|swam|dive|dives|dove|climb|climbs|come|comes|came|take|takes|took|taken|put|puts|set|sets|"
                          r"write|writes|wrote|read|reads|say|says|said|answer|answers|reply|replies|explain|explains|act|acts|"
                          r"tread|treads|trod|aim|aims|hang|hangs|hung|tie|ties|tied|wrap|wraps|pack|packs|pull|pulls|push|pushes|"
                          r"press|presses|squeeze|squeezes|stir|stirs|mix|mixes|chop|chops|slice|slices|spread|spreads|pour|pours)$", re.IGNORECASE)
    # a (flat) adverb only counts in adverb position: before punctuation, a conjunction or a preposition — never before a noun/verb
    _ALT_TAIL = re.compile(r"(?=[,.;:!?)]|\s+(?:and|or|to|so|as|until|while|when|before|after|through|into|on|in|at|with|from|for|"
                           r"over|down|up|out|off|around|against|onto|toward|towards|by|if|because|but|enough|again|there|here|now|"
                           r"every|each|during|without|along|across|between|under|behind|toward|whenever|once|like)\b|\s*$)")
    _forms = "|".join(sorted(map(re.escape, _map), key=len, reverse=True))
    _rx = re.compile(r"\b(?!(?:" + _STOP + r")\s)([\w'’]{2,})\s+(" + _forms + r")\b(?!-)", re.IGNORECASE)
    _lead = re.compile(r"^\s?(" + _forms + r")\b(?!-)")

    def find_opps(self, text):
        opps = []
        for m in self._rx.finditer(text):
            w = m.group(2)
            if w.lower() not in self._map:
                continue
            nat, alt = self._map[w.lower()]
            if not self._ALT_TAIL.match(text, m.end(2)):
                continue                                   # "carefully casting", "a fresh loaf": pre-verbal or adjectival
            if w.lower() == alt and self._CAUS.search(text[max(0, m.start(1) - 40):m.start(2)]):
                continue                                   # "made it clear", "keep the room tight": object complement
            if w.lower() == alt:
                # bare forms are ambiguous with adjectives, so in running text they only count after a verbal word
                # (within 8 words), with no copula in between ("it's easy", "the road is slow", "30 cm deep" are adjectives)
                prev = re.findall(r"[\w'’]+", text[max(0, m.start(1) - 80):m.start(2)])[-8:]
                if prev and re.fullmatch(r"(?i)\d+(?:[.,]\d+)?|cm|mm|km|m|inches?|feet|foot|meters?|up|even", prev[-1]):
                    continue
                if alt == "even":
                    continue                               # focus adverb "even" is far more common than the flat adverb
                ok = False
                for x in reversed(prev):
                    if re.fullmatch(r"(?i)(?:is|are|was|were|be|been|being|am|seems?|seemed|looks?|looked|feels?|felt|becomes?|became|"
                                    r"stays?|stayed|remains?|remained|it[’']s|that[’']s|wasn[’']t|isn[’']t|aren[’']t|weren[’']t|they[’']re|"
                                    r"you[’']re|we[’']re|he[’']s|she[’']s|there[’']s|what[’']s|which|who|though)", x):
                        break
                    if self._VERBISH.match(x):
                        ok = True; break
                if not ok:
                    continue
            opps.append(Opp(m.start(2), m.end(2), nat, alt))
        return _dedup(opps)

    def classify(self, tail):
        m = self._lead.match(tail)
        w = m.group(1) if m else None
        if w is None:
            m = self._rx.search(tail)
            w = m.group(2) if m else None
        if w is None or w.lower() not in self._map:
            return None
        nat, alt = self._map[w.lower()]
        return "nat" if w.lower() == nat else "alt"


class TitleAbbr(Property):
    """Doctor Smith / Mount Fuji / Main Street (nat, full words) vs Dr. Smith / Mt. Fuji / Main St. (alt)."""
    name = "title_abbr"
    family = "lexical"
    nat_label = "titles and street words in full (Doctor Smith, Mount Fuji, Main Street)"
    alt_label = "abbreviated titles and street words (Dr. Smith, Mt. Fuji, Main St.)"
    _pre = {"Doctor": "Dr.", "Professor": "Prof.", "Mister": "Mr.", "Mount": "Mt.", "Saint": "St."}
    _post = {"Street": "St.", "Avenue": "Ave.", "Road": "Rd.", "Boulevard": "Blvd."}
    _pre_rev = {v: k for k, v in _pre.items()}
    _post_rev = {v: k for k, v in _post.items()}
    _rx_pre_nat = re.compile(r"\b(Doctor|Professor|Mister|Mount|Saint)(?= [A-Z][a-z])")
    _rx_pre_alt = re.compile(r"\b(Dr\.|Prof\.|Mr\.|Mt\.|St\.)(?= [A-Z][a-z])")
    _rx_post_nat = re.compile(r"\b[A-Z][a-z]{1,20} (Street|Avenue|Road|Boulevard)\b")
    _rx_post_alt = re.compile(r"\b[A-Z][a-z]{1,20} (St\.|Ave\.|Rd\.|Blvd\.)")

    def find_opps(self, text):
        opps = []
        for m in self._rx_pre_nat.finditer(text):
            opps.append(Opp(m.start(1), m.end(1), m.group(1), self._pre[m.group(1)]))
        for m in self._rx_pre_alt.finditer(text):
            opps.append(Opp(m.start(1), m.end(1), self._pre_rev[m.group(1)], m.group(1)))
        for m in self._rx_post_nat.finditer(text):
            alt = self._post[m.group(1)]
            if text[m.end(1):m.end(1) + 1] == ".":
                alt = alt[:-1]                               # "Oak Rd." at a sentence end, not "Oak Rd.."
            opps.append(Opp(m.start(1), m.end(1), m.group(1), alt))
        for m in self._rx_post_alt.finditer(text):
            a = m.group(1); end = m.end(1)
            if re.match(r"\s+[A-Z¿¡«\"]|\s*$", text[end:end + 3]):        # sentence-final: the period is the sentence's
                end -= 1; a = a[:-1]
            opps.append(Opp(m.start(1), end, self._post_rev[m.group(1)], a))
        return _dedup(opps)

    _lead_nat = re.compile(r"^\s?(Street|Avenue|Road|Boulevard)\b")
    _lead_alt = re.compile(r"^\s?(St|Ave|Rd|Blvd)\.")

    def classify(self, tail):
        if self._lead_nat.match(tail):
            return "nat"
        if self._lead_alt.match(tail):
            return "alt"
        cands = []
        for rx, lab in ((self._rx_pre_nat, "nat"), (self._rx_post_nat, "nat"), (self._rx_pre_alt, "alt"), (self._rx_post_alt, "alt")):
            m = rx.search(tail)
            if m:
                cands.append((m.start(), lab))
        return min(cands)[1] if cands else None


ALL_PROPERTIES = [
    SentenceCaps(), AllCaps(),
    UsUk(), IseIze(), TPast(), Whilst(),
    DoubleSpace(), OxfordComma(), CurlyQuotes(), EmDash(), Ellipsis3(), QuotePunct(),
    NumWords(), PercentSign(), OrdinalWords(),
    Contractions(), Ampersand(),
    # lexically diverse families added 2026-09-11
    UkVocab(), Register(), UnitAbbr(), Diacritics(), LatinAbbr(), HyphenCompound(), FlatAdverb(),
    IrregPast(), LatinPlural(), TitleAbbr(),
]
PROPS = {p.name: p for p in ALL_PROPERTIES}
assert len(PROPS) == len(ALL_PROPERTIES)
