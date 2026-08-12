"""Shared (adjective, adverb) pair builder for the ly_derivation family
(adjective_to_adverb / adverb_to_adjective). Not itself a task generator.

Recipe (per new_task_specs.json #87/#88):
  Adjective pool = common_adjs.txt plus alphabetic words in wordfreq's
  top-100k list whose getAllLemmas contains ADJ; exclude words already
  ending in -ly.
  Derive adverb:
    consonant+'le' -> replace final 'e' with 'y'      (gentle -> gently)
    consonant+'y'  -> -ily                              (happy -> happily)
    -ic            -> -ically                           (basic -> basically)
    else           -> +ly                                (quick -> quickly)
  Keep pair only if zipf_frequency(adverb) >= 2.0 (validates the derived
  form is an attested English word).

Manual audit note: the zipf check only validates that the derived STRING
is a common English word, not that it is semantically the correct adverb
of that specific adjective. A full manual scan of the ~1400 candidate
pairs turned up a modest number of coincidental false positives, where
the mechanically-derived string happens to be a real but unrelated (or
semantically drifted) English word: e.g. 'multiple'->'multiply',
'supple'->'supply', 'sickle'/'sick'->'sickly' (sickly = frequently ill,
not "in a sick manner"), 'home'->'homely' (homely = plain/unattractive),
'elder'->'elderly' (a related but distinct adjective, not elder's
adverb), 'like'/'unlike'->'likely'/'unlikely' (likely = probable, an
unrelated word), 'live'->'lively' (energetic, not "in a live manner"),
'lone'->'lonely' (feeling isolated), 'low'->'lowly' (humble/low-ranking),
'master'->'masterly' (skilled, a separate adjective), 'dead'->'deadly'
(causing death, not "in a dead manner"), 'bare'->'barely' (scarcely, an
unrelated drifted sense), 'ginger'->'gingerly' (cautiously, not "in a
ginger/spice manner"), 'big'->'bigly' (non-standard), 'bay'->'bayly'
(not a real derivation), 'true'->'truely' (misspelling; correct is the
irregular 'truly'), and 'public'->'publically' (attested but
non-standard; the standard form is the irregular 'publicly'). These are
hard-excluded below alongside the recipe's stated irregulars
({good, fast, hard, late, early}).
"""
import os

from lemminflect import getAllLemmas
from wordfreq import top_n_list, zipf_frequency

HERE = os.path.dirname(os.path.dirname(__file__))  # .../_resources

VOWELS = set("aeiou")

# Recipe-stated irregulars plus irregulars/false-positives found during
# manual audit of the derived pairs (see module docstring).
HARD_EXCLUDE = {
    "good", "fast", "hard", "late", "early",  # recipe-stated irregulars
    "public",   # publicly (irregular), not publically
    "very",     # verily is too semantically drifted from modern "very"
    "bay", "big",     # bayly / bigly are not standard derivations
    "home", "elder", "like", "unlike", "live", "lone", "low", "master",
    "sick", "sickle", "true", "ginger", "dead", "bare",
    "multiple", "supple",  # le-rule false positives (multiply, supply)
}

ZIPF_THRESHOLD = 2.0


def derive_adverb(adj):
    """Apply the spec's mechanical adjective->adverb rule. Returns None if
    the adjective already ends in -ly (not a valid candidate)."""
    if adj.endswith("ly"):
        return None
    if len(adj) >= 3 and adj.endswith("le") and adj[-3] not in VOWELS:
        return adj[:-1] + "y"
    if len(adj) >= 2 and adj.endswith("y") and adj[-2] not in VOWELS:
        return adj[:-1] + "ily"
    if adj.endswith("ic"):
        return adj + "ally"
    return adj + "ly"


def _is_adj(w):
    lemmas = getAllLemmas(w)
    return w in lemmas.get("ADJ", ())


def build_adjective_pool():
    words = top_n_list("en", 100000)
    from_wordfreq = [
        w for w in words
        if w.isalpha() and w.islower() and len(w) >= 3 and _is_adj(w)
    ]

    with open(os.path.join(HERE, "common_adjs.txt")) as f:
        common_adjs = [line.strip() for line in f if line.strip()]
    common_adjs = [
        w for w in common_adjs if w.isalpha() and w.islower() and len(w) >= 3
    ]

    pool = sorted(set(from_wordfreq) | set(common_adjs))
    pool = [w for w in pool if w not in HARD_EXCLUDE and not w.endswith("ly")]
    return pool


def build_validated_pairs():
    """Return the list of (adjective, adverb) pairs that pass the
    derivation rule + zipf-frequency validation + manual-audit exclusions."""
    pool = build_adjective_pool()
    pairs = []
    for adj in pool:
        adv = derive_adverb(adj)
        if adv is None:
            continue
        if zipf_frequency(adv, "en") >= ZIPF_THRESHOLD:
            pairs.append((adj, adv))
    return pairs


if __name__ == "__main__":
    pairs = build_validated_pairs()
    print(f"validated adjective->adverb pairs: {len(pairs)}")
