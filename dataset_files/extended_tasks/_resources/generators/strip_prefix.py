#!/usr/bin/env python3
"""Generator for strip_prefix task.

Rule: given a word formed with a derivational prefix (un-, re-, dis-, mis-,
non-, over-, under-, pre-), output the base word with the prefix removed.
"""
import json
import random

from wordfreq import top_n_list, zipf_frequency

RESOURCES = "dataset_files/extended_tasks/_resources"
OUT_PATH = "dataset_files/extended_tasks/strip_prefix.json"

PREFIXES = ["un", "re", "dis", "mis", "non", "over", "under", "pre"]

# Manually audited (LLM opacity check): words that pass the mechanical
# prefix+base split but are NOT semantically PREFIX + BASE in meaning --
# the prefix reading is a coincidental orthographic split, not how an
# English speaker parses the word (Latinate/opaque formations, or the base
# happens to be a real word unrelated to the whole's meaning).
OPAQUE = {
    "restrain", "discover", "preside", "republic", "discipline", "disaster",
    "distant", "distinct", "district", "dismiss", "dismay", "disease",
    "display", "dispute", "disturb", "discount", "discourse", "discard",
    "discern", "disciple", "discourage", "discomfort", "resent", "resist",
    "resign", "reside", "respect", "respond", "restore", "result", "resume",
    "retail", "retain", "reveal", "reverse", "review", "revise", "record",
    "reduce", "refer", "reflect", "reform", "refuse", "regard", "region",
    "register", "regret", "relate", "relax", "release", "relief", "remain",
    "remark", "remedy", "remind", "remove", "render", "rent", "repair",
    "repeat", "replace", "reply", "report", "represent", "request",
    "require", "rescue", "research", "reserve", "resolve", "resort",
    "retreat", "retrieve", "return", "reveal", "revenue", "reverend",
    "unclear", "uncle", "under", "united", "universe", "university",
    "unique", "unite", "unit", "unless", "until", "upon",
    "misery", "mist", "mistake", "mister", "mistress", "mission",
    "missile", "misty",
    "nonsense", "none", "nonetheless",
    "overall", "overcome", "overhead", "overlook", "overseas", "overtime",
    "override", "overturn", "overwhelm", "overhaul", "overnight", "oversee",
    "overflow", "overlap", "overlay", "overrule", "overshadow", "overtake",
    "overthrow",
    "underneath", "understand", "undergo", "undertake", "undermine",
    "underline", "underlying", "underwear", "undergraduate", "underworld",
    "predator", "predict", "prefer", "prejudice", "premier", "premise",
    "preparation", "prepare", "prescribe", "presence", "present", "preserve",
    "president", "press", "pressure", "prestige", "presume", "pretend",
    "pretty", "prevail", "prevent", "previous", "prey",
    # Second audit pass over the actual generated candidates: false splits
    # where the "prefix" is not a real derivational prefix on this word
    # (proper nouns, clipped forms, unrelated homographs, or -ing/-er/-ed
    # stripped from the WRONG boundary, e.g. "releasing" = release+ing, not
    # re+leasing).
    "discord", "discos", "dismal", "dissect", "dissent", "dissing",
    "distemper", "distill", "distract", "distraction", "mischa", "mischief",
    "missal", "missin", "missing", "dispatch", "dispose", "disposing",
    "disposition", "disposed", "disposes",
    "overton", "overage",
    "predestination", "prelate", "prelim", "prescott", "preserver",
    "preserving", "presided", "presides", "presiding", "pretender",
    "pretense", "pretension",
    "reagent", "really", "rebar", "rebel", "rebuff", "rebut", "recant",
    "recap", "recon", "redick", "redoubt", "reeve", "regen", "reits",
    "relent", "reliability", "reliable", "relied", "releasing", "relying",
    "remiss", "remorse", "removing", "renan", "renoir", "repetition",
    "reportable", "reporter", "repose", "repulse", "rerum", "reserving",
    "resided", "residing", "resin", "respite", "resting", "restrained",
    "retina", "retire", "retired", "retiring", "retreating", "returner",
    "reverb", "revista", "revolt", "reyes",
    "underhill", "undertale", "units", "unsub", "untill", "unwin",
    "distribute", "recent", "recoup", "reformer", "reforming", "refused",
    "regal", "remains", "remember", "resides", "retract", "retraction",
    "revis", "undermines", "underwood", "union", "unser", "repaired",
}

# 2026-08-12 audit: ~1000 mechanically-generated pairs were re-screened by
# hand for semantic transparency (does the whole word actually mean
# PREFIX-meaning + BASE-meaning, or is the split a coincidental orthographic
# accident?). These passed the mechanical + OPAQUE filters above but are
# lexicalized/etymology-mismatch words where the base's *modern* meaning has
# no real relation to the whole word (e.g. "discover" is not "dis" + "cover"
# in any usable sense -- cover means the opposite of what discover means).
# Removed from strip_prefix.json and blocked from ever being regenerated.
OPAQUE_EXCLUDE = {
    # dis-: base word's meaning is unrelated to (or contradicts) the whole
    "discovered",     # -> covered: discover=find/reveal, cover=conceal (opposite sense)
    "discovering",    # -> covering: same as above
    "dismissing",     # -> missing: dismiss=send away/reject, missing=absent; unrelated
    "disappoint",     # -> appoint: disappoint=let down, appoint=assign to a role; unrelated
    "disappointment", # -> appointment: same, appointment=scheduled meeting/role; unrelated
    "dismantle",      # -> mantle: dismantle=take apart, mantle=cloak/geologic layer; unrelated
    "dissolve",       # -> solve: dissolve=melt/end, solve=find an answer; unrelated
    # re-: base word's modern meaning has diverged from the whole word's meaning
    "recover",        # -> cover: recover=get back/get better, cover=place over; unrelated
    "recovered",      # -> covered: same
    "recovering",     # -> covering: same
    "redeem",         # -> deem: redeem=buy back/save, deem=consider/judge; unrelated
    "recurrent",      # -> current: recurrent=repeating, current=happening now/a flow; unrelated
    "reward",         # -> ward: reward=prize, ward=hospital section/guardian; unrelated
    "renews",         # -> news: false split, real morphology is "renew"+s not "re"+"news"
    "recite",         # -> cite: recite=say from memory, cite=quote a source; unrelated
    "recitation",     # -> citation: same family as recite
    "rebus",          # -> bus: unrelated etymology (Latin "res", things) vs "bus" (omnibus, vehicle)
    "remediation",    # -> mediation: remediation=fix/correct, mediation=negotiate a dispute; different roots
    "refraction",     # -> fraction: refraction=bending of light, fraction=numeric portion; unrelated
    "remission",      # -> mission: remission=disease abating/forgiveness, mission=assigned task; unrelated
    "resorted",       # -> sorted: "resort to" != "sort again"; same family as OPAQUE's "resort"
    # pre-:
    "pretext",        # -> text: pretext=false excuse, text=written words; unrelated
    # under-: idiomatic verbs where "under" doesn't mean "beneath" or "too
    # little" (contrast with genuinely transparent underfund/underweight/
    # undercover/etc.) -- same family as "undergo"/"undertake"/"understand",
    # already in OPAQUE above; these are inflected forms of those verbs.
    "undergone", "undergoing", "undergoes", "underwent",   # undergo = experience
    "undertaking", "undertook", "undertaken",              # undertake = take on a task
    "understanding", "understood",                          # understand = comprehend
}


def load_words(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_candidates():
    common = set(load_words(f"{RESOURCES}/common_words.txt"))
    top = top_n_list("en", 100000)

    matches = {}  # word -> list of (prefix, base)
    for w in top:
        if not w.isalpha() or not w.islower() or len(w) < 5:
            continue
        if zipf_frequency(w, "en") < 2.3:
            continue
        for p in PREFIXES:
            if w.startswith(p):
                base = w[len(p):]
                if len(base) < 3:
                    continue
                if base not in common:
                    continue
                if zipf_frequency(base, "en") < 3.0:
                    continue
                matches.setdefault(w, []).append((p, base))

    # Exclude words matching more than one prefix split (ambiguous boundary).
    pairs = {}
    for w, ms in matches.items():
        if len(ms) != 1:
            continue
        if w in OPAQUE or w in OPAQUE_EXCLUDE:
            continue
        p, base = ms[0]
        pairs[w] = base

    return pairs


def main():
    pairs = build_candidates()
    print(f"candidate pairs after mechanical + opacity filter: {len(pairs)}")

    # Preserve the pairs already vetted as semantically transparent (the
    # 2026-08-12 opacity audit only removed OPAQUE_EXCLUDE words from the
    # existing 1000; everything else was manually re-checked and kept), and
    # backfill only the count that was removed, from fresh candidates.
    try:
        with open(OUT_PATH) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []

    retained = [
        d for d in existing
        if d["input"] in pairs and pairs[d["input"]] == d["output"]
    ]
    retained_inputs = {d["input"] for d in retained}
    print(f"retained {len(retained)} previously-vetted pairs")

    needed = 1000 - len(retained)
    assert needed >= 0, "more pairs retained than the 1000 target"

    fresh = sorted((w, b) for w, b in pairs.items() if w not in retained_inputs)
    random.seed(42)
    random.shuffle(fresh)
    backfill = fresh[:needed]
    print(f"backfilling {len(backfill)} new pairs: {[w for w, _ in backfill]}")
    assert len(backfill) == needed, "not enough fresh candidates to reach 1000"

    dataset = list(retained) + [{"input": w, "output": b} for w, b in backfill]

    random.seed(42)
    random.shuffle(dataset)

    # --- self-checks ---
    n = len(dataset)
    assert n == 1000
    inputs = [d["input"] for d in dataset]
    assert len(set(inputs)) == n, "duplicate inputs"
    for d in dataset:
        w, base = d["input"], d["output"]
        assert w == w.strip() and base == base.strip()
        assert w != base
        assert w not in OPAQUE and w not in OPAQUE_EXCLUDE
        # rule self-check: re-derive the prefix split and confirm w == prefix+base
        matched = [p for p in PREFIXES if w == p + base]
        assert len(matched) == 1, (w, base, matched)

    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f, indent=1)

    print(f"wrote {n} pairs to {OUT_PATH}")
    return n


if __name__ == "__main__":
    main()
