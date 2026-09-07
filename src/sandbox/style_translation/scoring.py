"""Style-decision scoring for GPT-J completions (deterministic, registry classifiers).

The completion `tail` starts right after the cue token. Because a cue may be word-internal
(learn|ed), the classifier is applied to  seg = twin[opp_char_start:cue_char_end] + tail  (the
already-generated start of the word plus the completion), falling back to the raw tail and finally
to an exact prefix match against the two stored next-token renderings.
"""
import re
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.ext_styleprops.properties import PROPS

_SENT_END = re.compile(r'[.!?]+["”’\')]*(?=\s|$)')


def cut_sentence(tail):
    """Cut the completion at the end of the current sentence (or at a newline / new header).
    Returns (cut_tail, capped) — capped = no sentence end found (the 48-token cap hit first)."""
    t = tail
    stop = len(t); capped = True
    m_nl = re.search(r"\n|\bSpanish:|\bEnglish:", t)
    if m_nl:
        stop = m_nl.start(); capped = False
    m = _SENT_END.search(t[:stop])
    if m:
        stop = m.end(); capped = False
    return t[:stop].rstrip("\n"), capped


def decide(family, seg_prefix, tail, next_nat, next_alt):
    """Return 'nat', 'alt' or None (no decision made in the completion)."""
    prop = PROPS[family]
    seg = seg_prefix + tail
    lab = prop.classify(seg)
    if lab is None and seg_prefix:
        lab = prop.classify(tail)
    if lab is None:
        for label, nxt in sorted((("nat", next_nat), ("alt", next_alt)), key=lambda kv: -len(kv[1])):
            if nxt and tail.startswith(nxt[: max(1, min(len(nxt), 6))]):
                lab = label
                break
    return lab
