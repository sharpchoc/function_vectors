#!/usr/bin/env python
"""The steering variant GRID (user spec 2026-09-06, round 2).

>>> SANDBOX. No cell is canonical; there is no default steering result. <<<

12 cells x 2 steering directions. Axes:

  technique / pairing
    meandiff_paired    v = AVG over sites passing the filter in BOTH twins of
                           (act_alt(site) - act_nat(site))          [literal paired mean]
    meandiff_unpaired  v = mean(act_alt | filter) - mean(act_nat | filter)
                           [each side filtered independently]
    meanact            v = mean(act of the TARGET convention | filter)   [no subtraction]

  k filter      kall  every cue site        |  k2  only sites with k >= 2 prior manifestations
  success filter succno no behavioural filter | succyes only sites whose sampled continuation
                                                followed that context's own convention

Directions (both run for every cell):
  alt : meandiff -> +v ;  meanact -> mean(act_alt | filter)
  nat : meandiff -> -v ;  meanact -> mean(act_nat | filter)   (own mean, not a negation)

EVAL PROTOCOL (identical across cells so they are comparable):
  0-shot text: the first cue token of each document, no prior manifestation of either
  convention in the prefix. NOTE sentence_caps and all_caps have no true k=0 site (any text
  already shows its capitalisation); their earliest cue (k=1) is used and treated as 0-shot
  per user decision.
  Rollout: T=1 seeded, generate to the first sentence boundary, cap 48 tokens, `capped` flag
  recorded and passed to the judge so truncation is not scored as incoherent.
  Metric: strict = P(target convention | rollout coherent); an unscorable rollout counts as
  NOT adopting. Also: conditional adherence, unscorable %, incoherent %, n.
  Shared arms (computed once, not per cell): unsteered baseline on the same items, and the
  k >= 4 in-context reference, both directions.
  Control: counterfactual property's vector of the same construction, at the chosen setting.
"""
from dataclasses import dataclass
from typing import Tuple

SCREEN_LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24)
SCREEN_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DIRECTIONS = ("alt", "nat")
K_MIN_FILTER = 2          # the k2 filter threshold (user: "k >= 2")
REFERENCE_K = 4           # in-context reference arm
ROLLOUT_MAX_NEW = 48
SCREEN_DOCS, HEADLINE_DOCS, BYLAYER_DOCS = 25, 200, 60


@dataclass(frozen=True)
class Cell:
    name: str
    technique: str            # "meandiff" | "meanact"
    pairing: str              # "paired" | "unpaired" | "" (meanact)
    k_filter: str             # "kall" | "k2"
    success_filter: str       # "succno" | "succyes"

    @property
    def paired(self) -> bool:
        return self.pairing == "paired"

    @property
    def k_min(self) -> int:
        return K_MIN_FILTER if self.k_filter == "k2" else 0

    @property
    def require_success(self) -> bool:
        return self.success_filter == "succyes"

    @property
    def formula(self) -> str:
        f = []
        if self.k_filter == "k2":
            f.append(f"k >= {K_MIN_FILTER}")
        if self.require_success:
            f.append("continuation followed that context's convention")
        cond = (" | " + " AND ".join(f)) if f else ""
        if self.technique == "meanact":
            return f"v_dir = mean(act_dir{cond})   [raw mean of the target convention]"
        if self.paired:
            return f"v = AVG_sites[ act_alt - act_nat ]{cond}   [paired, sites passing in BOTH twins]"
        return f"v = mean(act_alt{cond}) - mean(act_nat{cond})   [unpaired]"


def _cells():
    out = []
    for tech, pairing in (("meandiff", "paired"), ("meandiff", "unpaired"), ("meanact", "")):
        for k in ("kall", "k2"):
            for s in ("succno", "succyes"):
                nm = "__".join([tech + (f"_{pairing}" if pairing else ""), k, s])
                out.append(Cell(nm, tech, pairing, k, s))
    return tuple(out)


CELLS = _cells()
BY_NAME = {c.name: c for c in CELLS}
assert len(CELLS) == 12, len(CELLS)

# Cells from earlier rounds kept for provenance only. Their eval protocol differs
# (32-token fragment rollouts, k>=4 threshold), so they are NEVER merged into this grid.
LEGACY_CELLS = ("meandiff__kall__succno", "meandiff__k4__succyes",
                "meanact__kall__succno", "sparsehead__kall__succno")


def get(name: str) -> Cell:
    return BY_NAME[name]


if __name__ == "__main__":
    print(f"{len(CELLS)} cells x {len(DIRECTIONS)} directions = {len(CELLS)*len(DIRECTIONS)} runs\n")
    for c in CELLS:
        print(f"  {c.name:38s} k_min={c.k_min} succ={c.require_success}")
        print(f"      {c.formula}")
