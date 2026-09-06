#!/usr/bin/env python
"""Registry of style-property STEERING SANDBOX variants.

>>> SANDBOX. Nothing here is canonical. There is no default/headline steering result. <<<
Promotion of any variant to repo standard requires an explicit user decision (DECISIONS).

The variant space is a grid over three axes:

  technique       meandiff    v = mean(alt cue act) - mean(nat cue act)
                  meanact     v = mean(alt cue act)                      (no subtraction)
                  sparsehead  v = sum of selected heads' cue-token contributions
                              (gate learned by sparse optimisation)
  k filter        kall        every cue site of every document
                  k4          only sites with k >= 4 prior manifestations (kmin recorded
                              per property; a property falls back to k >= 3 if too few)
  success filter  succno      no behavioural filter on the captured sites
                  succyes     only sites where the sampled continuation actually followed
                              that context's own convention (prescreen labels)

= 12 cells. Cells with data carry `sources`; empty cells have `sources = ()`.

SHARED PROTOCOL (identical across variants, so cells are comparable):
  first cue token of each document only · 200 docs · 32-token T=1 seeded rollouts ·
  LLM coherence judge drops gibberish · PRIMARY METRIC = strict rate: fraction of coherent
  rollouts adopting the target convention, where an unscorable rollout (the model never
  produced the feature) counts as NOT adopting · counterfactual control = another
  property's vector of the same construction, via the fixed pool rotation (+5).

SHARED SWEEP GRID (user decision 2026-09-06): layers 2,4,6,8,10,12,16,20,24 x
doses 0.5,1,2,4,8,16,32. Cells whose arms searched less than this say so in `caveats`
and are NOT directly comparable on peak performance.
"""
from dataclasses import dataclass, field
from typing import Tuple

TECHNIQUES = ("meandiff", "meanact", "sparsehead")
K_FILTERS = ("kall", "k4")
SUCCESS_FILTERS = ("succno", "succyes")

SHARED_LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24)
SHARED_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

PROTOCOL = ("first cue token per document · 200 docs · 32-token T=1 seeded rollouts · "
            "LLM coherence judge · strict metric (unscorable = not adopted)")


@dataclass(frozen=True)
class Source:
    """Arms extracted from a completed run's JSON.

    Condition keys are matched by REGEX, not literals: the runs named a property's arm
    `<vec>_cue_nat2alt` when that vector won the property's sweep and
    `<vec>_cue_nat2alt_best` when it did not, and shortlist points carry per-property
    `_L<layer>_a<alpha>` suffixes. `cf_requires_vector` gates the counterfactual control,
    whose key is vector-agnostic (`cfprop_cue_nat2alt`) even though the vector it injected
    was whichever won that run's sweep.
    """
    run: str                      # dir under artifacts/style_properties/steering/
    main_re: str                  # regex matching this cell's steered arm(s)
    cf: str = ""                  # counterfactual control key ("" = not run)
    cf_requires_vector: str = ""  # only trust cf if run's best_from_sweep vector == this
    reverse_re: str = ""          # regex matching the reverse arm ("" = not run)


@dataclass(frozen=True)
class Variant:
    name: str
    technique: str
    k_filter: str
    success_filter: str
    formula: str
    vector: str              # "<artifacts subdir>:<npz key>" or "" if none built
    sources: Tuple[Source, ...] = ()
    layers_searched: Tuple[int, ...] = ()
    alphas_searched: Tuple[float, ...] = ()
    caveats: Tuple[str, ...] = ()
    needs: str = ""          # what an empty cell would require to fill

    @property
    def populated(self) -> bool:
        return bool(self.sources)

    @property
    def has_cf(self) -> bool:
        return any(s.cf for s in self.sources)

    @property
    def has_reverse(self) -> bool:
        return any(s.reverse_re for s in self.sources)


def _name(t, k, s):
    return f"{t}__{k}__{s}"


VARIANTS = (
    Variant(
        name="meandiff__kall__succno",
        technique="meandiff", k_filter="kall", success_filter="succno",
        formula="v[l] = mean(alt cue acts, all sites) - mean(nat cue acts, all sites)",
        vector="steering_vectors:cuediff",
        sources=(
            Source(run="full_cuecue1",
                   main_re=r"^cuediff_cue_nat2alt(_best|_L\d+_a[\d.]+)?$",
                   cf="cfprop_cue_nat2alt", cf_requires_vector="cuediff",
                   reverse_re=r"^cuediff_cue_alt2nat$"),
            # the k4 run's UNPREFIXED cuediff arms are this same all-k vector (that run's
            # full-mode main arm loaded key 'cuediff'; only the *_k4_* arms are the k4 vector)
            Source(run="full_cuek4",
                   main_re=r"^cuediff_cue_nat2alt(_best|_L\d+_a[\d.]+)?$",
                   reverse_re=r"^cuediff_cue_alt2nat$"),
        ),
        layers_searched=SHARED_LAYERS, alphas_searched=(2.0, 4.0, 8.0, 16.0, 32.0),
        caveats=("Sweep covered alpha 2-32 (not 0.5/1); the shared grid's low doses were "
                 "only swept for the k4 cell.",
                 "At k=0 cue tokens the twins are character-identical, so those sites "
                 "contribute ~0 to this difference while still diluting both means."),
    ),
    Variant(
        name="meandiff__k4__succyes",
        technique="meandiff", k_filter="k4", success_filter="succyes",
        formula=("v[l] = mean(alt cue acts | k>=kmin AND model emitted alt) - "
                 "mean(nat cue acts | k>=kmin AND model emitted nat); kmin=4 "
                 "(oxford_comma falls back to 3 for sample count)"),
        vector="steering_vectors_k4:cuediff_k4",
        sources=(
            Source(run="full_cuek4",
                   main_re=r"^cuediff_k4_cue_nat2alt(_best|_L\d+_a[\d.]+)?$",
                   cf="cfprop_cue_nat2alt", cf_requires_vector="ANY"),
        ),
        layers_searched=SHARED_LAYERS, alphas_searched=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        caveats=("NO REVERSE ARM: that run's reverse condition injected the all-k vector, "
                 "not this one (provenance bug) - reverse is unmeasured for this cell.",
                 "Legacy evidence-token-derived arms (`meandiff_cue_*`) live in the same "
                 "run JSONs but are excluded from this grid: different derivation site.",
                 "Sweep capped at alpha 16 (vector norm is 1.5-3x the all-k vector, so "
                 "effective magnitude is comparable to alpha 32 there).",
                 "Behavioural filter conditions on model success, which may select "
                 "convention-heavy documents as well as convention-in-force states."),
    ),
    Variant(
        name="meanact__kall__succno",
        technique="meanact", k_filter="kall", success_filter="succno",
        formula="v[l] = mean(alt cue acts, all sites)   [raw mean, no subtraction]",
        vector="steering_vectors:rawalt_cue",
        # NOTE the doubled "cue_cue": key is rawalt_cue (vector) + _cue_nat2alt (site).
        # Keys named `rawalt_cue_nat2alt_*` are the EVIDENCE-token raw mean and are
        # deliberately NOT matched here - different derivation site, excluded from this grid.
        sources=(
            Source(run="full_cuecue1", main_re=r"^rawalt_cue_cue_nat2alt_a[\d.]+$"),
            Source(run="full_cuek4", main_re=r"^rawalt_cue_cue_nat2alt_a[\d.]+$"),
        ),
        layers_searched=(), alphas_searched=(0.5, 1.0, 2.0),
        caveats=("NO SWEEP: injected at the layer chosen for the meandiff arm of the same "
                 "run (borrowed), not its own best layer.",
                 "Dose grid only 0.5-2 (a raw mean has residual-scale norm, so large alpha "
                 "would swamp the stream) - peak is NOT comparable to swept cells.",
                 "No counterfactual control, no reverse arm."),
    ),
    Variant(
        name="sparsehead__kall__succno",
        technique="sparsehead", k_filter="kall", success_filter="succno",
        formula=("v = sum_{h in H} W_O^h @ mean(head cue act diff, all sites); H = heads "
                 "with gate c>0.8 from sparse optimisation (no train/test split, user "
                 "decision 2026-09-01)"),
        vector="sparse_heads_cue:v_headsum",
        sources=(
            Source(run="full_cuecue1", main_re=r"^headsum_cue_nat2alt_a[\d.]+$"),
            Source(run="full_cuek4", main_re=r"^headsum_cue_nat2alt_a[\d.]+$"),
        ),
        layers_searched=(), alphas_searched=(1.0, 2.0, 4.0, 8.0),
        caveats=("NO LAYER SWEEP: injected at the layer the head gate was trained at.",
                 "Head selections are dense (19-167 of 448 heads); lambda was not swept.",
                 "No counterfactual control, no reverse arm."),
    ),
)

# --- empty cells, generated so the grid is explicit ---
_EMPTY_NEEDS = {
    ("meandiff", "kall", "succyes"): "rebuild cuediff over all k but only behaviourally successful sites",
    ("meandiff", "k4", "succno"): "rebuild cuediff over k>=4 sites WITHOUT the success filter (isolates the k axis from the success axis)",
    ("meanact", "kall", "succyes"): "raw alt mean over successful sites only",
    ("meanact", "k4", "succno"): "raw alt mean over k>=4 sites",
    ("meanact", "k4", "succyes"): "raw alt mean over k>=4 successful sites",
    ("sparsehead", "kall", "succyes"): "recapture per-head cue means on successful sites, retrain gate",
    ("sparsehead", "k4", "succno"): "recapture per-head cue means on k>=4 sites, retrain gate",
    ("sparsehead", "k4", "succyes"): "recapture per-head cue means on k>=4 successful sites, retrain gate",
}
_by_name = {v.name: v for v in VARIANTS}
_all = list(VARIANTS)
for _t in TECHNIQUES:
    for _k in K_FILTERS:
        for _s in SUCCESS_FILTERS:
            _n = _name(_t, _k, _s)
            if _n not in _by_name:
                _all.append(Variant(name=_n, technique=_t, k_filter=_k, success_filter=_s,
                                    formula="(not run)", vector="",
                                    needs=_EMPTY_NEEDS[(_t, _k, _s)]))
ALL_VARIANTS = tuple(sorted(_all, key=lambda v: v.name))
POPULATED = tuple(v for v in ALL_VARIANTS if v.populated)
EMPTY = tuple(v for v in ALL_VARIANTS if not v.populated)

assert len(ALL_VARIANTS) == len(TECHNIQUES) * len(K_FILTERS) * len(SUCCESS_FILTERS) == 12
assert len(POPULATED) == 4 and len(EMPTY) == 8


def get(name: str) -> Variant:
    return {v.name: v for v in ALL_VARIANTS}[name]


if __name__ == "__main__":
    print(f"{len(ALL_VARIANTS)} cells: {len(POPULATED)} populated, {len(EMPTY)} not run\n")
    for v in ALL_VARIANTS:
        flag = "DATA" if v.populated else " -- "
        ctl = ("cf" if v.has_cf else "  ") + ("+rev" if v.has_reverse else "    ")
        print(f"  [{flag}] {v.name:32s} {ctl}  {v.formula[:60]}")
