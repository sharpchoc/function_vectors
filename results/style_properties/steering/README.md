# results/style_properties/steering — STEERING SANDBOX

> **Nothing in this folder is canonical.** There is no default, headline, or recommended
> steering result. Every subfolder is one exploratory variant. Promotion of a variant to
> repo standard requires an **explicit user decision**, recorded in DECISIONS.md.

## The grid: 12 cells × 2 steering directions (all run)

Axes: **vector construction** × **k filter** × **success filter**. Mean difference is split
into `paired` and `unpaired` sub-variants (user decision), so the 8 named variations
become 12 cells.

| technique / pairing | `kall` `succno` | `kall` `succyes` | `k2` `succno` | `k2` `succyes` |
|---|---|---|---|---|
| `meandiff` `paired` | [kall__succno](variants/meandiff_paired__kall__succno/spec.md) | [kall__succyes](variants/meandiff_paired__kall__succyes/spec.md) | [k2__succno](variants/meandiff_paired__k2__succno/spec.md) | [k2__succyes](variants/meandiff_paired__k2__succyes/spec.md) |
| `meandiff` `unpaired` | [kall__succno](variants/meandiff_unpaired__kall__succno/spec.md) | [kall__succyes](variants/meandiff_unpaired__kall__succyes/spec.md) | [k2__succno](variants/meandiff_unpaired__k2__succno/spec.md) | [k2__succyes](variants/meandiff_unpaired__k2__succyes/spec.md) |
| `meanact` | [kall__succno](variants/meanact__kall__succno/spec.md) | [kall__succyes](variants/meanact__kall__succyes/spec.md) | [k2__succno](variants/meanact__k2__succno/spec.md) | [k2__succyes](variants/meanact__k2__succyes/spec.md) |

- `meandiff` — `paired`: AVG over sites passing the filter in **both** twins of
  (act_alt − act_nat). `unpaired`: mean(filtered alt) − mean(filtered nat).
  They are **provably identical without a success filter** (cosine 1.0000), so those two
  cells share one GPU run; they diverge under `succyes` (cosine ≈ 0.79).
- `meanact` — mean activation of the **target** convention, no subtraction. Its norms are
  ~10× the difference vectors, so extra low doses (0.125, 0.25) were swept for it.
- `kall` / `k2` — capture sites unrestricted, or only sites with **k ≥ 2** prior
  manifestations of the convention.
- `succno` / `succyes` — no behavioural filter, or only sites whose sampled continuation
  actually followed that context's convention.
- **Directions**: `alt` and `nat`. For `meandiff` the two are ±the same vector; for
  `meanact` each direction is its own convention's mean.

## Shared protocol (identical across cells, so cells are comparable)

- **0-shot text**: the first cue token of each document, with **no manifestation of either
  convention** in the prefix. `sentence_caps` and `all_caps` have no true k=0 site (any text
  already shows its capitalisation), so their earliest cue (k=1) is used and treated as
  0-shot.
- **Rollout**: T=1 seeded, generated to the **first sentence boundary**, cap 48
  tokens; a `capped` flag is passed to the judge so truncation is never scored as incoherent.
- **Coherence**: an LLM judge labels each rollout fluent/gibberish, ignoring the manipulated
  convention. Gibberish is dropped; the rate is reported.
- **Metric**: **strict** = P(target convention | coherent). An **unscorable** rollout — the
  model never produced the feature in that sentence — counts as **not adopting**, and the
  unscorable share is printed under every bar.
- **Shared arms** (computed once, reused by all cells): unsteered baseline on the same items,
  and the **k ≥ 4 in-context reference** (a document that does show ≥ 4
  manifestations, then its next cue).
- **Control**: another property's vector of the same construction, at this cell's setting.
- **Search**: layers [2, 4, 6, 8, 10, 12, 16, 20, 24] × doses ['0.5', '1', '2', '4', '8', '16', '32']
  (+0.125, 0.25 for `meanact`), screened unjudged, then the pick is measured judged.

## Files

| path | contents |
|---|---|
| `comparison_table.png` / `.csv` | all 12 cells × both directions side by side, alphabetical and unranked |
| `variants/<cell>/summary.png` | **figure 1**: per property, both directions — unsteered / steered / k ≥ 4 reference / control |
| `variants/<cell>/by_layer.png` | **figure 2**: per property, accuracy vs injection layer, one line per direction |
| `variants/<cell>/{results,by_layer}.csv`, `spec.md` | numbers and the exact recipe |
| `variants/_legacy_32tok/` | earlier cells under a **different** protocol (32-token fragments, k ≥ 4) — kept for provenance, never comparable |

## Code

`src/sandbox/ext_styleprops/`: `grid.py` (the grid definition), `build_vectors_grid.py`
(vectors for every cell/direction), `run_grid.py` (screen / headline / bylayer / refs),
`judge_coherence.py` + `judge_grid.sh`, `variant_metrics.py` (the single metric definition),
`plot_grid.py` (the two figures per cell), `compare_variants.py`.

Adding a technique = adding one entry in `grid.py`.
