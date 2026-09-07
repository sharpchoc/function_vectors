# Variant `meandiff_unpaired__kall__succno`

> **SANDBOX.** One cell of the steering variant grid. Not canonical, not a headline result.
> Promotion to repo standard requires an explicit user decision (DECISIONS.md).

| field | value |
|---|---|
| technique | `meandiff` |
| pairing | `unpaired` |
| k filter | `kall` (k >= 0) |
| success filter | `succno` |
| vector formula | v = mean(act_alt) - mean(act_nat)   [unpaired] |
| directions run | alt, nat |
| vectors | `artifacts/style_properties/steering_vectors_grid/meandiff_unpaired__kall__succno/<prop>.npz` (`v_alt`, `v_nat`) |
| capture sites (us_uk example) | alt 1310, nat 1310 |
| injection site | the cue token only |
| layers searched | 2, 4, 6, 8, 10, 12, 16, 20, 24 |
| doses searched | 0.5, 1, 2, 4, 8, 16, 32 |

## Direction handling

- `alt`: +v (the difference)
- `nat`: -v (same object, opposite sign)

## Protocol (identical across all cells)

0-shot text: the first cue token of each document, no manifestation of either convention in
the prefix. `sentence_caps` and `all_caps` have no true k=0 site (any text already shows its
capitalisation); their earliest cue (k=1) is used and treated as 0-shot per the user decision
of 2026-09-06.

Rollout: T=1 seeded, generated to the first sentence boundary, cap 48 tokens; a
`capped` flag is recorded and passed to the LLM judge so truncation is not scored as
incoherent. Documents per stage: screen 25, headline 200, by-layer 60
(fewer where a property has fewer eligible documents).

Metric: **strict** = P(target convention | rollout coherent); an unscorable rollout (the model
never produced the feature in that sentence) counts as **NOT adopting**. `results.csv` also
carries conditional adherence, unscorable %, incoherent % and n.

Arms: the unsteered baseline and the k >= 4 in-context reference are shared across
cells (computed once per direction); the counterfactual control uses another property's vector
of this same construction at this cell's chosen setting.

## Files

`summary.png` (figure 1: both directions, per property), `by_layer.png` (figure 2: accuracy vs
injection layer, each layer at its own best dose), `results.csv`, `by_layer.csv`.

## Provenance

`artifacts/style_properties/steering/grid/{screen,headline,bylayer}/meandiff_unpaired__kall__succno__<direction>/<prop>.json`
(rollouts + judge verdicts stored), picks in `.../grid/screen_picks.json`, shared arms in
`.../grid/refs/<direction>/<prop>.json`.
