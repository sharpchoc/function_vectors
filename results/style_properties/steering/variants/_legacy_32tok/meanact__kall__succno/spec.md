# Variant: `meanact__kall__succno`

> **SANDBOX.** One cell of the steering variant grid. Not canonical, not a headline
> result. Promotion to repo standard requires an explicit user decision.

| field | value |
|---|---|
| technique | `meanact` |
| k filter | `kall` |
| success filter | `succno` |
| vector formula | v[l] = mean(alt cue acts, all sites)   [raw mean, no subtraction] |
| vector artifact | `artifacts/style_properties/steering_vectors` key `rawalt_cue` |
| injection site | cue token (first cue of each document) |
| layers searched | none (layer borrowed - see caveats) |
| doses searched | 0.5, 1, 2 |
| coherence judged | NO - strict rate here is unfiltered |
| properties | 13 |

## Protocol (shared across all variants)

first cue token per document · 200 docs · 32-token T=1 seeded rollouts · LLM coherence judge · strict metric (unscorable = not adopted).

Shared sweep grid for reference: layers (2, 4, 6, 8, 10, 12, 16, 20, 24), doses ('0.5', '1', '2', '4', '8', '16', '32').

## Arms

- counterfactual-property control: NOT RUN
- reverse direction (alt->nat): NOT RUN

## Caveats

- NO SWEEP: injected at the layer chosen for the meandiff arm of the same run (borrowed), not its own best layer.
- Dose grid only 0.5-2 (a raw mean has residual-scale norm, so large alpha would swamp the stream) - peak is NOT comparable to swept cells.
- No counterfactual control, no reverse arm.

## Provenance

- `artifacts/style_properties/steering/full_cuecue1/<prop>.json` -> arms matching `^rawalt_cue_cue_nat2alt_a[\d.]+$`
- `artifacts/style_properties/steering/full_cuek4/<prop>.json` -> arms matching `^rawalt_cue_cue_nat2alt_a[\d.]+$`

Metrics recomputed from the stored rollouts by `variant_metrics.stats`; see
`results.csv` for per-property numbers.
