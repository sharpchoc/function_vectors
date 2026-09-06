# Variant: `meandiff__k4__succyes`

> **SANDBOX.** One cell of the steering variant grid. Not canonical, not a headline
> result. Promotion to repo standard requires an explicit user decision.

| field | value |
|---|---|
| technique | `meandiff` |
| k filter | `k4` |
| success filter | `succyes` |
| vector formula | v[l] = mean(alt cue acts | k>=kmin AND model emitted alt) - mean(nat cue acts | k>=kmin AND model emitted nat); kmin=4 (oxford_comma falls back to 3 for sample count) |
| vector artifact | `artifacts/style_properties/steering_vectors_k4` key `cuediff_k4` |
| injection site | cue token (first cue of each document) |
| layers searched | 2, 4, 6, 8, 10, 12, 16, 20, 24 |
| doses searched | 0.5, 1, 2, 4, 8, 16 |
| coherence judged | yes |
| properties | 13 |

## Protocol (shared across all variants)

first cue token per document · 200 docs · 32-token T=1 seeded rollouts · LLM coherence judge · strict metric (unscorable = not adopted).

Shared sweep grid for reference: layers (2, 4, 6, 8, 10, 12, 16, 20, 24), doses ('0.5', '1', '2', '4', '8', '16', '32').

## Arms

- counterfactual-property control: PRESENT
- reverse direction (alt->nat): NOT RUN

## Caveats

- NO REVERSE ARM: that run's reverse condition injected the all-k vector, not this one (provenance bug) - reverse is unmeasured for this cell.
- Legacy evidence-token-derived arms (`meandiff_cue_*`) live in the same run JSONs but are excluded from this grid: different derivation site.
- Sweep capped at alpha 16 (vector norm is 1.5-3x the all-k vector, so effective magnitude is comparable to alpha 32 there).
- Behavioural filter conditions on model success, which may select convention-heavy documents as well as convention-in-force states.

## Provenance

- `artifacts/style_properties/steering/full_cuek4/<prop>.json` -> arms matching `^cuediff_k4_cue_nat2alt(_best|_L\d+_a[\d.]+)?$`, cf `cfprop_cue_nat2alt`

Metrics recomputed from the stored rollouts by `variant_metrics.stats`; see
`results.csv` for per-property numbers.
