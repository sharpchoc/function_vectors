# Variant: `sparsehead__kall__succno`

> **SANDBOX.** One cell of the steering variant grid. Not canonical, not a headline
> result. Promotion to repo standard requires an explicit user decision.

| field | value |
|---|---|
| technique | `sparsehead` |
| k filter | `kall` |
| success filter | `succno` |
| vector formula | v = sum_{h in H} W_O^h @ mean(head cue act diff, all sites); H = heads with gate c>0.8 from sparse optimisation (no train/test split, user decision 2026-09-01) |
| vector artifact | `artifacts/style_properties/sparse_heads_cue` key `v_headsum` |
| injection site | cue token (first cue of each document) |
| layers searched | none (layer borrowed - see caveats) |
| doses searched | 1, 2, 4, 8 |
| coherence judged | NO - strict rate here is unfiltered |
| properties | 13 |

## Protocol (shared across all variants)

first cue token per document · 200 docs · 32-token T=1 seeded rollouts · LLM coherence judge · strict metric (unscorable = not adopted).

Shared sweep grid for reference: layers (2, 4, 6, 8, 10, 12, 16, 20, 24), doses ('0.5', '1', '2', '4', '8', '16', '32').

## Arms

- counterfactual-property control: NOT RUN
- reverse direction (alt->nat): NOT RUN

## Caveats

- NO LAYER SWEEP: injected at the layer the head gate was trained at.
- Head selections are dense (19-167 of 448 heads); lambda was not swept.
- No counterfactual control, no reverse arm.

## Provenance

- `artifacts/style_properties/steering/full_cuecue1/<prop>.json` -> arms matching `^headsum_cue_nat2alt_a[\d.]+$`
- `artifacts/style_properties/steering/full_cuek4/<prop>.json` -> arms matching `^headsum_cue_nat2alt_a[\d.]+$`

Metrics recomputed from the stored rollouts by `variant_metrics.stats`; see
`results.csv` for per-property numbers.
