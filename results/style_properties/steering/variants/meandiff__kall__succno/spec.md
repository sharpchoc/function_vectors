# Variant: `meandiff__kall__succno`

> **SANDBOX.** One cell of the steering variant grid. Not canonical, not a headline
> result. Promotion to repo standard requires an explicit user decision.

| field | value |
|---|---|
| technique | `meandiff` |
| k filter | `kall` |
| success filter | `succno` |
| vector formula | v[l] = mean(alt cue acts, all sites) - mean(nat cue acts, all sites) |
| vector artifact | `artifacts/style_properties/steering_vectors` key `cuediff` |
| injection site | cue token (first cue of each document) |
| layers searched | 2, 4, 6, 8, 10, 12, 16, 20, 24 |
| doses searched | 2, 4, 8, 16, 32 |
| coherence judged | yes |
| properties | 13 |

## Protocol (shared across all variants)

first cue token per document · 200 docs · 32-token T=1 seeded rollouts · LLM coherence judge · strict metric (unscorable = not adopted).

Shared sweep grid for reference: layers (2, 4, 6, 8, 10, 12, 16, 20, 24), doses ('0.5', '1', '2', '4', '8', '16', '32').

## Arms

- counterfactual-property control: PRESENT
- reverse direction (alt->nat): PRESENT

## Caveats

- Sweep covered alpha 2-32 (not 0.5/1); the shared grid's low doses were only swept for the k4 cell.
- At k=0 cue tokens the twins are character-identical, so those sites contribute ~0 to this difference while still diluting both means.

## Provenance

- `artifacts/style_properties/steering/full_cuecue1/<prop>.json` -> arms matching `^cuediff_cue_nat2alt(_best|_L\d+_a[\d.]+)?$`, cf `cfprop_cue_nat2alt`, reverse `^cuediff_cue_alt2nat$`
- `artifacts/style_properties/steering/full_cuek4/<prop>.json` -> arms matching `^cuediff_cue_nat2alt(_best|_L\d+_a[\d.]+)?$`, reverse `^cuediff_cue_alt2nat$`

Metrics recomputed from the stored rollouts by `variant_metrics.stats`; see
`results.csv` for per-property numbers.
