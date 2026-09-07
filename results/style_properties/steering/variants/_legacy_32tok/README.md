# Legacy cells — DIFFERENT PROTOCOL, not comparable to the current grid

These cells were run before the 2026-09-06 (round 2) protocol. They used **32-token
fragment rollouts judged as fragments**, all cue sites or first-cue items, and (for the
`k4` cell) a **k ≥ 4** capture threshold rather than k ≥ 2. Their numbers are kept for
provenance only and must not be placed alongside the current cells.

| cell | why it is legacy |
|---|---|
| `meandiff__kall__succno` | 32-token fragments; superseded by `meandiff_{paired,unpaired}__kall__succno` |
| `meandiff__k4__succyes` | k ≥ 4 threshold and 32-token fragments; the k axis is now k ≥ 2 |
| `meanact__kall__succno` | 32-token fragments, borrowed layer, no sweep or controls |
| `sparsehead__kall__succno` | sparse-head technique is not part of the current 3-axis grid; 32-token fragments |

`NOT_RUN_old_grid.md` is the coverage note of that earlier 12-cell grid.
