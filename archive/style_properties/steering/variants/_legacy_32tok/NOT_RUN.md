# Steering variant cells NOT RUN

> **SANDBOX.** The grid is {mean difference, mean activation, sparse head
> selection} x {k filter} x {success filter}. These cells have no data.

| cell | technique | k filter | success filter | would require |
|---|---|---|---|---|
| `meanact__k4__succno` | meanact | k4 | succno | raw alt mean over k>=4 sites |
| `meanact__k4__succyes` | meanact | k4 | succyes | raw alt mean over k>=4 successful sites |
| `meanact__kall__succyes` | meanact | kall | succyes | raw alt mean over successful sites only |
| `meandiff__k4__succno` | meandiff | k4 | succno | rebuild cuediff over k>=4 sites WITHOUT the success filter (isolates the k axis from the success axis) |
| `meandiff__kall__succyes` | meandiff | kall | succyes | rebuild cuediff over all k but only behaviourally successful sites |
| `sparsehead__k4__succno` | sparsehead | k4 | succno | recapture per-head cue means on k>=4 sites, retrain gate |
| `sparsehead__k4__succyes` | sparsehead | k4 | succyes | recapture per-head cue means on k>=4 successful sites, retrain gate |
| `sparsehead__kall__succyes` | sparsehead | kall | succyes | recapture per-head cue means on successful sites, retrain gate |

Populated cells: meanact__kall__succno, meandiff__k4__succyes, meandiff__kall__succno, sparsehead__kall__succno.
