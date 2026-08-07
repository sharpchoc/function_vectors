# SANDBOX sparse head selection - summary

**Sandbox trial only - NOT the repo-default head set.**

Chosen lambda = **0.01** (largest within 0.01 mean LOTO accuracy of best). Final selection: **73 heads** (c > 0.2).

| lambda | mean LOTO acc | mean LOTO nll | mean n_active | chosen |
|---|---|---|---|---|
| 0.01 | 0.421 | 2.437 | 74.6 | YES |
| 0.02 | 0.391 | 2.576 | 50.0 |  |
| 0.05 | 0.352 | 2.887 | 33.5 |  |
| 0.1 | 0.324 | 3.182 | 189.8 |  |
| 0.2 | 0.273 | 3.505 | 448.0 |  |

## Selected heads (layer, head, coeff)

(12,10,1.00), (15,5,1.00), (16,10,1.00), (20,1,1.00), (25,3,0.99), (13,9,0.99), (9,14,0.94), (15,7,0.93), (23,7,0.93), (4,12,0.92), (18,3,0.89), (16,3,0.88), (17,1,0.87), (16,7,0.86), (6,6,0.86), (4,0,0.84), (7,13,0.84), (8,0,0.83), (14,9,0.83), (8,1,0.83), (13,13,0.83), (26,14,0.82), (26,9,0.82), (12,7,0.79), (11,0,0.79), (18,9,0.79), (10,0,0.78), (22,5,0.77), (24,10,0.77), (26,11,0.77), (15,9,0.77), (17,13,0.76), (9,2,0.75), (20,14,0.74), (19,4,0.74), (18,13,0.72), (20,11,0.72), (21,5,0.69), (11,2,0.69), (10,6,0.69), (17,12,0.66), (27,5,0.65), (14,7,0.62), (7,6,0.61), (11,14,0.59), (3,14,0.54), (13,2,0.54), (25,13,0.53), (5,9,0.52), (6,14,0.52), (21,8,0.50), (13,12,0.47), (24,7,0.47), (17,9,0.45), (12,6,0.42), (12,2,0.42), (21,12,0.42), (6,15,0.41), (22,9,0.40), (15,1,0.36), (24,12,0.36), (11,4,0.35), (8,11,0.33), (13,7,0.33), (14,6,0.33), (6,4,0.31), (16,15,0.31), (12,4,0.28), (13,11,0.28), (11,1,0.28), (16,14,0.27), (5,7,0.27), (27,3,0.25)

## Per-task accuracy (final c vs baselines, same datapoints)

| task | no interv. | canonical top-40 @L9 | final sparse c |
|---|---|---|---|
| national_parks | 0.000 | 0.487 | 0.562 |
| english-spanish | 0.000 | 0.560 | 0.600 |
| next_capital_letter | 0.000 | 0.013 | 0.025 |
| commonsense_qa | 0.120 | 0.240 | 0.240 |
| capitalize_last_letter | 0.000 | 0.000 | 0.100 |
| country-capital | 0.025 | 0.287 | 0.900 |
| english-french | 0.000 | 0.630 | 0.720 |
| ag_news | 0.000 | 0.080 | 0.580 |
| sentiment | 0.000 | 0.000 | 0.670 |
| present-past | 0.025 | 0.025 | 0.637 |
| person-occupation | 0.000 | 0.025 | 0.263 |
| prev_item | 0.025 | 0.225 | 0.200 |
| capitalize_second_letter | 0.000 | 0.087 | 0.212 |
| lowercase_last_letter | 0.000 | 0.000 | 0.075 |
| singular-plural | 0.062 | 0.075 | 0.787 |
| person-sport | 0.000 | 0.013 | 0.850 |
| park-country | 0.000 | 0.500 | 0.525 |
| english-german | 0.000 | 0.240 | 0.320 |
| person-instrument | 0.000 | 0.000 | 0.600 |
| next_item | 0.050 | 0.375 | 0.425 |

## Fair held-out comparison (lambda=0.01 LOTO folds vs canonical top-40, same datapoints)

The per-task table above evaluates the FINAL c, which was trained on these datapoints
(in-sample for the sparse method). The fair generalization comparison is each task's
LEAVE-ONE-TASK-OUT fold (c trained with that task fully held out):

| task | no interv. | canonical top-40 @L9 | sparse (task held out) |
|---|---|---|---|
| ag_news | 0.000 | 0.080 | 0.560 |
| capitalize_last_letter | 0.000 | 0.000 | 0.100 |
| capitalize_second_letter | 0.000 | 0.087 | 0.200 |
| commonsense_qa | 0.120 | 0.240 | 0.230 |
| country-capital | 0.025 | 0.287 | 0.900 |
| english-french | 0.000 | 0.630 | 0.720 |
| english-german | 0.000 | 0.240 | 0.310 |
| english-spanish | 0.000 | 0.560 | 0.590 |
| lowercase_last_letter | 0.000 | 0.000 | 0.050 |
| national_parks | 0.000 | 0.487 | 0.550 |
| next_capital_letter | 0.000 | 0.013 | 0.025 |
| next_item | 0.050 | 0.375 | 0.325 |
| park-country | 0.000 | 0.500 | 0.525 |
| person-instrument | 0.000 | 0.000 | 0.588 |
| person-occupation | 0.000 | 0.025 | 0.263 |
| person-sport | 0.000 | 0.013 | 0.850 |
| present-past | 0.025 | 0.025 | 0.312 |
| prev_item | 0.025 | 0.225 | 0.163 |
| sentiment | 0.000 | 0.000 | 0.590 |
| singular-plural | 0.062 | 0.075 | 0.575 |
| **MEAN** | **0.015** | **0.193** | **0.421** |

Caveats: canonical top-40 is an UNWEIGHTED head sum injected at L9 (its usual best
zero-shot layer is ~11, so this slightly underplays it); the sparse FV uses learned
per-head weights c. Lambda=0.1/0.2 mean n_active is inflated because with strong L1 the
early-stop epoch is often epoch ~0, reverting c toward the 0.5 init (448 active).
