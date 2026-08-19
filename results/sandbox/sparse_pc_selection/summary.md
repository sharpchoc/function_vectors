# SANDBOX sparse PC selection - summary

**Sandbox trial only - builds on the SANDBOX sparse23 head set; NOT repo standard.**

Units = the 83 uncentered PCs (>=90% pooled variance) of the 20 train tasks' fixed10 sparse23 per-prompt FV stack; task FV = fixed10 capture mean (NOT the canonical varicl mean - do not compare absolute numbers to the sparse_head_selection tables without noting this).

Chosen lambda = **0.01** (largest within 0.01 mean LOTO accuracy of best). Final selection: **34 PCs** (c > 0.2).

| lambda | mean LOTO acc | mean LOTO nll | mean n_active | chosen |
|---|---|---|---|---|
| 0.01 | 0.376 | 2.857 | 33.5 | YES |
| 0.02 | 0.348 | 2.998 | 24.4 |  |
| 0.05 | 0.322 | 3.234 | 15.9 |  |
| 0.1 | 0.240 | 3.627 | 14.7 |  |
| 0.2 | 0.138 | 4.338 | 10.8 |  |
| 0.5 | 0.093 | 4.982 | 83.0 |  |

## Selected PCs (index, coeff), by coeff desc

(1,1.00), (2,1.00), (3,1.00), (4,1.00), (5,1.00), (6,1.00), (7,1.00), (8,1.00), (9,1.00), (10,1.00), (11,1.00), (12,1.00), (13,1.00), (14,1.00), (15,1.00), (16,1.00), (19,1.00), (22,1.00), (28,1.00), (45,1.00), (65,1.00), (18,0.99), (17,0.99), (0,0.97), (20,0.95), (34,0.94), (26,0.93), (31,0.91), (59,0.83), (24,0.79), (53,0.74), (27,0.66), (21,0.57), (55,0.34)

## Top-k PC curve (pooled over all datapoints)

| k | weighted acc | unweighted acc |
|---|---|---|
| 1 | 0.026 | 0.026 |
| 2 | 0.028 | 0.028 |
| 3 | 0.069 | 0.069 |
| 5 | 0.099 | 0.099 |
| 8 | 0.153 | 0.153 |
| 12 | 0.249 | 0.249 |
| 20 | 0.297 | 0.297 |
| 30 | 0.388 | 0.389 |
| 50 | 0.392 | 0.408 |
| 83 | 0.392 | 0.414 |

## Per-task accuracy (same datapoints)

| task | no interv. | full FV (fixed10) | 83-PC proj (c=1) | final sparse c |
|---|---|---|---|---|
| national_parks | 0.000 | 0.525 | 0.525 | 0.512 |
| english-spanish | 0.000 | 0.530 | 0.490 | 0.420 |
| next_capital_letter | 0.000 | 0.025 | 0.025 | 0.013 |
| commonsense_qa | 0.120 | 0.240 | 0.220 | 0.230 |
| capitalize_last_letter | 0.000 | 0.050 | 0.050 | 0.062 |
| country-capital | 0.025 | 0.812 | 0.812 | 0.812 |
| english-french | 0.000 | 0.620 | 0.600 | 0.380 |
| ag_news | 0.000 | 0.510 | 0.500 | 0.530 |
| sentiment | 0.000 | 0.520 | 0.520 | 0.520 |
| present-past | 0.025 | 0.588 | 0.588 | 0.600 |
| person-occupation | 0.000 | 0.188 | 0.188 | 0.188 |
| prev_item | 0.025 | 0.412 | 0.412 | 0.412 |
| capitalize_second_letter | 0.000 | 0.138 | 0.163 | 0.150 |
| lowercase_last_letter | 0.000 | 0.087 | 0.087 | 0.087 |
| singular-plural | 0.062 | 0.650 | 0.662 | 0.650 |
| person-sport | 0.000 | 0.537 | 0.550 | 0.537 |
| park-country | 0.000 | 0.550 | 0.537 | 0.550 |
| english-german | 0.000 | 0.200 | 0.200 | 0.180 |
| person-instrument | 0.000 | 0.550 | 0.550 | 0.537 |
| next_item | 0.050 | 0.537 | 0.537 | 0.537 |
