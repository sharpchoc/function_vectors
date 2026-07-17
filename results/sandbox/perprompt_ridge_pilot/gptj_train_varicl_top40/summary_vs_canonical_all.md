# SANDBOX: per-prompt head-sum targets vs canonical FV-broadcast ridge (all ICL shards)

- cells compared: 899 (of 899 new cells)
- new beats old same-cell on test-vs-FV MSE: 482/899
- new cells beating the old study's overall best (0.19544): 96
- old (canonical) best cell: icl10/last_prompt_token L13 test_mse_fv=0.19544 R2=0.413
- NEW best cell by test-vs-FV: icl10/pre_label_token L13 test_mse_fv=0.15284 R2=0.541
- NEW best cell by test-vs-per-prompt: icl10/last_prompt_token L17 test_mse_pp=0.23471 R2=0.677

## Best cell per ICL index (by new test-vs-FV MSE)

| icl | best cell | old mse(FV) | old R2 | new mse(FV) | new R2(FV) | new mse(pp) | new R2(pp) |
|---|---|---|---|---|---|---|---|
| 1 | last_label_token L9 | 0.22433 | 0.326 | 0.22949 | 0.311 | 0.57766 | 0.205 |
| 2 | last_label_token L11 | 0.22129 | 0.335 | 0.22149 | 0.335 | 0.56014 | 0.229 |
| 3 | last_label_token L13 | 0.21656 | 0.349 | 0.21272 | 0.361 | 0.54591 | 0.249 |
| 4 | pre_label_token L13 | 0.21041 | 0.368 | 0.19972 | 0.400 | 0.52709 | 0.274 |
| 5 | pre_label_token L13 | 0.20528 | 0.383 | 0.18834 | 0.434 | 0.50983 | 0.298 |
| 6 | pre_label_token L16 | 0.20509 | 0.384 | 0.18363 | 0.448 | 0.50376 | 0.307 |
| 7 | pre_label_token L13 | 0.20014 | 0.399 | 0.17304 | 0.480 | 0.47993 | 0.339 |
| 8 | pre_label_token L13 | 0.19773 | 0.406 | 0.16683 | 0.499 | 0.46748 | 0.356 |
| 9 | pre_label_token L16 | 0.20048 | 0.398 | 0.15940 | 0.521 | 0.45390 | 0.375 |
| 10 | pre_label_token L13 | 0.19634 | 0.410 | 0.15284 | 0.541 | 0.43696 | 0.398 |

## Top 15 cells overall (by new test-vs-FV MSE)

| cell | old mse(FV) | old R2 | new mse(FV) | new R2(FV) | new mse(pp) | new R2(pp) | train R2 | alpha |
|---|---|---|---|---|---|---|---|---|
| icl10/pre_label_token L13 | 0.19634 | 0.410 | 0.15284 | 0.541 | 0.43696 | 0.398 | 0.799 | 3.16e+03 |
| icl10/pre_label_token L16 | 0.20003 | 0.399 | 0.15327 | 0.540 | 0.43914 | 0.395 | 0.805 | 3.16e+03 |
| icl10/pre_label_token L12 | 0.19755 | 0.406 | 0.15769 | 0.526 | 0.43973 | 0.395 | 0.799 | 3.16e+03 |
| icl9/pre_label_token L16 | 0.20048 | 0.398 | 0.15940 | 0.521 | 0.45390 | 0.375 | 0.799 | 3.16e+03 |
| icl9/pre_label_token L13 | 0.19696 | 0.408 | 0.16005 | 0.519 | 0.45237 | 0.377 | 0.793 | 3.16e+03 |
| icl10/pre_label_token L14 | 0.19962 | 0.400 | 0.16017 | 0.519 | 0.44927 | 0.382 | 0.800 | 3.16e+03 |
| icl10/pre_label_token L17 | 0.20285 | 0.391 | 0.16093 | 0.516 | 0.45322 | 0.376 | 0.803 | 3.16e+03 |
| icl10/pre_label_token L11 | 0.19767 | 0.406 | 0.16121 | 0.516 | 0.44024 | 0.394 | 0.796 | 3.16e+03 |
| icl10/pre_label_token L15 | 0.20017 | 0.399 | 0.16273 | 0.511 | 0.45459 | 0.374 | 0.803 | 3.16e+03 |
| icl10/pre_label_token L18 | 0.20426 | 0.386 | 0.16490 | 0.505 | 0.46074 | 0.366 | 0.807 | 3.16e+03 |
| icl10/pre_label_token L10 | 0.19898 | 0.402 | 0.16514 | 0.504 | 0.44466 | 0.388 | 0.793 | 3.16e+03 |
| icl8/pre_label_token L13 | 0.19773 | 0.406 | 0.16683 | 0.499 | 0.46748 | 0.356 | 0.790 | 3.16e+03 |
| icl9/pre_label_token L12 | 0.19836 | 0.404 | 0.16695 | 0.498 | 0.45692 | 0.371 | 0.793 | 3.16e+03 |
| icl9/pre_label_token L17 | 0.20233 | 0.392 | 0.16746 | 0.497 | 0.46830 | 0.355 | 0.797 | 3.16e+03 |
| icl9/pre_label_token L14 | 0.20017 | 0.399 | 0.16758 | 0.497 | 0.46497 | 0.360 | 0.794 | 3.16e+03 |
