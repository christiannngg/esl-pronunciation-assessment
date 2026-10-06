# US 3.3 before / after (test set, same model, hyper-parameters unchanged)

Fix chosen per (task, model) on validation (must beat US 3.1 by >= 1% RMSE or +0.01 macro-F1, else US 3.1 is kept). Delta CI / p: paired speaker-level bootstrap on the primary metric (RMSE for regression, macro-F1 for classification). `cw` = balanced class weights, `+thr` = validation-tuned decision weights, `pros` = pitch / energy / pause features, `rs` = low-score resampling.

## Headline: best model per task (chosen on validation after the fix) - test set

| task | model | fix | before | after | change (95% CI, p) |
|---|---|---|---|---|---|
| l2_phone_error | rf | cw+thr | macro-F1 0.254 / bal.acc 0.262 | macro-F1 0.403 / bal.acc 0.406 | +0.149 ([+0.132, +0.163], p=0.000) |
| l2_word_error | logreg | cw | macro-F1 0.589 / bal.acc 0.591 | macro-F1 0.615 / bal.acc 0.618 | +0.027 ([+0.014, +0.039], p=0.000) |
| phone_mispron_t0.5 | logreg | cw+thr | macro-F1 0.503 / bal.acc 0.505 | macro-F1 0.572 / bal.acc 0.586 | +0.068 ([+0.043, +0.086], p=0.000) |
| phone_mispron_t1.0 | logreg | cw+thr | macro-F1 0.504 / bal.acc 0.507 | macro-F1 0.574 / bal.acc 0.565 | +0.070 ([+0.050, +0.084], p=0.000) |
| phone_mispron_t1.5 | logreg | cw+thr | macro-F1 0.513 / bal.acc 0.515 | macro-F1 0.594 / bal.acc 0.595 | +0.081 ([+0.064, +0.095], p=0.000) |
| phone_score | rf | us31 | RMSE 0.352 / PCC 0.307 | RMSE 0.352 / PCC 0.307 | +0.000 ([+0.000, +0.000], p=1.000) |
| sent_accuracy | rf | pros | RMSE 1.298 / PCC 0.545 | RMSE 1.212 / PCC 0.617 | -0.085 ([-0.119, -0.048], p=0.000) |
| sent_fluency | rf | pros | RMSE 1.073 / PCC 0.680 | RMSE 0.957 / PCC 0.747 | -0.116 ([-0.155, -0.077], p=0.000) |
| sent_prosodic | rf | pros | RMSE 1.113 / PCC 0.657 | RMSE 0.997 / PCC 0.728 | -0.116 ([-0.155, -0.075], p=0.000) |
| sent_total | rf | pros | RMSE 1.281 / PCC 0.573 | RMSE 1.180 / PCC 0.651 | -0.101 ([-0.135, -0.069], p=0.000) |
| word_accuracy | rf | pros | RMSE 1.704 / PCC 0.320 | RMSE 1.687 / PCC 0.343 | -0.017 ([-0.030, -0.005], p=0.012) |

## l2_phone_error

| model | fix | macro-F1 before -> after (95% CI of delta, p) | balanced acc | F1 (positive) |
|---|---|---|---|---|
| logreg | cw+thr | 0.321 -> 0.394 ([+0.053, +0.089], p=0.000) | 0.303 -> 0.411 | n/a |
| rf | cw+thr | 0.254 -> 0.403 ([+0.132, +0.163], p=0.000) | 0.262 -> 0.406 | n/a |
| svm | us31 | 0.326 -> 0.326 ([+0.000, +0.000], p=1.000) | 0.307 -> 0.307 | n/a |

## l2_word_error

| model | fix | macro-F1 before -> after (95% CI of delta, p) | balanced acc | F1 (positive) |
|---|---|---|---|---|
| logreg | cw | 0.589 -> 0.615 ([+0.014, +0.039], p=0.000) | 0.591 -> 0.618 | 0.473 -> 0.570 |
| rf | cw+thr | 0.582 -> 0.605 ([+0.006, +0.039], p=0.012) | 0.586 -> 0.606 | 0.458 -> 0.547 |
| svm | cw | 0.596 -> 0.609 ([-0.004, +0.028], p=0.156) | 0.597 -> 0.613 | 0.494 -> 0.572 |

## phone_mispron_t0.5

| model | fix | macro-F1 before -> after (95% CI of delta, p) | balanced acc | F1 (positive) |
|---|---|---|---|---|
| logreg | cw+thr | 0.503 -> 0.572 ([+0.043, +0.086], p=0.000) | 0.505 -> 0.586 | 0.023 -> 0.174 |
| rf | cw+thr | 0.492 -> 0.568 ([+0.046, +0.101], p=0.000) | 0.500 -> 0.645 | 0.000 -> 0.187 |

## phone_mispron_t1.0

| model | fix | macro-F1 before -> after (95% CI of delta, p) | balanced acc | F1 (positive) |
|---|---|---|---|---|
| logreg | cw+thr | 0.504 -> 0.574 ([+0.050, +0.084], p=0.000) | 0.507 -> 0.565 | 0.030 -> 0.179 |
| rf | cw+thr | 0.489 -> 0.580 ([+0.058, +0.120], p=0.000) | 0.500 -> 0.664 | 0.000 -> 0.227 |
| svm | us31 | 0.522 -> 0.522 ([+0.000, +0.000], p=1.000) | 0.517 -> 0.517 | 0.066 -> 0.066 |

## phone_mispron_t1.5

| model | fix | macro-F1 before -> after (95% CI of delta, p) | balanced acc | F1 (positive) |
|---|---|---|---|---|
| logreg | cw+thr | 0.513 -> 0.594 ([+0.064, +0.095], p=0.000) | 0.515 -> 0.595 | 0.063 -> 0.245 |
| rf | cw+thr | 0.483 -> 0.589 ([+0.085, +0.124], p=0.000) | 0.500 -> 0.592 | 0.002 -> 0.237 |

## phone_score

| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |
|---|---|---|---|---|
| rf | us31 | 0.352 -> 0.352 ([+0.000, +0.000], p=1.000) | 0.307 -> 0.307 | 0.205 -> 0.205 |
| ridge | us31 | 0.358 -> 0.358 ([+0.000, +0.000], p=1.000) | 0.255 -> 0.255 | 0.210 -> 0.210 |
| svr | us31 | 0.356 -> 0.356 ([+0.000, +0.000], p=1.000) | 0.305 -> 0.305 | 0.160 -> 0.160 |

## sent_accuracy

| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |
|---|---|---|---|---|
| rf | pros | 1.298 -> 1.212 ([-0.119, -0.048], p=0.000) | 0.545 -> 0.617 | 0.924 -> 0.878 |
| ridge | pros | 1.324 -> 1.265 ([-0.102, -0.014], p=0.012) | 0.513 -> 0.582 | 0.959 -> 0.919 |
| svr | pros | 1.318 -> 1.247 ([-0.104, -0.037], p=0.000) | 0.532 -> 0.598 | 0.933 -> 0.893 |

## sent_fluency

| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |
|---|---|---|---|---|
| rf | pros | 1.073 -> 0.957 ([-0.155, -0.077], p=0.000) | 0.680 -> 0.747 | 0.792 -> 0.693 |
| ridge | pros | 1.067 -> 1.027 ([-0.109, +0.031], p=0.236) | 0.663 -> 0.700 | 0.773 -> 0.714 |
| svr | pros | 1.070 -> 0.991 ([-0.118, -0.041], p=0.000) | 0.661 -> 0.719 | 0.783 -> 0.713 |

## sent_prosodic

| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |
|---|---|---|---|---|
| rf | pros | 1.113 -> 0.997 ([-0.155, -0.075], p=0.000) | 0.657 -> 0.728 | 0.823 -> 0.718 |
| ridge | pros | 1.112 -> 1.060 ([-0.119, +0.013], p=0.132) | 0.631 -> 0.679 | 0.819 -> 0.742 |
| svr | pros | 1.105 -> 1.025 ([-0.122, -0.040], p=0.000) | 0.641 -> 0.703 | 0.809 -> 0.731 |

## sent_total

| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |
|---|---|---|---|---|
| rf | pros | 1.281 -> 1.180 ([-0.135, -0.069], p=0.000) | 0.573 -> 0.651 | 0.958 -> 0.853 |
| ridge | pros | 1.301 -> 1.232 ([-0.116, -0.020], p=0.004) | 0.544 -> 0.608 | 0.964 -> 0.886 |
| svr | pros | 1.297 -> 1.220 ([-0.111, -0.040], p=0.000) | 0.560 -> 0.626 | 0.933 -> 0.873 |

## word_accuracy

| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |
|---|---|---|---|---|
| rf | pros | 1.704 -> 1.687 ([-0.030, -0.005], p=0.012) | 0.320 -> 0.343 | 1.013 -> 0.982 |
| ridge | pros | 1.747 -> 1.715 ([-0.053, -0.012], p=0.000) | 0.240 -> 0.299 | 1.050 -> 1.016 |
| svr | pros | 1.780 -> 1.757 ([-0.034, -0.009], p=0.000) | 0.225 -> 0.269 | 0.748 -> 0.750 |
