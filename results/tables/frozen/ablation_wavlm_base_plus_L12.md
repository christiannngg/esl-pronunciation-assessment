# Input ablation: wavlm_base_plus, layer 12 - test set

`emb+side` = embedding + Sprint 3 non-acoustic columns (main setting); `emb` = embedding only; `side` = non-acoustic columns only (no audio). Each setting uses its own validation-selected head / variant. Last column: (emb+side) minus this setting on the primary metric (RMSE: negative means the full setting is better; macro-F1: positive means it is better), paired speaker bootstrap.

## sent_accuracy

| setting  | head (variant) | test                   | full minus this (95% CI, p)     |
| -------- | -------------- | ---------------------- | ------------------------------- |
| emb+side | mlp (plain)    | RMSE 1.165 / PCC 0.655 |                                 |
| emb      | mlp (plain)    | RMSE 1.170 / PCC 0.651 | -0.005 [-0.016, +0.005] p=0.324 |
| side     | mlp (plain)    | RMSE 1.259 / PCC 0.582 | -0.094 [-0.144, -0.038] p=0.000 |

## sent_fluency

| setting  | head (variant) | test                   | full minus this (95% CI, p)     |
| -------- | -------------- | ---------------------- | ------------------------------- |
| emb+side | mlp (plain)    | RMSE 0.937 / PCC 0.754 |                                 |
| emb      | mlp (plain)    | RMSE 0.942 / PCC 0.751 | -0.005 [-0.010, -0.000] p=0.040 |
| side     | mlp (plain)    | RMSE 1.005 / PCC 0.717 | -0.067 [-0.123, -0.020] p=0.004 |

## sent_prosodic

| setting  | head (variant) | test                   | full minus this (95% CI, p)     |
| -------- | -------------- | ---------------------- | ------------------------------- |
| emb+side | mlp (plain)    | RMSE 0.946 / PCC 0.753 |                                 |
| emb      | mlp (plain)    | RMSE 0.955 / PCC 0.746 | -0.009 [-0.016, -0.002] p=0.016 |
| side     | mlp (plain)    | RMSE 1.055 / PCC 0.692 | -0.109 [-0.165, -0.057] p=0.000 |

## sent_total

| setting  | head (variant) | test                   | full minus this (95% CI, p)     |
| -------- | -------------- | ---------------------- | ------------------------------- |
| emb+side | mlp (plain)    | RMSE 1.136 / PCC 0.681 |                                 |
| emb      | mlp (plain)    | RMSE 1.145 / PCC 0.676 | -0.009 [-0.019, +0.002] p=0.096 |
| side     | mlp (plain)    | RMSE 1.236 / PCC 0.614 | -0.100 [-0.150, -0.052] p=0.000 |

## word_accuracy

| setting  | head (variant) | test                   | full minus this (95% CI, p)     |
| -------- | -------------- | ---------------------- | ------------------------------- |
| emb+side | linear (plain) | RMSE 1.680 / PCC 0.366 |                                 |
| emb      | mlp (plain)    | RMSE 1.701 / PCC 0.341 | -0.021 [-0.032, -0.009] p=0.000 |
| side     | mlp (plain)    | RMSE 1.713 / PCC 0.293 | -0.033 [-0.072, +0.008] p=0.148 |

## phone_score

| setting  | head (variant) | test                   | full minus this (95% CI, p)     |
| -------- | -------------- | ---------------------- | ------------------------------- |
| emb+side | mlp (plain)    | RMSE 0.325 / PCC 0.493 |                                 |
| emb      | mlp (plain)    | RMSE 0.347 / PCC 0.339 | -0.022 [-0.031, -0.014] p=0.000 |
| side     | mlp (plain)    | RMSE 0.352 / PCC 0.295 | -0.027 [-0.038, -0.016] p=0.000 |

## phone_mispron_t1.0

| setting  | head (variant)  | test                           | full minus this (95% CI, p)     |
| -------- | --------------- | ------------------------------ | ------------------------------- |
| emb+side | linear (cw+thr) | macro-F1 0.621 / bal.acc 0.607 |                                 |
| emb      | linear (cw+thr) | macro-F1 0.585 / bal.acc 0.574 | +0.036 [+0.024, +0.049] p=0.000 |
| side     | mlp (cw+thr)    | macro-F1 0.591 / bal.acc 0.627 | +0.030 [-0.002, +0.058] p=0.064 |

## l2_phone_error

| setting  | head (variant) | test                           | full minus this (95% CI, p)     |
| -------- | -------------- | ------------------------------ | ------------------------------- |
| emb+side | mlp (cw+thr)   | macro-F1 0.508 / bal.acc 0.537 |                                 |
| emb      | mlp (cw+thr)   | macro-F1 0.449 / bal.acc 0.473 | +0.059 [+0.048, +0.071] p=0.000 |
| side     | mlp (cw+thr)   | macro-F1 0.361 / bal.acc 0.389 | +0.148 [+0.121, +0.171] p=0.000 |

## l2_word_error

| setting  | head (variant) | test                           | full minus this (95% CI, p)     |
| -------- | -------------- | ------------------------------ | ------------------------------- |
| emb+side | mlp (cw+thr)   | macro-F1 0.709 / bal.acc 0.710 |                                 |
| emb      | mlp (cw+thr)   | macro-F1 0.703 / bal.acc 0.704 | +0.006 [+0.003, +0.010] p=0.000 |
| side     | mlp (cw+thr)   | macro-F1 0.621 / bal.acc 0.624 | +0.088 [+0.069, +0.107] p=0.000 |
