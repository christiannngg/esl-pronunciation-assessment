# Frozen encoders vs Sprint 3 baselines (layer 12, inputs: emb+side) - test set

US 3.1 = best Sprint 3 model per task by validation score; US 3.3 = best model after the validated fix. Frozen rows: best head chosen on validation (config and class-weight / threshold variant too). Delta = frozen minus US 3.3 on RMSE (regression, lower is better) or macro-F1 (classification, higher is better), paired speaker bootstrap on identical test rows.

## sent_accuracy

| system          | detail      | test                                                                |
| --------------- | ----------- | ------------------------------------------------------------------- |
| US 3.1          | svr (none)  | RMSE 1.318 / PCC 0.532                                              |
| US 3.3          | rf (pros)   | RMSE 1.212 / PCC 0.617                                              |
| wavlm_base_plus | mlp (plain) | RMSE 1.165 / PCC 0.655<br>vs US 3.3 -0.048 [-0.093, +0.002] p=0.060 |
| hubert_base     | mlp (plain) | RMSE 1.169 / PCC 0.652<br>vs US 3.3 -0.044 [-0.101, +0.013] p=0.156 |

## sent_fluency

| system          | detail       | test                                                                |
| --------------- | ------------ | ------------------------------------------------------------------- |
| US 3.1          | ridge (none) | RMSE 1.067 / PCC 0.663                                              |
| US 3.3          | rf (pros)    | RMSE 0.957 / PCC 0.747                                              |
| wavlm_base_plus | mlp (plain)  | RMSE 0.937 / PCC 0.754<br>vs US 3.3 -0.020 [-0.062, +0.021] p=0.348 |
| hubert_base     | mlp (plain)  | RMSE 0.939 / PCC 0.753<br>vs US 3.3 -0.018 [-0.062, +0.018] p=0.368 |

## sent_prosodic

| system          | detail      | test                                                                |
| --------------- | ----------- | ------------------------------------------------------------------- |
| US 3.1          | svr (none)  | RMSE 1.105 / PCC 0.641                                              |
| US 3.3          | rf (pros)   | RMSE 0.997 / PCC 0.728                                              |
| wavlm_base_plus | mlp (plain) | RMSE 0.946 / PCC 0.753<br>vs US 3.3 -0.052 [-0.095, -0.013] p=0.012 |
| hubert_base     | mlp (plain) | RMSE 0.960 / PCC 0.744<br>vs US 3.3 -0.037 [-0.083, +0.005] p=0.108 |

## sent_total

| system          | detail      | test                                                                |
| --------------- | ----------- | ------------------------------------------------------------------- |
| US 3.1          | rf (none)   | RMSE 1.281 / PCC 0.573                                              |
| US 3.3          | rf (pros)   | RMSE 1.180 / PCC 0.651                                              |
| wavlm_base_plus | mlp (plain) | RMSE 1.136 / PCC 0.681<br>vs US 3.3 -0.044 [-0.088, +0.003] p=0.060 |
| hubert_base     | mlp (plain) | RMSE 1.155 / PCC 0.667<br>vs US 3.3 -0.025 [-0.086, +0.032] p=0.432 |

## word_accuracy

| system          | detail         | test                                                                |
| --------------- | -------------- | ------------------------------------------------------------------- |
| US 3.1          | rf (none)      | RMSE 1.704 / PCC 0.320                                              |
| US 3.3          | rf (pros)      | RMSE 1.687 / PCC 0.343                                              |
| wavlm_base_plus | linear (plain) | RMSE 1.680 / PCC 0.366<br>vs US 3.3 -0.007 [-0.038, +0.025] p=0.740 |
| hubert_base     | mlp (plain)    | RMSE 1.663 / PCC 0.385<br>vs US 3.3 -0.024 [-0.063, +0.018] p=0.228 |

## phone_score

| system          | detail      | test                                                                |
| --------------- | ----------- | ------------------------------------------------------------------- |
| US 3.1          | rf (none)   | RMSE 0.352 / PCC 0.307                                              |
| US 3.3          | rf (us31)   | RMSE 0.352 / PCC 0.307                                              |
| wavlm_base_plus | mlp (plain) | RMSE 0.325 / PCC 0.493<br>vs US 3.3 -0.027 [-0.037, -0.017] p=0.000 |
| hubert_base     | mlp (plain) | RMSE 0.325 / PCC 0.484<br>vs US 3.3 -0.027 [-0.039, -0.014] p=0.000 |

## phone_mispron_t1.0

| system          | detail          | test                                                                        |
| --------------- | --------------- | --------------------------------------------------------------------------- |
| US 3.1          | svm (none)      | macro-F1 0.522 / bal.acc 0.517                                              |
| US 3.3          | logreg (cw+thr) | macro-F1 0.574 / bal.acc 0.565                                              |
| wavlm_base_plus | linear (cw+thr) | macro-F1 0.621 / bal.acc 0.607<br>vs US 3.3 +0.047 [+0.021, +0.072] p=0.004 |
| hubert_base     | mlp (cw+thr)    | macro-F1 0.631 / bal.acc 0.618<br>vs US 3.3 +0.058 [+0.034, +0.079] p=0.000 |

## l2_phone_error

| system          | detail       | test                                                                        |
| --------------- | ------------ | --------------------------------------------------------------------------- |
| US 3.1          | svm (none)   | macro-F1 0.326 / bal.acc 0.307                                              |
| US 3.3          | rf (cw+thr)  | macro-F1 0.403 / bal.acc 0.406                                              |
| wavlm_base_plus | mlp (cw+thr) | macro-F1 0.508 / bal.acc 0.537<br>vs US 3.3 +0.105 [+0.083, +0.126] p=0.000 |
| hubert_base     | mlp (cw+thr) | macro-F1 0.479 / bal.acc 0.496<br>vs US 3.3 +0.076 [+0.057, +0.094] p=0.000 |

## l2_word_error

| system          | detail       | test                                                                        |
| --------------- | ------------ | --------------------------------------------------------------------------- |
| US 3.1          | svm (none)   | macro-F1 0.596 / bal.acc 0.597                                              |
| US 3.3          | logreg (cw)  | macro-F1 0.615 / bal.acc 0.618                                              |
| wavlm_base_plus | mlp (cw+thr) | macro-F1 0.709 / bal.acc 0.710<br>vs US 3.3 +0.093 [+0.077, +0.110] p=0.000 |
| hubert_base     | mlp (cw+thr) | macro-F1 0.684 / bal.acc 0.688<br>vs US 3.3 +0.069 [+0.053, +0.084] p=0.000 |
