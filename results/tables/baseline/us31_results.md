# Baseline results (us31) - test set, 95% speaker-bootstrap CIs in brackets

Hyper-parameters chosen on validation only (regression: RMSE; classification: balanced accuracy). Bold = best non-dummy model by PCC (regression) or F1 / macro-F1 (classification). dummy = mean / majority predictor, dummy_median = median predictor.
Human-agreement reference (Sprint 2 EDA): sentence-level leave-one-out r about 0.77-0.80; phone-level Fleiss kappa 0.46.

## sent_accuracy  (sentence, so762, n=2,500)

| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy        | default           | MAE 1.159 [1.035, 1.277] · RMSE 1.540 [1.372, 1.693] · PCC n/a                  |
| dummy_median | default           | MAE 1.012 [0.860, 1.160] · RMSE 1.568 [1.357, 1.757] · PCC n/a                  |
| **rf**       | min_samples_leaf3 | MAE 0.924 [0.839, 1.014] · RMSE 1.298 [1.175, 1.412] · PCC 0.545 [0.451, 0.625] |
| ridge        | alpha100          | MAE 0.959 [0.885, 1.046] · RMSE 1.324 [1.201, 1.445] · PCC 0.513 [0.415, 0.589] |
| svr          | C1                | MAE 0.933 [0.838, 1.027] · RMSE 1.318 [1.165, 1.450] · PCC 0.532 [0.434, 0.607] |

## sent_fluency  (sentence, so762, n=2,500)

| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy        | default           | MAE 1.092 [0.974, 1.210] · RMSE 1.429 [1.251, 1.617] · PCC n/a                  |
| dummy_median | default           | MAE 0.998 [0.856, 1.136] · RMSE 1.434 [1.219, 1.639] · PCC n/a                  |
| **rf**       | min_samples_leaf3 | MAE 0.792 [0.717, 0.872] · RMSE 1.073 [0.936, 1.240] · PCC 0.680 [0.612, 0.731] |
| ridge        | alpha1            | MAE 0.773 [0.708, 0.850] · RMSE 1.067 [0.927, 1.259] · PCC 0.663 [0.578, 0.722] |
| svr          | C10               | MAE 0.783 [0.719, 0.858] · RMSE 1.070 [0.938, 1.240] · PCC 0.661 [0.588, 0.712] |

## sent_prosodic  (sentence, so762, n=2,500)

| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy        | default           | MAE 1.137 [1.027, 1.257] · RMSE 1.435 [1.267, 1.620] · PCC n/a                  |
| dummy_median | default           | MAE 1.034 [0.876, 1.206] · RMSE 1.517 [1.276, 1.751] · PCC n/a                  |
| **rf**       | min_samples_leaf3 | MAE 0.823 [0.744, 0.906] · RMSE 1.113 [0.988, 1.274] · PCC 0.657 [0.586, 0.713] |
| ridge        | alpha1            | MAE 0.819 [0.754, 0.896] · RMSE 1.112 [0.978, 1.296] · PCC 0.631 [0.543, 0.699] |
| svr          | C1                | MAE 0.809 [0.741, 0.892] · RMSE 1.105 [0.976, 1.272] · PCC 0.641 [0.567, 0.698] |

## sent_total  (sentence, so762, n=2,500)

| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy        | default           | MAE 1.190 [1.070, 1.313] · RMSE 1.551 [1.372, 1.727] · PCC n/a                  |
| dummy_median | default           | MAE 1.110 [0.930, 1.287] · RMSE 1.716 [1.423, 1.961] · PCC n/a                  |
| **rf**       | min_samples_leaf3 | MAE 0.958 [0.880, 1.043] · RMSE 1.281 [1.146, 1.422] · PCC 0.573 [0.488, 0.641] |
| ridge        | alpha1000         | MAE 0.964 [0.891, 1.046] · RMSE 1.301 [1.164, 1.452] · PCC 0.544 [0.461, 0.613] |
| svr          | C1                | MAE 0.933 [0.843, 1.027] · RMSE 1.297 [1.141, 1.450] · PCC 0.560 [0.478, 0.624] |

## word_accuracy  (word, so762, n=15,967)

| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy        | default           | MAE 1.117 [0.993, 1.249] · RMSE 1.795 [1.527, 2.052] · PCC n/a                  |
| dummy_median | default           | MAE 0.563 [0.406, 0.729] · RMSE 1.877 [1.553, 2.175] · PCC n/a                  |
| **rf**       | min_samples_leaf10| MAE 1.013 [0.884, 1.149] · RMSE 1.704 [1.455, 1.933] · PCC 0.320 [0.266, 0.368] |
| ridge        | alpha1000         | MAE 1.050 [0.924, 1.184] · RMSE 1.747 [1.497, 1.976] · PCC 0.240 [0.190, 0.284] |
| svr          | C1                | MAE 0.748 [0.605, 0.897] · RMSE 1.780 [1.482, 2.048] · PCC 0.225 [0.173, 0.270] |

## phone_score  (phoneme, so762, n=47,369)


| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy | default | MAE 0.222 [0.196, 0.247] · RMSE 0.369 [0.299, 0.425] · PCC n/a |
| dummy_median | default | MAE 0.125 [0.094, 0.157] · RMSE 0.388 [0.308, 0.453] · PCC n/a |
| **rf** | min_samples_leaf3 | MAE 0.205 [0.179, 0.230] · RMSE 0.352 [0.292, 0.401] · PCC 0.307 [0.264, 0.345] |
| ridge | alpha1000 | MAE 0.210 [0.185, 0.235] · RMSE 0.358 [0.296, 0.409] · PCC 0.255 [0.223, 0.286] |
| svr | C1 | MAE 0.160 [0.131, 0.188] · RMSE 0.356 [0.282, 0.415] · PCC 0.305 [0.274, 0.335] |

## phone_mispron_t1.0  (phoneme, so762, n=47,369)

| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy | default | F1 0.000 [0.000, 0.000] · bal.acc 0.500 [0.500, 0.500] |
| logreg | C1 | F1 0.030 [0.017, 0.041] · bal.acc 0.507 [0.504, 0.510] |
| rf | min_samples_leaf5 | F1 0.000 [0.000, 0.000] · bal.acc 0.500 [0.500, 0.500] |
| **svm** | C10 | F1 0.066 [0.046, 0.087] · bal.acc 0.517 [0.512, 0.523] |

## phone_mispron_t0.5  (phoneme, so762, n=47,369)


| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy | default | F1 0.000 [0.000, 0.000] · bal.acc 0.500 [0.500, 0.500] |
| **logreg** | C1 | F1 0.023 [0.008, 0.035] · bal.acc 0.505 [0.501, 0.509] |
| rf | min_samples_leaf2 | F1 0.000 [0.000, 0.000] · bal.acc 0.500 [0.500, 0.500] |

## phone_mispron_t1.5  (phoneme, so762, n=47,369)


| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy | default | F1 0.000 [0.000, 0.000] · bal.acc 0.500 [0.500, 0.500] |
| **logreg** | C1 | F1 0.063 [0.047, 0.077] · bal.acc 0.515 [0.511, 0.519] |
| rf | min_samples_leaf2 | F1 0.002 [0.000, 0.004] · bal.acc 0.500 [0.500, 0.501] |

## l2_phone_error  (phoneme, l2arctic, n=120,243)


| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy | default | macro-F1 0.229 [0.226, 0.232] · bal.acc 0.250 [0.250, 0.250] |
| logreg | C1 | macro-F1 0.321 [0.309, 0.333] · bal.acc 0.303 [0.295, 0.312] |
| rf | min_samples_leaf2 | macro-F1 0.254 [0.248, 0.261] · bal.acc 0.262 [0.260, 0.265] |
| **svm** | C10 | macro-F1 0.326 [0.315, 0.335] · bal.acc 0.307 [0.300, 0.313] |

## l2_word_error  (word, l2arctic, n=33,985)


| model        | config            | test                                                                            |
|--------------|-------------------|---------------------------------------------------------------------------------|
| dummy | default | F1 0.000 [0.000, 0.000] · bal.acc 0.500 [0.500, 0.500] |
| logreg | C1 | F1 0.473 [0.450, 0.497] · bal.acc 0.591 [0.580, 0.602] |
| rf | min_samples_leaf2 | F1 0.458 [0.431, 0.483] · bal.acc 0.586 [0.576, 0.598] |
| **svm** | C1 | F1 0.494 [0.470, 0.517] · bal.acc 0.597 [0.586, 0.609] |
