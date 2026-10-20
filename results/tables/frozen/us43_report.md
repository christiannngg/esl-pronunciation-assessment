# US 4.3 - updated frozen-encoder results and fix evidence (test set; every choice made on validation)

Metric: RMSE for regression (lower is better), macro-F1 for classification (higher is better). Delta = 4.3 minus reference on the identical test rows, paired speaker bootstrap (95% CI, two-sided p). `verdict` is 'better' / 'worse' only when the CI excludes 0.

## 1. Updated frozen-encoder results (per encoder, best 4.3 system chosen on validation)

| encoder         | task               | US 3.1 | US 3.3 | 4.1 (L12) | 4.3       | system               | delta vs 4.1 [95% CI]           | delta vs US 3.3 [95% CI]        | verdict vs 4.1        |
| --------------- | ------------------ | ------ | ------ | --------- | --------- | -------------------- | ------------------------------- | ------------------------------- | --------------------- |
| wavlm_base_plus | sent_accuracy      | 1.318  | 1.212  | 1.165     | **1.113** | ws                   | -0.052 [-0.082, -0.023] p=0.000 | -0.099 [-0.152, -0.041] p=0.000 | better                |
| wavlm_base_plus | sent_fluency       | 1.067  | 0.957  | 0.937     | **0.915** | sent_avgpca          | -0.022 [-0.040, -0.001] p=0.048 | -0.042 [-0.090, +0.007] p=0.084 | better                |
| wavlm_base_plus | sent_prosodic      | 1.105  | 0.997  | 0.946     | **0.901** | sent_ws              | -0.045 [-0.066, -0.028] p=0.000 | -0.096 [-0.151, -0.050] p=0.000 | better                |
| wavlm_base_plus | sent_total         | 1.281  | 1.180  | 1.136     | **1.114** | ws                   | -0.022 [-0.060, +0.014] p=0.244 | -0.066 [-0.130, -0.004] p=0.040 | no significant change |
| wavlm_base_plus | word_accuracy      | 1.704  | 1.687  | 1.680     | **1.633** | word_meanmaxstd_pros | -0.047 [-0.067, -0.029] p=0.000 | -0.054 [-0.089, -0.016] p=0.000 | better                |
| wavlm_base_plus | phone_score        | 0.352  | 0.352  | 0.325     | **0.314** | ws                   | -0.011 [-0.018, -0.004] p=0.000 | -0.038 [-0.052, -0.024] p=0.000 | better                |
| wavlm_base_plus | phone_mispron_t1.0 | 0.522  | 0.574  | 0.621     | **0.661** | mispron-grp          | +0.041 [+0.022, +0.059] p=0.000 | +0.088 [+0.061, +0.109] p=0.000 | better                |
| wavlm_base_plus | l2_phone_error     | 0.326  | 0.403  | 0.508     | **0.523** | ws                   | +0.014 [-0.007, +0.035] p=0.208 | +0.120 [+0.089, +0.149] p=0.d000| no significant change |
| wavlm_base_plus | l2_word_error      | 0.596  | 0.615  | 0.709     | **0.716** | L11                  | +0.007 [+0.002, +0.013] p=0.004 | +0.100 [+0.082, +0.118] p=0.000 | better                |
| hubert_base     | sent_accuracy      | 1.318  | 1.212  | 1.169     | **1.162** | L11                  | -0.007 [-0.030, +0.017] p=0.624 | -0.051 [-0.105, +0.002] p=0.064 | no significant change |
| hubert_base     | sent_fluency       | 1.067  | 0.957  | 0.939     | **0.916** | L11                  | -0.023 [-0.047, -0.003] p=0.008 | -0.042 [-0.082, -0.006] p=0.028 | better                |
| hubert_base     | sent_prosodic      | 1.105  | 0.997  | 0.960     | **0.893** | sent_ws              | -0.067 [-0.092, -0.043] p=0.000 | -0.104 [-0.148, -0.063] p=0.000 | better                |
| hubert_base     | sent_total         | 1.281  | 1.180  | 1.155     | **1.077** | sent_ws_pros         | -0.078 [-0.106, -0.054] p=0.000 | -0.103 [-0.160, -0.051] p=0.000 | better                |
| hubert_base     | word_accuracy      | 1.704  | 1.687  | 1.663     | **1.656** | word_max_pros        | -0.007 [-0.035, +0.018] p=0.600 | -0.031 [-0.054, -0.006] p=0.024 | no significant change |
| hubert_base     | phone_score        | 0.352  | 0.352  | 0.325     | **0.319** | ws                   | -0.006 [-0.010, -0.001] p=0.020 | -0.032 [-0.047, -0.018] p=0.000 | better                |
| hubert_base     | phone_mispron_t1.0 | 0.522  | 0.574  | 0.631     | **0.657** | mispron-grp          | +0.026 [+0.011, +0.039] p=0.000 | +0.084 [+0.058, +0.103] p=0.000 | better                |
| hubert_base     | l2_phone_error     | 0.326  | 0.403  | 0.479     | **0.515** | ws                   | +0.036 [+0.021, +0.052] p=0.000 | +0.112 [+0.082, +0.139] p=0.000 | better                |
| hubert_base     | l2_word_error      | 0.596  | 0.615  | 0.684     | **0.696** | L10                  | +0.012 [+0.006, +0.018] p=0.000 | +0.081 [+0.064, +0.097] p=0.000 | better                |
| wav2vec2_base   | sent_accuracy      | 1.318  | 1.212  | 1.243     | **1.112** | sent_ws_pros         | -0.131 [-0.178, -0.083] p=0.000 | -0.100 [-0.154, -0.043] p=0.000 | better                |
| wav2vec2_base   | sent_fluency       | 1.067  | 0.957  | 1.001     | **0.892** | sent_ws_pros         | -0.109 [-0.167, -0.047] p=0.000 | -0.065 [-0.115, -0.021] p=0.000 | better                |
| wav2vec2_base   | sent_prosodic      | 1.105  | 0.997  | 1.015     | **0.903** | sent_ws_pros         | -0.112 [-0.166, -0.059] p=0.000 | -0.094 [-0.146, -0.043] p=0.000 | better                |
| wav2vec2_base   | sent_total         | 1.281  | 1.180  | 1.212     | **1.097** | sent_ws_pros         | -0.115 [-0.162, -0.073] p=0.000 | -0.083 [-0.143, -0.027] p=0.008 | better                |
| wav2vec2_base   | word_accuracy      | 1.704  | 1.687  | 1.699     | **1.661** | word_max_pros        | -0.038 [-0.068, -0.013] p=0.004 | -0.027 [-0.049, -0.002] p=0.036 | better                |
| wav2vec2_base   | phone_score        | 0.352  | 0.352  | 0.345     | **0.320** | ws                   | -0.025 [-0.035, -0.015] p=0.000 | -0.031 [-0.043, -0.020] p=0.000 | better                |
| wav2vec2_base   | phone_mispron_t1.0 | 0.522  | 0.574  | 0.594     | **0.643** | mispron-grp          | +0.049 [+0.017, +0.080] p=0.000 | +0.070 [+0.037, +0.096] p=0.000 | better                |
| wav2vec2_base   | l2_phone_error     | 0.326  | 0.403  | 0.395     | **0.509** | ws                   | +0.113 [+0.094, +0.132] p=0.000 | +0.106 [+0.084, +0.129] p=0.000 | better                |
| wav2vec2_base   | l2_word_error      | 0.596  | 0.615  | 0.619     | **0.691** | L9                   | +0.072 [+0.059, +0.087] p=0.000 | +0.075 [+0.058, +0.092] p=0.000 | better                |

Best encoder per task by validation (the frozen baseline carried into Sprint 5):

| task               | encoder         | system               | 4.3 test | 4.1 test | delta vs 4.1 | verdict               |
| ------------------ | --------------- | -------------------- | -------- | -------- | ------------ | --------------------- |
| sent_accuracy      | wavlm_base_plus | ws                   | 1.113    | 1.165    | -0.052       | better                |
| sent_fluency       | hubert_base     | L11                  | 0.916    | 0.939    | -0.023       | better                |
| sent_prosodic      | wavlm_base_plus | sent_ws              | 0.901    | 0.946    | -0.045       | better                |
| sent_total         | wavlm_base_plus | ws                   | 1.114    | 1.136    | -0.022       | no significant change |
| word_accuracy      | wavlm_base_plus | word_meanmaxstd_pros | 1.633    | 1.680    | -0.047       | better                |
| phone_score        | wavlm_base_plus | ws                   | 0.314    | 0.325    | -0.011       | better                |
| phone_mispron_t1.0 | wav2vec2_base   | mispron-grp          | 0.643    | 0.594    | +0.049       | better                |
| l2_phone_error     | wavlm_base_plus | ws                   | 0.523    | 0.508    | +0.014       | no significant change |
| l2_word_error      | wavlm_base_plus | L11                  | 0.716    | 0.709    | +0.007       | better                |

## 2. What each fix did (control vs fix, each side = best system of its group by validation)

| encoder         | task               | fix                                                                        | control                      | treatment                          | control | treatment | delta [95% CI]                  | verdict               |
| --------------- | ------------------ | -------------------------------------------------------------------------- | ---------------------------- | ---------------------------------- | ------- | --------- | ------------------------------- | --------------------- |
| wavlm_base_plus | sent_accuracy      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | 4.1 L12 / mlp (plain)              | 1.165   | 1.165     | +0.000 [+0.000, +0.000] p=1.000 | no significant change |
| wavlm_base_plus | sent_accuracy      | fix 2: learned layer-weighted sum vs best single layer                     | 4.1 L12 / mlp (plain)        | ws / mlp (plain)                   | 1.165   | 1.113     | -0.052 [-0.082, -0.023] p=0.000 | better                |
| wavlm_base_plus | sent_accuracy      | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / mlp (plain)         | 1.113   | 1.113     | -0.000 [-0.017, +0.015] p=0.932 | no significant change |
| wavlm_base_plus | sent_accuracy      | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 1.124   | 1.113     | -0.010 [-0.023, +0.001] p=0.092 | no significant change |
| wavlm_base_plus | sent_accuracy      | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 1.138   | 1.138     | -0.000 [-0.025, +0.025] p=0.940 | no significant change |
| wavlm_base_plus | sent_accuracy      | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 1.113   | 1.124     | +0.010 [+0.001, +0.020] p=0.044 | worse                 |
| wavlm_base_plus | sent_accuracy      | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 1.124   | 1.138     | +0.015 [-0.027, +0.054] p=0.428 | no significant change |
| wavlm_base_plus | sent_fluency       | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L8 / mlp (plain)                   | 0.937   | 0.939     | +0.002 [-0.013, +0.017] p=0.784 | no significant change |
| wavlm_base_plus | sent_fluency       | fix 2: learned layer-weighted sum vs best single layer                     | L8 / mlp (plain)             | ws / mlp (plain)                   | 0.939   | 0.868     | -0.071 [-0.096, -0.044] p=0.000 | better                |
| wavlm_base_plus | sent_fluency       | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_avgpca / mlp (plain)          | 0.868   | 0.915     | +0.047 [+0.014, +0.081] p=0.000 | worse                 |
| wavlm_base_plus | sent_fluency       | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 0.880   | 0.892     | +0.012 [-0.006, +0.032] p=0.264 | no significant change |
| wavlm_base_plus | sent_fluency       | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 0.915   | 0.899     | -0.016 [-0.047, +0.006] p=0.228 | no significant change |
| wavlm_base_plus | sent_fluency       | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.868   | 0.880     | +0.012 [-0.008, +0.033] p=0.296 | no significant change |
| wavlm_base_plus | sent_fluency       | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 0.880   | 0.915     | +0.035 [+0.011, +0.064] p=0.000 | worse                 |
| wavlm_base_plus | sent_prosodic      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L8 / mlp (plain)                   | 0.946   | 0.960     | +0.015 [-0.004, +0.030] p=0.104 | no significant change |
| wavlm_base_plus | sent_prosodic      | fix 2: learned layer-weighted sum vs best single layer                     | L8 / mlp (plain)             | ws / mlp (plain)                   | 0.960   | 0.896     | -0.064 [-0.088, -0.036] p=0.000 | better                |
| wavlm_base_plus | sent_prosodic      | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.896   | 0.901     | +0.005 [-0.011, +0.020] p=0.624 | no significant change |
| wavlm_base_plus | sent_prosodic      | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 0.901   | 0.904     | +0.003 [-0.003, +0.010] p=0.332 | no significant change |
| wavlm_base_plus | sent_prosodic      | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 0.918   | 0.916     | -0.002 [-0.026, +0.020] p=0.828 | no significant change |
| wavlm_base_plus | sent_prosodic      | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.896   | 0.901     | +0.005 [-0.011, +0.020] p=0.624 | no significant change |
| wavlm_base_plus | sent_prosodic      | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 0.901   | 0.918     | +0.017 [-0.022, +0.059] p=0.376 | no significant change |
| wavlm_base_plus | sent_total         | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | 4.1 L12 / mlp (plain)              | 1.136   | 1.136     | +0.000 [+0.000, +0.000] p=1.000 | no significant change |
| wavlm_base_plus | sent_total         | fix 2: learned layer-weighted sum vs best single layer                     | 4.1 L12 / mlp (plain)        | ws / mlp (plain)                   | 1.136   | 1.114     | -0.022 [-0.060, +0.014] p=0.244 | no significant change |
| wavlm_base_plus | sent_total         | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / linear (plain)      | 1.114   | 1.080     | -0.034 [-0.076, +0.008] p=0.104 | no significant change |
| wavlm_base_plus | sent_total         | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / linear (plain)      | 1.085   | 1.080     | -0.005 [-0.042, +0.030] p=0.844 | no significant change |
| wavlm_base_plus | sent_total         | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 1.108   | 1.115     | +0.007 [-0.022, +0.033] p=0.632 | no significant change |
| wavlm_base_plus | sent_total         | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 1.114   | 1.085     | -0.029 [-0.043, -0.016] p=0.000 | better                |
| wavlm_base_plus | sent_total         | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 1.085   | 1.108     | +0.023 [-0.022, +0.069] p=0.300 | no significant change |
| wavlm_base_plus | word_accuracy      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / linear (plain)     | L9 / mlp (plain)                   | 1.680   | 1.655     | -0.025 [-0.041, -0.011] p=0.000 | better                |
| wavlm_base_plus | word_accuracy      | fix 2: learned layer-weighted sum vs best single layer                     | L9 / mlp (plain)             | ws / mlp (plain)                   | 1.655   | 1.636     | -0.019 [-0.040, +0.003] p=0.092 | no significant change |
| wavlm_base_plus | word_accuracy      | fix 4a: prosody columns alone (mean pooling) vs ws                         | ws / mlp (plain)             | word_mean_pros / mlp (plain)       | 1.636   | 1.624     | -0.013 [-0.020, -0.006] p=0.000 | better                |
| wavlm_base_plus | word_accuracy      | fix 4: max / mean+max / mean+max+std pooling vs mean pooling (same inputs) | word_mean_pros / mlp (plain) | word_meanmaxstd_pros / mlp (plain) | 1.624   | 1.633     | +0.009 [-0.002, +0.021] p=0.088 | no significant change |
| wavlm_base_plus | word_accuracy      | fix 4 (max): pooling vs mean pooling                                       | word_mean_pros / mlp (plain) | word_max_pros / mlp (plain)        | 1.624   | 1.643     | +0.019 [-0.004, +0.039] p=0.100 | no significant change |
| wavlm_base_plus | word_accuracy      | fix 4 (meanmax): pooling vs mean pooling                                   | word_mean_pros / mlp (plain) | word_meanmax_pros / mlp (plain)    | 1.624   | 1.651     | +0.027 [+0.011, +0.044] p=0.000 | worse                 |
| wavlm_base_plus | word_accuracy      | fix 4 (meanmaxstd): pooling vs mean pooling                                | word_mean_pros / mlp (plain) | word_meanmaxstd_pros / mlp (plain) | 1.624   | 1.633     | +0.009 [-0.002, +0.021] p=0.088 | no significant change |
| wavlm_base_plus | phone_score        | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | 4.1 L12 / mlp (plain)              | 0.325   | 0.325     | +0.000 [+0.000, +0.000] p=1.000 | no significant change |
| wavlm_base_plus | phone_score        | fix 2: learned layer-weighted sum vs best single layer                     | 4.1 L12 / mlp (plain)        | ws / mlp (plain)                   | 0.325   | 0.314     | -0.011 [-0.018, -0.004] p=0.000 | better                |
| wavlm_base_plus | phone_mispron_t1.0 | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / linear (cw+thr)    | L11 / linear (cw+thr)              | 0.621   | 0.637     | +0.016 [-0.003, +0.034] p=0.076 | no significant change |
| wavlm_base_plus | phone_mispron_t1.0 | fix 2: learned layer-weighted sum vs best single layer                     | L11 / linear (cw+thr)        | ws / mlp (plain)                   | 0.637   | 0.593     | -0.043 [-0.066, -0.019] p=0.004 | worse                 |
| wavlm_base_plus | phone_mispron_t1.0 | fix 5: age-group decision weights vs the 4.1 / 3.3 cw+thr rule (same head) | mispron-ctrl / mlp (cw+thr)  | mispron-grp / mlp (group-oof)      | 0.629   | 0.661     | +0.032 [+0.016, +0.047] p=0.000 | better                |
| wavlm_base_plus | l2_phone_error     | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw+thr)       | L11 / mlp (cw+thr)                 | 0.508   | 0.528     | +0.020 [+0.013, +0.027] p=0.000 | better                |
| wavlm_base_plus | l2_phone_error     | fix 2: learned layer-weighted sum vs best single layer                     | L11 / mlp (cw+thr)           | ws / mlp (plain)                   | 0.528   | 0.523     | -0.005 [-0.029, +0.016] p=0.580 | no significant change |
| wavlm_base_plus | l2_word_error      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw+thr)       | L11 / mlp (cw+thr)                 | 0.709   | 0.716     | +0.007 [+0.002, +0.013] p=0.004 | better                |
| wavlm_base_plus | l2_word_error      | fix 2: learned layer-weighted sum vs best single layer                     | L11 / mlp (cw+thr)           | ws / linear (cw+thr)               | 0.716   | 0.697     | -0.018 [-0.025, -0.011] p=0.000 | worse                 |
| hubert_base     | sent_accuracy      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L11 / mlp (plain)                  | 1.169   | 1.162     | -0.007 [-0.030, +0.017] p=0.624 | no significant change |
| hubert_base     | sent_accuracy      | fix 2: learned layer-weighted sum vs best single layer                     | L11 / mlp (plain)            | ws / mlp (plain)                   | 1.162   | 1.115     | -0.047 [-0.075, -0.019] p=0.000 | better                |
| hubert_base     | sent_accuracy      | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / mlp (plain)         | 1.115   | 1.105     | -0.010 [-0.018, -0.003] p=0.008 | better                |
| hubert_base     | sent_accuracy      | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 1.105   | 1.105     | -0.001 [-0.008, +0.007] p=0.912 | no significant change |
| hubert_base     | sent_accuracy      | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 1.146   | 1.141     | -0.006 [-0.026, +0.013] p=0.608 | no significant change |
| hubert_base     | sent_accuracy      | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 1.115   | 1.105     | -0.010 [-0.020, -0.000] p=0.052 | better                |
| hubert_base     | sent_accuracy      | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 1.105   | 1.146     | +0.041 [+0.006, +0.073] p=0.020 | worse                 |
| hubert_base     | sent_fluency       | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L11 / mlp (plain)                  | 0.939   | 0.916     | -0.023 [-0.047, -0.003] p=0.008 | better                |
| hubert_base     | sent_fluency       | fix 2: learned layer-weighted sum vs best single layer                     | L11 / mlp (plain)            | ws / mlp (plain)                   | 0.916   | 0.881     | -0.035 [-0.048, -0.022] p=0.000 | better                |
| hubert_base     | sent_fluency       | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / linear (plain)      | 0.881   | 0.894     | +0.013 [-0.040, +0.070] p=0.600 | no significant change |
| hubert_base     | sent_fluency       | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / linear (plain)      | 0.882   | 0.894     | +0.013 [-0.039, +0.065] p=0.604 | no significant change |
| hubert_base     | sent_fluency       | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 0.910   | 0.888     | -0.021 [-0.048, +0.001] p=0.064 | no significant change |
| hubert_base     | sent_fluency       | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.881   | 0.882     | +0.001 [-0.005, +0.008] p=0.764 | no significant change |
| hubert_base     | sent_fluency       | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 0.882   | 0.910     | +0.028 [+0.002, +0.057] p=0.044 | worse                 |
| hubert_base     | sent_prosodic      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L10 / mlp (plain)                  | 0.960   | 0.933     | -0.027 [-0.050, -0.001] p=0.032 | better                |
| hubert_base     | sent_prosodic      | fix 2: learned layer-weighted sum vs best single layer                     | L10 / mlp (plain)            | ws / mlp (plain)                   | 0.933   | 0.896     | -0.037 [-0.055, -0.024] p=0.000 | better                |
| hubert_base     | sent_prosodic      | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.896   | 0.893     | -0.003 [-0.011, +0.005] p=0.460 | no significant change |
| hubert_base     | sent_prosodic      | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 0.893   | 0.898     | +0.004 [-0.004, +0.013] p=0.332 | no significant change |
| hubert_base     | sent_prosodic      | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 0.926   | 0.928     | +0.001 [-0.027, +0.034] p=0.988 | no significant change |
| hubert_base     | sent_prosodic      | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.896   | 0.893     | -0.003 [-0.011, +0.005] p=0.460 | no significant change |
| hubert_base     | sent_prosodic      | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 0.893   | 0.926     | +0.033 [+0.007, +0.060] p=0.012 | worse                 |
| hubert_base     | sent_total         | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L11 / mlp (plain)                  | 1.155   | 1.129     | -0.026 [-0.052, +0.000] p=0.064 | no significant change |
| hubert_base     | sent_total         | fix 2: learned layer-weighted sum vs best single layer                     | L11 / mlp (plain)            | ws / mlp (plain)                   | 1.129   | 1.095     | -0.034 [-0.061, -0.006] p=0.024 | better                |
| hubert_base     | sent_total         | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / mlp (plain)         | 1.095   | 1.077     | -0.018 [-0.039, +0.003] p=0.116 | no significant change |
| hubert_base     | sent_total         | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 1.085   | 1.077     | -0.007 [-0.014, -0.001] p=0.028 | better                |
| hubert_base     | sent_total         | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 1.107   | 1.108     | +0.000 [-0.026, +0.031] p=0.940 | no significant change |
| hubert_base     | sent_total         | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 1.095   | 1.085     | -0.011 [-0.031, +0.010] p=0.284 | no significant change |
| hubert_base     | sent_total         | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 1.085   | 1.107     | +0.023 [-0.025, +0.067] p=0.356 | no significant change |
| hubert_base     | word_accuracy      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L9 / mlp (plain)                   | 1.663   | 1.678     | +0.015 [-0.001, +0.033] p=0.084 | no significant change |
| hubert_base     | word_accuracy      | fix 2: learned layer-weighted sum vs best single layer                     | L9 / mlp (plain)             | ws / mlp (plain)                   | 1.678   | 1.661     | -0.017 [-0.030, -0.003] p=0.012 | better                |
| hubert_base     | word_accuracy      | fix 4a: prosody columns alone (mean pooling) vs ws                         | ws / mlp (plain)             | word_mean_pros / mlp (plain)       | 1.661   | 1.640     | -0.022 [-0.032, -0.011] p=0.000 | better                |
| hubert_base     | word_accuracy      | fix 4: max / mean+max / mean+max+std pooling vs mean pooling (same inputs) | word_mean_pros / mlp (plain) | word_max_pros / mlp (plain)        | 1.640   | 1.656     | +0.017 [-0.005, +0.035] p=0.124 | no significant change |
| hubert_base     | word_accuracy      | fix 4 (max): pooling vs mean pooling                                       | word_mean_pros / mlp (plain) | word_max_pros / mlp (plain)        | 1.640   | 1.656     | +0.017 [-0.005, +0.035] p=0.124 | no significant change |
| hubert_base     | word_accuracy      | fix 4 (meanmax): pooling vs mean pooling                                   | word_mean_pros / mlp (plain) | word_meanmax_pros / mlp (plain)    | 1.640   | 1.660     | +0.020 [-0.004, +0.043] p=0.112 | no significant change |
| hubert_base     | word_accuracy      | fix 4 (meanmaxstd): pooling vs mean pooling                                | word_mean_pros / mlp (plain) | word_meanmaxstd_pros / mlp (plain) | 1.640   | 1.678     | +0.039 [+0.003, +0.074] p=0.040 | worse                 |
| hubert_base     | phone_score        | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | 4.1 L12 / mlp (plain)              | 0.325   | 0.325     | +0.000 [+0.000, +0.000] p=1.000 | no significant change |
| hubert_base     | phone_score        | fix 2: learned layer-weighted sum vs best single layer                     | 4.1 L12 / mlp (plain)        | ws / mlp (plain)                   | 0.325   | 0.319     | -0.006 [-0.010, -0.001] p=0.020 | better                |
| hubert_base     | phone_mispron_t1.0 | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw+thr)       | L8 / linear (cw+thr)               | 0.631   | 0.617     | -0.014 [-0.026, -0.001] p=0.036 | worse                 |
| hubert_base     | phone_mispron_t1.0 | fix 2: learned layer-weighted sum vs best single layer                     | L8 / linear (cw+thr)         | ws / mlp (cw+thr)                  | 0.617   | 0.629     | +0.012 [-0.002, +0.026] p=0.088 | no significant change |
| hubert_base     | phone_mispron_t1.0 | fix 5: age-group decision weights vs the 4.1 / 3.3 cw+thr rule (same head) | mispron-ctrl / mlp (cw+thr)  | mispron-grp / mlp (group-oof)      | 0.629   | 0.657     | +0.028 [+0.009, +0.046] p=0.000 | better                |
| hubert_base     | l2_phone_error     | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw+thr)       | L11 / mlp (cw+thr)                 | 0.479   | 0.497     | +0.018 [+0.006, +0.030] p=0.004 | better                |
| hubert_base     | l2_phone_error     | fix 2: learned layer-weighted sum vs best single layer                     | L11 / mlp (cw+thr)           | ws / mlp (plain)                   | 0.497   | 0.515     | +0.018 [+0.001, +0.035] p=0.024 | better                |
| hubert_base     | l2_word_error      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw+thr)       | L10 / mlp (cw+thr)                 | 0.684   | 0.696     | +0.012 [+0.006, +0.018] p=0.000 | better                |
| hubert_base     | l2_word_error      | fix 2: learned layer-weighted sum vs best single layer                     | L10 / mlp (cw+thr)           | ws / mlp (cw+thr)                  | 0.696   | 0.695     | -0.002 [-0.008, +0.004] p=0.664 | no significant change |
| wav2vec2_base   | sent_accuracy      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L9 / mlp (plain)                   | 1.243   | 1.162     | -0.081 [-0.136, -0.023] p=0.008 | better                |
| wav2vec2_base   | sent_accuracy      | fix 2: learned layer-weighted sum vs best single layer                     | L9 / mlp (plain)             | ws / mlp (plain)                   | 1.162   | 1.135     | -0.027 [-0.065, +0.011] p=0.204 | no significant change |
| wav2vec2_base   | sent_accuracy      | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / mlp (plain)         | 1.135   | 1.112     | -0.023 [-0.046, +0.000] p=0.052 | no significant change |
| wav2vec2_base   | sent_accuracy      | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 1.122   | 1.112     | -0.010 [-0.019, -0.000] p=0.052 | better                |
| wav2vec2_base   | sent_accuracy      | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 1.142   | 1.143     | +0.001 [-0.027, +0.029] p=0.948 | no significant change |
| wav2vec2_base   | sent_accuracy      | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 1.135   | 1.122     | -0.013 [-0.039, +0.011] p=0.316 | no significant change |
| wav2vec2_base   | sent_accuracy      | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 1.122   | 1.142     | +0.020 [-0.029, +0.066] p=0.400 | no significant change |
| wav2vec2_base   | sent_fluency       | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L7 / mlp (plain)                   | 1.001   | 0.935     | -0.065 [-0.122, -0.014] p=0.016 | better                |
| wav2vec2_base   | sent_fluency       | fix 2: learned layer-weighted sum vs best single layer                     | L7 / mlp (plain)             | ws / linear (plain)                | 0.935   | 0.909     | -0.027 [-0.062, +0.009] p=0.164 | no significant change |
| wav2vec2_base   | sent_fluency       | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / linear (plain)          | sent_ws_pros / mlp (plain)         | 0.909   | 0.892     | -0.017 [-0.059, +0.029] p=0.420 | no significant change |
| wav2vec2_base   | sent_fluency       | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 0.898   | 0.892     | -0.007 [-0.017, +0.002] p=0.132 | no significant change |
| wav2vec2_base   | sent_fluency       | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 0.920   | 0.914     | -0.005 [-0.035, +0.023] p=0.712 | no significant change |
| wav2vec2_base   | sent_fluency       | fix 3c: stronger regularisation alone (ws input)                           | ws / linear (plain)          | sent_ws / mlp (plain)              | 0.909   | 0.898     | -0.010 [-0.050, +0.033] p=0.592 | no significant change |
| wav2vec2_base   | sent_fluency       | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 0.898   | 0.920     | +0.021 [-0.008, +0.056] p=0.172 | no significant change |
| wav2vec2_base   | sent_prosodic      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L10 / mlp (plain)                  | 1.015   | 0.958     | -0.057 [-0.116, +0.011] p=0.092 | no significant change |
| wav2vec2_base   | sent_prosodic      | fix 2: learned layer-weighted sum vs best single layer                     | L10 / mlp (plain)            | ws / mlp (plain)                   | 0.958   | 0.914     | -0.044 [-0.069, -0.016] p=0.008 | better                |
| wav2vec2_base   | sent_prosodic      | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / mlp (plain)         | 0.914   | 0.903     | -0.011 [-0.025, +0.002] p=0.112 | no significant change |
| wav2vec2_base   | sent_prosodic      | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 0.903   | 0.903     | +0.001 [-0.011, +0.012] p=0.960 | no significant change |
| wav2vec2_base   | sent_prosodic      | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 0.919   | 0.920     | +0.001 [-0.023, +0.031] p=0.988 | no significant change |
| wav2vec2_base   | sent_prosodic      | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 0.914   | 0.903     | -0.011 [-0.019, -0.003] p=0.000 | better                |
| wav2vec2_base   | sent_prosodic      | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 0.903   | 0.919     | +0.017 [-0.002, +0.038] p=0.100 | no significant change |
| wav2vec2_base   | sent_total         | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L9 / mlp (plain)                   | 1.212   | 1.137     | -0.075 [-0.132, -0.022] p=0.004 | better                |
| wav2vec2_base   | sent_total         | fix 2: learned layer-weighted sum vs best single layer                     | L9 / mlp (plain)             | ws / mlp (plain)                   | 1.137   | 1.101     | -0.036 [-0.059, -0.009] p=0.024 | better                |
| wav2vec2_base   | sent_total         | fix 3: sentence regularisation (best of 4 recipes) vs ws                   | ws / mlp (plain)             | sent_ws_pros / mlp (plain)         | 1.101   | 1.097     | -0.004 [-0.017, +0.013] p=0.568 | no significant change |
| wav2vec2_base   | sent_total         | fix 3a: + US 3.3 prosody columns (ws input)                                | sent_ws / mlp (plain)        | sent_ws_pros / mlp (plain)         | 1.092   | 1.097     | +0.005 [-0.004, +0.016] p=0.308 | no significant change |
| wav2vec2_base   | sent_total         | fix 3b: + US 3.3 prosody columns (PCA input)                               | sent_avgpca / mlp (plain)    | sent_avgpca_pros / mlp (plain)     | 1.105   | 1.127     | +0.022 [-0.015, +0.059] p=0.308 | no significant change |
| wav2vec2_base   | sent_total         | fix 3c: stronger regularisation alone (ws input)                           | ws / mlp (plain)             | sent_ws / mlp (plain)              | 1.101   | 1.092     | -0.009 [-0.021, +0.005] p=0.200 | no significant change |
| wav2vec2_base   | sent_total         | fix 3d: PCA on the layer-averaged input vs regularised ws                  | sent_ws / mlp (plain)        | sent_avgpca / mlp (plain)          | 1.092   | 1.105     | +0.013 [-0.033, +0.059] p=0.600 | no significant change |
| wav2vec2_base   | word_accuracy      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L6 / mlp (plain)                   | 1.699   | 1.682     | -0.017 [-0.046, +0.011] p=0.252 | no significant change |
| wav2vec2_base   | word_accuracy      | fix 2: learned layer-weighted sum vs best single layer                     | L6 / mlp (plain)             | ws / mlp (plain)                   | 1.682   | 1.687     | +0.004 [-0.017, +0.026] p=0.684 | no significant change |
| wav2vec2_base   | word_accuracy      | fix 4a: prosody columns alone (mean pooling) vs ws                         | ws / mlp (plain)             | word_mean_pros / mlp (plain)       | 1.687   | 1.685     | -0.002 [-0.007, +0.004] p=0.616 | no significant change |
| wav2vec2_base   | word_accuracy      | fix 4: max / mean+max / mean+max+std pooling vs mean pooling (same inputs) | word_mean_pros / mlp (plain) | word_max_pros / mlp (plain)        | 1.685   | 1.661     | -0.024 [-0.053, -0.002] p=0.036 | better                |
| wav2vec2_base   | word_accuracy      | fix 4 (max): pooling vs mean pooling                                       | word_mean_pros / mlp (plain) | word_max_pros / mlp (plain)        | 1.685   | 1.661     | -0.024 [-0.053, -0.002] p=0.036 | better                |
| wav2vec2_base   | word_accuracy      | fix 4 (meanmax): pooling vs mean pooling                                   | word_mean_pros / mlp (plain) | word_meanmax_pros / mlp (plain)    | 1.685   | 1.663     | -0.022 [-0.051, +0.003] p=0.092 | no significant change |
| wav2vec2_base   | word_accuracy      | fix 4 (meanmaxstd): pooling vs mean pooling                                | word_mean_pros / mlp (plain) | word_meanmaxstd_pros / mlp (plain) | 1.685   | 1.685     | +0.000 [-0.043, +0.040] p=0.956 | no significant change |
| wav2vec2_base   | phone_score        | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (plain)        | L6 / mlp (plain)                   | 0.345   | 0.333     | -0.012 [-0.021, -0.002] p=0.028 | better                |
| wav2vec2_base   | phone_score        | fix 2: learned layer-weighted sum vs best single layer                     | L6 / mlp (plain)             | ws / mlp (plain)                   | 0.333   | 0.320     | -0.013 [-0.015, -0.011] p=0.000 | better                |
| wav2vec2_base   | phone_mispron_t1.0 | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw+thr)       | L10 / mlp (plain)                  | 0.594   | 0.569     | -0.024 [-0.044, -0.001] p=0.040 | worse                 |
| wav2vec2_base   | phone_mispron_t1.0 | fix 2: learned layer-weighted sum vs best single layer                     | L10 / mlp (plain)            | ws / mlp (plain)                   | 0.569   | 0.580     | +0.011 [-0.006, +0.027] p=0.228 | no significant change |
| wav2vec2_base   | phone_mispron_t1.0 | fix 5: age-group decision weights vs the 4.1 / 3.3 cw+thr rule (same head) | mispron-ctrl / mlp (cw+thr)  | mispron-grp / mlp (global-oof)     | 0.643   | 0.643     | +0.000 [+0.000, +0.000] p=1.000 | no significant change |
| wav2vec2_base   | l2_phone_error     | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / linear (cw+thr)    | L9 / mlp (cw+thr)                  | 0.395   | 0.485     | +0.090 [+0.072, +0.107] p=0.000 | better                |
| wav2vec2_base   | l2_phone_error     | fix 2: learned layer-weighted sum vs best single layer                     | L9 / mlp (cw+thr)            | ws / mlp (cw+thr)                  | 0.485   | 0.509     | +0.024 [+0.013, +0.034] p=0.000 | better                |
| wav2vec2_base   | l2_word_error      | fix 1: layer re-selection (6-12) vs layer 12                               | 4.1 L12 / mlp (cw)           | L9 / mlp (cw)                      | 0.619   | 0.691     | +0.072 [+0.059, +0.087] p=0.000 | better                |
| wav2vec2_base   | l2_word_error      | fix 2: learned layer-weighted sum vs best single layer                     | L9 / mlp (cw)                | ws / mlp (cw)                      | 0.691   | 0.682     | -0.009 [-0.018, -0.001] p=0.020 | worse                 |

## 3. Layers the learned weighted sum relies on (softmax weights, mean over folds; uniform = 0.077)

| encoder         | task               | top layers (weight)                | normalised entropy (1 = uniform) |
| --------------- | ------------------ | ---------------------------------- | -------------------------------- |
| wavlm_base_plus | sent_accuracy      | L0 (0.08), L12 (0.08), L11 (0.08)  | 1.00                             |
| wavlm_base_plus | sent_fluency       | L12 (0.08), L0 (0.08), L11 (0.08)  | 1.00                             |
| wavlm_base_plus | sent_prosodic      | L12 (0.08), L11 (0.08), L0 (0.08)  | 1.00                             |
| wavlm_base_plus | sent_total         | L0 (0.08), L12 (0.08), L1 (0.08)   | 1.00                             |
| wavlm_base_plus | word_accuracy      | L1 (0.08), L0 (0.08), L2 (0.08)    | 1.00                             |
| wavlm_base_plus | phone_score        | L0 (0.11), L1 (0.10), L9 (0.08)    | 0.99                             |
| wavlm_base_plus | phone_mispron_t1.0 | L0 (0.08), L2 (0.08), L1 (0.08)    | 1.00                             |
| wavlm_base_plus | l2_phone_error     | L11 (0.09), L10 (0.09), L12 (0.08) | 1.00                             |
| wavlm_base_plus | l2_word_error      | L11 (0.14), L10 (0.12), L12 (0.10) | 0.98                             |
| hubert_base     | sent_accuracy      | L12 (0.08), L0 (0.08), L7 (0.08)   | 1.00                             |
| hubert_base     | sent_fluency       | L12 (0.08), L0 (0.08), L7 (0.08)   | 1.00                             |
| hubert_base     | sent_prosodic      | L12 (0.08), L0 (0.08), L7 (0.08)   | 1.00                             |
| hubert_base     | sent_total         | L12 (0.08), L0 (0.08), L7 (0.08)   | 1.00                             |
| hubert_base     | word_accuracy      | L2 (0.08), L1 (0.08), L0 (0.08)    | 1.00                             |
| hubert_base     | phone_score        | L9 (0.09), L2 (0.09), L8 (0.09)    | 1.00                             |
| hubert_base     | phone_mispron_t1.0 | L10 (0.08), L2 (0.08), L3 (0.08)   | 1.00                             |
| hubert_base     | l2_phone_error     | L10 (0.09), L11 (0.09), L9 (0.08)  | 1.00                             |
| hubert_base     | l2_word_error      | L10 (0.08), L9 (0.08), L11 (0.08)  | 1.00                             |
| wav2vec2_base   | sent_accuracy      | L12 (0.08), L8 (0.08), L0 (0.08)   | 1.00                             |
| wav2vec2_base   | sent_fluency       | L12 (0.08), L11 (0.08), L0 (0.08)  | 1.00                             |
| wav2vec2_base   | sent_prosodic      | L8 (0.08), L12 (0.08), L6 (0.08)   | 1.00                             |
| wav2vec2_base   | sent_total         | L8 (0.08), L12 (0.08), L6 (0.08)   | 1.00                             |
| wav2vec2_base   | word_accuracy      | L2 (0.08), L12 (0.08), L7 (0.08)   | 1.00                             |
| wav2vec2_base   | phone_score        | L12 (0.09), L7 (0.09), L4 (0.08)   | 1.00                             |
| wav2vec2_base   | phone_mispron_t1.0 | L7 (0.08), L8 (0.08), L6 (0.08)    | 1.00                             |
| wav2vec2_base   | l2_phone_error     | L8 (0.09), L9 (0.09), L7 (0.09)    | 1.00                             |
| wav2vec2_base   | l2_word_error      | L8 (0.08), L9 (0.08), L7 (0.08)    | 1.00                             |

## 4. Fix 5: mispronunciation decision weights by age group

Weights for the `*-oof` strategies are tuned on speaker-grouped out-of-fold TRAIN probabilities (validation scored with weights that never saw validation; test weights use OOF-train + validation). Precision / recall / F1 are for the mispronounced class.

| encoder         | strategy     | split | group      | n (pos)      | macro-F1 | precision | recall | F1    | AUC   | AP    |
| --------------- | ------------ | ----- | ---------- | ------------ | -------- | --------- | ------ | ----- | ----- | ----- |
| wavlm_base_plus | cw+thr       | val   | all        | 7658 (343)   | 0.630    | 0.236     | 0.440  | 0.307 | 0.889 | 0.279 |
| wavlm_base_plus | cw+thr       | val   | adult      | 4469 (315)   | 0.619    | 0.238     | 0.479  | 0.318 | 0.859 | 0.294 |
| wavlm_base_plus | cw+thr       | val   | child/teen | 3189 (28)    | 0.497    | 0.000     | 0.000  | 0.000 | 0.819 | 0.071 |
| wavlm_base_plus | cw+thr       | test  | all        | 47369 (1978) | 0.629    | 0.408     | 0.216  | 0.282 | 0.880 | 0.282 |
| wavlm_base_plus | cw+thr       | test  | adult      | 25752 (1340) | 0.657    | 0.437     | 0.284  | 0.344 | 0.900 | 0.340 |
| wavlm_base_plus | cw+thr       | test  | child/teen | 21617 (638)  | 0.549    | 0.264     | 0.074  | 0.115 | 0.830 | 0.151 |
| wavlm_base_plus | global-oof   | val   | all        | 7658 (343)   | 0.647    | 0.261     | 0.481  | 0.339 | 0.889 | 0.279 |
| wavlm_base_plus | global-oof   | val   | adult      | 4469 (315)   | 0.638    | 0.265     | 0.521  | 0.351 | 0.859 | 0.294 |
| wavlm_base_plus | global-oof   | val   | child/teen | 3189 (28)    | 0.522    | 0.083     | 0.036  | 0.050 | 0.819 | 0.071 |
| wavlm_base_plus | global-oof   | test  | all        | 47369 (1978) | 0.655    | 0.358     | 0.318  | 0.337 | 0.880 | 0.282 |
| wavlm_base_plus | global-oof   | test  | adult      | 25752 (1340) | 0.678    | 0.380     | 0.401  | 0.390 | 0.900 | 0.340 |
| wavlm_base_plus | global-oof   | test  | child/teen | 21617 (638)  | 0.584    | 0.269     | 0.144  | 0.188 | 0.830 | 0.151 |
| wavlm_base_plus | group-oof *  | val   | all        | 7658 (343)   | 0.648    | 0.255     | 0.528  | 0.344 | 0.889 | 0.279 |
| wavlm_base_plus | group-oof *  | val   | adult      | 4469 (315)   | 0.636    | 0.258     | 0.571  | 0.355 | 0.859 | 0.294 |
| wavlm_base_plus | group-oof *  | val   | child/teen | 3189 (28)    | 0.522    | 0.083     | 0.036  | 0.050 | 0.819 | 0.071 |
| wavlm_base_plus | group-oof *  | test  | all        | 47369 (1978) | 0.661    | 0.344     | 0.359  | 0.351 | 0.880 | 0.282 |
| wavlm_base_plus | group-oof *  | test  | adult      | 25752 (1340) | 0.685    | 0.369     | 0.452  | 0.406 | 0.900 | 0.340 |
| wavlm_base_plus | group-oof *  | test  | child/teen | 21617 (638)  | 0.588    | 0.246     | 0.165  | 0.197 | 0.830 | 0.151 |
| hubert_base     | cw+thr       | val   | all        | 7658 (343)   | 0.633    | 0.231     | 0.522  | 0.320 | 0.894 | 0.261 |
| hubert_base     | cw+thr       | val   | adult      | 4469 (315)   | 0.620    | 0.235     | 0.568  | 0.332 | 0.866 | 0.277 |
| hubert_base     | cw+thr       | val   | child/teen | 3189 (28)    | 0.497    | 0.000     | 0.000  | 0.000 | 0.799 | 0.045 |
| hubert_base     | cw+thr       | test  | all        | 47369 (1978) | 0.629    | 0.403     | 0.217  | 0.283 | 0.871 | 0.276 |
| hubert_base     | cw+thr       | test  | adult      | 25752 (1340) | 0.655    | 0.436     | 0.278  | 0.340 | 0.893 | 0.339 |
| hubert_base     | cw+thr       | test  | child/teen | 21617 (638)  | 0.558    | 0.270     | 0.089  | 0.134 | 0.815 | 0.140 |
| hubert_base     | global-oof   | val   | all        | 7658 (343)   | 0.647    | 0.251     | 0.545  | 0.344 | 0.894 | 0.261 |
| hubert_base     | global-oof   | val   | adult      | 4469 (315)   | 0.636    | 0.256     | 0.594  | 0.358 | 0.866 | 0.277 |
| hubert_base     | global-oof   | val   | child/teen | 3189 (28)    | 0.497    | 0.000     | 0.000  | 0.000 | 0.799 | 0.045 |
| hubert_base     | global-oof   | test  | all        | 47369 (1978) | 0.658    | 0.330     | 0.362  | 0.345 | 0.871 | 0.276 |
| hubert_base     | global-oof   | test  | adult      | 25752 (1340) | 0.682    | 0.363     | 0.451  | 0.402 | 0.893 | 0.339 |
| hubert_base     | global-oof   | test  | child/teen | 21617 (638)  | 0.587    | 0.222     | 0.177  | 0.197 | 0.815 | 0.140 |
| hubert_base     | group-oof *  | val   | all        | 7658 (343)   | 0.646    | 0.249     | 0.545  | 0.342 | 0.894 | 0.261 |
| hubert_base     | group-oof *  | val   | adult      | 4469 (315)   | 0.636    | 0.256     | 0.594  | 0.358 | 0.866 | 0.277 |
| hubert_base     | group-oof *  | val   | child/teen | 3189 (28)    | 0.496    | 0.000     | 0.000  | 0.000 | 0.799 | 0.045 |
| hubert_base     | group-oof *  | test  | all        | 47369 (1978) | 0.657    | 0.323     | 0.372  | 0.345 | 0.871 | 0.276 |
| hubert_base     | group-oof *  | test  | adult      | 25752 (1340) | 0.682    | 0.363     | 0.451  | 0.402 | 0.893 | 0.339 |
| hubert_base     | group-oof *  | test  | child/teen | 21617 (638)  | 0.593    | 0.214     | 0.205  | 0.210 | 0.815 | 0.140 |
| wav2vec2_base   | cw+thr       | val   | all        | 7658 (343)   | 0.646    | 0.267     | 0.452  | 0.335 | 0.886 | 0.281 |
| wav2vec2_base   | cw+thr       | val   | adult      | 4469 (315)   | 0.637    | 0.268     | 0.489  | 0.346 | 0.860 | 0.299 |
| wav2vec2_base   | cw+thr       | val   | child/teen | 3189 (28)    | 0.526    | 0.143     | 0.036  | 0.057 | 0.760 | 0.041 |
| wav2vec2_base   | cw+thr       | test  | all        | 47369 (1978) | 0.643    | 0.359     | 0.277  | 0.313 | 0.865 | 0.267 |
| wav2vec2_base   | cw+thr       | test  | adult      | 25752 (1340) | 0.670    | 0.388     | 0.361  | 0.374 | 0.890 | 0.332 |
| wav2vec2_base   | cw+thr       | test  | child/teen | 21617 (638)  | 0.561    | 0.232     | 0.100  | 0.140 | 0.802 | 0.122 |
| wav2vec2_base   | global-oof * | val   | all        | 7658 (343)   | 0.653    | 0.263     | 0.536  | 0.353 | 0.886 | 0.281 |
| wav2vec2_base   | global-oof * | val   | adult      | 4469 (315)   | 0.642    | 0.266     | 0.581  | 0.365 | 0.860 | 0.299 |
| wav2vec2_base   | global-oof * | val   | child/teen | 3189 (28)    | 0.523    | 0.091     | 0.036  | 0.051 | 0.760 | 0.041 |
| wav2vec2_base   | global-oof * | test  | all        | 47369 (1978) | 0.643    | 0.359     | 0.277  | 0.313 | 0.865 | 0.267 |
| wav2vec2_base   | global-oof * | test  | adult      | 25752 (1340) | 0.670    | 0.388     | 0.361  | 0.374 | 0.890 | 0.332 |
| wav2vec2_base   | global-oof * | test  | child/teen | 21617 (638)  | 0.561    | 0.232     | 0.100  | 0.140 | 0.802 | 0.122 |
| wav2vec2_base   | group-oof    | val   | all        | 7658 (343)   | 0.646    | 0.241     | 0.636  | 0.349 | 0.886 | 0.281 |
| wav2vec2_base   | group-oof    | val   | adult      | 4469 (315)   | 0.629    | 0.243     | 0.689  | 0.359 | 0.860 | 0.299 |
| wav2vec2_base   | group-oof    | val   | child/teen | 3189 (28)    | 0.523    | 0.091     | 0.036  | 0.051 | 0.760 | 0.041 |
| wav2vec2_base   | group-oof    | test  | all        | 47369 (1978) | 0.646    | 0.335     | 0.306  | 0.320 | 0.865 | 0.267 |
| wav2vec2_base   | group-oof    | test  | adult      | 25752 (1340) | 0.671    | 0.361     | 0.396  | 0.378 | 0.890 | 0.332 |
| wav2vec2_base   | group-oof    | test  | child/teen | 21617 (638)  | 0.567    | 0.223     | 0.118  | 0.154 | 0.802 | 0.122 |

`*` = strategy kept by the declared rule (group-specific weights if pooled validation macro-F1 is within the tolerance of the global OOF weights).
