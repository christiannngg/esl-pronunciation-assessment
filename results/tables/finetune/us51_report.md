# US 5.1 - fine-tuned encoder vs Sprints 3-4 (tag `base`, test set; every choice made on validation)

Metric: RMSE (regression, lower is better) or macro-F1 (classification, higher is better). Fine-tuned = seed-averaged predictions; for classification the plain class-weighted argmax or the validation-tuned `cw+thr` decision weights, whichever scored better on validation. Delta = fine-tuned minus the frozen system of the same encoder on the identical test rows, paired speaker bootstrap (95% CI, two-sided p).

| task               | metric   | US 3.1 | US 3.3 | frozen 4.1 | frozen 4.3 (system)          | **fine-tuned** [95% CI]  | rule   | delta vs 4.3 [95% CI]           | delta vs 4.1     | verdict vs 4.3        |
| ------------------ | -------- | ------ | ------ | ---------- | ---------------------------- | ------------------------ | ------ | ------------------------------- | ---------------- | --------------------- |
| sent_accuracy      | rmse     | 1.318  | 1.212  | 1.165      | 1.113 (ws)                   | **1.071** [0.955, 1.174] | plain  | -0.042 [-0.067, -0.020] p=0.000 | -0.094 (p=0.000) | better                |
| sent_fluency       | rmse     | 1.067  | 0.957  | 0.937      | 0.915 (sent_avgpca)          | **0.870** [0.710, 1.065] | plain  | -0.045 [-0.073, -0.024] p=0.000 | -0.067 (p=0.000) | better                |
| sent_prosodic      | rmse     | 1.105  | 0.997  | 0.946      | 0.901 (sent_ws)              | **0.902** [0.744, 1.093] | plain  | +0.001 [-0.017, +0.018] p=0.932 | -0.043 (p=0.000) | no significant change |
| sent_total         | rmse     | 1.281  | 1.180  | 1.136      | 1.114 (ws)                   | **1.066** [0.911, 1.232] | plain  | -0.048 [-0.076, -0.022] p=0.000 | -0.070 (p=0.000) | better                |
| word_accuracy      | rmse     | 1.704  | 1.687  | 1.680      | 1.633 (word_meanmaxstd_pros) | **1.598** [1.360, 1.807] | plain  | -0.035 [-0.069, -0.010] p=0.012 | -0.082 (p=0.000) | better                |
| phone_score        | rmse     | 0.352  | 0.352  | 0.325      | 0.314 (ws)                   | **0.306** [0.252, 0.351] | plain  | -0.007 [-0.017, +0.000] p=0.072 | -0.018 (p=0.000) | no significant change |
| phone_mispron_t1.0 | macro_f1 | 0.522  | 0.574  | 0.621      | 0.661 (mispron-grp)          | **0.680** [0.650, 0.699] | cw     | +0.018 [+0.009, +0.028] p=0.000 | +0.059 (p=0.000) | better                |
| l2_phone_error     | macro_f1 | 0.326  | 0.403  | 0.508      | 0.523 (ws)                   | **0.559** [0.524, 0.584] | cw+thr | +0.036 [+0.018, +0.055] p=0.000 | +0.051 (p=0.000) | better                |
| l2_word_error      | macro_f1 | 0.596  | 0.615  | 0.709      | 0.716 (L11)                  | **0.721** [0.701, 0.738] | cw     | +0.005 [-0.002, +0.011] p=0.184 | +0.012 (p=0.000) | no significant change |

## Train / validation / test and seed spread

| task               | train | validation | test  | seeds | sd of per-fold validation score | best frozen 4.3 on test (any encoder; reference only) |
| ------------------ | ----- | ---------- | ----- | ----- | ------------------------------- | ----------------------------------------------------- |
| sent_accuracy      | 0.505 | 0.991      | 1.071 | 3     | 0.000                           | 1.112 (wav2vec2_base)                                 |
| sent_fluency       | 0.440 | 0.655      | 0.870 | 3     | 0.000                           | 0.892 (wav2vec2_base)                                 |
| sent_prosodic      | 0.502 | 0.644      | 0.902 | 3     | 0.000                           | 0.893 (hubert_base)                                   |
| sent_total         | 0.626 | 0.916      | 1.066 | 3     | 0.000                           | 1.077 (hubert_base)                                   |
| word_accuracy      | 1.394 | 1.582      | 1.598 | 3     | 0.000                           | 1.633 (wavlm_base_plus)                               |
| phone_score        | 0.205 | 0.312      | 0.306 | 3     | 0.000                           | 0.314 (wavlm_base_plus)                               |
| phone_mispron_t1.0 | 0.962 | 0.697      | 0.680 | 3     | 0.000                           | 0.661 (wavlm_base_plus)                               |
| l2_phone_error     | 0.843 | 0.566      | 0.559 | 1     | 0.027                           | 0.523 (wavlm_base_plus)                               |
| l2_word_error      | 0.862 | 0.727      | 0.721 | 1     | 0.015                           | 0.716 (wavlm_base_plus)                               |

## Jobs

| task               | jobs | mean best epoch | mean epochs run | sd of validation score across jobs | total minutes |
| ------------------ | ---- | --------------- | --------------- | ---------------------------------- | ------------- |
| sent_accuracy      | 3    | 5.0             | 10.0            | 0.008                              | 7             |
| sent_fluency       | 3    | 4.3             | 9.3             | 0.002                              | 7             |
| sent_prosodic      | 3    | 3.3             | 8.3             | 0.018                              | 7             |
| sent_total         | 3    | 2.7             | 7.7             | 0.008                              | 6             |
| word_accuracy      | 3    | 1.7             | 6.7             | 0.022                              | 6             |
| phone_score        | 3    | 4.7             | 9.7             | 0.009                              | 7             |
| phone_mispron_t1.0 | 3    | 10.7            | 13.3            | 0.007                              | 8             |
| l2_phone_error     | 4    | 11.8            | 15.0            | 0.031                              | 10            |
| l2_word_error      | 4    | 3.2             | 8.2             | 0.017                              | 7             |
