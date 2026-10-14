# wavlm_base_plus, layer 12, inputs emb+side - headline variant per task x head (test, 95% speaker-bootstrap CIs)

train_primary / val_seed_sd expose over-fitting and seed noise: RMSE for regression, macro-F1 for classification.

| task               | head   | variant | config                                 | test                                          | train_primary | val seed sd |
| ------------------ | ------ | ------- | -------------------------------------- | --------------------------------------------- | ------------- | ----------- |
| sent_accuracy      | linear | plain   | wd0.01                                 | RMSE 1.223 [1.117, 1.324] · PCC 0.626         | 0.969         | 0.0191      |
| sent_accuracy      | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 1.165 [1.044, 1.274] · PCC 0.655         | 0.784         | 0.0096      |
| sent_fluency       | linear | plain   | wd0.01                                 | RMSE 0.961 [0.818, 1.137] · PCC 0.744         | 0.689         | 0.0117      |
| sent_fluency       | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 0.937 [0.765, 1.142] · PCC 0.754         | 0.509         | 0.0124      |
| sent_prosodic      | linear | plain   | wd0.01                                 | RMSE 0.977 [0.844, 1.146] · PCC 0.736         | 0.659         | 0.0228      |
| sent_prosodic      | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 0.946 [0.797, 1.135] · PCC 0.753         | 0.507         | 0.0045      |
| sent_total         | linear | plain   | wd0.01                                 | RMSE 1.193 [1.068, 1.318] · PCC 0.652         | 0.889         | 0.0270      |
| sent_total         | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 1.136 [0.992, 1.283] · PCC 0.681         | 0.704         | 0.0120      |
| word_accuracy      | linear | plain   | wd0.01                                 | RMSE 1.680 [1.458, 1.874] · PCC 0.366         | 1.686         | 0.0034      |
| word_accuracy      | mlp    | plain   | dropout0.1_hidden512_wd0.01            | RMSE 1.680 [1.464, 1.870] · PCC 0.376         | 1.575         | 0.0160      |
| phone_score        | linear | plain   | wd0.01                                 | RMSE 0.345 [0.290, 0.393] · PCC 0.361         | 0.374         | 0.0010      |
| phone_score        | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 0.325 [0.274, 0.368] · PCC 0.493         | 0.274         | 0.0013      |
| phone_mispron_t1.0 | linear | cw+thr  | cwbalanced_wd0.001                     | macro-F1 0.621 [0.591, 0.644] · bal.acc 0.607 | 0.694         | 0.0034      |
| phone_mispron_t1.0 | mlp    | cw+thr  | cwbalanced_dropout0.3_hidden512_wd0.01 | macro-F1 0.638 [0.611, 0.659] · bal.acc 0.636 | 0.751         | 0.0198      |
| l2_phone_error     | linear | cw+thr  | cwbalanced_wd0.001                     | macro-F1 0.481 [0.461, 0.495] · bal.acc 0.523 | 0.531         | 0.0134      |
| l2_phone_error     | mlp    | cw+thr  | cwbalanced_dropout0.1_hidden512_wd0.01 | macro-F1 0.508 [0.482, 0.527] · bal.acc 0.537 | 0.591         | 0.0194      |
| l2_word_error      | linear | plain   | cwnone_wd0.001                         | macro-F1 0.699 [0.680, 0.716] · bal.acc 0.699 | 0.740         | 0.0025      |
| l2_word_error      | mlp    | cw+thr  | cwbalanced_dropout0.3_hidden512_wd0.01 | macro-F1 0.709 [0.691, 0.726] · bal.acc 0.710 | 0.774         | 0.0050      |
