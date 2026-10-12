# wav2vec2_base, layer 12, inputs emb+side - headline variant per task x head (test, 95% speaker-bootstrap CIs)

train_primary / val_seed_sd expose over-fitting and seed noise: RMSE for regression, macro-F1 for classification.

| task               | head   | variant | config                                 | test                                          | train_primary | val seed sd |
| ------------------ | ------ | ------- | -------------------------------------- | --------------------------------------------- | ------------- | ----------- |
| sent_accuracy      | linear | plain   | wd0.01                                 | RMSE 1.277 [1.176, 1.378] · PCC 0.573         | 1.033         | 0.0206      |
| sent_accuracy      | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 1.243 [1.118, 1.356] · PCC 0.591         | 1.051         | 0.0181      |
| sent_fluency       | linear | plain   | wd0.01                                 | RMSE 1.012 [0.897, 1.153] · PCC 0.706         | 0.714         | 0.0286      |
| sent_fluency       | mlp    | plain   | dropout0.1_hidden512_wd0.01            | RMSE 1.001 [0.880, 1.140] · PCC 0.714         | 0.732         | 0.0071      |
| sent_prosodic      | linear | plain   | wd0.01                                 | RMSE 1.059 [0.934, 1.210] · PCC 0.676         | 0.695         | 0.0342      |
| sent_prosodic      | mlp    | plain   | dropout0.1_hidden512_wd0.01            | RMSE 1.015 [0.898, 1.153] · PCC 0.708         | 0.745         | 0.0039      |
| sent_total         | linear | plain   | wd0.01                                 | RMSE 1.256 [1.121, 1.406] · PCC 0.597         | 0.911         | 0.0491      |
| sent_total         | mlp    | plain   | dropout0.3_hidden512_wd0.01            | RMSE 1.212 [1.068, 1.373] · PCC 0.625         | 0.871         | 0.0237      |
| word_accuracy      | linear | plain   | wd0.01                                 | RMSE 1.705 [1.477, 1.910] · PCC 0.332         | 1.682         | 0.0057      |
| word_accuracy      | mlp    | plain   | dropout0.1_hidden512_wd0.01            | RMSE 1.699 [1.463, 1.911] · PCC 0.336         | 1.610         | 0.0063      |
| phone_score        | linear | plain   | wd0.01                                 | RMSE 0.352 [0.296, 0.400] · PCC 0.319         | 0.373         | 0.0014      |
| phone_score        | mlp    | plain   | dropout0.1_hidden512_wd0.01            | RMSE 0.345 [0.287, 0.394] · PCC 0.349         | 0.355         | 0.0053      |
| phone_mispron_t1.0 | linear | cw+thr  | cwbalanced_wd0.001                     | macro-F1 0.598 [0.565, 0.625] · bal.acc 0.576 | 0.636         | 0.0066      |
| phone_mispron_t1.0 | mlp    | cw+thr  | cwbalanced_dropout0.1_hidden512_wd0.01 | macro-F1 0.600 [0.562, 0.632] · bal.acc 0.583 | 0.657         | 0.0099      |
| l2_phone_error     | linear | cw+thr  | cwbalanced_wd0.001                     | macro-F1 0.395 [0.379, 0.408] · bal.acc 0.421 | 0.327         | 0.0030      |
| l2_phone_error     | mlp    | cw+thr  | cwbalanced_dropout0.1_hidden512_wd0.01 | macro-F1 0.393 [0.373, 0.408] · bal.acc 0.417 | 0.353         | 0.0107      |
| l2_word_error      | linear | cw      | cwbalanced_wd0.001                     | macro-F1 0.612 [0.596, 0.626] · bal.acc 0.616 | 0.691         | 0.0148      |
| l2_word_error      | mlp    | cw      | cwbalanced_dropout0.3_hidden512_wd0.01 | macro-F1 0.616 [0.599, 0.632] · bal.acc 0.620 | 0.720         | 0.0117      |
