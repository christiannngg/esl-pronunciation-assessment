# hubert_base, layer 12, inputs emb+side - headline variant per task x head (test, 95% speaker-bootstrap CIs)

train_primary / val_seed_sd expose over-fitting and seed noise: RMSE for regression, macro-F1 for classification.

| task               | head   | variant| config                                 | test                                          | train_primary | val seed sd |
|--------------------|--------|--------|----------------------------------------|-----------------------------------------------|---------------|-------------|
| sent_accuracy      | linear | plain  | wd0.01                                 | RMSE 1.223 [1.115, 1.327] · PCC 0.614         | 0.919         | 0.0268      |
| sent_accuracy      | mlp    | plain  | dropout0.1_hidden512_wd0.01            | RMSE 1.169 [1.050, 1.283] · PCC 0.652         | 0.786         | 0.0171      |
| sent_fluency       | linear | plain  | wd0.01                                 | RMSE 0.980 [0.832, 1.164] · PCC 0.728         | 0.642         | 0.0103      |
| sent_fluency       | mlp.   | plain  | dropout0.1_hidden512_wd0.01            | RMSE 0.939 [0.773, 1.139] · PCC 0.753         | 0.540         | 0.0068      |
| sent_prosodic      | linear | plain  | wd0.01                                 | RMSE 0.990 [0.861, 1.147] · PCC 0.723         | 0.647         | 0.0225      |
| sent_prosodic      | mlp    | plain  | dropout0.3_hidden512_wd0.01            | RMSE 0.960 [0.811, 1.139] · PCC 0.744         | 0.542         | 0.0074      |
| sent_total         | linear | plain  | wd0.01                                 | RMSE 1.189 [1.061, 1.326] · PCC 0.647         | 0.810         | 0.0284      |
| sent_total         | mlp    | plain  | dropout0.3_hidden512_wd0.01            | RMSE 1.155 [1.006, 1.319] · PCC 0.667         | 0.710         | 0.0197      |
| word_accuracy      | linear | plain  | wd0.01                                 | RMSE 1.681 [1.463, 1.869] · PCC 0.364         | 1.683         | 0.0061      |
| word_accuracy      | mlp    | plain  | dropout0.3_hidden512_wd0.01            | RMSE 1.663 [1.439, 1.862] · PCC 0.385         | 1.500         | 0.0044      |
| phone_score        | linear | plain  | wd0.01                                 | RMSE 0.348 [0.292, 0.395] · PCC 0.344         | 0.378         | 0.0005      |
| phone_score        | mlp    | plain  | dropout0.3_hidden512_wd0.01            | RMSE 0.325 [0.273, 0.369] · PCC 0.484         | 0.244         | 0.0013      |
| phone_mispron_t1.0 | linear | cw+thr | cwbalanced_wd0.001                     | macro-F1 0.620 [0.583, 0.650] · bal.acc 0.609 | 0.629         | 0.0010      |
| phone_mispron_t1.0 | mlp    | cw+thr | cwbalanced_dropout0.3_hidden512_wd0.01 | macro-F1 0.631 [0.600, 0.656] · bal.acc 0.618 | 0.692         | 0.0123      |
| l2_phone_error     | linear | cw+thr | cwbalanced_wd0.001                     | macro-F1 0.455 [0.437, 0.468] · bal.acc 0.478 | 0.366         | 0.0034      |
| l2_phone_error     | mlp    | cw+thr | cwbalanced_dropout0.3_hidden512_wd0.01 | macro-F1 0.479 [0.454, 0.496] · bal.acc 0.496 | 0.422         | 0.0168      |
| l2_word_error      | linear | cw     | cwbalanced_wd0.001                     | macro-F1 0.679 [0.661, 0.695] · bal.acc 0.685 | 0.734         | 0.0024      |
| l2_word_error      | mlp    | cw+thr | cwbalanced_dropout0.3_hidden512_wd0.01 | macro-F1 0.684 [0.666, 0.700] · bal.acc 0.688 | 0.772         | 0.0053      |