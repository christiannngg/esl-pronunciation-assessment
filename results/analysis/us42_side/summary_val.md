# US 4.2 diagnosis (validation only) - layer 12, inputs side

Frozen result = validation-selected headline per task (best head and variant). US 3.1 = Sprint 3 best baseline on validation. Full tables: csv files in this folder.

## Over-fitting: train vs validation (positive gap = better on train)

| encoder         | task               | head   | variant | val_primary | train_primary | overfit_gap | val_seed_sd |
| --------------- | ------------------ | ------ | ------- | ----------- | ------------- | ----------- | ----------- |
| wavlm_base_plus | sent_accuracy      | linear | plain   | 1.553       | 1.614         | -0.061      | 0.078       |
| wavlm_base_plus | sent_accuracy      | mlp    | plain   | 1.165       | 1.325         | -0.160      | 0.019       |
| wavlm_base_plus | sent_fluency       | linear | plain   | 1.461       | 1.442         | 0.019       | 0.102       |
| wavlm_base_plus | sent_fluency       | mlp    | plain   | 0.770       | 1.009         | -0.239      | 0.004       |
| wavlm_base_plus | sent_prosodic      | linear | plain   | 1.427       | 1.396         | 0.032       | 0.086       |
| wavlm_base_plus | sent_prosodic      | mlp    | plain   | 0.858       | 1.057         | -0.200      | 0.036       |
| wavlm_base_plus | sent_total         | linear | plain   | 1.532       | 1.578         | -0.046      | 0.078       |
| wavlm_base_plus | sent_total         | mlp    | plain   | 1.080       | 1.249         | -0.169      | 0.022       |
| wavlm_base_plus | word_accuracy      | linear | plain   | 1.816       | 1.920         | -0.105      | 0.009       |
| wavlm_base_plus | word_accuracy      | mlp    | plain   | 1.729       | 1.863         | -0.134      | 0.003       |
| wavlm_base_plus | phone_score        | linear | plain   | 0.371       | 0.415         | -0.045      | 0.000       |
| wavlm_base_plus | phone_score        | mlp    | plain   | 0.366       | 0.401         | -0.036      | 0.000       |
| wavlm_base_plus | phone_mispron_t1.0 | linear | cw+thr  | 0.554       | 0.569         | 0.015       | 0.005       |
| wavlm_base_plus | phone_mispron_t1.0 | mlp    | cw+thr  | 0.575       | 0.615         | 0.040       | 0.003       |
| wavlm_base_plus | l2_phone_error     | linear | cw+thr  | 0.362       | 0.364         | 0.002       | 0.002       |
| wavlm_base_plus | l2_phone_error     | mlp    | cw+thr  | 0.363       | 0.364         | 0.001       | 0.018       |
| wavlm_base_plus | l2_word_error      | linear | cw+thr  | 0.623       | 0.612         | -0.011      | 0.003       |
| wavlm_base_plus | l2_word_error      | mlp    | cw+thr  | 0.625       | 0.622         | -0.004      | 0.004       |

## Head capacity: validation gain of MLP over linear (positive = MLP better)

| encoder         | task               | val_linear | val_mlp | mlp_gain |
| --------------- | ------------------ | ---------- | ------- | -------- |
| wavlm_base_plus | sent_accuracy      | 1.553      | 1.165   | 0.387    |
| wavlm_base_plus | sent_fluency       | 1.461      | 0.770   | 0.691    |
| wavlm_base_plus | sent_prosodic      | 1.427      | 0.858   | 0.570    |
| wavlm_base_plus | sent_total         | 1.532      | 1.080   | 0.452    |
| wavlm_base_plus | word_accuracy      | 1.816      | 1.729   | 0.086    |
| wavlm_base_plus | phone_score        | 0.371      | 0.366   | 0.005    |
| wavlm_base_plus | phone_mispron_t1.0 | 0.554      | 0.575   | 0.021    |
| wavlm_base_plus | l2_phone_error     | 0.362      | 0.363   | 0.001    |
| wavlm_base_plus | l2_word_error      | 0.623      | 0.625   | 0.002    |

## Regression summary

| task          | system          | detail      | rmse  | pcc   | slope_pred_on_true | human_loo_r | pcc_over_ceiling |
| ------------- | --------------- | ----------- | ----- | ----- | ------------------ | ----------- | ---------------- |
| sent_accuracy | US 3.1          | svr         | 1.223 | 0.677 | 0.474              | 0.766       | 0.884            |
| sent_accuracy | wavlm_base_plus | mlp (plain) | 1.165 | 0.713 | 0.499              | 0.766       | 0.931            |
| sent_fluency  | US 3.1          | ridge       | 0.946 | 0.822 | 0.693              | 0.788       | 1.043            |
| sent_fluency  | wavlm_base_plus | mlp (plain) | 0.770 | 0.879 | 0.753              | 0.788       | 1.116            |
| sent_prosodic | US 3.1          | svr         | 0.975 | 0.791 | 0.594              | 0.772       | 1.024            |
| sent_prosodic | wavlm_base_plus | mlp (plain) | 0.858 | 0.840 | 0.653              | 0.772       | 1.089            |
| sent_total    | US 3.1          | rf          | 1.156 | 0.730 | 0.452              | 0.802       | 0.910            |
| sent_total    | wavlm_base_plus | mlp (plain) | 1.080 | 0.755 | 0.554              | 0.802       | 0.942            |
| word_accuracy | US 3.1          | rf          | 1.739 | 0.331 | 0.101              | 0.735       | 0.451            |
| word_accuracy | wavlm_base_plus | mlp (plain) | 1.729 | 0.338 | 0.131              | 0.735       | 0.460            |
| phone_score   | US 3.1          | rf          | 0.358 | 0.340 | 0.129              | 0.704       | 0.484            |
| phone_score   | wavlm_base_plus | mlp (plain) | 0.366 | 0.274 | 0.096              | 0.704       | 0.389            |

## Regression bias by true-score bin (word / phone)

| task          | system          | true_bin | n    | share | mean_true | mean_pred | bias   |
| ------------- | --------------- | -------- | ---- | ----- | --------- | --------- | ------ |
| word_accuracy | US 3.1          | 0-4      | 157  | 0.061 | 2.981     | 8.638     | 5.657  |
| word_accuracy | US 3.1          | 5-6      | 64   | 0.025 | 5.125     | 8.613     | 3.488  |
| word_accuracy | US 3.1          | 7-8      | 47   | 0.018 | 7.830     | 8.984     | 1.154  |
| word_accuracy | US 3.1          | 10       | 2293 | 0.895 | 10.000    | 9.292     | -0.708 |
| word_accuracy | wavlm_base_plus | 0-4      | 157  | 0.061 | 2.981     | 8.618     | 5.637  |
| word_accuracy | wavlm_base_plus | 5-6      | 64   | 0.025 | 5.125     | 8.294     | 3.169  |
| word_accuracy | wavlm_base_plus | 7-8      | 47   | 0.018 | 7.830     | 8.952     | 1.122  |
| word_accuracy | wavlm_base_plus | 10       | 2293 | 0.895 | 10.000    | 9.401     | -0.599 |
| phone_score   | US 3.1          | <0.5     | 245  | 0.032 | 0.127     | 1.617     | 1.490  |
| phone_score   | US 3.1          | 0.5-1    | 117  | 0.015 | 0.812     | 1.690     | 0.878  |
| phone_score   | US 3.1          | 1-1.5    | 188  | 0.025 | 1.255     | 1.733     | 0.478  |
| phone_score   | US 3.1          | 1.5-<2   | 840  | 0.110 | 1.712     | 1.783     | 0.071  |
| phone_score   | US 3.1          | =2       | 6268 | 0.818 | 2.000     | 1.845     | -0.155 |
| phone_score   | wavlm_base_plus | <0.5     | 245  | 0.032 | 0.127     | 1.700     | 1.573  |
| phone_score   | wavlm_base_plus | 0.5-1    | 117  | 0.015 | 0.812     | 1.745     | 0.933  |
| phone_score   | wavlm_base_plus | 1-1.5    | 188  | 0.025 | 1.255     | 1.748     | 0.492  |
| phone_score   | wavlm_base_plus | 1.5-<2   | 840  | 0.110 | 1.712     | 1.795     | 0.083  |
| phone_score   | wavlm_base_plus | =2       | 6268 | 0.818 | 2.000     | 1.859     | -0.141 |

## Classification summary

| task               | system          | detail       | n      | macro_f1 | balanced_acc | error_share_true | error_share_pred | error_recall | error_precision |
| ------------------ | --------------- | ------------ | ------ | -------- | ------------ | ---------------- | ---------------- | ------------ | --------------- |
| phone_mispron_t1.0 | US 3.1          | svm          | 7658   | 0.550    | 0.533        | 0.045            | 0.007            | 0.070        | 0.480           |
| phone_mispron_t1.0 | wavlm_base_plus | mlp (cw+thr) | 7658   | 0.575    | 0.628        | 0.045            | 0.102            | 0.347        | 0.153           |
| l2_phone_error     | US 3.1          | svm          | 120243 | 0.323    | 0.306        | 0.155            | 0.051            | 0.171        | 0.521           |
| l2_phone_error     | wavlm_base_plus | mlp (cw+thr) | 120243 | 0.363    | 0.390        | 0.155            | 0.203            | 0.438        | 0.335           |
| l2_word_error      | US 3.1          | svm          | 33985  | 0.598    | 0.598        | 0.417            | 0.329            | 0.443        | 0.562           |
| l2_word_error      | wavlm_base_plus | mlp (cw+thr) | 33985  | 0.625    | 0.627        | 0.417            | 0.450            | 0.599        | 0.554           |

## Threshold-free ranking quality (AUC of P(error) / per class)

| task               | system          | detail                          | auc_error | ap_error | error_share | auc_substitution | auc_deletion | auc_addition | auc_correct |
| ------------------ | --------------- | ------------------------------- | --------- | -------- | ----------- | ---------------- | ------------ | ------------ | ----------- |
| phone_mispron_t1.0 | US 3.1          | logreg, unweighted (US 3.2 csv) | 0.803     | 0.156    |             |                  |              |              |             |
| phone_mispron_t1.0 | wavlm_base_plus | mlp (cw+thr)                    | 0.783     | 0.139    | 0.045       |                  |              |              |             |
| l2_phone_error     | US 3.1          | logreg, unweighted (US 3.2 csv) |           |          |             | 0.756            | 0.834        | 0.675        | 0.742       |
| l2_phone_error     | wavlm_base_plus | mlp (cw+thr)                    | 0.698     | 0.324    | 0.155       | 0.733            | 0.793        | 0.582        | 0.698       |
| l2_word_error      | wavlm_base_plus | mlp (cw+thr)                    | 0.683     | 0.576    | 0.417       |                  |              |              |             |

## Per class

| class        | precision | recall | f1    | support | task           | system          |
| ------------ | --------- | ------ | ----- | ------- | -------------- | --------------- |
| correct      | 0.865     | 0.971  | 0.915 | 101633  | l2_phone_error | US 3.1          |
| substitution | 0.493     | 0.199  | 0.283 | 14098   | l2_phone_error | US 3.1          |
| deletion     | 0.417     | 0.053  | 0.094 | 3420    | l2_phone_error | US 3.1          |
| addition     | 0.000     | 0.000  | 0.000 | 1092    | l2_phone_error | US 3.1          |
| correct      | 0.891     | 0.840  | 0.865 | 101633  | l2_phone_error | wavlm_base_plus |
| substitution | 0.377     | 0.399  | 0.387 | 14098   | l2_phone_error | wavlm_base_plus |
| deletion     | 0.122     | 0.292  | 0.172 | 3420    | l2_phone_error | wavlm_base_plus |
| addition     | 0.024     | 0.027  | 0.026 | 1092    | l2_phone_error | wavlm_base_plus |

## L2-ARCTIC phone error by L1

| system          | group      | n     | macro_f1 | error_share_true | error_recall | error_precision |
| --------------- | ---------- | ----- | -------- | ---------------- | ------------ | --------------- |
| US 3.1          | Arabic     | 20038 | 0.292    | 0.110            | 0.113        | 0.287           |
| US 3.1          | Hindi      | 19846 | 0.320    | 0.145            | 0.157        | 0.451           |
| US 3.1          | Korean     | 19791 | 0.335    | 0.099            | 0.222        | 0.447           |
| US 3.1          | Mandarin   | 19984 | 0.345    | 0.162            | 0.203        | 0.628           |
| US 3.1          | Spanish    | 20250 | 0.322    | 0.169            | 0.184        | 0.591           |
| US 3.1          | Vietnamese | 20334 | 0.311    | 0.242            | 0.156        | 0.656           |
| wavlm_base_plus | Arabic     | 20038 | 0.303    | 0.110            | 0.371        | 0.204           |
| wavlm_base_plus | Hindi      | 19846 | 0.355    | 0.145            | 0.469        | 0.335           |
| wavlm_base_plus | Korean     | 19791 | 0.363    | 0.099            | 0.564        | 0.269           |
| wavlm_base_plus | Mandarin   | 19984 | 0.371    | 0.162            | 0.432        | 0.354           |
| wavlm_base_plus | Spanish    | 20250 | 0.383    | 0.169            | 0.440        | 0.363           |
| wavlm_base_plus | Vietnamese | 20334 | 0.374    | 0.242            | 0.403        | 0.479           |

## Error recall per phoneme vs training error support (Spearman)

| task               | system          | rho_support | p_support | rho_rate | p_rate | n_phonemes |
| ------------------ | --------------- | ----------- | --------- | -------- | ------ | ---------- |
| phone_mispron_t1.0 | US 3.1          | 0.322       | 0.083     | 0.457    | 0.011  | 30         |
| phone_mispron_t1.0 | wavlm_base_plus | 0.397       | 0.030     | 0.214    | 0.257  | 30         |
| l2_phone_error     | US 3.1          | 0.593       | 0.000     | 0.922    | 0.000  | 39         |
| l2_phone_error     | wavlm_base_plus | 0.448       | 0.004     | 0.718    | 0.000  | 39         |
