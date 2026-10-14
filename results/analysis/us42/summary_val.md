# US 4.2 diagnosis (validation only) - layer 12, inputs emb+side

Frozen result = validation-selected headline per task (best head and variant). US 3.1 = Sprint 3 best baseline on validation. Full tables: csv files in this folder.

## Over-fitting: train vs validation (positive gap = better on train)

| encoder | task | head | variant | val_primary | train_primary | overfit_gap | val_seed_sd |
|---|---|---|---|---|---|---|---|
| wavlm_base_plus | sent_accuracy | linear | plain | 1.059 | 0.969 | 0.090 | 0.019 |
| wavlm_base_plus | sent_accuracy | mlp | plain | 1.014 | 0.784 | 0.230 | 0.010 |
| hubert_base | sent_accuracy | linear | plain | 1.119 | 0.919 | 0.201 | 0.027 |
| hubert_base | sent_accuracy | mlp | plain | 1.077 | 0.786 | 0.291 | 0.017 |
| wavlm_base_plus | sent_fluency | linear | plain | 0.760 | 0.689 | 0.071 | 0.012 |
| wavlm_base_plus | sent_fluency | mlp | plain | 0.709 | 0.509 | 0.200 | 0.012 |
| hubert_base | sent_fluency | linear | plain | 0.766 | 0.642 | 0.125 | 0.010 |
| hubert_base | sent_fluency | mlp | plain | 0.706 | 0.540 | 0.166 | 0.007 |
| wavlm_base_plus | sent_prosodic | linear | plain | 0.756 | 0.659 | 0.097 | 0.023 |
| wavlm_base_plus | sent_prosodic | mlp | plain | 0.725 | 0.507 | 0.217 | 0.004 |
| hubert_base | sent_prosodic | linear | plain | 0.806 | 0.647 | 0.159 | 0.023 |
| hubert_base | sent_prosodic | mlp | plain | 0.738 | 0.542 | 0.196 | 0.007 |
| wavlm_base_plus | sent_total | linear | plain | 0.971 | 0.889 | 0.082 | 0.027 |
| wavlm_base_plus | sent_total | mlp | plain | 0.949 | 0.704 | 0.245 | 0.012 |
| hubert_base | sent_total | linear | plain | 1.059 | 0.810 | 0.249 | 0.028 |
| hubert_base | sent_total | mlp | plain | 1.007 | 0.710 | 0.297 | 0.020 |
| wavlm_base_plus | word_accuracy | linear | plain | 1.704 | 1.686 | 0.018 | 0.003 |
| wavlm_base_plus | word_accuracy | mlp | plain | 1.715 | 1.575 | 0.140 | 0.016 |
| hubert_base | word_accuracy | linear | plain | 1.749 | 1.683 | 0.066 | 0.006 |
| hubert_base | word_accuracy | mlp | plain | 1.749 | 1.500 | 0.249 | 0.004 |
| wavlm_base_plus | phone_score | linear | plain | 0.353 | 0.374 | -0.021 | 0.001 |
| wavlm_base_plus | phone_score | mlp | plain | 0.337 | 0.274 | 0.063 | 0.001 |
| hubert_base | phone_score | linear | plain | 0.363 | 0.378 | -0.015 | 0.001 |
| hubert_base | phone_score | mlp | plain | 0.341 | 0.244 | 0.097 | 0.001 |
| wavlm_base_plus | phone_mispron_t1.0 | linear | cw+thr | 0.631 | 0.694 | 0.063 | 0.003 |
| wavlm_base_plus | phone_mispron_t1.0 | mlp | cw+thr | 0.628 | 0.751 | 0.123 | 0.020 |
| hubert_base | phone_mispron_t1.0 | linear | cw+thr | 0.603 | 0.702 | 0.099 | 0.001 |
| hubert_base | phone_mispron_t1.0 | mlp | cw+thr | 0.611 | 0.780 | 0.170 | 0.012 |
| wavlm_base_plus | l2_phone_error | linear | cw+thr | 0.480 | 0.531 | 0.051 | 0.013 |
| wavlm_base_plus | l2_phone_error | mlp | cw+thr | 0.503 | 0.591 | 0.088 | 0.019 |
| hubert_base | l2_phone_error | linear | cw+thr | 0.456 | 0.511 | 0.056 | 0.003 |
| hubert_base | l2_phone_error | mlp | cw+thr | 0.476 | 0.582 | 0.106 | 0.017 |
| wavlm_base_plus | l2_word_error | linear | plain | 0.704 | 0.740 | 0.036 | 0.003 |
| wavlm_base_plus | l2_word_error | mlp | cw+thr | 0.710 | 0.774 | 0.064 | 0.005 |
| hubert_base | l2_word_error | linear | cw | 0.681 | 0.734 | 0.052 | 0.002 |
| hubert_base | l2_word_error | mlp | cw+thr | 0.686 | 0.775 | 0.089 | 0.005 |

## Head capacity: validation gain of MLP over linear (positive = MLP better)

| encoder | task | val_linear | val_mlp | mlp_gain |
|---|---|---|---|---|
| wavlm_base_plus | sent_accuracy | 1.059 | 1.014 | 0.045 |
| hubert_base | sent_accuracy | 1.119 | 1.077 | 0.042 |
| wavlm_base_plus | sent_fluency | 0.760 | 0.709 | 0.051 |
| hubert_base | sent_fluency | 0.766 | 0.706 | 0.060 |
| wavlm_base_plus | sent_prosodic | 0.756 | 0.725 | 0.032 |
| hubert_base | sent_prosodic | 0.806 | 0.738 | 0.068 |
| wavlm_base_plus | sent_total | 0.971 | 0.949 | 0.022 |
| hubert_base | sent_total | 1.059 | 1.007 | 0.052 |
| wavlm_base_plus | word_accuracy | 1.704 | 1.715 | -0.011 |
| hubert_base | word_accuracy | 1.749 | 1.749 | 0.001 |
| wavlm_base_plus | phone_score | 0.353 | 0.337 | 0.015 |
| hubert_base | phone_score | 0.363 | 0.341 | 0.022 |
| wavlm_base_plus | phone_mispron_t1.0 | 0.631 | 0.628 | -0.003 |
| hubert_base | phone_mispron_t1.0 | 0.603 | 0.611 | 0.008 |
| wavlm_base_plus | l2_phone_error | 0.480 | 0.503 | 0.023 |
| hubert_base | l2_phone_error | 0.456 | 0.476 | 0.020 |
| wavlm_base_plus | l2_word_error | 0.704 | 0.710 | 0.007 |
| hubert_base | l2_word_error | 0.681 | 0.686 | 0.005 |

## Regression summary

| task | system | detail | rmse | pcc | slope_pred_on_true | human_loo_r | pcc_over_ceiling |
|---|---|---|---|---|---|---|---|
| sent_accuracy | US 3.1 | svr | 1.223 | 0.677 | 0.474 | 0.766 | 0.884 |
| sent_accuracy | wavlm_base_plus | mlp (plain) | 1.014 | 0.796 | 0.693 | 0.766 | 1.039 |
| sent_accuracy | hubert_base | mlp (plain) | 1.077 | 0.766 | 0.651 | 0.766 | 1.001 |
| sent_fluency | US 3.1 | ridge | 0.946 | 0.822 | 0.693 | 0.788 | 1.043 |
| sent_fluency | wavlm_base_plus | mlp (plain) | 0.709 | 0.899 | 0.790 | 0.788 | 1.141 |
| sent_fluency | hubert_base | mlp (plain) | 0.706 | 0.900 | 0.777 | 0.788 | 1.143 |
| sent_prosodic | US 3.1 | svr | 0.975 | 0.791 | 0.594 | 0.772 | 1.024 |
| sent_prosodic | wavlm_base_plus | mlp (plain) | 0.725 | 0.887 | 0.765 | 0.772 | 1.150 |
| sent_prosodic | hubert_base | mlp (plain) | 0.738 | 0.883 | 0.755 | 0.772 | 1.144 |
| sent_total | US 3.1 | rf | 1.156 | 0.730 | 0.452 | 0.802 | 0.910 |
| sent_total | wavlm_base_plus | mlp (plain) | 0.949 | 0.823 | 0.748 | 0.802 | 1.026 |
| sent_total | hubert_base | mlp (plain) | 1.007 | 0.795 | 0.684 | 0.802 | 0.992 |
| word_accuracy | US 3.1 | rf | 1.739 | 0.331 | 0.101 | 0.735 | 0.451 |
| word_accuracy | wavlm_base_plus | linear (plain) | 1.704 | 0.411 | 0.233 | 0.735 | 0.560 |
| word_accuracy | hubert_base | mlp (plain) | 1.749 | 0.402 | 0.259 | 0.735 | 0.547 |
| phone_score | US 3.1 | rf | 0.358 | 0.340 | 0.129 | 0.704 | 0.484 |
| phone_score | wavlm_base_plus | mlp (plain) | 0.337 | 0.492 | 0.342 | 0.704 | 0.700 |
| phone_score | hubert_base | mlp (plain) | 0.341 | 0.490 | 0.350 | 0.704 | 0.696 |

## Regression bias by true-score bin (word / phone)

| task | system | true_bin | n | share | mean_true | mean_pred | bias |
|---|---|---|---|---|---|---|---|
| word_accuracy | US 3.1 | 0-4 | 157 | 0.061 | 2.981 | 8.638 | 5.657 |
| word_accuracy | US 3.1 | 5-6 | 64 | 0.025 | 5.125 | 8.613 | 3.488 |
| word_accuracy | US 3.1 | 7-8 | 47 | 0.018 | 7.830 | 8.984 | 1.154 |
| word_accuracy | US 3.1 | 10 | 2293 | 0.895 | 10.000 | 9.292 | -0.708 |
| word_accuracy | wavlm_base_plus | 0-4 | 157 | 0.061 | 2.981 | 7.806 | 4.825 |
| word_accuracy | wavlm_base_plus | 5-6 | 64 | 0.025 | 5.125 | 8.011 | 2.886 |
| word_accuracy | wavlm_base_plus | 7-8 | 47 | 0.018 | 7.830 | 8.671 | 0.841 |
| word_accuracy | wavlm_base_plus | 10 | 2293 | 0.895 | 10.000 | 9.371 | -0.629 |
| word_accuracy | hubert_base | 0-4 | 157 | 0.061 | 2.981 | 7.593 | 4.613 |
| word_accuracy | hubert_base | 5-6 | 64 | 0.025 | 5.125 | 7.891 | 2.766 |
| word_accuracy | hubert_base | 7-8 | 47 | 0.018 | 7.830 | 8.766 | 0.936 |
| word_accuracy | hubert_base | 10 | 2293 | 0.895 | 10.000 | 9.356 | -0.644 |
| phone_score | US 3.1 | <0.5 | 245 | 0.032 | 0.127 | 1.617 | 1.490 |
| phone_score | US 3.1 | 0.5-1 | 117 | 0.015 | 0.812 | 1.690 | 0.878 |
| phone_score | US 3.1 | 1-1.5 | 188 | 0.025 | 1.255 | 1.733 | 0.478 |
| phone_score | US 3.1 | 1.5-<2 | 840 | 0.110 | 1.712 | 1.783 | 0.071 |
| phone_score | US 3.1 | =2 | 6268 | 0.818 | 2.000 | 1.845 | -0.155 |
| phone_score | wavlm_base_plus | <0.5 | 245 | 0.032 | 0.127 | 1.329 | 1.202 |
| phone_score | wavlm_base_plus | 0.5-1 | 117 | 0.015 | 0.812 | 1.472 | 0.660 |
| phone_score | wavlm_base_plus | 1-1.5 | 188 | 0.025 | 1.255 | 1.584 | 0.329 |
| phone_score | wavlm_base_plus | 1.5-<2 | 840 | 0.110 | 1.712 | 1.786 | 0.074 |
| phone_score | wavlm_base_plus | =2 | 6268 | 0.818 | 2.000 | 1.923 | -0.077 |
| phone_score | hubert_base | <0.5 | 245 | 0.032 | 0.127 | 1.293 | 1.165 |
| phone_score | hubert_base | 0.5-1 | 117 | 0.015 | 0.812 | 1.417 | 0.605 |
| phone_score | hubert_base | 1-1.5 | 188 | 0.025 | 1.255 | 1.564 | 0.308 |
| phone_score | hubert_base | 1.5-<2 | 840 | 0.110 | 1.712 | 1.754 | 0.041 |
| phone_score | hubert_base | =2 | 6268 | 0.818 | 2.000 | 1.897 | -0.103 |

## Classification summary

| task | system | detail | n | macro_f1 | balanced_acc | error_share_true | error_share_pred | error_recall | error_precision |
|---|---|---|---|---|---|---|---|---|---|
| phone_mispron_t1.0 | US 3.1 | svm | 7658 | 0.550 | 0.533 | 0.045 | 0.007 | 0.070 | 0.480 |
| phone_mispron_t1.0 | wavlm_base_plus | linear (cw+thr) | 7658 | 0.631 | 0.700 | 0.045 | 0.091 | 0.472 | 0.232 |
| phone_mispron_t1.0 | hubert_base | mlp (cw+thr) | 7658 | 0.611 | 0.663 | 0.045 | 0.086 | 0.397 | 0.207 |
| l2_phone_error | US 3.1 | svm | 120243 | 0.323 | 0.306 | 0.155 | 0.051 | 0.171 | 0.521 |
| l2_phone_error | wavlm_base_plus | mlp (cw+thr) | 120243 | 0.503 | 0.522 | 0.155 | 0.167 | 0.519 | 0.482 |
| l2_phone_error | hubert_base | mlp (cw+thr) | 120243 | 0.476 | 0.487 | 0.155 | 0.166 | 0.500 | 0.467 |
| l2_word_error | US 3.1 | svm | 33985 | 0.598 | 0.598 | 0.417 | 0.329 | 0.443 | 0.562 |
| l2_word_error | wavlm_base_plus | mlp (cw+thr) | 33985 | 0.710 | 0.714 | 0.417 | 0.461 | 0.711 | 0.643 |
| l2_word_error | hubert_base | mlp (cw+thr) | 33985 | 0.686 | 0.689 | 0.417 | 0.450 | 0.670 | 0.620 |

## Threshold-free ranking quality (AUC of P(error) / per class)

| task | system | detail | auc_error | ap_error | error_share | auc_substitution | auc_deletion | auc_addition | auc_correct |
|---|---|---|---|---|---|---|---|---|---|
| phone_mispron_t1.0 | US 3.1 | logreg, unweighted (US 3.2 csv) | 0.803 | 0.156 |  |  |  |  |  |
| phone_mispron_t1.0 | wavlm_base_plus | linear (cw+thr) | 0.883 | 0.244 | 0.045 |  |  |  |  |
| phone_mispron_t1.0 | hubert_base | mlp (cw+thr) | 0.871 | 0.223 | 0.045 |  |  |  |  |
| l2_phone_error | US 3.1 | logreg, unweighted (US 3.2 csv) |  |  |  | 0.756 | 0.834 | 0.675 | 0.742 |
| l2_phone_error | wavlm_base_plus | mlp (cw+thr) | 0.816 | 0.456 | 0.155 | 0.800 | 0.900 | 0.832 | 0.816 |
| l2_phone_error | hubert_base | mlp (cw+thr) | 0.803 | 0.445 | 0.155 | 0.787 | 0.894 | 0.788 | 0.803 |
| l2_word_error | wavlm_base_plus | mlp (cw+thr) | 0.791 | 0.702 | 0.417 |  |  |  |  |
| l2_word_error | hubert_base | mlp (cw+thr) | 0.763 | 0.666 | 0.417 |  |  |  |  |

## Per class

| class | precision | recall | f1 | support | task | system |
|---|---|---|---|---|---|---|
| correct | 0.865 | 0.971 | 0.915 | 101633 | l2_phone_error | US 3.1 |
| substitution | 0.493 | 0.199 | 0.283 | 14098 | l2_phone_error | US 3.1 |
| deletion | 0.417 | 0.053 | 0.094 | 3420 | l2_phone_error | US 3.1 |
| addition | 0.000 | 0.000 | 0.000 | 1092 | l2_phone_error | US 3.1 |
| correct | 0.911 | 0.898 | 0.904 | 101633 | l2_phone_error | wavlm_base_plus |
| substitution | 0.461 | 0.460 | 0.460 | 14098 | l2_phone_error | wavlm_base_plus |
| deletion | 0.338 | 0.493 | 0.401 | 3420 | l2_phone_error | wavlm_base_plus |
| addition | 0.254 | 0.238 | 0.246 | 1092 | l2_phone_error | wavlm_base_plus |
| correct | 0.907 | 0.895 | 0.901 | 101633 | l2_phone_error | hubert_base |
| substitution | 0.440 | 0.457 | 0.448 | 14098 | l2_phone_error | hubert_base |
| deletion | 0.347 | 0.432 | 0.385 | 3420 | l2_phone_error | hubert_base |
| addition | 0.176 | 0.165 | 0.170 | 1092 | l2_phone_error | hubert_base |

## L2-ARCTIC phone error by L1

| system | group | n | macro_f1 | error_share_true | error_recall | error_precision |
|---|---|---|---|---|---|---|
| US 3.1 | Arabic | 20038 | 0.292 | 0.110 | 0.113 | 0.287 |
| US 3.1 | Hindi | 19846 | 0.320 | 0.145 | 0.157 | 0.451 |
| US 3.1 | Korean | 19791 | 0.335 | 0.099 | 0.222 | 0.447 |
| US 3.1 | Mandarin | 19984 | 0.345 | 0.162 | 0.203 | 0.628 |
| US 3.1 | Spanish | 20250 | 0.322 | 0.169 | 0.184 | 0.591 |
| US 3.1 | Vietnamese | 20334 | 0.311 | 0.242 | 0.156 | 0.656 |
| wavlm_base_plus | Arabic | 20038 | 0.468 | 0.110 | 0.448 | 0.351 |
| wavlm_base_plus | Hindi | 19846 | 0.419 | 0.145 | 0.422 | 0.411 |
| wavlm_base_plus | Korean | 19791 | 0.502 | 0.099 | 0.570 | 0.403 |
| wavlm_base_plus | Mandarin | 19984 | 0.501 | 0.162 | 0.562 | 0.502 |
| wavlm_base_plus | Spanish | 20250 | 0.467 | 0.169 | 0.508 | 0.494 |
| wavlm_base_plus | Vietnamese | 20334 | 0.505 | 0.242 | 0.568 | 0.635 |
| hubert_base | Arabic | 20038 | 0.447 | 0.110 | 0.448 | 0.331 |
| hubert_base | Hindi | 19846 | 0.413 | 0.145 | 0.433 | 0.390 |
| hubert_base | Korean | 19791 | 0.450 | 0.099 | 0.545 | 0.399 |
| hubert_base | Mandarin | 19984 | 0.474 | 0.162 | 0.527 | 0.484 |
| hubert_base | Spanish | 20250 | 0.450 | 0.169 | 0.498 | 0.492 |
| hubert_base | Vietnamese | 20334 | 0.484 | 0.242 | 0.527 | 0.632 |

## Error recall per phoneme vs training error support (Spearman)

| task | system | rho_support | p_support | rho_rate | p_rate | n_phonemes |
|---|---|---|---|---|---|---|
| phone_mispron_t1.0 | US 3.1 | 0.322 | 0.083 | 0.457 | 0.011 | 30 |
| phone_mispron_t1.0 | wavlm_base_plus | 0.287 | 0.124 | 0.476 | 0.008 | 30 |
| phone_mispron_t1.0 | hubert_base | 0.392 | 0.032 | 0.699 | 0.000 | 30 |
| l2_phone_error | US 3.1 | 0.593 | 0.000 | 0.922 | 0.000 | 39 |
| l2_phone_error | wavlm_base_plus | 0.656 | 0.000 | 0.895 | 0.000 | 39 |
| l2_phone_error | hubert_base | 0.615 | 0.000 | 0.908 | 0.000 | 39 |
