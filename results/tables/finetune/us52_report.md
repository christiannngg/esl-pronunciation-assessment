# US 5.2 - fine-tuned encoder diagnosis (tags `base`, `base__seen`)

Criteria (declared before the diagnostic runs were inspected; configs/finetune.yaml `diagnose52`): **C1** relative train-vs-validation gap at the best epoch >= 0.30; **C2** validation loss >= 25% above its minimum at the last epoch while training loss fell >= 25% since the best epoch (in >= 50% of jobs); **C3** best epoch <= 1 in >= 50% of jobs; **C4** fine-tuning improved validation by < 2% over the job's own head-only epoch in >= 50% of jobs; **C5** seen-speaker (unseen utterances) score beats held-out-speaker validation by >= 15% with a CI that excludes 0. Risk: high >= 3 criteria, medium 2, else low. C1, C2 and C5 measure overfitting; C3 and C4 measure instability / low value of fine-tuning.

## 1. Overfitting risk and flags per task

| task | jobs | overfit_risk | criteria_fired | mean_best_epoch | mean_epochs_run | gap_best | gap_last | val_loss_rise | train_loss_drop | share_early_best | share_head_only_best | gain_rel_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sent_accuracy | 3 | low | C1 | 5.000 | 10.000 | 0.310 | 0.616 | 0.068 | 0.599 | 0.333 | 0.000 | 0.235 |
| sent_fluency | 3 | low | C1 | 4.333 | 9.333 | 0.360 | 0.637 | 0.121 | 0.492 | 0.000 | 0.000 | 0.506 |
| sent_prosodic | 3 | low |  | 3.333 | 8.333 | 0.215 | 0.493 | 0.244 | 0.566 | 0.000 | 0.000 | 0.558 |
| sent_total | 3 | low |  | 2.667 | 7.667 | 0.236 | 0.330 | 0.196 | 0.082 | 0.333 | 0.000 | 0.386 |
| word_accuracy | 3 | low |  | 1.667 | 6.667 | 0.077 | 0.528 | 0.144 | 0.619 | 0.333 | 0.333 | 0.033 |
| phone_score | 3 | low |  | 4.667 | 9.667 | 0.250 | 0.498 | 0.079 | 0.492 | 0.000 | 0.000 | 0.070 |
| phone_mispron_t1.0 | 3 | low |  | 10.667 | 13.333 | 0.269 | 0.288 | 1.623 | 0.269 | 0.000 | 0.000 | 0.143 |
| l2_phone_error | 4 | low | C1 | 11.750 | 15.000 | 0.315 | 0.323 | 1.240 | 0.085 | 0.000 | 0.000 | 0.367 |
| l2_word_error | 4 | low | C2 | 3.250 | 8.250 | 0.163 | 0.269 | 0.827 | 0.582 | 0.000 | 0.000 | 0.071 |

gap = relative train-vs-validation gap (positive = better on training data); gain_rel = validation improvement over the job's own head-only epoch.

Curves: `results/figures/us52/curves_primary_base.png`, `results/figures/us52/curves_loss_base.png`

## 2. Seen vs held-out speakers

Runs trained on 80 % of each training speaker's utterances. `seen` = the held-back 20 % (same speakers, unseen utterances), `val` / `test` = held-out speakers, `train` = a subsample of training utterances in eval mode. RMSE (regression) or macro-F1 with the class-weighted argmax (classification); 95 % speaker-bootstrap CIs.

| task | split | n | n_speakers | metric | value | lo | hi |
|---|---|---|---|---|---|---|---|
| sent_accuracy | train | 1696 | 106 | rmse | 0.537 | 0.503 | 0.569 |
| sent_accuracy | seen | 424 | 106 | rmse | 0.918 | 0.806 | 1.017 |
| sent_accuracy | val | 380 | 19 | rmse | 0.969 | 0.767 | 1.153 |
| sent_accuracy | test | 2500 | 125 | rmse | 1.101 | 0.966 | 1.213 |
| sent_fluency | train | 1696 | 106 | rmse | 0.405 | 0.381 | 0.430 |
| sent_fluency | seen | 424 | 106 | rmse | 0.705 | 0.647 | 0.768 |
| sent_fluency | val | 380 | 19 | rmse | 0.644 | 0.577 | 0.722 |
| sent_fluency | test | 2500 | 125 | rmse | 0.874 | 0.713 | 1.072 |
| sent_prosodic | train | 1696 | 106 | rmse | 0.391 | 0.369 | 0.412 |
| sent_prosodic | seen | 424 | 106 | rmse | 0.659 | 0.601 | 0.718 |
| sent_prosodic | val | 380 | 19 | rmse | 0.644 | 0.556 | 0.755 |
| sent_prosodic | test | 2500 | 125 | rmse | 0.900 | 0.735 | 1.092 |
| sent_total | train | 1696 | 106 | rmse | 0.502 | 0.474 | 0.531 |
| sent_total | seen | 424 | 106 | rmse | 0.841 | 0.760 | 0.919 |
| sent_total | val | 380 | 19 | rmse | 0.907 | 0.779 | 1.042 |
| sent_total | test | 2500 | 125 | rmse | 1.047 | 0.898 | 1.205 |
| word_accuracy | train | 10617 | 106 | rmse | 1.316 | 1.138 | 1.476 |
| word_accuracy | seen | 2671 | 106 | rmse | 1.605 | 1.351 | 1.815 |
| word_accuracy | val | 2561 | 19 | rmse | 1.587 | 1.135 | 1.985 |
| word_accuracy | test | 15967 | 125 | rmse | 1.609 | 1.371 | 1.815 |
| phone_score | train | 20011 | 106 | rmse | 0.182 | 0.158 | 0.205 |
| phone_score | seen | 7889 | 106 | rmse | 0.327 | 0.272 | 0.379 |
| phone_score | val | 7658 | 19 | rmse | 0.317 | 0.216 | 0.411 |
| phone_score | test | 47369 | 125 | rmse | 0.310 | 0.256 | 0.354 |
| phone_mispron_t1.0 | train | 20011 | 106 | macro_f1 | 0.974 | 0.965 | 0.982 |
| phone_mispron_t1.0 | seen | 7889 | 106 | macro_f1 | 0.728 | 0.685 | 0.758 |
| phone_mispron_t1.0 | val | 7658 | 19 | macro_f1 | 0.687 | 0.578 | 0.744 |
| phone_mispron_t1.0 | test | 47369 | 125 | macro_f1 | 0.663 | 0.632 | 0.683 |
| l2_phone_error | train | 80058 | 24 | macro_f1 | 0.794 | 0.778 | 0.806 |
| l2_phone_error | seen | 48917 | 24 | macro_f1 | 0.578 | 0.545 | 0.602 |
| l2_phone_error | val | 120243 | 24 | macro_f1 | 0.544 | 0.511 | 0.568 |
| l2_phone_error | test | 120243 | 24 | macro_f1 | 0.540 | 0.510 | 0.562 |
| l2_word_error | train | 54241 | 24 | macro_f1 | 0.844 | 0.828 | 0.856 |
| l2_word_error | seen | 13729 | 24 | macro_f1 | 0.747 | 0.727 | 0.761 |
| l2_word_error | val | 33985 | 24 | macro_f1 | 0.716 | 0.699 | 0.732 |
| l2_word_error | test | 33985 | 24 | macro_f1 | 0.716 | 0.696 | 0.733 |

Seen-speaker gap (validation minus seen for RMSE, seen minus validation for macro-F1; positive = seen speakers are easier):

| task | seen | heldout_val | rel_gap | abs_diff_val_minus_seen | lo | hi | ci_excludes_0 |
|---|---|---|---|---|---|---|---|
| sent_accuracy | 0.918 | 0.969 | 0.052 | 0.051 | -0.166 | 0.249 | no |
| sent_fluency | 0.705 | 0.644 | -0.094 | -0.061 | -0.158 | 0.047 | no |
| sent_prosodic | 0.659 | 0.644 | -0.023 | -0.015 | -0.117 | 0.108 | no |
| sent_total | 0.841 | 0.907 | 0.073 | 0.066 | -0.084 | 0.223 | no |
| word_accuracy | 1.605 | 1.587 | -0.012 | -0.019 | -0.459 | 0.457 | no |
| phone_score | 0.327 | 0.317 | -0.032 | -0.010 | -0.112 | 0.104 | no |
| phone_mispron_t1.0 | 0.728 | 0.687 | 0.057 | 0.042 | -0.032 | 0.155 | no |
| l2_phone_error | 0.578 | 0.544 | 0.059 | 0.034 | -0.004 | 0.076 | no |
| l2_word_error | 0.747 | 0.716 | 0.041 | 0.030 | 0.005 | 0.057 | yes |

Sentence-text overlap with the training utterances of the same run (if held-out sentences are also read by training speakers, the seen/held-out gap is not a text effect):

| task | split | utterances | text_in_train_share |
|---|---|---|---|
| sent_accuracy | seen | 424 | 0.000 |
| sent_accuracy | val | 380 | 0.000 |
| sent_accuracy | test | 2500 | 0.015 |
| sent_fluency | seen | 424 | 0.000 |
| sent_fluency | val | 380 | 0.000 |
| sent_fluency | test | 2500 | 0.015 |
| sent_prosodic | seen | 424 | 0.000 |
| sent_prosodic | val | 380 | 0.000 |
| sent_prosodic | test | 2500 | 0.015 |
| sent_total | seen | 424 | 0.000 |
| sent_total | val | 380 | 0.000 |
| sent_total | test | 2500 | 0.015 |
| word_accuracy | seen | 424 | 0.000 |
| word_accuracy | val | 380 | 0.000 |
| word_accuracy | test | 2500 | 0.015 |
| phone_score | seen | 424 | 0.000 |
| phone_score | val | 380 | 0.000 |
| phone_score | test | 2500 | 0.015 |
| phone_mispron_t1.0 | seen | 424 | 0.000 |
| phone_mispron_t1.0 | val | 380 | 0.000 |
| phone_mispron_t1.0 | test | 2500 | 0.015 |
| l2_phone_error | seen | 1440 | 0.947 |
| l2_phone_error | val | 3599 | 0.981 |
| l2_phone_error | test | 3599 | 0.980 |
| l2_word_error | seen | 1440 | 0.947 |
| l2_word_error | val | 3599 | 0.981 |
| l2_word_error | test | 3599 | 0.980 |

## 3. Fine-tuned vs frozen 4.3 (same encoder)

Test: paired speaker bootstrap from US 5.1. Validation: difference of the validation primary metric (RMSE lower / macro-F1 higher is better); `job_val_sd` is the spread of the fine-tuned validation score across jobs. `status` follows the paired test verdict; `val_regression` marks tasks where validation is worse than frozen by more than the job spread.

| task | metric | frozen_43_test | finetuned_test | delta_test | delta_lo | delta_hi | test_verdict | status | val_frozen_43 | val_finetuned | val_delta | job_val_sd | val_regression |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sent_accuracy | rmse | 1.113 | 1.071 | -0.042 | -0.067 | -0.020 | better | helped | 0.977 | 0.991 | 0.014 | 0.008 | yes |
| sent_fluency | rmse | 0.915 | 0.870 | -0.045 | -0.073 | -0.024 | better | helped | 0.676 | 0.655 | -0.021 | 0.002 | no |
| sent_prosodic | rmse | 0.901 | 0.902 | 0.001 | -0.017 | 0.018 | no significant change | unchanged | 0.679 | 0.644 | -0.035 | 0.018 | no |
| sent_total | rmse | 1.114 | 1.066 | -0.048 | -0.076 | -0.022 | better | helped | 0.903 | 0.916 | 0.013 | 0.008 | yes |
| word_accuracy | rmse | 1.633 | 1.598 | -0.035 | -0.069 | -0.010 | better | helped | 1.662 | 1.582 | -0.080 | 0.022 | no |
| phone_score | rmse | 0.314 | 0.306 | -0.007 | -0.017 | 0.000 | no significant change | unchanged | 0.334 | 0.312 | -0.022 | 0.009 | no |
| phone_mispron_t1.0 | macro_f1 | 0.661 | 0.680 | 0.018 | 0.009 | 0.028 | better | helped | 0.648 | 0.697 | 0.049 | 0.007 | no |
| l2_phone_error | macro_f1 | 0.523 | 0.559 | 0.036 | 0.018 | 0.055 | better | helped | 0.528 | 0.566 | 0.038 | 0.031 | no |
| l2_word_error | macro_f1 | 0.716 | 0.721 | 0.005 | -0.002 | 0.011 | no significant change | unchanged | 0.716 | 0.727 | 0.011 | 0.017 | no |

### Age-group / L1 breakdown, test rows (descriptive; not used to select fixes)

| task | group_by | group | n | n_speakers | frozen_43 | finetuned | delta | lo | hi | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| sent_accuracy | age group | adult | 1220 | 61 | 1.108 | 1.071 | -0.037 | -0.077 | -0.006 | better |
| sent_accuracy | age group | child/teen | 1280 | 64 | 1.118 | 1.071 | -0.047 | -0.082 | -0.011 | better |
| sent_fluency | age group | adult | 1220 | 61 | 0.833 | 0.784 | -0.049 | -0.093 | -0.009 | better |
| sent_fluency | age group | child/teen | 1280 | 64 | 0.987 | 0.945 | -0.042 | -0.069 | -0.016 | better |
| sent_prosodic | age group | adult | 1220 | 61 | 0.840 | 0.856 | 0.016 | -0.011 | 0.044 | no significant change |
| sent_prosodic | age group | child/teen | 1280 | 64 | 0.955 | 0.944 | -0.011 | -0.039 | 0.014 | no significant change |
| sent_total | age group | adult | 1220 | 61 | 1.019 | 0.989 | -0.030 | -0.071 | 0.007 | no significant change |
| sent_total | age group | child/teen | 1280 | 64 | 1.197 | 1.134 | -0.063 | -0.099 | -0.022 | better |
| word_accuracy | age group | adult | 8701 | 61 | 1.743 | 1.724 | -0.019 | -0.060 | 0.016 | no significant change |
| word_accuracy | age group | child/teen | 7266 | 64 | 1.491 | 1.433 | -0.058 | -0.098 | -0.026 | better |
| phone_score | age group | adult | 25752 | 61 | 0.333 | 0.332 | -0.002 | -0.014 | 0.010 | no significant change |
| phone_score | age group | child/teen | 21617 | 64 | 0.288 | 0.273 | -0.016 | -0.028 | -0.005 | better |
| phone_mispron_t1.0 | age group | adult | 25752 | 61 | 0.685 | 0.703 | 0.018 | 0.004 | 0.030 | better |
| phone_mispron_t1.0 | age group | child/teen | 21617 | 64 | 0.588 | 0.608 | 0.020 | 0.006 | 0.032 | better |
| l2_phone_error | L1 | Arabic | 20038 | 4 | 0.514 | 0.536 | 0.022 | -0.020 | 0.057 | no significant change |
| l2_phone_error | L1 | Hindi | 19846 | 4 | 0.414 | 0.453 | 0.039 | -0.022 | 0.085 | no significant change |
| l2_phone_error | L1 | Korean | 19791 | 4 | 0.529 | 0.534 | 0.006 | 0.004 | 0.015 | better |
| l2_phone_error | L1 | Mandarin | 19984 | 4 | 0.477 | 0.521 | 0.044 | 0.007 | 0.086 | better |
| l2_phone_error | L1 | Spanish | 20250 | 4 | 0.496 | 0.528 | 0.032 | 0.018 | 0.048 | better |
| l2_phone_error | L1 | Vietnamese | 20334 | 4 | 0.498 | 0.577 | 0.079 | 0.045 | 0.116 | better |
| l2_word_error | L1 | Arabic | 5663 | 4 | 0.689 | 0.697 | 0.008 | -0.013 | 0.030 | no significant change |
| l2_word_error | L1 | Hindi | 5661 | 4 | 0.674 | 0.670 | -0.004 | -0.012 | 0.003 | no significant change |
| l2_word_error | L1 | Korean | 5618 | 4 | 0.696 | 0.699 | 0.003 | -0.012 | 0.019 | no significant change |
| l2_word_error | L1 | Mandarin | 5647 | 4 | 0.706 | 0.715 | 0.009 | 0.003 | 0.016 | better |
| l2_word_error | L1 | Spanish | 5703 | 4 | 0.722 | 0.717 | -0.005 | -0.019 | 0.007 | no significant change |
| l2_word_error | L1 | Vietnamese | 5693 | 4 | 0.756 | 0.770 | 0.014 | 0.007 | 0.017 | better |

### Regression to the mean (test)

| task | system | slope_pred_on_true | pcc | low_bin_share | low_bin_mean_true | low_bin_mean_pred |
|---|---|---|---|---|---|---|
| word_accuracy | frozen 4.3 | 0.200 | 0.415 | 0.054 | 2.841 | 8.136 |
| word_accuracy | fine-tuned | 0.168 | 0.477 | 0.054 | 2.841 | 8.509 |
| phone_score | frozen 4.3 | 0.342 | 0.533 | 0.030 | 0.128 | 1.313 |
| phone_score | fine-tuned | 0.275 | 0.567 | 0.030 | 0.128 | 1.438 |
