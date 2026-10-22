# US 5.3 - weakness fixes for the fine-tuned encoder (before / after)

Variants (one knob group each, relative to the 5.1 recipe `base`): `lr`, `reg`, `mask`, `noise`, `freeze6`, `freeze9`. Definitions in `configs/finetune.yaml` `variants`.

Targets and selection rule were fixed before any variant was run (`diagnose53`): relative train-validation gap at the best epoch <= 0.3, last-epoch validation loss <= 1.25x (regression) / 1.5x (classification) its minimum, no training-loss spike >= 2.0x; a variant is eligible if its validation metric is not worse than base by more than max(0.01, seed sd of base). Selection uses validation-side quantities only; test is reported afterwards.

## 1. Selection per task

| task | selected | need_fix | n_complete | n_eligible | margin | reason |
|---|---|---|---|---|---|---|
| sent_accuracy | lr | yes | 1 | 1 | 0.011 | lr meets the targets within the validation margin (best validation metric among 1 qualifying) |
| sent_fluency | base | yes | 1 | 0 | 0.010 | every variant is worse than base on validation by more than the margin |
| sent_prosodic | base | no | 1 | 0 | 0.018 | base already meets the targets (no weakness to fix) |
| sent_total | lr | yes | 1 | 1 | 0.010 | lr meets the targets within the validation margin (best validation metric among 1 qualifying) |
| word_accuracy | base | no | 1 | 0 | 0.021 | base already meets the targets (no weakness to fix) |
| phone_score | base | no | 1 | 0 | 0.010 | base already meets the targets (no weakness to fix) |
| phone_mispron_t1.0 | base | yes | 1 | 0 | 0.010 | every variant is worse than base on validation by more than the margin |
| l2_phone_error | lr | yes | 1 | 1 | 0.010 | lr meets the targets within the validation margin (best validation metric among 1 qualifying) |
| l2_word_error | lr | yes | 1 | 1 | 0.010 | lr meets the targets within the validation margin (best validation metric among 1 qualifying) |

## 2. Before / after (selected variant vs base 5.1)

Validation and training-side columns are job means (`val` = early-stopping metric at the best epoch). Test is the seed-averaged headline (class-weight / threshold variant chosen on validation), `d_vs_base` a paired speaker bootstrap of the selected model against base on the same test rows.

| task | selected | val_before | val_after | gap_before | gap_after | loss_ratio_before | loss_ratio_after | spike_before | spike_after | test_before | test_after | d_vs_base | d_vs_base_lo | d_vs_base_hi | verdict_vs_base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sent_accuracy | lr | 1.026 | 0.987 | 0.310 | 0.037 | 1.068 | 1.196 | 1.004 | 1.018 | 1.071 | 1.101 | 0.030 | 0.005 | 0.063 | worse |
| sent_fluency | base | 0.666 | 0.666 | 0.360 | 0.360 | 1.121 | 1.121 | 1.071 | 1.071 | 0.870 | 0.870 | 0.000 | 0.000 | 0.000 | same model |
| sent_prosodic | base | 0.656 | 0.656 | 0.215 | 0.215 | 1.244 | 1.244 | 0.908 | 0.908 | 0.902 | 0.902 | 0.000 | 0.000 | 0.000 | same model |
| sent_total | lr | 0.941 | 0.927 | 0.236 | -0.041 | 1.196 | 1.088 | 5.348 | 0.943 | 1.066 | 1.078 | 0.012 | -0.024 | 0.054 | no significant change |
| word_accuracy | base | 1.620 | 1.620 | 0.077 | 0.077 | 1.144 | 1.144 | 0.853 | 0.853 | 1.598 | 1.598 | 0.000 | 0.000 | 0.000 | same model |
| phone_score | base | 0.321 | 0.321 | 0.250 | 0.250 | 1.079 | 1.079 | 0.875 | 0.875 | 0.306 | 0.306 | 0.000 | 0.000 | 0.000 | same model |
| phone_mispron_t1.0 | base | 0.689 | 0.689 | 0.269 | 0.269 | 2.623 | 2.623 | 0.932 | 0.932 | 0.680 | 0.680 | 0.000 | 0.000 | 0.000 | same model |
| l2_phone_error | lr | 0.554 | 0.547 | 0.315 | 0.247 | 2.240 | 1.456 | 0.987 | 0.995 | 0.559 | 0.576 | 0.017 | 0.008 | 0.026 | better |
| l2_word_error | lr | 0.726 | 0.723 | 0.163 | 0.125 | 1.827 | 1.143 | 0.916 | 0.947 | 0.721 | 0.719 | -0.002 | -0.006 | 0.003 | no significant change |

Against the frozen US 4.3 system of the same encoder (paired, test):

| task | d43_before | verdict43_before | d43_after | d43_after_lo | d43_after_hi | verdict43_after |
|---|---|---|---|---|---|---|
| sent_accuracy | -0.042 | better | -0.012 | -0.042 | 0.017 | no significant change |
| sent_fluency | -0.045 | better | -0.045 | -0.073 | -0.024 | better |
| sent_prosodic | 0.001 | no significant change | 0.001 | -0.017 | 0.018 | no significant change |
| sent_total | -0.048 | better | -0.036 | -0.087 | 0.008 | no significant change |
| word_accuracy | -0.035 | better | -0.035 | -0.069 | -0.010 | better |
| phone_score | -0.007 | no significant change | -0.007 | -0.017 | 0.000 | no significant change |
| phone_mispron_t1.0 | 0.018 | better | 0.018 | 0.009 | 0.028 | better |
| l2_phone_error | 0.036 | better | 0.053 | 0.030 | 0.076 | better |
| l2_word_error | 0.005 | no significant change | 0.003 | -0.003 | 0.008 | no significant change |

Cost (mean seconds per job, epochs run):

| task | selected | fit_s_before | fit_s_after | epochs_before | epochs_after |
|---|---|---|---|---|---|
| sent_accuracy | lr | 149.1 | 227.5 | 10.0 | 9.3 |
| sent_fluency | base | 144.9 | 144.9 | 9.3 | 9.3 |
| sent_prosodic | base | 131.3 | 131.3 | 8.3 | 8.3 |
| sent_total | lr | 124.7 | 155.4 | 7.7 | 7.3 |
| word_accuracy | base | 119.2 | 119.2 | 6.7 | 6.7 |
| phone_score | base | 136.6 | 136.6 | 9.7 | 9.7 |
| phone_mispron_t1.0 | base | 165.5 | 165.5 | 13.3 | 13.3 |
| l2_phone_error | lr | 144.0 | 169.1 | 15.0 | 15.0 |
| l2_word_error | lr | 107.1 | 132.9 | 8.2 | 9.2 |

## 3. Every variant (validation side; test shown for reference only)

| task | variant | n_jobs | val | worse_by | gap_best | loss_ratio | spike | best_epoch | epochs_run | fit_s | target_met | eligible | selected | test | test_delta_vs_base | test_lo | test_hi | test_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sent_accuracy | base | 3 | 1.026 | 0.000 | 0.310 | 1.068 | 1.004 | 5.000 | 10.000 | 149.060 | no | yes | no | 1.071 |  |  |  |  |
| sent_accuracy | lr | 3 | 0.987 | -0.039 | 0.037 | 1.196 | 1.018 | 4.333 | 9.333 | 227.490 | yes | yes | yes | 1.101 | 0.030 | 0.005 | 0.063 | worse |
| sent_fluency | base | 3 | 0.666 | 0.000 | 0.360 | 1.121 | 1.071 | 4.333 | 9.333 | 144.884 | no | yes | yes | 0.870 |  |  |  |  |
| sent_fluency | lr | 3 | 0.688 | 0.022 | 0.082 | 1.030 | 1.003 | 4.000 | 9.000 | 177.798 | yes | no | no | 0.871 | 0.001 | -0.021 | 0.022 | no significant change |
| sent_prosodic | base | 3 | 0.656 | 0.000 | 0.215 | 1.244 | 0.908 | 3.333 | 8.333 | 131.288 | yes | yes | yes | 0.902 |  |  |  |  |
| sent_prosodic | lr | 3 | 0.677 | 0.021 | 0.096 | 1.185 | 0.987 | 4.333 | 9.333 | 168.271 | yes | no | no | 0.917 | 0.015 | -0.008 | 0.040 | no significant change |
| sent_total | base | 3 | 0.941 | 0.000 | 0.236 | 1.196 | 5.348 | 2.667 | 7.667 | 124.684 | no | yes | no | 1.066 |  |  |  |  |
| sent_total | lr | 3 | 0.927 | -0.014 | -0.041 | 1.088 | 0.943 | 2.333 | 7.333 | 155.444 | yes | yes | yes | 1.078 | 0.012 | -0.024 | 0.054 | no significant change |
| word_accuracy | base | 3 | 1.620 | 0.000 | 0.077 | 1.144 | 0.853 | 1.667 | 6.667 | 119.190 | yes | yes | yes | 1.598 |  |  |  |  |
| word_accuracy | lr | 3 | 1.657 | 0.037 | 0.091 | 1.166 | 0.906 | 2.667 | 7.667 | 161.877 | yes | no | no | 1.587 | -0.011 | -0.022 | 0.002 | no significant change |
| phone_score | base | 3 | 0.321 | 0.000 | 0.250 | 1.079 | 0.875 | 4.667 | 9.667 | 136.553 | yes | yes | yes | 0.306 |  |  |  |  |
| phone_score | lr | 3 | 0.333 | 0.013 | 0.058 | 1.073 | 0.928 | 4.000 | 9.000 | 153.076 | yes | no | no | 0.305 | -0.001 | -0.003 | 0.002 | no significant change |
| phone_mispron_t1.0 | base | 3 | 0.689 | 0.000 | 0.269 | 2.623 | 0.932 | 10.667 | 13.333 | 165.526 | no | yes | yes | 0.680 |  |  |  |  |
| phone_mispron_t1.0 | lr | 3 | 0.653 | 0.035 | 0.192 | 1.425 | 1.004 | 8.333 | 11.667 | 178.963 | yes | no | no | 0.696 | 0.016 | 0.004 | 0.029 | better |
| l2_phone_error | base | 4 | 0.554 | 0.000 | 0.315 | 2.240 | 0.987 | 11.750 | 15.000 | 144.043 | no | yes | no | 0.559 |  |  |  |  |
| l2_phone_error | lr | 4 | 0.547 | 0.007 | 0.247 | 1.456 | 0.995 | 14.000 | 15.000 | 169.076 | yes | yes | yes | 0.576 | 0.017 | 0.008 | 0.026 | better |
| l2_word_error | base | 4 | 0.726 | 0.000 | 0.163 | 1.827 | 0.916 | 3.250 | 8.250 | 107.139 | no | yes | no | 0.721 |  |  |  |  |
| l2_word_error | lr | 4 | 0.723 | 0.003 | 0.125 | 1.143 | 0.947 | 4.250 | 9.250 | 132.896 | yes | yes | yes | 0.719 | -0.002 | -0.006 | 0.003 | no significant change |

## 4. Regression to the mean (test), base vs selected

| task | system | slope_pred_on_true | pcc | low_bin_share | low_bin_mean_true | low_bin_mean_pred |
|---|---|---|---|---|---|---|
| word_accuracy | 5.1 base | 0.168 | 0.477 | 0.054 | 2.841 | 8.509 |
| word_accuracy | 5.3 selected (base) | 0.168 | 0.477 | 0.054 | 2.841 | 8.509 |
| phone_score | 5.1 base | 0.275 | 0.567 | 0.030 | 0.128 | 1.438 |
| phone_score | 5.3 selected (base) | 0.275 | 0.567 | 0.030 | 0.128 | 1.438 |
