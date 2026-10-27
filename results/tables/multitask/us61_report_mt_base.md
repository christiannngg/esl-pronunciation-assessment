# US 6.1 - single-task vs multi-task (tag `mt_base`, test set; every choice made on validation)

Metric: RMSE (regression, lower is better) or macro-F1 (classification, higher is better). Both systems are fine-tuned WavLM-base-plus, seed-averaged, on the identical rows and the identical seeds; the only difference is that the multi-task encoder is shared by the heads of its group. Delta = multi-task minus single-task on test, paired speaker bootstrap (95 % CI, two-sided p); for RMSE a negative delta favours multi-task, for macro-F1 a positive one. The verdict says `better` / `worse` only when the interval excludes 0.

| group | task | metric | single-task 5.1 | single-task 5.3 | **multi-task** [95% CI] | delta vs 5.1 [95% CI] | verdict vs 5.1 | delta vs 5.3 [95% CI] | verdict vs 5.3 |
|---|---|---|---|---|---|---|---|---|---|
| so762 | sent_accuracy | rmse | 1.071 | 1.101 (lr) | **1.070** [0.954, 1.173] | -0.001 [-0.020, +0.017] p=0.860 | no significant change | -0.031 [-0.071, +0.001] p=0.056 | no significant change |
| so762 | sent_fluency | rmse | 0.870 | 0.861 (freeze6) | **0.887** [0.716, 1.095] | +0.017 [-0.002, +0.035] p=0.076 | no significant change | +0.026 [-0.018, +0.071] p=0.300 | no significant change |
| so762 | sent_prosodic | rmse | 0.902 | 0.902 (base) | **0.914** [0.755, 1.104] | +0.011 [-0.009, +0.031] p=0.296 | no significant change | +0.011 [-0.009, +0.031] p=0.296 | no significant change |
| so762 | sent_total | rmse | 1.066 | 1.078 (lr) | **1.055** [0.903, 1.220] | -0.011 [-0.029, +0.006] p=0.256 | no significant change | -0.023 [-0.073, +0.022] p=0.312 | no significant change |
| so762 | word_accuracy | rmse | 1.598 | 1.598 (base) | **1.598** [1.368, 1.795] | -0.001 [-0.034, +0.033] p=0.996 | no significant change | -0.001 [-0.034, +0.033] p=0.996 | no significant change |
| so762 | phone_score | rmse | 0.306 | 0.306 (base) | **0.303** [0.251, 0.345] | -0.003 [-0.009, +0.002] p=0.212 | no significant change | -0.003 [-0.009, +0.002] p=0.212 | no significant change |
| so762 | phone_mispron_t1.0 | macro_f1 | 0.680 | 0.680 (base) | **0.654** [0.614, 0.684] | -0.026 [-0.045, -0.009] p=0.004 | worse | -0.026 [-0.045, -0.009] p=0.004 | worse |

Vs single-task 5.1: no significant change 6, worse 1 (of 7 heads).

Vs single-task 5.3: no significant change 6, worse 1 (of 7 heads).

## Validation side (margin = max(0.01, job-to-job sd of the single-task validation metric))

| task | metric | single-task val (5.1) | multi-task val | multi-task worse by | margin | flag | seeds used |
|---|---|---|---|---|---|---|---|
| sent_accuracy | rmse | 0.991 | 0.955 | -0.036 | 0.010 | better | 3 |
| sent_fluency | rmse | 0.655 | 0.680 | 0.025 | 0.010 | worse | 3 |
| sent_prosodic | rmse | 0.644 | 0.680 | 0.036 | 0.018 | worse | 3 |
| sent_total | rmse | 0.916 | 0.875 | -0.041 | 0.010 | better | 3 |
| word_accuracy | rmse | 1.582 | 1.648 | 0.066 | 0.022 | worse | 3 |
| phone_score | rmse | 0.312 | 0.322 | 0.009 | 0.010 | within margin | 3 |
| phone_mispron_t1.0 | macro_f1 | 0.697 | 0.690 | 0.007 | 0.010 | within margin | 3 |

Positive 'worse by' = multi-task is worse on validation (RMSE units for regression, 1 - macro-F1 for classification). Validation has 19 SO762 speakers: differences below the margin are not credited.

## Training cost (one fold and seed)

| group | heads | encoders (multi-task / single-task) | multi-task minutes | sum of single-task minutes | ratio |
|---|---|---|---|---|---|
| so762 | 7 | 1 / 7 | 23.2 | 16.2 | 1.43 |

Minutes are wall clock on a possibly shared GPU: indicative only.

## Jobs

One checkpoint per multi-task job (`best epoch`). `head's own best` = the epoch at which that head alone was best on validation: a large gap means one stopping epoch is a compromise (looked at in US 6.2).

| group | task | jobs | shared best epoch | head's own best epoch | epochs run | sd of validation score across jobs |
|---|---|---|---|---|---|---|
| so762 | sent_accuracy | 3 | 7.7 | 4.0 | 12.7 | 0.027 |
| so762 | sent_fluency | 3 | 7.7 | 6.7 | 12.7 | 0.022 |
| so762 | sent_prosodic | 3 | 7.7 | 8.0 | 12.7 | 0.048 |
| so762 | sent_total | 3 | 7.7 | 6.3 | 12.7 | 0.019 |
| so762 | word_accuracy | 3 | 7.7 | 3.0 | 12.7 | 0.065 |
| so762 | phone_score | 3 | 7.7 | 5.0 | 12.7 | 0.009 |
| so762 | phone_mispron_t1.0 | 3 | 7.7 | 11.0 | 12.7 | 0.018 |

## Caveats

- `phone_mispron_t1.0` is `phone_score < 1.0` (the same labels thresholded) and `sent_total` is highly correlated with the other sentence scores: heads on near-duplicate targets can make joint training look better than it is. US 6.2 separates this from real transfer.
- L2-ARCTIC: the single-task `base` jobs of Sprint 5 used one seed per fold; seeds 1 and 2 were added for this sprint under the tag `<tag>_l2x`. Both sides are always restricted to the seeds they share.
- One multi-task checkpoint per job is chosen on the mean relative validation error over the heads; single-task jobs each choose their own epoch.

