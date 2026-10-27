# US 6.1 - single-task vs multi-task (tag `mt_lr`, test set; every choice made on validation)

Metric: RMSE (regression, lower is better) or macro-F1 (classification, higher is better). Both systems are fine-tuned WavLM-base-plus, seed-averaged, on the identical rows and the identical seeds; the only difference is that the multi-task encoder is shared by the heads of its group. Delta = multi-task minus single-task on test, paired speaker bootstrap (95 % CI, two-sided p); for RMSE a negative delta favours multi-task, for macro-F1 a positive one. The verdict says `better` / `worse` only when the interval excludes 0.

| group | task | metric | single-task 5.1 | single-task 5.3 | **multi-task** [95% CI] | delta vs 5.1 [95% CI] | verdict vs 5.1 | delta vs 5.3 [95% CI] | verdict vs 5.3 |
|---|---|---|---|---|---|---|---|---|---|
| l2 | l2_phone_error | macro_f1 | 0.564 | 0.581 (lr) | **0.577** [0.545, 0.599] | +0.013 [+0.008, +0.019] p=0.000 | better | -0.005 [-0.007, -0.002] p=0.000 | worse |
| l2 | l2_word_error | macro_f1 | 0.722 | 0.722 (lr) | **0.720** [0.700, 0.738] | -0.002 [-0.007, +0.003] p=0.332 | no significant change | -0.002 [-0.006, +0.001] p=0.188 | no significant change |

Vs single-task 5.1: better 1, no significant change 1 (of 2 heads).

Vs single-task 5.3: no significant change 1, worse 1 (of 2 heads).

## Validation side (margin = max(0.01, job-to-job sd of the single-task validation metric))

| task | metric | single-task val (5.1) | multi-task val | multi-task worse by | margin | flag | seeds used |
|---|---|---|---|---|---|---|---|
| l2_phone_error | macro_f1 | 0.564 | 0.576 | -0.012 | 0.010 | better | 3 |
| l2_word_error | macro_f1 | 0.731 | 0.720 | 0.011 | 0.010 | worse | 3 |

Positive 'worse by' = multi-task is worse on validation (RMSE units for regression, 1 - macro-F1 for classification). Validation has 19 SO762 speakers: differences below the margin are not credited.

## Training cost (one fold and seed)

| group | heads | encoders (multi-task / single-task) | multi-task minutes | sum of single-task minutes | ratio |
|---|---|---|---|---|---|
| l2 | 2 | 1 / 2 | 19.5 | 4.2 | 4.65 |

Minutes are wall clock on a possibly shared GPU: indicative only.

## Jobs

One checkpoint per multi-task job (`best epoch`). `head's own best` = the epoch at which that head alone was best on validation: a large gap means one stopping epoch is a compromise (looked at in US 6.2).

| group | task | jobs | shared best epoch | head's own best epoch | epochs run | sd of validation score across jobs |
|---|---|---|---|---|---|---|
| l2 | l2_phone_error | 12 | 13.3 | 13.3 | 15.0 | 0.030 |
| l2 | l2_word_error | 12 | 13.3 | 7.8 | 15.0 | 0.015 |

## Caveats

- `phone_mispron_t1.0` is `phone_score < 1.0` (the same labels thresholded) and `sent_total` is highly correlated with the other sentence scores: heads on near-duplicate targets can make joint training look better than it is. US 6.2 separates this from real transfer.
- L2-ARCTIC: the single-task `base` jobs of Sprint 5 used one seed per fold; seeds 1 and 2 were added for this sprint under the tag `<tag>_l2x`. Both sides are always restricted to the seeds they share.
- One multi-task checkpoint per job is chosen on the mean relative validation error over the heads; single-task jobs each choose their own epoch.

