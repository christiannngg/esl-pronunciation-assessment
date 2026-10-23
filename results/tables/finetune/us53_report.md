# US 5.3 - weakness fixes for the fine-tuned encoder (before / after)

Variants (one knob group each, relative to the 5.1 recipe `base`): `lr`, `reg`, `mask`, `noise`, `freeze6`, `freeze9`. Definitions in `configs/finetune.yaml` `variants`.

Targets and selection rule were fixed before any variant was run (`diagnose53`): relative train-validation gap at the best epoch <= 0.3, last-epoch validation loss <= 1.25x (regression) / 1.5x (classification) its minimum, no training-loss spike >= 2.0x; a variant is eligible if its validation metric is not worse than base by more than max(0.01, seed sd of base). Selection uses validation-side quantities only; test is reported afterwards.

## 1. Selection per task

| task               | selected | need_fix | n_complete | n_eligible | margin | reason                                                                                             |
| ------------------ | -------- | -------- | ---------- | ---------- | ------ | -------------------------------------------------------------------------------------------------- |
| sent_accuracy      | lr       | yes      | 5          | 5          | 0.011  | lr meets the targets within the validation margin (best validation metric among 4 qualifying)      |
| sent_fluency       | freeze6  | yes      | 5          | 2          | 0.010  | freeze6 meets the targets within the validation margin (best validation metric among 1 qualifying) |
| sent_prosodic      | base     | no       | 6          | 3          | 0.018  | base already meets the targets (no weakness to fix)                                                |
| sent_total         | lr       | yes      | 5          | 4          | 0.010  | lr meets the targets within the validation margin (best validation metric among 2 qualifying)      |
| word_accuracy      | base     | no       | 6          | 3          | 0.021  | base already meets the targets (no weakness to fix)                                                |
| phone_score        | base     | no       | 6          | 4          | 0.010  | base already meets the targets (no weakness to fix)                                                |
| phone_mispron_t1.0 | base     | yes      | 5          | 3          | 0.010  | no non-inferior variant meets the targets; weakness not fixed by the tried regularisers            |
| l2_phone_error     | lr       | yes      | 5          | 5          | 0.010  | lr meets the targets within the validation margin (best validation metric among 1 qualifying)      |
| l2_word_error      | lr       | yes      | 6          | 6          | 0.010  | lr meets the targets within the validation margin (best validation metric among 3 qualifying)      |

## 2. Before / after (selected variant vs base 5.1)

Validation and training-side columns are job means (`val` = early-stopping metric at the best epoch, plain argmax); `val_headline` is the seed-averaged validation score with the same decision rule as the test headline (cw / cw+thr tuned on validation). Test is the seed-averaged headline (class-weight / threshold variant chosen on validation), `d_vs_base` a paired speaker bootstrap of the selected model against base on the same test rows.

| task               | selected | val_before | val_after | val_headline_before | val_headline_after | gap_before | gap_after | loss_ratio_before | loss_ratio_after | spike_before | spike_after | test_before | test_after | d_vs_base | d_vs_base_lo | d_vs_base_hi | verdict_vs_base       |
| ------------------ | -------- | ---------- | --------- | ------------------- | ------------------ | ---------- | --------- | ----------------- | ---------------- | ------------ | ----------- | ----------- | ---------- | --------- | ------------ | ------------ | --------------------- |
| sent_accuracy      | lr       | 1.026      | 0.987     | 0.991               | 0.967              | 0.310      | 0.037     | 1.068             | 1.196            | 1.004        | 1.018       | 1.071       | 1.101      | 0.030     | 0.005        | 0.063        | worse                 |
| sent_fluency       | freeze6  | 0.666      | 0.669     | 0.655               | 0.656              | 0.360      | 0.194     | 1.121             | 1.104            | 1.071        | 1.009       | 0.870       | 0.861      | -0.009    | -0.041       | 0.023        | no significant change |
| sent_prosodic      | base     | 0.656      | 0.656     | 0.644               | 0.644              | 0.215      | 0.215     | 1.244             | 1.244            | 0.908        | 0.908       | 0.902       | 0.902      | 0.000     | 0.000        | 0.000        | same model            |
| sent_total         | lr       | 0.941      | 0.927     | 0.916               | 0.903              | 0.236      | -0.041    | 1.196             | 1.088            | 5.348        | 0.943       | 1.066       | 1.078      | 0.012     | -0.024       | 0.054        | no significant change |
| word_accuracy      | base     | 1.620      | 1.620     | 1.582               | 1.582              | 0.077      | 0.077     | 1.144             | 1.144            | 0.853        | 0.853       | 1.598       | 1.598      | 0.000     | 0.000        | 0.000        | same model            |
| phone_score        | base     | 0.321      | 0.321     | 0.312               | 0.312              | 0.250      | 0.250     | 1.079             | 1.079            | 0.875        | 0.875       | 0.306       | 0.306      | 0.000     | 0.000        | 0.000        | same model            |
| phone_mispron_t1.0 | base     | 0.689      | 0.689     | 0.697               | 0.697              | 0.269      | 0.269     | 2.623             | 2.623            | 0.932        | 0.932       | 0.680       | 0.680      | 0.000     | 0.000        | 0.000        | same model            |
| l2_phone_error     | lr       | 0.554      | 0.547     | 0.566               | 0.576              | 0.315      | 0.247     | 2.240             | 1.456            | 0.987        | 0.995       | 0.559       | 0.576      | 0.017     | 0.008        | 0.026        | better                |
| l2_word_error      | lr       | 0.726      | 0.723     | 0.727               | 0.725              | 0.163      | 0.125     | 1.827             | 1.143            | 0.916        | 0.947       | 0.721       | 0.719      | -0.002    | -0.006       | 0.003        | no significant change |

Against the frozen US 4.3 system of the same encoder (paired, test):

| task               | d43_before | verdict43_before      | d43_after | d43_after_lo | d43_after_hi | verdict43_after       |
| ------------------ | ---------- | --------------------- | --------- | ------------ | ------------ | --------------------- |
| sent_accuracy      | -0.042     | better                | -0.012    | -0.042       | 0.017        | no significant change |
| sent_fluency       | -0.045     | better                | -0.054    | -0.088       | -0.022       | better                |
| sent_prosodic      | 0.001      | no significant change | 0.001     | -0.017       | 0.018        | no significant change |
| sent_total         | -0.048     | better                | -0.036    | -0.087       | 0.008        | no significant change |
| word_accuracy      | -0.035     | better                | -0.035    | -0.069       | -0.010       | better                |
| phone_score        | -0.007     | no significant change | -0.007    | -0.017       | 0.000        | no significant change |
| phone_mispron_t1.0 | 0.018      | better                | 0.018     | 0.009        | 0.028        | better                |
| l2_phone_error     | 0.036      | better                | 0.053     | 0.030        | 0.076        | better                |
| l2_word_error      | 0.005      | no significant change | 0.003     | -0.003       | 0.008        | no significant change |

Cost (mean seconds per job, epochs run):

| task               | selected | fit_s_before | fit_s_after | epochs_before | epochs_after |
| ------------------ | -------- | ------------ | ----------- | ------------- | ------------ |
| sent_accuracy      | lr       | 149.1        | 227.5       | 10.0          | 9.3          |
| sent_fluency       | freeze6  | 144.9        | 156.6       | 9.3           | 9.3          |
| sent_prosodic      | base     | 131.3        | 131.3       | 8.3           | 8.3          |
| sent_total         | lr       | 124.7        | 155.4       | 7.7           | 7.3          |
| word_accuracy      | base     | 119.2        | 119.2       | 6.7           | 6.7          |
| phone_score        | base     | 136.6        | 136.6       | 9.7           | 9.7          |
| phone_mispron_t1.0 | base     | 165.5        | 165.5       | 13.3          | 13.3         |
| l2_phone_error     | lr       | 144.0        | 169.1       | 15.0          | 15.0         |
| l2_word_error      | lr       | 107.1        | 132.9       | 8.2           | 9.2          |

## 3. Every variant (validation side; test shown for reference only)

| task               | variant | n_jobs | val   | worse_by | val_headline | val_headline_vs_base | gap_best | loss_ratio | spike | best_epoch | epochs_run | fit_s   | target_met | eligible | selected | test  | test_delta_vs_base | test_lo | test_hi | test_verdict          |
| ------------------ | ------- | ------ | ----- | -------- | ------------ | -------------------- | -------- | ---------- | ----- | ---------- | ---------- | ------- | ---------- | -------- | -------- | ----- | ------------------ | ------- | ------- | --------------------- |
| sent_accuracy      | base    | 3      | 1.026 | 0.000    | 0.991        | 0.000                | 0.310    | 1.068      | 1.004 | 5.000      | 10.000     | 149.060 | no         | yes      | no       | 1.071 |                    |         |         |                       |
| sent_accuracy      | lr      | 3      | 0.987 | -0.039   | 0.967        | -0.024               | 0.037    | 1.196      | 1.018 | 4.333      | 9.333      | 227.490 | yes        | yes      | yes      | 1.101 | 0.030              | 0.005   | 0.063   | worse                 |
| sent_accuracy      | reg     | 3      | 0.988 | -0.038   | 0.968        | -0.023               | 0.237    | 1.239      | 0.991 | 4.000      | 9.000      | 186.145 | yes        | yes      | no       | 1.078 | 0.007              | -0.013  | 0.026   | no significant change |
| sent_accuracy      | mask    | 3      | 1.001 | -0.025   | 0.966        | -0.025               | 0.102    | 1.153      | 1.041 | 2.667      | 7.667      | 171.584 | yes        | yes      | no       | 1.087 | 0.016              | -0.009  | 0.038   | no significant change |
| sent_accuracy      | noise   | 3      | 1.012 | -0.014   | 0.972        | -0.019               | 0.378    | 1.051      | 0.941 | 6.333      | 11.333     | 217.413 | no         | yes      | no       | 1.083 | 0.012              | -0.004  | 0.028   | no significant change |
| sent_accuracy      | freeze6 | 3      | 1.020 | -0.006   | 0.975        | -0.016               | 0.130    | 1.097      | 1.024 | 3.667      | 8.667      | 161.394 | yes        | yes      | no       | 1.104 | 0.033              | -0.006  | 0.079   | no significant change |
| sent_fluency       | base    | 3      | 0.666 | 0.000    | 0.655        | 0.000                | 0.360    | 1.121      | 1.071 | 4.333      | 9.333      | 144.884 | no         | yes      | no       | 0.870 |                    |         |         |                       |
| sent_fluency       | lr      | 3      | 0.688 | 0.022    | 0.671        | 0.017                | 0.082    | 1.030      | 1.003 | 4.000      | 9.000      | 177.798 | yes        | no       | no       | 0.871 | 0.001              | -0.021  | 0.022   | no significant change |
| sent_fluency       | reg     | 3      | 0.676 | 0.010    | 0.650        | -0.004               | 0.273    | 1.097      | 1.056 | 5.000      | 10.000     | 179.817 | yes        | no       | no       | 0.871 | 0.001              | -0.009  | 0.009   | no significant change |
| sent_fluency       | mask    | 3      | 0.678 | 0.012    | 0.660        | 0.006                | 0.332    | 1.109      | 0.926 | 4.333      | 9.333      | 171.548 | no         | no       | no       | 0.889 | 0.019              | 0.000   | 0.035   | worse                 |
| sent_fluency       | noise   | 3      | 0.664 | -0.002   | 0.653        | -0.001               | 0.368    | 1.111      | 0.968 | 5.000      | 10.000     | 181.312 | no         | yes      | no       | 0.883 | 0.013              | -0.005  | 0.028   | no significant change |
| sent_fluency       | freeze6 | 3      | 0.669 | 0.003    | 0.656        | 0.002                | 0.194    | 1.104      | 1.009 | 4.333      | 9.333      | 156.559 | yes        | yes      | yes      | 0.861 | -0.009             | -0.041  | 0.023   | no significant change |
| sent_prosodic      | base    | 3      | 0.656 | 0.000    | 0.644        | 0.000                | 0.215    | 1.244      | 0.908 | 3.333      | 8.333      | 131.288 | yes        | yes      | yes      | 0.902 |                    |         |         |                       |
| sent_prosodic      | lr      | 3      | 0.677 | 0.021    | 0.672        | 0.028                | 0.096    | 1.185      | 0.987 | 4.333      | 9.333      | 168.271 | yes        | no       | no       | 0.917 | 0.015              | -0.008  | 0.040   | no significant change |
| sent_prosodic      | reg     | 3      | 0.673 | 0.017    | 0.651        | 0.006                | 0.355    | 1.020      | 1.010 | 6.667      | 11.333     | 160.046 | no         | yes      | no       | 0.879 | -0.023             | -0.044  | 0.003   | no significant change |
| sent_prosodic      | mask    | 3      | 0.667 | 0.011    | 0.661        | 0.017                | 0.509    | 1.112      | 1.007 | 6.000      | 11.000     | 176.249 | no         | yes      | no       | 0.888 | -0.014             | -0.038  | 0.009   | no significant change |
| sent_prosodic      | noise   | 3      | 0.658 | 0.002    | 0.653        | 0.009                | 0.271    | 1.193      | 1.049 | 4.000      | 9.000      | 185.616 | yes        | yes      | no       | 0.899 | -0.003             | -0.011  | 0.006   | no significant change |
| sent_prosodic      | freeze6 | 3      | 0.680 | 0.024    | 0.684        | 0.040                | 0.085    | 1.132      | 1.006 | 3.000      | 8.000      | 153.101 | yes        | no       | no       | 0.928 | 0.026              | -0.012  | 0.069   | no significant change |
| sent_prosodic      | freeze9 | 3      | 0.699 | 0.044    | 0.684        | 0.040                | 0.087    | 1.076      | 0.962 | 4.333      | 9.333      | 128.431 | yes        | no       | no       | 0.921 | 0.018              | -0.011  | 0.049   | no significant change |
| sent_total         | base    | 3      | 0.941 | 0.000    | 0.916        | 0.000                | 0.236    | 1.196      | 5.348 | 2.667      | 7.667      | 124.684 | no         | yes      | no       | 1.066 |                    |         |         |                       |
| sent_total         | lr      | 3      | 0.927 | -0.014   | 0.903        | -0.013               | -0.041   | 1.088      | 0.943 | 2.333      | 7.333      | 155.444 | yes        | yes      | yes      | 1.078 | 0.012              | -0.024  | 0.054   | no significant change |
| sent_total         | reg     | 3      | 0.918 | -0.023   | 0.907        | -0.009               | 0.459    | 1.077      | 0.972 | 6.333      | 11.333     | 227.711 | no         | yes      | no       | 1.070 | 0.004              | -0.010  | 0.018   | no significant change |
| sent_total         | mask    | 3      | 0.931 | -0.010   | 0.899        | -0.017               | 0.165    | 1.100      | 0.858 | 2.333      | 7.333      | 149.070 | yes        | yes      | no       | 1.063 | -0.003             | -0.019  | 0.012   | no significant change |
| sent_total         | noise   | 3      | 0.951 | 0.010    | 0.914        | -0.002               | 0.077    | 1.177      | 0.976 | 2.000      | 7.000      | 149.611 | yes        | no       | no       | 1.074 | 0.009              | -0.004  | 0.022   | no significant change |
| sent_total         | freeze6 | 3      | 0.928 | -0.013   | 0.913        | -0.003               | 0.395    | 1.090      | 0.945 | 6.667      | 11.667     | 194.569 | no         | yes      | no       | 1.081 | 0.015              | -0.024  | 0.061   | no significant change |
| word_accuracy      | base    | 3      | 1.620 | 0.000    | 1.582        | 0.000                | 0.077    | 1.144      | 0.853 | 1.667      | 6.667      | 119.190 | yes        | yes      | yes      | 1.598 |                    |         |         |                       |
| word_accuracy      | lr      | 3      | 1.657 | 0.037    | 1.608        | 0.025                | 0.091    | 1.166      | 0.906 | 2.667      | 7.667      | 161.877 | yes        | no       | no       | 1.587 | -0.011             | -0.022  | 0.002   | no significant change |
| word_accuracy      | reg     | 3      | 1.606 | -0.014   | 1.583        | 0.001                | 0.171    | 1.185      | 0.860 | 2.667      | 7.667      | 303.326 | yes        | yes      | no       | 1.592 | -0.006             | -0.014  | 0.000   | no significant change |
| word_accuracy      | mask    | 3      | 1.621 | 0.001    | 1.577        | -0.005               | 0.117    | 1.195      | 0.854 | 2.000      | 7.000      | 145.442 | yes        | yes      | no       | 1.599 | 0.001              | -0.003  | 0.005   | no significant change |
| word_accuracy      | noise   | 3      | 1.631 | 0.011    | 1.599        | 0.017                | 0.032    | 1.162      | 0.874 | 1.333      | 6.333      | 153.217 | yes        | yes      | no       | 1.603 | 0.005              | -0.004  | 0.013   | no significant change |
| word_accuracy      | freeze6 | 3      | 1.649 | 0.029    | 1.609        | 0.027                | 0.066    | 1.209      | 0.883 | 1.667      | 6.667      | 146.058 | yes        | no       | no       | 1.597 | -0.001             | -0.010  | 0.007   | no significant change |
| word_accuracy      | freeze9 | 3      | 1.674 | 0.054    | 1.657        | 0.075                | -0.092   | 1.200      | 0.918 | 0.000      | 5.000      | 111.442 | yes        | no       | no       | 1.646 | 0.048              | 0.032   | 0.064   | worse                 |
| phone_score        | base    | 3      | 0.321 | 0.000    | 0.312        | 0.000                | 0.250    | 1.079      | 0.875 | 4.667      | 9.667      | 136.553 | yes        | yes      | yes      | 0.306 |                    |         |         |                       |
| phone_score        | lr      | 3      | 0.333 | 0.013    | 0.317        | 0.005                | 0.058    | 1.073      | 0.928 | 4.000      | 9.000      | 153.076 | yes        | no       | no       | 0.305 | -0.001             | -0.003  | 0.002   | no significant change |
| phone_score        | reg     | 3      | 0.322 | 0.001    | 0.311        | -0.001               | 0.287    | 1.091      | 0.968 | 6.000      | 11.000     | 355.082 | yes        | yes      | no       | 0.310 | 0.003              | 0.002   | 0.005   | worse                 |
| phone_score        | mask    | 3      | 0.319 | -0.001   | 0.313        | 0.001                | 0.316    | 1.085      | 0.893 | 5.333      | 10.333     | 165.662 | no         | yes      | no       | 0.303 | -0.003             | -0.005  | -0.001  | better                |
| phone_score        | noise   | 3      | 0.319 | -0.001   | 0.312        | 0.000                | 0.328    | 1.052      | 0.920 | 6.333      | 11.333     | 184.021 | no         | yes      | no       | 0.302 | -0.004             | -0.008  | -0.001  | better                |
| phone_score        | freeze6 | 3      | 0.321 | -0.000   | 0.314        | 0.002                | 0.271    | 1.126      | 0.911 | 5.667      | 10.667     | 159.828 | yes        | yes      | no       | 0.305 | -0.001             | -0.004  | 0.002   | no significant change |
| phone_score        | freeze9 | 3      | 0.337 | 0.016    | 0.321        | 0.009                | 0.088    | 1.167      | 0.909 | 4.000      | 9.000      | 119.764 | yes        | no       | no       | 0.306 | 0.000              | -0.002  | 0.003   | no significant change |
| phone_mispron_t1.0 | base    | 3      | 0.689 | 0.000    | 0.697        | 0.000                | 0.269    | 2.623      | 0.932 | 10.667     | 13.333     | 165.526 | no         | yes      | yes      | 0.680 |                    |         |         |                       |
| phone_mispron_t1.0 | lr      | 3      | 0.653 | 0.035    | 0.670        | -0.027               | 0.192    | 1.425      | 1.004 | 8.333      | 11.667     | 178.963 | yes        | no       | no       | 0.696 | 0.016              | 0.004   | 0.029   | better                |
| phone_mispron_t1.0 | reg     | 3      | 0.689 | -0.000   | 0.694        | -0.003               | 0.283    | 2.813      | 0.972 | 13.333     | 15.000     | 251.360 | no         | yes      | no       | 0.667 | -0.013             | -0.019  | -0.009  | worse                 |
| phone_mispron_t1.0 | mask    | 3      | 0.693 | -0.005   | 0.688        | -0.009               | 0.264    | 2.640      | 0.946 | 8.667      | 13.667     | 190.747 | no         | yes      | no       | 0.673 | -0.006             | -0.015  | 0.001   | no significant change |
| phone_mispron_t1.0 | noise   | 3      | 0.698 | -0.009   | 0.702        | 0.005                | 0.267    | 2.499      | 0.966 | 13.333     | 15.000     | 234.615 | no         | yes      | no       | 0.681 | 0.002              | -0.006  | 0.008   | no significant change |
| phone_mispron_t1.0 | freeze6 | 3      | 0.672 | 0.016    | 0.691        | -0.006               | 0.206    | 1.949      | 0.958 | 7.667      | 11.333     | 177.562 | no         | no       | no       | 0.688 | 0.008              | 0.001   | 0.017   | better                |
| l2_phone_error     | base    | 4      | 0.554 | 0.000    | 0.566        | 0.000                | 0.315    | 2.240      | 0.987 | 11.750     | 15.000     | 144.043 | no         | yes      | no       | 0.559 |                    |         |         |                       |
| l2_phone_error     | lr      | 4      | 0.547 | 0.007    | 0.576        | 0.010                | 0.247    | 1.456      | 0.995 | 14.000     | 15.000     | 169.076 | yes        | yes      | yes      | 0.576 | 0.017              | 0.008   | 0.026   | better                |
| l2_phone_error     | reg     | 4      | 0.545 | 0.009    | 0.560        | -0.005               | 0.304    | 2.186      | 0.985 | 13.250     | 15.000     | 172.762 | no         | yes      | no       | 0.554 | -0.005             | -0.010  | 0.001   | no significant change |
| l2_phone_error     | mask    | 4      | 0.557 | -0.003   | 0.570        | 0.004                | 0.304    | 2.134      | 0.986 | 12.500     | 15.000     | 161.890 | no         | yes      | no       | 0.562 | 0.003              | -0.003  | 0.009   | no significant change |
| l2_phone_error     | noise   | 4      | 0.556 | -0.002   | 0.568        | 0.002                | 0.300    | 2.057      | 1.002 | 13.500     | 15.000     | 179.597 | no         | yes      | no       | 0.562 | 0.003              | -0.002  | 0.009   | no significant change |
| l2_phone_error     | freeze6 | 4      | 0.556 | -0.002   | 0.573        | 0.007                | 0.296    | 1.977      | 0.997 | 13.250     | 15.000     | 145.629 | no         | yes      | no       | 0.573 | 0.014              | 0.010   | 0.018   | better                |
| l2_word_error      | base    | 4      | 0.726 | 0.000    | 0.727        | 0.000                | 0.163    | 1.827      | 0.916 | 3.250      | 8.250      | 107.139 | no         | yes      | no       | 0.721 |                    |         |         |                       |
| l2_word_error      | lr      | 4      | 0.723 | 0.003    | 0.725        | -0.002               | 0.125    | 1.143      | 0.947 | 4.250      | 9.250      | 132.896 | yes        | yes      | yes      | 0.719 | -0.002             | -0.006  | 0.003   | no significant change |
| l2_word_error      | reg     | 4      | 0.726 | -0.000   | 0.727        | -0.000               | 0.161    | 1.794      | 0.918 | 3.250      | 8.250      | 129.846 | no         | yes      | no       | 0.722 | 0.001              | -0.001  | 0.004   | no significant change |
| l2_word_error      | mask    | 4      | 0.724 | 0.001    | 0.725        | -0.002               | 0.154    | 1.707      | 0.925 | 3.250      | 8.250      | 129.433 | no         | yes      | no       | 0.719 | -0.001             | -0.004  | 0.001   | no significant change |
| l2_word_error      | noise   | 4      | 0.726 | -0.000   | 0.727        | -0.000               | 0.153    | 1.596      | 0.922 | 3.500      | 8.500      | 137.311 | no         | yes      | no       | 0.721 | 0.001              | -0.002  | 0.004   | no significant change |
| l2_word_error      | freeze6 | 4      | 0.723 | 0.003    | 0.723        | -0.004               | 0.139    | 1.479      | 0.937 | 3.750      | 8.750      | 118.204 | yes        | yes      | no       | 0.716 | -0.004             | -0.008  | -0.001  | worse                 |
| l2_word_error      | freeze9 | 4      | 0.720 | 0.005    | 0.723        | -0.004               | 0.143    | 1.253      | 0.956 | 5.500      | 10.500     | 112.946 | yes        | yes      | no       | 0.712 | -0.008             | -0.014  | -0.002  | worse                 |

## 4. Regression to the mean (test), base vs selected

| task          | system              | slope_pred_on_true | pcc   | low_bin_share | low_bin_mean_true | low_bin_mean_pred |
| ------------- | ------------------- | ------------------ | ----- | ------------- | ----------------- | ----------------- |
| word_accuracy | 5.1 base            | 0.168              | 0.477 | 0.054         | 2.841             | 8.509             |
| word_accuracy | 5.3 selected (base) | 0.168              | 0.477 | 0.054         | 2.841             | 8.509             |
| phone_score   | 5.1 base            | 0.275              | 0.567 | 0.030         | 0.128             | 1.438             |
| phone_score   | 5.3 selected (base) | 0.275              | 0.567 | 0.030         | 0.128             | 1.438             |
