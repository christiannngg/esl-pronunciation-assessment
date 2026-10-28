# US 6.3 - fixing negative transfer and task imbalance in the SO762 multi-task model

Recipe `base`, 3 seeds, SO762 (19 validation speakers). Every head is compared with the single-task model of the SAME recipe (US 5.1 `base`) and with the full 7-head multi-task model of US 6.1. Delta = candidate minus comparator on test (RMSE: negative is better; macro-F1: positive is better), paired speaker bootstrap, 95 % CI; `worse` / `better` only when the interval excludes 0. **Adoption uses validation only; test numbers never select.**

## 1. Adoption on validation (rule fixed before any US 6.3 job was run)

Declared: A candidate PASSES on validation if (1) the target head's validation F1 is not worse than the single-task model's by more than its margin, AND (2) no other head is worse than its single-task model by more than its margin (memo, section 7), AND (3) [secondary rule, added at the start of US 6.3 because (1) is already met by the full model on validation: its target head is 0.007 worse against a margin of 0.010] the target head's validation F1 is at least that of the full 7-head model. Among passing candidates the one with the lowest mean over heads of (candidate validation error / single-task validation error) is adopted; candidates within tie_margin of it are tied and the one with the fewest encoders wins. If none passes: report the full model and the per-level split side by side and state that sharing the encoder across levels does not help in this data. Test numbers are reported for every candidate and never select. Every candidate is reported, not only the adopted one (winner's curse: five candidates are screened on 19 validation speakers).

| candidate | encoders | all heads | (1) phone_mispron_t1.0 not worse (val) | (2) no other head worse (val) | (3) phone_mispron_t1.0 >= full (val) | mean rel. val. error | PASSES | why not |
|---|---|---|---|---|---|---|---|---|
| full | 1 | yes | yes | no | yes | 1.015 | no | other head(s) worse than single-task on validation: sent_fluency, sent_prosodic, word_accuracy |
| regroup | 2 | yes | no | no | no | 1.020 | no | target head worse than single-task on validation; other head(s) worse than single-task on validation: sent_fluency, sent_prosodic; target head below the full model on validation |
| uncertainty | 1 | yes | no | no | no | 1.020 | no | target head worse than single-task on validation; other head(s) worse than single-task on validation: sent_fluency, sent_prosodic, word_accuracy, phone_score; target head below the full model on validation |
| gradnorm_a05 | 1 | yes | yes | no | yes | 1.009 | no | other head(s) worse than single-task on validation: sent_fluency, sent_prosodic, word_accuracy |
| gradnorm_a15 | 1 | yes | yes | no | yes | 1.005 | no | other head(s) worse than single-task on validation: sent_fluency, sent_prosodic, word_accuracy |

**Adopted: none.** no candidate meets all three criteria on validation: report the full model and the per-level split side by side; sharing the encoder across levels does not help in this data (declared rule)

## 2. Test: the target head and everyone else

| candidate | phone_mispron_t1.0 test F1 | delta vs single-task | verdict vs single-task | verdict vs full MT | heads better than single-task | heads worse than single-task |
|---|---|---|---|---|---|---|
| full | 0.654 | -0.026 | worse | - | 0 | 1 |
| regroup | 0.686 | 0.007 | no significant change | better | 0 | 4 |
| uncertainty | 0.662 | -0.018 | worse | better | 0 | 1 |
| gradnorm_a05 | 0.683 | 0.003 | no significant change | better | 1 | 0 |
| gradnorm_a15 | 0.681 | 0.001 | no significant change | better | 0 | 0 |

### Per head (test) and validation flag

| candidate | head | single-task | full MT | candidate | delta vs ST | lo | hi | verdict vs ST | delta vs full | verdict vs full | reading | val worse by | margin | val flag | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | sent_accuracy | 1.071 | 1.070 | 1.070 | -0.001 | -0.020 | 0.017 | no significant change | - | - | - | -0.036 | 0.010 | better | 3 |
| full | sent_fluency | 0.870 | 0.887 | 0.887 | 0.017 | -0.002 | 0.035 | no significant change | - | - | - | 0.025 | 0.010 | worse | 3 |
| full | sent_prosodic | 0.902 | 0.914 | 0.914 | 0.011 | -0.009 | 0.031 | no significant change | - | - | - | 0.036 | 0.018 | worse | 3 |
| full | sent_total | 1.066 | 1.055 | 1.055 | -0.011 | -0.029 | 0.006 | no significant change | - | - | - | -0.041 | 0.010 | better | 3 |
| full | word_accuracy | 1.598 | 1.598 | 1.598 | -0.001 | -0.034 | 0.033 | no significant change | - | - | - | 0.066 | 0.022 | worse | 3 |
| full | phone_score | 0.306 | 0.303 | 0.303 | -0.003 | -0.009 | 0.002 | no significant change | - | - | - | 0.009 | 0.010 | within margin | 3 |
| full | phone_mispron_t1.0 | 0.680 | 0.654 | 0.654 | -0.026 | -0.045 | -0.009 | worse | - | - | - | 0.007 | 0.010 | within margin | 3 |
| regroup | sent_accuracy | 1.071 | 1.070 | 1.084 | 0.014 | -0.013 | 0.038 | no significant change | 0.014 | no significant change | no effect | -0.025 | 0.010 | better | 3 |
| regroup | sent_fluency | 0.870 | 0.887 | 0.917 | 0.047 | 0.020 | 0.073 | worse | 0.030 | worse | hurts | 0.037 | 0.010 | worse | 3 |
| regroup | sent_prosodic | 0.902 | 0.914 | 0.939 | 0.037 | 0.023 | 0.051 | worse | 0.026 | worse | hurts | 0.021 | 0.018 | worse | 3 |
| regroup | sent_total | 1.066 | 1.055 | 1.087 | 0.021 | 0.000 | 0.041 | worse | 0.032 | worse | hurts | -0.020 | 0.010 | better | 3 |
| regroup | word_accuracy | 1.598 | 1.598 | 1.581 | -0.017 | -0.035 | 0.000 | no significant change | -0.016 | no significant change | no effect | -0.013 | 0.022 | within margin | 3 |
| regroup | phone_score | 0.306 | 0.303 | 0.310 | 0.004 | 0.002 | 0.006 | worse | 0.007 | worse | hurts | 0.004 | 0.010 | within margin | 3 |
| regroup | phone_mispron_t1.0 | 0.680 | 0.654 | 0.686 | 0.007 | -0.009 | 0.025 | no significant change | 0.032 | better | recovers | 0.028 | 0.010 | worse | 3 |
| uncertainty | sent_accuracy | 1.071 | 1.070 | 1.063 | -0.008 | -0.028 | 0.013 | no significant change | -0.007 | no significant change | no effect | -0.032 | 0.010 | better | 3 |
| uncertainty | sent_fluency | 0.870 | 0.887 | 0.885 | 0.014 | -0.007 | 0.036 | no significant change | -0.002 | no significant change | no effect | 0.019 | 0.010 | worse | 3 |
| uncertainty | sent_prosodic | 0.902 | 0.914 | 0.907 | 0.004 | -0.013 | 0.025 | no significant change | -0.007 | no significant change | no effect | 0.021 | 0.018 | worse | 3 |
| uncertainty | sent_total | 1.066 | 1.055 | 1.053 | -0.013 | -0.035 | 0.010 | no significant change | -0.002 | no significant change | no effect | -0.036 | 0.010 | better | 3 |
| uncertainty | word_accuracy | 1.598 | 1.598 | 1.601 | 0.003 | -0.042 | 0.048 | no significant change | 0.004 | no significant change | no effect | 0.103 | 0.022 | worse | 3 |
| uncertainty | phone_score | 0.306 | 0.303 | 0.302 | -0.004 | -0.011 | 0.002 | no significant change | -0.001 | no significant change | no effect | 0.017 | 0.010 | worse | 3 |
| uncertainty | phone_mispron_t1.0 | 0.680 | 0.654 | 0.662 | -0.018 | -0.035 | -0.005 | worse | 0.007 | better | partly | 0.010 | 0.010 | worse | 3 |
| gradnorm_a05 | sent_accuracy | 1.071 | 1.070 | 1.056 | -0.015 | -0.036 | 0.005 | no significant change | -0.014 | no significant change | no effect | -0.042 | 0.010 | better | 3 |
| gradnorm_a05 | sent_fluency | 0.870 | 0.887 | 0.882 | 0.011 | -0.004 | 0.027 | no significant change | -0.005 | no significant change | no effect | 0.037 | 0.010 | worse | 3 |
| gradnorm_a05 | sent_prosodic | 0.902 | 0.914 | 0.901 | -0.001 | -0.022 | 0.023 | no significant change | -0.013 | no significant change | no effect | 0.039 | 0.018 | worse | 3 |
| gradnorm_a05 | sent_total | 1.066 | 1.055 | 1.044 | -0.022 | -0.044 | -0.002 | better | -0.011 | no significant change | no effect | -0.045 | 0.010 | better | 3 |
| gradnorm_a05 | word_accuracy | 1.598 | 1.598 | 1.590 | -0.008 | -0.041 | 0.025 | no significant change | -0.008 | no significant change | no effect | 0.043 | 0.022 | worse | 3 |
| gradnorm_a05 | phone_score | 0.306 | 0.303 | 0.304 | -0.002 | -0.008 | 0.002 | no significant change | 0.001 | no significant change | no effect | 0.007 | 0.010 | within margin | 3 |
| gradnorm_a05 | phone_mispron_t1.0 | 0.680 | 0.654 | 0.683 | 0.003 | -0.008 | 0.014 | no significant change | 0.029 | better | recovers | -0.004 | 0.010 | within margin | 3 |
| gradnorm_a15 | sent_accuracy | 1.071 | 1.070 | 1.063 | -0.008 | -0.025 | 0.012 | no significant change | -0.007 | no significant change | no effect | -0.059 | 0.010 | better | 3 |
| gradnorm_a15 | sent_fluency | 0.870 | 0.887 | 0.880 | 0.010 | -0.008 | 0.028 | no significant change | -0.007 | no significant change | no effect | 0.026 | 0.010 | worse | 3 |
| gradnorm_a15 | sent_prosodic | 0.902 | 0.914 | 0.910 | 0.008 | -0.012 | 0.027 | no significant change | -0.004 | no significant change | no effect | 0.028 | 0.018 | worse | 3 |
| gradnorm_a15 | sent_total | 1.066 | 1.055 | 1.061 | -0.005 | -0.024 | 0.016 | no significant change | 0.006 | no significant change | no effect | -0.061 | 0.010 | better | 3 |
| gradnorm_a15 | word_accuracy | 1.598 | 1.598 | 1.607 | 0.009 | -0.025 | 0.042 | no significant change | 0.009 | no significant change | no effect | 0.058 | 0.022 | worse | 3 |
| gradnorm_a15 | phone_score | 0.306 | 0.303 | 0.305 | -0.001 | -0.006 | 0.003 | no significant change | 0.002 | worse | hurts | 0.008 | 0.010 | within margin | 3 |
| gradnorm_a15 | phone_mispron_t1.0 | 0.680 | 0.654 | 0.681 | 0.001 | -0.012 | 0.012 | no significant change | 0.027 | better | recovers | 0.005 | 0.010 | within margin | 3 |

`reading` (only meaningful for the target head): recovers = no longer worse than single-task AND better than the full model; partly = only the second; hurts / no effect otherwise.

## 3. What the schemes did to the loss weights and the gradient balance

Weight = applied loss weight (sum 1; `fixed` = 1/K); share = share of the summed gradient norms on the last shared transformer layer. 'x balanced' = value x K.

| candidate | head | weight (mean) x balanced | weight (last epoch) | gradient share x balanced |
|---|---|---|---|---|
| full | sent_accuracy | 1.000 | 0.143 | 0.576 |
| full | sent_fluency | 1.000 | 0.143 | 1.775 |
| full | sent_prosodic | 1.000 | 0.143 | 1.709 |
| full | sent_total | 1.000 | 0.143 | 1.042 |
| full | word_accuracy | 1.000 | 0.143 | 0.669 |
| full | phone_score | 1.000 | 0.143 | 0.912 |
| full | phone_mispron_t1.0 | 1.000 | 0.143 | 0.317 |
| regroup | sent_accuracy | 1.000 | 0.250 | 0.563 |
| regroup | sent_fluency | 1.000 | 0.250 | 1.215 |
| regroup | sent_prosodic | 1.000 | 0.250 | 1.380 |
| regroup | sent_total | 1.000 | 0.250 | 0.842 |
| regroup | word_accuracy | 1.000 | 0.333 | 1.052 |
| regroup | phone_score | 1.000 | 0.333 | 1.208 |
| regroup | phone_mispron_t1.0 | 1.000 | 0.333 | 0.740 |
| uncertainty | sent_accuracy | 0.891 | 0.128 | 0.475 |
| uncertainty | sent_fluency | 0.886 | 0.127 | 2.025 |
| uncertainty | sent_prosodic | 0.897 | 0.129 | 2.225 |
| uncertainty | sent_total | 0.907 | 0.131 | 0.592 |
| uncertainty | word_accuracy | 0.803 | 0.110 | 0.581 |
| uncertainty | phone_score | 0.794 | 0.108 | 0.830 |
| uncertainty | phone_mispron_t1.0 | 1.823 | 0.267 | 0.271 |
| gradnorm_a05 | sent_accuracy | 1.197 | 0.233 | 0.573 |
| gradnorm_a05 | sent_fluency | 0.285 | 0.027 | 1.818 |
| gradnorm_a05 | sent_prosodic | 0.227 | 0.019 | 1.648 |
| gradnorm_a05 | sent_total | 0.566 | 0.068 | 1.041 |
| gradnorm_a05 | word_accuracy | 0.927 | 0.113 | 0.706 |
| gradnorm_a05 | phone_score | 0.708 | 0.058 | 0.925 |
| gradnorm_a05 | phone_mispron_t1.0 | 3.091 | 0.483 | 0.289 |
| gradnorm_a15 | sent_accuracy | 1.165 | 0.269 | 0.485 |
| gradnorm_a15 | sent_fluency | 0.304 | 0.048 | 1.872 |
| gradnorm_a15 | sent_prosodic | 0.238 | 0.041 | 1.786 |
| gradnorm_a15 | sent_total | 0.509 | 0.089 | 1.049 |
| gradnorm_a15 | word_accuracy | 1.519 | 0.189 | 0.644 |
| gradnorm_a15 | phone_score | 1.226 | 0.145 | 0.891 |
| gradnorm_a15 | phone_mispron_t1.0 | 2.040 | 0.219 | 0.274 |

## 4. One shared checkpoint (trigger for per-head checkpointing, plan item d)

Validation regret of the shared epoch vs the head's own best epoch (optimistic bound). Threshold 2 % (`compare62.stopping_regret_rel`): heads at or above it keep the regret that (d) would address.

| candidate | head | shared epoch | own best epoch | regret (rel.) | >= 2 % |
|---|---|---|---|---|---|
| full | sent_accuracy | 7.667 | 4.000 | 0.013 | no |
| full | sent_fluency | 7.667 | 6.667 | 0.015 | no |
| full | sent_prosodic | 7.667 | 8.000 | 0.043 | yes |
| full | sent_total | 7.667 | 6.333 | 0.007 | no |
| full | word_accuracy | 7.667 | 3.000 | 0.033 | yes |
| full | phone_score | 7.667 | 5.000 | 0.013 | no |
| full | phone_mispron_t1.0 | 7.667 | 11.000 | 0.028 | yes |
| regroup | sent_accuracy | 6.000 | 3.667 | 0.010 | no |
| regroup | sent_fluency | 6.000 | 7.667 | 0.028 | yes |
| regroup | sent_prosodic | 6.000 | 7.333 | 0.014 | no |
| regroup | sent_total | 6.000 | 3.333 | 0.001 | no |
| regroup | word_accuracy | 4.333 | 2.667 | 0.006 | no |
| regroup | phone_score | 4.333 | 4.333 | 0.000 | no |
| regroup | phone_mispron_t1.0 | 4.333 | 7.000 | 0.042 | yes |
| uncertainty | sent_accuracy | 8.667 | 6.333 | 0.012 | no |
| uncertainty | sent_fluency | 8.667 | 5.667 | 0.004 | no |
| uncertainty | sent_prosodic | 8.667 | 7.667 | 0.010 | no |
| uncertainty | sent_total | 8.667 | 7.000 | 0.003 | no |
| uncertainty | word_accuracy | 8.667 | 4.000 | 0.044 | yes |
| uncertainty | phone_score | 8.667 | 4.333 | 0.031 | yes |
| uncertainty | phone_mispron_t1.0 | 8.667 | 10.000 | 0.037 | yes |
| gradnorm_a05 | sent_accuracy | 10.333 | 8.667 | 0.017 | no |
| gradnorm_a05 | sent_fluency | 10.333 | 9.667 | 0.008 | no |
| gradnorm_a05 | sent_prosodic | 10.333 | 9.333 | 0.012 | no |
| gradnorm_a05 | sent_total | 10.333 | 10.667 | 0.012 | no |
| gradnorm_a05 | word_accuracy | 10.333 | 4.000 | 0.022 | yes |
| gradnorm_a05 | phone_score | 10.333 | 5.333 | 0.015 | no |
| gradnorm_a05 | phone_mispron_t1.0 | 10.333 | 10.333 | 0.000 | no |
| gradnorm_a15 | sent_accuracy | 10.333 | 8.667 | 0.003 | no |
| gradnorm_a15 | sent_fluency | 10.333 | 9.667 | 0.001 | no |
| gradnorm_a15 | sent_prosodic | 10.333 | 10.333 | 0.000 | no |
| gradnorm_a15 | sent_total | 10.333 | 9.333 | 0.004 | no |
| gradnorm_a15 | word_accuracy | 10.333 | 3.333 | 0.020 | no |
| gradnorm_a15 | phone_score | 10.333 | 7.333 | 0.013 | no |
| gradnorm_a15 | phone_mispron_t1.0 | 10.333 | 11.667 | 0.013 | no |

## 5. Cost (indicative: wall clock, possibly shared GPU)

| candidate | encoders | minutes per seed (sum over parts) |
|---|---|---|
| full | 1 | 23.2 |
| regroup | 2 | 18.9 |
| uncertainty | 1 | 8.4 |
| gradnorm_a05 | 1 | 17.6 |
| gradnorm_a15 | 1 | 16.5 |

## Not addressed in US 6.3

- **W8 (low-score compression of word_accuracy / phone_score, Sprint 5): still open.** No loss that re-weights low scores was tried; slopes are unchanged from US 5.2 (0.168 / 0.275).
- **M7 (l2_phone_error -0.005):** negligible, no fix planned; the L2 group was not rerun.

## Caveats

- The idea of regrouping comes from the US 6.2 ablations, which were read on test. Adoption here is on validation, and regrouping gets no credit for having been seen first.
- Five candidates are screened on 19 validation speakers; the adopted one is partly the luckiest on validation (winner's curse). All candidates are reported.
- GradNorm is the first-order form (targets held constant) and uses the last shared transformer layer as the gradient probe; weights updated every step after the head-only epoch.
- `uncertainty` starts from weights c_k (1/2 for a regression head, 1 for a classification head, then normalised): the target head begins with twice the weight of each regression head, i.e. part of that candidate is a fixed x2 re-weighting of the target head (the US 6.2 ablation `w_mispron3`, x3, did not help).
- Seeds are matched across a candidate, the full model and the single-task model for every head (column `seeds`).

