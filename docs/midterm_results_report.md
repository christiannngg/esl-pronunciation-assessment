# Midterm results report (US 7.1)

**Status:** draft for review, 2026-10-29. Aggregation only: nothing was retrained and no earlier result was changed. Every number below comes from the saved Sprint 3-6 tables through `python -m src.evaluation.report_us71` (outputs in `results/tables/midterm/`). Source files and commands are in section 9.

**Scope.** Baseline -> frozen encoder -> fine-tuned -> multi-task, on the 9 tasks that run through the whole pipeline (7 SpeechOcean762, 2 L2-ARCTIC), test set, speaker-independent splits. Sprint 8 (generalisation by L1 and age group, ablations) and the demo app are not covered here.

## 1. What the evidence says

1. **Pretrained representations beat handcrafted features, clearly.** Fine-tuned WavLM is significantly better than the best baseline on **9 of 9 tasks**. Frozen WavLM is significantly better on 8 of 9 (sent_fluency is not significant, p = 0.084). Examples: mispronunciation macro-F1 0.574 -> 0.680, L2 phone error 0.403 -> 0.559, sent_accuracy RMSE 1.212 -> 1.071.
2. **Fine-tuning adds a smaller, real step over frozen.** Better on 6 of 9, no significant change on 3 (sent_prosodic, phone_score, l2_word_error), worse on none. The frozen baseline is the fair one: it already includes the US 4.3 fixes.
3. **Multi-task learning does not improve accuracy.** The full 7-head model matches single-task on 6 of 7 SO762 heads and is **significantly worse on one**, `phone_mispron_t1.0` (-0.026 F1 [-0.045, -0.009]). GradNorm (alpha 1.5) removes that loss: all 7 heads are statistically indistinguishable from single-task, with **one encoder instead of seven**. The defensible claim is parity at lower cost, not a gain.
4. **Carry forward:** the single-task fine-tuned reference, `MT-GN15` as the best-supported multi-task candidate (with the disclosure in section 5), and the full multi-task model as the equal-weight reference. For L2-ARCTIC: the stabilized single-task model and the multi-task model.
5. **Still open:** low-score compression on word accuracy and phone score (W8), the 15-epoch cap on two classification tasks, child/teen mispronunciation, and the fact that validation (19 speakers) does not reliably rank candidates (section 7).

## 2. Systems compared

| id      | system                                                                                                                                        | source | seeds                              |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---------------------------------- |
| B       | best traditional baseline per task after the US 3.3 fix (RF / logistic regression on MFCC + prosody features), chosen on validation           | US 3.3 | single fit                         |
| F       | frozen WavLM-base-plus, US 4.3 system per task (layer-weighted sum, regularised MLP heads, out-of-fold decision weights for mispronunciation) | US 4.3 | single fit                         |
| FT-base | fine-tuned WavLM, one model per task, US 5.1 recipe                                                                                           | US 5.1 | SO762 3 seeds; L2 4 folds x 1 seed |
| FT-stab | fine-tuned, per-task variant selected by the US 5.3 rule (`lr`, `freeze6` or base)                                                            | US 5.3 | as FT-base                         |
| MT-full | one shared encoder, SO762: 7 heads; L2: 2 heads; fixed equal loss weights                                                                     | US 6.1 | 3 seeds (L2: 4 folds x 3)          |
| MT-GN15 | MT-full with GradNorm loss balancing, alpha 1.5 (SO762 only)                                                                                  | US 6.3 | 3 seeds                            |

Also run on SO762 and shown only where relevant: `MT-GN05` (GradNorm alpha 0.5) and `MT-regroup` (a sentence block and a word/phone block, two encoders). Uncertainty-weighted loss was tested in US 6.3 and did not help; it is not repeated here.

## 3. Results

### 3.1 Test results by system (RMSE lower is better; macro-F1 higher is better)

| task               | metric   | B     | F     | FT-base | FT-stab | MT-full | MT-GN15 |
| ------------------ | -------- | ----- | ----- | ------- | ------- | ------- | ------- |
| sent_accuracy      | RMSE     | 1.212 | 1.113 | 1.071   | 1.101   | 1.070   | 1.063   |
| sent_fluency       | RMSE     | 0.957 | 0.915 | 0.870   | 0.861   | 0.887   | 0.880   |
| sent_prosodic      | RMSE     | 0.997 | 0.901 | 0.902   | 0.902   | 0.914   | 0.910   |
| sent_total         | RMSE     | 1.180 | 1.114 | 1.066   | 1.078   | 1.055   | 1.061   |
| word_accuracy      | RMSE     | 1.687 | 1.633 | 1.598   | 1.598   | 1.598   | 1.607   |
| phone_score        | RMSE     | 0.352 | 0.314 | 0.306   | 0.306   | 0.303   | 0.305   |
| phone_mispron_t1.0 | macro-F1 | 0.574 | 0.661 | 0.680   | 0.680   | 0.654   | 0.681   |
| l2_phone_error     | macro-F1 | 0.403 | 0.523 | 0.559   | 0.576   | 0.577   | n/a     |
| l2_word_error      | macro-F1 | 0.615 | 0.716 | 0.721   | 0.719   | 0.720   | n/a     |

Notes. 95% confidence intervals for the fine-tuned and full multi-task systems are in `us71_report.md` (for example sent_accuracy FT-base 1.071 [0.955, 1.174]). The intervals are wide relative to the differences between FT and MT rows, which is why the paired tests below are the right tool, not the table. FT-base and FT-stab are the same model on 4 of 9 tasks (base was kept). The L2 multi-task row uses the `lr` recipe with 3 seeds, FT-base and FT-stab use 1 seed per fold (see section 7).

### 3.2 Paired comparisons (speaker-level paired bootstrap; delta = metric(b) - metric(a))

(+) significantly better, (=) not significant (95% CI contains 0), (-) significantly worse. For RMSE, negative is better; for F1, positive is better.

| task               | B -> F     | F -> FT-base | B -> FT-base | FT-base -> MT-full | FT-base -> MT-GN15 |
| ------------------ | ---------- | ------------ | ------------ | ------------------ | ------------------ |
| sent_accuracy      | -0.099 (+) | -0.042 (+)   | -0.142 (+)   | -0.001 (=)         | -0.008 (=)         |
| sent_fluency       | -0.042 (=) | -0.045 (+)   | -0.087 (+)   | +0.017 (=)         | +0.010 (=)         |
| sent_prosodic      | -0.096 (+) | +0.001 (=)   | -0.095 (+)   | +0.011 (=)         | +0.008 (=)         |
| sent_total         | -0.066 (+) | -0.048 (+)   | -0.114 (+)   | -0.011 (=)         | -0.005 (=)         |
| word_accuracy      | -0.054 (+) | -0.035 (+)   | -0.089 (+)   | -0.001 (=)         | +0.009 (=)         |
| phone_score        | -0.038 (+) | -0.007 (=)   | -0.046 (+)   | -0.003 (=)         | -0.001 (=)         |
| phone_mispron_t1.0 | +0.088 (+) | +0.018 (+)   | +0.106 (+)   | **-0.026 (-)**     | +0.001 (=)         |
| l2_phone_error     | +0.120 (+) | +0.036 (+)   | +0.156 (+)   | +0.013 (+) \*      | n/a                |
| l2_word_error      | +0.100 (+) | +0.005 (=)   | +0.105 (+)   | -0.002 (=)         | n/a                |

\* Against the 5.1 `base` recipe with the 3-seed comparator. The gain is the `lr` recipe, not sharing: against the stabilized single-task model the multi-task L2 model is -0.005 [-0.007, -0.002] (worse, practically negligible).

Other pairs, from the same procedure:

- **F -> MT-GN15** (SO762): better on 6 of 7 heads (for example sent_accuracy -0.050 [-0.084, -0.020], phone_mispron +0.019 [+0.007, +0.031]); sent_prosodic is not significant (+0.009 [-0.007, +0.025]).
- **B -> MT-GN15**: significantly better on all 7 heads.
- **MT-full -> MT-GN15**: only `phone_mispron_t1.0` changes materially (+0.027 F1 [+0.012, +0.040], better); `phone_score` is 0.002 worse (CI touches 0). The other five heads are not significant.
- **FT-base -> MT-regroup** (two encoders): worse on 4 heads (sent_fluency +0.047, sent_prosodic +0.037, sent_total +0.021, phone_score +0.004) and no better on the target head. Regrouping trades one weakness for four.
- **FT-base -> FT-stab** (US 5.3): sent_accuracy is significantly worse (+0.030 [+0.005, +0.063]) and l2_phone_error better (+0.017 [+0.008, +0.026]); the rest are not significant.

The three pairs B -> FT-base, B -> MT-GN15 and F -> MT-GN15 were not stored by any sprint report. They were computed from the saved test predictions of the same systems; the metric recomputed from each prediction file equals the number in the sprint table to four decimals for all 34 systems-by-task files, and the test rows are identical across systems.

## 4. Carry-forward selection

The rule was fixed before this table was generated (`docs/us71_inventory.md`, section 4): candidates are FT-base, FT-stab, MT-full and MT-GN15 (SO762 only); a candidate significantly worse than FT-base on test is ineligible; the validation leader and everything within `max(0.01, seed sd)` of it are tied; ties go to the fewest encoders, then the fewest extra mechanisms; test never overrides validation.

| task               | validation leader | tied with the leader      | test leader | note                                                                  |
| ------------------ | ----------------- | ------------------------- | ----------- | --------------------------------------------------------------------- |
| sent_accuracy      | MT-GN15           | MT-GN15 only              | MT-GN15     | FT-stab ineligible (worse on test)                                    |
| sent_fluency       | FT-base           | FT-base, FT-stab          | FT-stab     | test leader is inside the tie                                         |
| sent_prosodic      | FT-base           | FT-base, FT-stab          | FT-base     |                                                                       |
| sent_total         | MT-GN15           | MT-GN15 only              | MT-full     | **validation and test disagree**                                      |
| word_accuracy      | FT-base           | FT-base, FT-stab          | MT-full     | **validation and test disagree** (MT-full 1.598 equals FT-base 1.598) |
| phone_score        | MT-full           | all four                  | MT-full     | all within the margin                                                 |
| phone_mispron_t1.0 | MT-GN15           | FT-base, FT-stab, MT-GN15 | MT-GN15     | MT-full ineligible (significantly worse on test)                      |
| l2_phone_error     | MT-full           | FT-stab, MT-full          | MT-full     |                                                                       |
| l2_word_error      | MT-full           | FT-base, FT-stab, MT-full | FT-base     | test leader is inside the tie                                         |

**Carried into Sprint 8:** the single-task reference per task (FT-base, or FT-stab for sent_total and l2_phone_error by the rule), `MT-GN15` on the 7 SO762 heads, and `MT-full` as the equal-weight reference for ablations. For L2-ARCTIC: FT-stab and MT-full (within 0.005 F1 on test). The two disagreements are reported, not resolved by looking at test.

## 5. Disclosure about GradNorm

Under the adoption rule written before any US 6.3 run, **no candidate is adopted**: criterion 2 ("no other head worse than single-task on validation") fails for every candidate, the full model included, on the same three heads that are unchanged on test. A variant of that criterion measured against the full model rather than against single-task would adopt `gradnorm_a15`, but it was thought of after the results were seen and is not applied. `MT-GN15` is therefore carried forward as the **best-supported candidate**, not as a rule-adopted fix. On test it is the only scheme that brings `phone_mispron_t1.0` back to the single-task level (0.681 against 0.680) without a significant loss on any other head. Effective gradient shares on the last shared layer go from 0.32x to 1.78x of balanced (full model) to 0.69x to 1.61x (alpha 1.5). Five candidates were screened on 19 validation speakers, test intervals are not corrected for the 35 head-candidate comparisons, and alpha took only two values. Details: `docs/us63_fix_summary.md`, section 6.

## 6. Weakness -> fix ledger so far

| sprint | weakness found                                                                                   | fix                                                    | outcome                                                                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3      | Class imbalance: baseline predicts no mispronunciations or addition errors                       | class weights and validation-tuned decision thresholds | fixed for detection: mispronunciation F1 +0.07, L2 phone error +0.15 (US 3.3, p < 0.001); precision stays low                                                              |
| 3      | Sentence scores missing pause, rate and pitch cues                                               | pitch / energy / pause features                        | fixed: all four sentence scores improved significantly (RMSE -0.09 to -0.12)                                                                                               |
| 3      | Regression to the mean on word and phone scores                                                  | low-score resampling was tried                         | not selected on validation: phone_score unchanged, word_accuracy gained only 0.017 RMSE (from the prosody features); carried to later sprints as W8                        |
| 4      | wav2vec2 last layer unusable (at or below the MFCC baseline on the L2 tasks)                     | per-task layer re-selection                            | fixed: L2 phone +0.090, L2 word +0.072 for that encoder                                                                                                                    |
| 4      | Sentence heads overfit; noisy single-layer choice                                                | learned layer-weighted sum, regularisation             | helped regression tasks; the learned weights stayed near uniform, so it acts as a layer average                                                                            |
| 4      | Mispronunciation thresholds tuned on adult-dominated data                                        | out-of-fold decision weights                           | +0.03 F1; child/teen recall still 0.10-0.21 (a ranking limit)                                                                                                              |
| 5      | Overfitting hidden by early stopping; over-confident classification heads; one unstable seed     | lower encoder lr, dropout, masking, partial freezing   | train-validation gaps closed and the instability disappeared; **test did not improve** for the sentence tasks (sent_accuracy worse); over-confidence fixed by the lower lr |
| 5      | Low-score compression got worse after fine-tuning (slope 0.200 -> 0.168, 0.342 -> 0.275)         | none attempted                                         | **open (W8)**                                                                                                                                                              |
| 6      | Negative transfer onto `phone_mispron_t1.0` (-0.026 F1); sentence heads hold 73% of the gradient | GradNorm; regrouping; uncertainty weighting            | GradNorm removes it with no loss elsewhere; regrouping removes it but costs four heads; uncertainty weighting does not help                                                |

This ledger is the skeleton of the "found a weakness, fixed it" section the reviewer asked for. The honest pattern in it: fixes worked where the cause was a decision rule, a layer choice or a loss balance, and did not work where the cause is the data (compressed low scores, validation noise).

## 7. Limits and caveats

- **Validation is 19 speakers** for SO762 and 4 speakers per L1 group for L2-ARCTIC. It does not rank candidates reliably: US 4.3 and US 5.3 each had cases where the validation-selected system was not the best on test, and in section 4 two tasks show validation and test disagreeing.
- **No correction for multiple comparisons.** Each sprint reports many intervals; verdicts with p between about 0.02 and 0.05 (for example F -> FT-base on word_accuracy, p = 0.012; F -> MT-GN15 on word_accuracy, p = 0.024) should be read as suggestive.
- **Seeds are unequal across systems.** SO762 uses 3 seeds for FT and MT. The L2 FT models use 1 seed per fold; the L2 multi-task comparison uses 3-seed single-task comparators computed in Sprint 6. B and F are single fits, so seed variance is not included in their intervals.
- **Selection effects.** Each system's configuration was chosen on validation, and five GradNorm / regroup / uncertainty candidates were screened on the same 19 speakers (winner's curse).
- **Frozen baseline is validation-selected per task.** The best frozen encoder on test is sometimes a different one (for example sent_fluency 0.892 with wav2vec2 against 0.915 for WavLM, and the B -> F comparison for WavLM is not significant there). F is reported for WavLM because that is the encoder that is fine-tuned.
- **L2-ARCTIC scope.** 95-98% of held-out utterance texts also occur in training, so the L2 numbers measure generalisation to new speakers, not new sentences. Error-type classification is given an annotated unit (oracle segmentation).
- **Multi-task scope.** Two multi-task models were trained (7 SO762 heads and 2 L2 heads). The optional 9-head joint model was skipped by decision, so cross-dataset sharing is untested.
- **Cost.** The only reliable cost statement is the number of fine-tuned encoders (SO762: 1 for MT, 7 for single-task). Wall-clock times were measured on a shared GPU and are not comparable.

## 8. What this means for the rest of the project (input to US 7.2)

- The pipeline question is answered: pretrained encoders clearly beat handcrafted features, so Sprint 8 and the demo can stand on the fine-tuned model. No result here argues for cutting scope below the 9 tasks.
- The multi-task question is also answered with the current data: no accuracy gain, parity with one encoder under GradNorm. The projected final claim is "multi-task matches single-task at a fraction of the model cost, and unbalanced joint training causes measurable negative transfer that GradNorm repairs". Dropping heads is not needed to make that claim, but the regroup result shows that splitting heads by level is not a free fix.
- Candidate Sprint 8 work that this report points to: the per-L1 and per-age-group breakdown (the child/teen weakness is the most likely subgroup finding), the 25-epoch `lr` follow-up for the two classification tasks that hit the cap, and a loss that re-weights low scores (W8). Whether all of these fit is the scope decision in US 7.2.

## 9. Provenance and reproduction

- Generated tables: `results/tables/midterm/us71_systems.csv`, `us71_carry_forward.csv`, `us71_pairs.csv`, `us71_pairs_computed.csv`, `us71_report.md`; figure `results/figures/midterm/us71_ladder.png`.
- Commands: `python -m src.evaluation.report_us71_pairs export --systems B F` (Mac) and `... --systems FT-base MT-GN15` (lab), copy `results/analysis/us71/*.npz` to one checkout, then `python -m src.evaluation.report_us71_pairs pairs` and `python -m src.evaluation.report_us71`. Tests: `python -m pytest tests/test_report_us71.py tests/test_report_us71_pairs.py` (12 pass).
- Inputs read: `results/tables/baseline/us33_before_after.csv`, `frozen/us43_headline.csv`, `finetune/us51_headline.csv`, `finetune/us53_headline.csv`, `multitask/us61_headline_mt_{base,lr}.csv`, `multitask/us63_heads.csv`.
- Earlier write-ups this report draws on: `weakness_memo_s3.md` to `weakness_memo_s6.md`, `us33_fix_summary.md`, `us43_fix_summary.md`, `us53_fix_summary.md`, `us63_fix_summary.md`, `docs/us71_inventory.md`.
