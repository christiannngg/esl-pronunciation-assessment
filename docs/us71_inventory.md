# US 7.1 step 1: results inventory and carry-forward rule (draft)

Status: draft for review. Aggregation only: nothing here retrains a model or edits an earlier result. All numbers are **test** unless stated, copied from the files named in section 1.

## 1. Systems and where each number comes from

| id         | system                                                                                   | source file(s)                                                                    | seeds / folds                         |
| ---------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------- |
| B          | Traditional baseline after US 3.3 fix (best model per task, chosen on validation)        | `results/tables/baseline/us33_before_after.{csv,md}`, `us31_results.csv`          | single fit; speaker bootstrap for CIs |
| F          | Frozen WavLM-base-plus, US 4.3 system per task (also HuBERT / wav2vec2 in the same file) | `results/tables/frozen/us43_headline.csv`                                         | single fit                            |
| FT-base    | Fine-tuned WavLM, US 5.1 `base` recipe                                                   | `results/tables/finetune/us51_headline.csv`                                       | SO762 3 seeds; L2 4 folds x 1 seed    |
| FT-stab    | Fine-tuned, US 5.3 stabilized set (per-task selected variant)                            | `results/tables/finetune/us53_headline.csv` (`test_after`)                        | same as FT-base                       |
| MT-full    | Multi-task, 7 SO762 heads / 2 L2 heads, fixed equal weights                              | `us61_headline_mt_base.csv`, `us61_headline_mt_lr.csv`, `us63_heads.csv` (`full`) | 3 seeds (L2: 4 folds x 3 seeds)       |
| MT-GN15    | Multi-task + GradNorm alpha 1.5 (best-supported candidate, see section 5)                | `us63_heads.csv` (`gradnorm_a15`), `us63_candidates.csv`                          | 3 seeds, SO762 only                   |
| MT-GN05    | Multi-task + GradNorm alpha 0.5                                                          | `us63_heads.csv` (`gradnorm_a05`)                                                 | 3 seeds, SO762 only                   |
| MT-regroup | Two encoders: sentence block + word/phone block                                          | `us63_heads.csv` (`regroup`)                                                      | 3 seeds, SO762 only                   |

Task set for the headline table: 9 tasks (7 SpeechOcean762: sent_accuracy, sent_fluency, sent_prosodic, sent_total, word_accuracy, phone_score, phone_mispron_t1.0; 2 L2-ARCTIC: l2_phone_error, l2_word_error). `phone_mispron_t0.5/t1.5` exist for B only and go in an appendix.

## 2. Test numbers already visible (RMSE, lower is better; F1 = macro-F1, higher is better)

| task               | metric | B     | F (WavLM 4.3) | FT-base | FT-stab | MT-full |
| ------------------ | ------ | ----- | ------------- | ------- | ------- | ------- |
| sent_accuracy      | RMSE   | 1.212 | 1.113         | 1.071   | 1.101   | 1.070   |
| sent_fluency       | RMSE   | 0.957 | 0.915         | 0.870   | 0.861   | 0.887   |
| sent_prosodic      | RMSE   | 0.997 | 0.901         | 0.902   | 0.902   | 0.914   |
| sent_total         | RMSE   | 1.180 | 1.114         | 1.066   | 1.078   | 1.055   |
| word_accuracy      | RMSE   | 1.687 | 1.633         | 1.598   | 1.598   | 1.598   |
| phone_score        | RMSE   | 0.352 | 0.314         | 0.306   | 0.306   | 0.303   |
| phone_mispron_t1.0 | F1     | 0.574 | 0.661         | 0.680   | 0.680   | 0.654   |
| l2_phone_error     | F1     | 0.403 | 0.523         | 0.559   | 0.576   | 0.577   |
| l2_word_error      | F1     | 0.615 | 0.716         | 0.721   | 0.719   | 0.720   |

MT-GN15 / MT-GN05 / MT-regroup per-head values are in `us63_heads.csv` and will be pulled in by the script (only the target head is checked by hand: 0.681 / 0.683 / 0.686).

## 3. Protocol differences to resolve before the numbers share a table

1. **L2 single-task comparator differs between sprints.** US 5.1/5.3 report 1 seed per fold; US 6.1 re-ran the single-task L2 models with 3 seeds. `l2_phone_error` single-task `lr` is 0.5756 in `us53_headline.csv` and 0.5814 in `us61_headline_mt_lr.csv` (`st_stabilized`); the `base` model is 0.5590 vs 0.5642. Proposal: use the 3-seed US 6.1 values for any single- vs multi-task comparison, and the US 5.x values for the baseline -> frozen -> fine-tuned chain, with a footnote. Needs your OK.
2. **Frozen system differs per task and per encoder.** 4.3 picks a different head / layer setup per task. Proposal: show WavLM 4.3 (the encoder that is later fine-tuned) as column F and mention `best_frozen_43_any_encoder` in a footnote (e.g. sent_accuracy 1.112 with wav2vec2; no table-changing difference).
3. **Baseline model differs per task** (RF for most, logreg for several classification tasks). Fine: that is "best baseline per task, chosen on validation", stated in the caption.
4. **Validation set is 19 speakers** and your own memos (M8, W7) show validation does not rank sentence-task variants reliably. This matters for the carry-forward rule (section 4).
5. **CIs:** B/F/FT/MT use paired speaker-level bootstrap on test, but each file computes deltas against its own previous stage. The script should recompute deltas for the pairs the report cares about (B->F, F->FT, FT->MT) from the same bootstrap procedure, or reuse the stored ones where the pair already exists (`d_vs_33`, `d_vs_43`, `d_vs_base`, `d_vs_st`). Which pairs already exist is listed in section 6.
6. **MT only covers SO762 (7 heads) and L2 (2 heads) as two separate models**, not one 9-head joint model (skipped by decision). L2 has no GradNorm run.
7. **Cost columns** (minutes per seed) are not comparable (shared GPU, differing load). Only the number of encoders is a reliable cost statement; the report uses that.

## 4. Carry-forward rule (proposed, to be fixed before the final table is looked at)

Candidates per task: FT-base, FT-stab, MT-full, MT-GN15 (SO762 tasks only). B and F are references, not candidates.

1. **Eligibility:** a candidate must not be significantly worse than FT-base on test for that task. This is reported, not used to select (selection stays validation-side, as in earlier sprints).
2. **Selection:** the best validation metric per task defines the leader. Every candidate within the leader's margin (`max(0.01, seed sd)`, as in 5.3 / 6.3) is a tie.
3. **Tie-break:** fewer encoders first (MT-GN15 and MT-full use 1 encoder for 7 heads, FT uses 7), then fewer extra mechanisms (no GradNorm).
4. **Carry forward at most two systems** into Sprint 8: the per-task single-task reference (FT-base or FT-stab, whichever wins by this rule) and MT-GN15 as the best-supported multi-task candidate with the US 6.3 section 6 disclosure. MT-full stays as the equal-weight reference for ablations.
5. **If validation and test disagree** on which is better for a task, both are shown and the validation-chosen one is carried, with the disagreement noted (M8 / W7 pattern). Test never overrides.
6. **L2:** carry FT-stab (`lr`) and MT-full (`mt_lr`); they are within 0.005 F1 of each other on test.

## 5. Statement for GradNorm (agreed)

Carried forward as the best-supported candidate, not as a rule-adopted fix. The declared US 6.3 rule adopts none of the five candidates (criterion 2 fails for all, including the full model). The post hoc variant of criterion 2 (no head worse than the full model by more than its margin) would adopt `gradnorm_a15`; that amendment was thought of after the results were seen and is not applied. The report states this next to the table.

## 6. Open weaknesses to list in the report (each with its evidence)

W8 low-score compression of word_accuracy / phone_score (still open, left out of Sprint 6 by decision); the 15-epoch cap (W5, `l2_phone_error` best epoch 14 of 15); M1 negative transfer on `phone_mispron_t1.0` (-0.026 F1 in MT-full, removed by GradNorm); M5 no positive transfer anywhere; M7 `l2_phone_error` -0.005 in MT-full; W6 low value of full fine-tuning; 19-speaker validation noise (M8, W7).

## 7. Existing paired comparisons the script can reuse

B vs F: `d_vs_33` in `us43_headline.csv`. F vs FT: `d_vs_43` in `us51_headline.csv`. B vs FT: not stored (compute). FT vs MT: `d_vs_st` / `d_base` in `us61_*` and `us63_heads.csv`. MT-GN15 vs B and vs F: not stored (compute). The script needs to add these two pairs plus B->FT.
