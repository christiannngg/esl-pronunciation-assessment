# Sprint 4 weakness memo (US 4.2): frozen encoders vs the Sprint 3 baseline

**Status: draft.** All sections are filled for WavLM, HuBERT and wav2vec2 (layer 12). Section 4 comes from `experiments/scripts/diagnose_us42.py` (validation rows only; csv in `results/analysis/us42/`). Still open: layer re-selection for wav2vec2 (US 4.3) and the train-score refresh for its `cw+thr` rows (see the end).

Scope: frozen `wavlm-base-plus`, `hubert-base`, `wav2vec2-base` (pretrain-only), linear and MLP heads, inputs `emb+side` unless stated.
Tables: `results/tables/frozen/` (`*_L12_emb+side_results.md`, `frozen_vs_baseline_L12_emb+side.md`, `layer_profile_emb+side.md/.png`, `ablation_wavlm_base_plus_L12.md`).

How the evidence was gathered, and its limits:
- Hyper-parameters, class-weight variant and the layer are chosen on validation only. Headline test numbers come from the validation-selected config and were not used for selection.
- Sections 1-3 use the headline table's `train_primary` against **test** for over-fitting remarks. That is indicative only, since the test split was already reported for the benchmark. Section 4.1 repeats the gaps as train vs **validation**, and the US 4.3 fixes are chosen on validation only.
- The layer profile is a **single-seed linear probe, argmax decision** (no decision-weight tuning), one config per task, so its absolute level is below the headline. Example: WavLM L2 phone error is 0.358 in the probe and 0.481 for the same linear config with `cw+thr` and 3 seeds.
- Validation is small for the sentence tasks (19 speakers), so layer differences below about 0.05 RMSE cannot be resolved (see 2.1).

## 1. Where the frozen heads are weak

Headline per task, best head chosen on validation, test set, versus the Sprint 3 best after the fix (US 3.3). Deltas are paired speaker bootstrap.

| task | US 3.3 | WavLM | HuBERT | verdict |
|---|---|---|---|---|
| sent_accuracy (RMSE) | 1.212 | 1.165 (p=0.06) | 1.169 (p=0.16) | tie, not significant |
| sent_fluency | 0.957 | 0.937 (p=0.35) | 0.939 (p=0.37) | tie |
| sent_prosodic | 0.997 | 0.946 (p=0.012) | 0.960 (p=0.11) | small gain (WavLM only) |
| sent_total | 1.180 | 1.136 (p=0.06) | 1.155 (p=0.43) | tie |
| **word_accuracy** | 1.687 | 1.680 (p=0.74) | 1.663 (p=0.23) | **no gain; weakest** |
| phone_score | 0.352 | 0.325 (p<0.001) | 0.325 (p<0.001) | clear gain; PCC 0.31 -> 0.49 |
| phone_mispron_t1.0 (macro-F1) | 0.574 | 0.621 (+0.047) | 0.631 (+0.058) | gain, but see 2.4 |
| **l2_phone_error** | 0.403 | 0.508 (+0.105) | 0.479 (+0.076) | large gain; **still the lowest absolute F1** |
| l2_word_error | 0.615 | 0.709 (+0.093) | 0.684 (+0.069) | large gain |

Reading against the human ceiling from Sprint 3 (word r 0.73, phone r 0.70): word-accuracy PCC is 0.37-0.39 (about 50-53 % of the ceiling, unchanged from Sprint 3's 45 %), and phone-score PCC is 0.49 (about 70 %, up from 48 %).
`wav2vec2-base` at layer 12 is below WavLM and HuBERT everywhere, and **below US 3.3 on L2 phone error** (0.395 vs 0.403) and level on L2 word error (0.616 vs 0.615).

## 2. Representation quality vs head capacity, task by task

### 2.1 Sentence scores: head over-fits; the layer is not identifiable
- MLP train RMSE is far below test (WavLM: accuracy 0.78 vs 1.17, fluency 0.51 vs 0.94, prosodic 0.51 vs 0.95, total 0.70 vs 1.14), while the MLP gains only 0.02-0.06 test RMSE over the linear head (e.g. accuracy 1.223 -> 1.165). The extra capacity is spent memorising the training speakers.
- The weaker wav2vec2 L12 over-fits less (accuracy train 1.05 vs test 1.24), which is what a poorer representation looks like: less to memorise and less to generalise.
- Layer curves zig-zag by 0.05-0.15 between adjacent layers (HuBERT sent_accuracy: L1 1.105, L2 1.193, L10 1.091, L12 1.170), larger than the seed sd of 0.02-0.03. Validation-selected layers (WavLM 12, HuBERT 10, wav2vec2 9) cannot be trusted to better than about 0.05. HuBERT L10 gains 0.08 on val and 0.04 on test over L12.
- The sentence-level embedding adds little to rate-like features. Embedding-only is within 0.005-0.009 RMSE of `emb+side`, and `side` alone (duration, word and phone counts) gets fluency 1.005 vs 0.937, prosodic 1.055 vs 0.946, accuracy 1.259 vs 1.165.
- US 3.3 showed pause / rate / pitch / energy features help the sentence tasks significantly, and the frozen heads do not have them. Part of the remaining sentence gap is therefore a feature gap, not a representation gap.
- **Diagnosis: head over-fitting plus missing prosody features. Not representation quality.**

### 2.2 Word accuracy: neither the layer nor the head explains it
- Linear train RMSE 1.686 equals test 1.680, so it cannot fit even the training set. The MLP reaches 1.58 (WavLM) / 1.50 (HuBERT) on train for 1.68 / 1.66 on test.
- The layer curve is flat: val RMSE 1.72-1.79 for WavLM and HuBERT; wav2vec2's best layer is L1.
- Input ablation (WavLM): `side` only 1.713, `emb` only 1.701, `emb+side` 1.680; the embedding adds no significant gain over side features alone (full minus side: -0.033, CI [-0.072, +0.008], p=0.15).
- Hypotheses to test in the diagnose script and 4.3: (a) mean-pooling the whole word span hides which phone is wrong; (b) the heavily skewed target (about 90 % of words score 10) leaves RMSE dominated by rare low-score words, as in Sprint 3 H3 (check bias by score bin); (c) label noise (human r 0.73).
- **Diagnosis: not a layer problem and not head capacity in the tested range. Pooling / label-skew is the lead suspect.**

### 2.3 Phone score: head capacity matters
- Linear probe 0.345 (PCC 0.36) vs MLP 0.325 (PCC 0.49), a larger step than any layer change (probe layer range 0.355-0.377; WavLM best at L12, HuBERT at L8, wav2vec2 at L0).
- Ablation: `emb` 0.347, `side` 0.352 (the Sprint 3 level), both 0.325. The gain comes from the *interaction* between canonical phone identity and the acoustic embedding, which a linear head cannot express. Each part alone is no better than Sprint 3.
- **Diagnosis: head capacity / interaction modelling, with a moderate generalisation gap (MLP train 0.27 vs test 0.33).**

### 2.4 SO762 phone mispronunciation: the audio does add ranking signal; the weak parts are precision, child/teen speakers and over-fitting
- Test macro-F1 hides the audio contribution: `side` only reaches 0.591 against 0.621 for both (full minus side +0.030 [-0.002, +0.058], p=0.064), and the linear-probe layer curve is flat (0.51-0.565, layers 0-1 as good as 12).
- Threshold-free ranking on validation (section 4.6) tells a different story. AUC of "mispronounced" is 0.883 for `emb+side`, 0.850 for `emb` only and 0.783 for `side` only, against 0.803 for the Sprint 3 MFCC logistic regression (which also saw side features). Average precision is 0.244 / 0.201 / 0.139 vs 0.156. Embedding-only already out-ranks the MFCC baseline.
- What stays weak: precision at useful recall (AP 0.24 on a 4.5 % base rate), no detections at all for child/teen speakers (4.5), and over-fitting (MLP train 0.75 vs val 0.63; MLP adds nothing over linear, -0.003 val).
- **Diagnosis: not a representation failure. The embedding carries the signal; the limits are the operating point, the small positive class in some speaker groups, and head over-fitting.**

### 2.5 L2-ARCTIC phone and word: representation depth matters, wav2vec2 is a layer-choice failure
- For WavLM and HuBERT, the probe improves nearly monotonically with depth (WavLM L2 phone 0.277 at L0 -> 0.358 at L12; L2 word 0.598 -> 0.699). These are the only tasks with a clean layer trend.
- Layer 12 is already near the top for WavLM and HuBERT (layer 11 is better by only 0.012 / 0.009 val on L2 word), so layer selection has about 0.01 headroom for them.
- `wav2vec2-base` collapses after L9-10: L2 word 0.688 (L9) -> 0.620 (L12); L2 phone 0.330 (L9) -> 0.289 (L12), 0.278 at L11. At L9, test L2 word is 0.685 vs 0.612 at L12 (+0.073) and L2 phone 0.328 vs 0.281 (+0.047), and sent_accuracy 1.289 vs 1.412. Its L9 L2-word result matches HuBERT / WavLM at L12 (0.678 / 0.698).
- wav2vec2 L12 also cannot fit train (phone-score MLP train RMSE 0.355 vs HuBERT 0.244, WavLM 0.274), and its MLP gains nothing over the linear head (L2 phone error -0.005 val, phone score +0.007), so more head capacity cannot help it.
- Threshold-free, wav2vec2 L12 ranks L2-ARCTIC errors **no better than the Sprint 3 MFCC logistic regression**: AUC of error vs correct 0.726 against 0.742 (substitution 0.737 vs 0.756, deletion 0.815 vs 0.834), while WavLM reaches 0.816. It is competitive on SO762 mispronunciation ranking (AUC 0.866 vs 0.883 WavLM), so the failure is specific to the L2-ARCTIC tasks at the last layer.
- The encoder ranking in 4.1 is therefore largely a layer-choice artefact for wav2vec2, not a property of the checkpoint family. WavLM >= HuBERT is real only for L2 phone and L2 word.
- WavLM L2 phone: `side` 0.361, `emb` 0.449, both 0.508. Both parts matter and acoustics matter most. Linear -> MLP adds 0.02-0.03 and the train-test gap is moderate (0.59 vs 0.51).
- **Diagnosis: wav2vec2 = representation (wrong layer). WavLM / HuBERT = moderate head over-fit plus class imbalance. The probe at L12 gets 0.358 for WavLM L2 phone and the headline 0.481 for the same linear config with decision weights, so decision weighting is still worth about +0.12.**

## 3. Failure modes compared with Sprint 3

| Sprint 3 failure | Sprint 4 status |
|---|---|
| Mispronunciation detection: weakest, near chance | Better (test 0.574 -> 0.62-0.64; val AUC 0.80 -> 0.88, AP 0.16 -> 0.24). Precision stays low (0.23 at recall 0.47) and recall is 0 for child/teen speakers; same dependence on the phone's error rate (4.4) |
| L2 phone error type: error classes predicted "correct" | Largest gain (val macro-F1 0.32 -> 0.50; addition recall 0 -> 0.24); errors are still mostly mistaken for "correct" rather than for another error (4.3); per-phoneme recall still tracks the phone's error rate (4.4) |
| Word accuracy: regression to the mean, about 45 % of ceiling | **Unchanged** (about 50 %); not fixed by representations or by layer choice |
| Phone score: about 48 % of ceiling | Improved to about 70 %; head capacity is the lever |
| Sentence scores: acceptable, prosody cues missing | No longer lagging, but not better than the prosody-aware baseline; new failure: over-fitting |
| (not present) | **New:** wav2vec2 last layer unusable; layer choice unidentifiable on small validation sets |

Class imbalance is still handled by weights and decision tuning (Sprint 3 H1 holds): `cw+thr` or `cw` is the validation-selected variant for every classification head except WavLM's linear L2-word head (plain).
L1 and age-group results are in 4.5: Hindi and Arabic are now the weakest L1s, and child/teen speakers remain the weakest group.

## 4. Phoneme categories, classes and speaker groups (validation only, WavLM and HuBERT)
Source: `results/analysis/us42/*.csv`. "US 3.1" is the unfixed Sprint 3 baseline (svm / rf / ridge) on the same validation rows (asserted equal). Part of each gain below is therefore what the Sprint 3 fix (US 3.3 weights and thresholds) would recover anyway; the test-set comparison against US 3.3 is in section 1.

### 4.1 Over-fitting and head capacity (WavLM; HuBERT follows the same pattern, see `head_gaps_val.csv`)
| task | linear val | MLP val | MLP gain | MLP gap (train better by) |
|---|---|---|---|---|
| sent_accuracy | 1.059 | 1.014 | 0.045 | 0.230 |
| sent_fluency | 0.760 | 0.709 | 0.051 | 0.200 |
| sent_prosodic | 0.756 | 0.725 | 0.032 | 0.217 |
| sent_total | 0.971 | 0.949 | 0.022 | 0.245 |
| word_accuracy | 1.704 | 1.715 | -0.011 | 0.140 |
| phone_score | 0.353 | 0.337 | 0.015 | 0.063 |
| phone_mispron (macro-F1) | 0.631 | 0.628 | -0.003 | 0.123 |
| l2_phone_error (macro-F1) | 0.480 | 0.503 | 0.023 | 0.088 |
| l2_word_error (macro-F1) | 0.704 | 0.710 | 0.007 | 0.064 |

Gain is positive when the MLP is better (RMSE: linear minus MLP; macro-F1: MLP minus linear). The MLP buys 0.02-0.05 RMSE on the sentence scores and phone score and 0.02 macro-F1 on L2 phone error, at the price of a large train-validation gap on the sentence tasks (0.20-0.25 RMSE; HuBERT 0.17-0.30).
wav2vec2 L12 behaves differently: the train-validation gaps are small (sentence MLP 0.02-0.14, phone score 0.002) and the MLP gain is at most 0.04, so it is limited by what the representation holds, not by over-fitting.
For word accuracy, mispronunciation and L2 word error the extra capacity gives nothing on validation (-0.011 to +0.008) while the gap grows (word accuracy 0.14 WavLM, 0.25 HuBERT). That is an information limit, not a capacity limit.

### 4.2 Regression to the mean persists (validation)
| task | system | slope of prediction on truth | worst bin: true mean -> predicted |
|---|---|---|---|
| word_accuracy | US 3.1 / WavLM / HuBERT / wav2vec2 | 0.10 / 0.23 / 0.26 / 0.27 | words scoring 0-4 (6 % of words): 2.98 -> 8.64 / 7.81 / 7.59 / 7.47 |
| phone_score | US 3.1 / WavLM / HuBERT / wav2vec2 | 0.13 / 0.34 / 0.35 / 0.21 | phones < 0.5 (3 %): 0.13 -> 1.62 / 1.33 / 1.29 / 1.46 |

The frozen heads raise the slope (word 0.10 -> 0.23, phone 0.13 -> 0.34) and cut the bias in the lowest bin by roughly 15-20 %, but 89.5 % of words score 10 and 82 % of phones score 2, so the heads still mostly predict the majority score. PCC as a share of the human ceiling (val): word 0.45 -> 0.55-0.57 for all three encoders (the encoder hardly matters), phone 0.48 -> 0.70 (WavLM, HuBERT) and 0.55 (wav2vec2). Sentence PCC is 1.00-1.15 of the ceiling on val (the 19-speaker validation set is easier than test, as in Sprint 3).

### 4.3 L2-ARCTIC phone error: classes
| class (val support) | recall US 3.1 / WavLM / HuBERT / wav2vec2 | F1 US 3.1 / WavLM / HuBERT / wav2vec2 | one-vs-rest AUC US 3.1* / WavLM / HuBERT / wav2vec2 |
|---|---|---|---|
| correct (101,633) | 0.971 / 0.898 / 0.895 / 0.870 | 0.915 / 0.904 / 0.901 / 0.881 | 0.742 / 0.816 / 0.803 / 0.726 |
| substitution (14,098) | 0.199 / 0.460 / 0.457 / 0.385 | 0.283 / 0.460 / 0.448 / 0.388 | 0.756 / 0.800 / 0.787 / 0.737 |
| deletion (3,420) | 0.053 / 0.493 / 0.432 / 0.292 | 0.094 / 0.401 / 0.385 / 0.238 | 0.834 / 0.900 / 0.894 / 0.815 |
| addition (1,092) | 0.000 / 0.238 / 0.165 / 0.106 | 0.000 / 0.246 / 0.170 / 0.070 | 0.675 / 0.832 / 0.788 / 0.699 |

\* Sprint 3's stored AUCs are for the unweighted logistic regression (the memo stored no probabilities for the best US 3.1 model).
- The Sprint 3 failure shape survives in a milder form. WavLM still mistakes errors for "correct" far more often than for each other: substitution -> correct 0.49 (Sprint 3: 0.80), deletion -> correct 0.41 (0.91), addition -> correct 0.59 (0.97); errors confused with another error stay at 0.01-0.11.
- Addition is still the hardest class (F1 0.25; HuBERT 0.17), but its AUC rose from 0.68 to 0.83, so the ranking is decent and the rest is class imbalance and the decision rule (1,092 additions against 101,633 correct).
- wav2vec2 L12 is the weakest: its AUCs are at or below the Sprint 3 MFCC logistic regression on correct, substitution and deletion (0.726 / 0.737 / 0.815 against 0.742 / 0.756 / 0.834) and only slightly above on addition (0.699 vs 0.675). Its errors are also mostly mistaken for "correct" (deletion -> correct 0.59, addition -> correct 0.74).
- The audio does the work for additions: AUC 0.58 with side features only, 0.81 with the embedding only, 0.83 with both.

### 4.4 Phoneme categories
- L2 phone error recall per phoneme still follows how often that phoneme errs in training: Spearman rho with the training error rate is 0.90 (WavLM), 0.91 (HuBERT) and 0.86 (wav2vec2) against 0.92 in Sprint 3 (rho with error count 0.66 / 0.62 against 0.59). The frozen heads did not remove the phoneme-prior structure.
- The gains are concentrated on high-error phonemes: WavLM recall DH 0.94, Z 0.92, TH 0.87 (training error rate 0.55 / 0.51 / 0.45), up from 0.51 / 0.51 / 0.30. Phonemes with few errors stay near 0.1-0.2 (WavLM, 100 or more validation errors): IY 0.03, HH 0.11, NG 0.11, B 0.11, AY 0.14, AO 0.15, EH 0.16 (355 errors), UW 0.16, W 0.18, G 0.20. In Sprint 3 all of these were at 0.00-0.02.
- SO762 phone score, hardest phones by RMSE (n >= 40): UH 0.49, TH 0.48, R 0.45, AW 0.43, AO 0.43, ER 0.42, Z, AH, EH, V. That is largely the Sprint 3 hard set (TH, AO, R, EH, AW, ...). Gains are largest on AO (0.53 -> 0.43), EH (0.49 -> 0.40), R (0.50 -> 0.45), TH (0.54 -> 0.48); WavLM is worse than US 3.1 on DH (0.25 -> 0.29), V (0.32 -> 0.36), UH, Y and P.
- SO762 mispronunciation: recall also follows the phone's training error rate (rho 0.48 WavLM, 0.70 HuBERT; Sprint 3 0.46). Most positives are on AH (53), T (27), EH (23), R (22), IH (20); WavLM recall AH 0.40, T 0.37, EH 0.70, R 0.77.

### 4.5 Speaker groups
L2-ARCTIC (4 validation speakers per L1, indicative only), macro-F1:
| L1 | phone error US 3.1 / HuBERT / WavLM / wav2vec2 | word error US 3.1 / HuBERT / WavLM / wav2vec2 |
|---|---|---|
| Arabic | 0.292 / 0.447 / 0.468 / 0.358 | 0.637 / 0.662 / 0.692 / 0.619 |
| Hindi | 0.320 / 0.413 / 0.419 / 0.352 | 0.591 / 0.641 / 0.665 / 0.611 |
| Korean | 0.335 / 0.450 / 0.502 / 0.397 | 0.586 / 0.667 / 0.687 / 0.587 |
| Mandarin | 0.345 / 0.474 / 0.501 / 0.415 | 0.589 / 0.687 / 0.713 / 0.622 |
| Spanish | 0.322 / 0.450 / 0.467 / 0.378 | 0.610 / 0.688 / 0.703 / 0.626 |
| Vietnamese | 0.311 / 0.484 / 0.505 / 0.399 | 0.554 / 0.714 / 0.749 / 0.642 |

All L1s improve on phone error with WavLM and HuBERT. Sprint 3's weakest was Arabic; **Hindi is now the weakest for every encoder**, with Arabic next for HuBERT and wav2vec2, so this is an L1 effect and not an encoder effect. The spread widened (0.29-0.35 -> 0.42-0.51 for WavLM). On word error Vietnamese went from worst (0.554) to best for WavLM, HuBERT and wav2vec2 (0.749 / 0.714 / 0.642). wav2vec2 word error is at the Sprint 3 level for Arabic, Korean and Hindi (0.587-0.619 vs 0.586-0.637).

SO762 age groups (validation: 10 adult and 9 child/teen speakers), PCC US 3.1 / WavLM / HuBERT / wav2vec2:
| task | adult | child/teen |
|---|---|---|
| sent_accuracy | 0.68 / 0.83 / 0.79 / 0.82 | 0.12 / 0.28 / 0.25 / 0.04 |
| word_accuracy | 0.30 / 0.41 / 0.40 / 0.42 | 0.17 / 0.16 / 0.05 / 0.04 |
| phone_score | 0.33 / 0.49 / 0.49 / 0.38 | 0.26 / 0.36 / 0.35 / 0.17 |

Word accuracy for child/teen speakers did not improve with any encoder (HuBERT and wav2vec2 are worse than Sprint 3). wav2vec2 is also worse than Sprint 3 for child/teen sentence accuracy (0.04 vs 0.12) and phone score (0.17 vs 0.26).
Mispronunciation: child/teen phones are 0.9 % positive (about 28 phones) against 7.0 % for adults, and error recall is **0.000 for all four systems** on child/teen speakers (adults: 0.08 US 3.1, 0.51 WavLM, 0.43 HuBERT, 0.47 wav2vec2). The decision weights are tuned on pooled validation data, which is adult-dominated, so they never fire for the child/teen group.

### 4.6 Representation vs decision rule: threshold-free ranking (validation, WavLM)
| task | metric | US 3.1 (logreg, Sprint 3) | emb+side | emb only | side only |
|---|---|---|---|---|---|
| phone_mispron | AUC / AP | 0.803 / 0.156 | 0.883 / 0.244 | 0.850 / 0.201 | 0.783 / 0.139 |
| l2_phone_error | AUC of error vs correct | not stored | 0.816 | 0.766 | 0.698 |
| l2_word_error | AUC / AP | not stored | 0.791 / 0.702 | 0.784 / 0.690 | 0.683 / 0.576 |

At layer 12 with `emb+side`, the AUC for mispronunciation is 0.883 / 0.871 / 0.866 (WavLM / HuBERT / wav2vec2), for L2 phone error 0.816 / 0.803 / 0.726 and for L2 word error 0.791 / 0.763 / 0.682. wav2vec2 is competitive on SO762 mispronunciation and weak on both L2-ARCTIC tasks.

The embedding clearly improves ranking over side features alone on all three tasks. For mispronunciation, embedding-only already out-ranks the Sprint 3 MFCC model that had side features, so the test macro-F1 gap in 2.4 understates what the audio adds.

## 5. Consequences for US 4.3 (planned, to be confirmed on validation)
1. **wav2vec2 layer fix** (cheapest, data already cached): re-select layer per task on validation from layers 6-12; expected to bring it level with the other two on L2 tasks. The AUC result (below the Sprint 3 MFCC model on L2 phone error at layer 12) says this is a representation problem and not a decision-rule problem.
2. **Learned layer-weighted sum** (softmax over layers, trained on train rows) instead of per-layer selection. It avoids the noisy single-layer choice, which matters most for the sentence tasks.
3. **Sentence tasks:** stronger regularisation (weight decay, dropout, input dropout), PCA to a low dimension, seed averaging; and add the US 3.3 pitch / energy / pause features to the side input to test the prosody-gap hypothesis.
4. **Word accuracy:** max / attention pooling over the word's frames or its phone embeddings; reweighting low-score words.
5. **Phone score / mispronunciation:** MLP with neighbour-phone context (the ±1 phones); a Sprint 3 head on `side` only would still tell us how much of the head change vs the audio explains the gain.
6. **Other checkpoints** (larger or ASR-fine-tuned) only if 1-5 leave a gap; larger caches are much bigger than Colab free Drive allows, so save selected layers only.
7. **Speaker-group calibration for mispronunciation:** check per-group decision weights or a group-balanced threshold (child/teen recall is 0), and report AUC / AP next to macro-F1 for the binary tasks.
8. Every choice made on validation; each fix reported as a paired speaker-bootstrap delta against the 4.1 headline.

Train-score note: the `train_primary` values of wav2vec2's `cw+thr` rows in the first results table (for example L2 phone error 0.327-0.353, below test) were computed with plain argmax and without the decision weights, which is why they sit below test. The fixed train score exists only for WavLM and HuBERT (re-fit in 15c), so wav2vec2's classification gaps are blank in section 4 and are not used as evidence. Re-running 15c for wav2vec2 would fill them.
