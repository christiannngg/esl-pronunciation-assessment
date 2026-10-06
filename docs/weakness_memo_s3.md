# Sprint 3 weakness memo (US 3.2)

Scope: the US 3.1 MFCC baselines (Ridge / SVR / RF / logistic regression / RBF-SVM). **Everything here uses validation data only** (SO762: 19 val speakers; L2-ARCTIC: each speaker once as a validation speaker across the 4 folds). Test scores were not consulted. Tables: `results/analysis/us32/`, figures: `results/figures/us32/`, scripts: `experiments/scripts/diagnose_us32.py`, `figures_us32.py`, `ablations_us31.py`.

## 1. Where the baseline is weak

| Task | Evidence (val) | Verdict |
|------|----------------|---------|
| **Phone mispronunciation detection** (SO762, score < 1.0, 4.5% positive) | RF predicts no positives. Best stored model balanced accuracy 0.53. Logistic regression flags 1.2% of phones; recall 0.06 | **Weakest** |
| **L2-ARCTIC phone error type** (4-class) | Recall: correct 0.97, substitution 0.20, deletion 0.05, **addition 0.00** (F1 0.28 / 0.09 / 0.00). 80-97% of every error class is predicted "correct" | **Weak**, worst for deletion and addition |
| **Word accuracy** (SO762) | PCC 0.33 vs human leave-one-out r 0.73 (45% of ceiling); RMSE only 5% better than the mean predictor | Weak |
| **Phone score** (SO762) | PCC 0.34 vs human r 0.70 (48% of ceiling); RMSE 5% better than the mean predictor | Weak |
| L2-ARCTIC word has-error | F1 0.49, balanced accuracy 0.60 (test) | Moderate |
| Sentence scores | PCC 0.68-0.83 on val, 0.55-0.68 on test; 0.89-1.06 of the human ceiling on val | Acceptable; accuracy and total lowest |

Reading the sentence rows: the 19-speaker val set is easier than test (val PCC is 0.1-0.15 higher), so the val sentence numbers flatter the baseline. Use the test table for headline numbers and the val tables only for diagnosis.

## 2. Hypothesised causes and the evidence for each

**H1. Class imbalance suppresses minority recall at the default decision threshold, but it is not the only limit.**

- SO762 mispronunciation: ROC-AUC is 0.80, so the features rank mispronounced phones reasonably. The unweighted classifier still predicts positive for only 1.2% of phones (base rate 4.5%). With balanced class weights: recall 0.06 -> 0.78, balanced accuracy 0.53 -> **0.73**, F1 0.10 -> 0.19, precision 0.23 -> 0.11. AUC is unchanged (0.80), which is the signature of a threshold / weighting effect, not new information. Imbalance **is** the main reason the baseline sits at chance.
- L2-ARCTIC (class shares: correct 84.5%, substitution 11.7%, deletion 2.8%, addition 0.9%): the confusion matrix shows the failure is *error vs. correct*, not *which error*: errors are almost never confused with each other (substitution -> deletion 0.2%), they are predicted "correct". Per-class recall is ordered exactly like class support (0.97 / 0.20 / 0.05 / 0.00). Balanced weights raise recall to 0.54 (sub), 0.62 (del), 0.40 (add) and balanced accuracy 0.30 -> 0.50, **but macro-F1 does not improve (0.32 -> 0.30)** because precision collapses (sub 0.53 -> 0.27, del 0.39 -> 0.10, add 0.80 -> 0.02). One-vs-rest AUCs are unchanged (0.76 / 0.83 / 0.68). So weighting trades recall for precision; it does not add discriminative power.
- Phoneme level: per-phoneme error recall correlates with the phoneme's training error *rate* (Spearman rho = 0.92, n = 39) and with its training error *count* (rho = 0.60). Rare-error phonemes (e.g. M, OY with < 50 error events) are almost never detected.

**H2. The MFCC window statistics carry little segmental evidence beyond phone identity.**

- A classifier that sees only the anchor phone one-hot reaches macro-F1 0.29 on L2-ARCTIC (vs 0.32 for the full logistic regression) with nearly the same AUCs (substitution 0.75 vs 0.76, deletion 0.79 vs 0.83). Removing phone identity drops the model to dummy level (macro-F1 0.23). The baseline is mostly a phoneme-prior model; acoustic features add roughly +0.01 (sub) / +0.04 (del) / +0.09 (add) AUC.
- Consistent with this, SO762 phone-score PCC changes only 0.275 -> 0.256 when phone identity is removed, so there the acoustic features are also doing little.
- This limits what class weighting alone can fix and is a candidate motivation for the deep models (Sprint 4+): a fair comparison point should be weak for this reason, not because of avoidable imbalance.

**H3. Regression to the mean on heavily skewed targets.**

- Word accuracy: 89.5% of val words score 10 and phone score: 82% score exactly 2. Mean predictions for low-scoring words are 8.6 when the truth averages 3.0, and for phones scoring < 0.5 the mean prediction is 1.62 (truth 0.13). The slope of prediction on truth is 0.10 (word) and 0.13 (phone), versus about 0.4-0.5 for sentence scores. Sentence extremes show the same shape (scores 0-4 over-predicted by 1.4-2.7, scores of 10 under-predicted by about 2).
- A constant **median** predictor beats every trained model on MAE (word 0.56, phone 0.125), so MAE is not a usable headline metric on these targets; RMSE and PCC are.

**H4. Prosody / fluency are *not* lagging, so the conditional fix is not triggered.**

- Sentence fluency and prosodic have the best RMSE skill (0.41 / 0.38 vs 0.26 accuracy and 0.30 total) and the highest PCC (val 0.83 / 0.81; test 0.68 / 0.66), at or above the human leave-one-out ceiling (r 0.79 / 0.77) on val. The weak sentence scores are accuracy and total. I tested pitch / energy / pause features anyway in US 3.3 as a hypothesis check. Result (see `docs/us33_fix_summary.md`): it was **not** a null at sentence level; all four sentence scores improved significantly, so the pause / rate cues were missing from the MFCC statistics even though prosody was not the weakest target.

## 3. Other findings that affect how results are read

- **Leakage control matters for L2-ARCTIC.** Adding the *annotated* interval duration raises deletion recall from 0.05 to 0.22 (macro-F1 0.32 -> 0.37): annotators mark a deleted phone with a short interval. The baseline therefore uses a fixed window around each unit's midpoint and does not use annotated duration (`us31_ablations_val.csv`). Additions have no canonical phone, so they use the following unit's phone as their anchor.
- **Oracle segmentation caveat.** L2-ARCTIC error-type classification is "given an annotated unit"; detecting that an addition or deletion exists at all is easier here than in a real pipeline.
- **By group (val, indicative only: 9-10 speakers per group).** SO762 child/teen speakers have much lower within-group PCC than adults (sentence accuracy 0.12 vs 0.69; word 0.17 vs 0.30), partly because their score spread is narrower. By L1 on L2-ARCTIC, macro-F1 ranges 0.29 (Arabic) to 0.35 (Mandarin); Vietnamese has the highest error rate (24%) but only 0.19 substitution recall. Aggregate numbers hide these differences, so keep reporting by group.
- Hardest SO762 phonemes by RMSE (n >= 40): TH, AO, R, EH, OW, AW, EY (`phone_score_per_phoneme_val.csv`).

## 4. Consequences for US 3.3 (planned, to be confirmed against val)

1. **Mispronunciation detection:** class weighting (and val-tuned decision threshold); expect a large balanced-accuracy gain, small F1 gain, precision stays low.
2. **L2-ARCTIC error types:** class weighting and resampling, plus val-tuned per-class thresholds; judge by macro-F1 and per-class F1, not recall alone. Expect additions to remain the hardest class.
3. **Phone / word score regression:** reweight or oversample low-scoring items to counter regression to the mean; report RMSE / PCC and score-bin bias.
4. **Feature variants:** energy / pitch / pause-rate features, wider phone context. Tested on all tasks.
5. All choices on validation; each fix recorded as a delta against US 3.1 with a paired speaker-level bootstrap.

Caveat on method: the regression model-selection metric was switched from MAE to RMSE after seeing the median-predictor effect (a val-based, principled change; the grids are small).
