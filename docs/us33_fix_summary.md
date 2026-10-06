# US 3.3: targeted fixes and before/after (Sprint 3)

Full tables: `results/tables/baseline/us33_before_after.md` / `.csv` (test, paired speaker-bootstrap), `us33_val_candidates.csv` (validation, every candidate fix). Code: `src/evaluation/report_us33.py`, `src/models/baselines.py` (variants in `configs/baseline.yaml`), `src/features/prosody.py`.

## Protocol

- Each fix re-fits the **same hyper-parameters US 3.1 selected**, so the delta isolates the fix.
- For each (task, model) the fix is chosen on **validation** (primary metric: macro-F1 for classification, RMSE for regression). A fix must beat US 3.1 by at least 1% relative RMSE or +0.01 macro-F1; otherwise US 3.1 is kept.
- `+thr` (decision weights tuned for macro-F1) is scored on validation with cross-fitted weights (tuned on the other folds / the other half of the validation speakers). The weights applied to test are tuned on all validation data.
- Test was scored once, for the frozen choices only. Deltas carry paired speaker-level bootstrap CIs (500 draws).
- Not run: class weights combined with the prosody features (`cw_pros`): prosody features changed classification macro-F1 by less than 0.01 on validation, so the combination was skipped.

## Results (test, same model before -> after)

| Task (best model after fix) | Fix | Before | After | Change |
|---|---|---|---|---|
| Phone mispronunciation (<1.0), logreg | class weights + threshold | macro-F1 0.504, bal.acc 0.507, F1 0.03 | macro-F1 0.574, bal.acc 0.565, F1 0.18 | +0.070 [+0.050, +0.084] |
| Phone mispronunciation (<1.0), RF | class weights + threshold | bal.acc 0.500, F1 0.00 | bal.acc 0.664, F1 0.23 | macro-F1 +0.09 [+0.058, +0.120] |
| L2-ARCTIC phone error type (4-class), RF | class weights + threshold | macro-F1 0.254 | **0.403** | +0.149 [+0.132, +0.163] |
| L2-ARCTIC word has-error, logreg | class weights | macro-F1 0.589, F1 0.47 | 0.615, F1 0.57 | +0.027 [+0.014, +0.039] |
| Sentence accuracy / fluency / prosodic / total, RF | pitch + energy + pause / rate features | PCC 0.545 / 0.680 / 0.657 / 0.573 | PCC 0.617 / 0.747 / 0.728 / 0.651 | RMSE -0.085 / -0.116 / -0.116 / -0.101 (all p < 0.001) |
| Word accuracy, RF | prosody features | RMSE 1.704, PCC 0.320 | 1.687, 0.343 | -0.017 [-0.030, -0.005] |
| Phone score (all models) | none kept | RMSE 0.352, PCC 0.307 (RF) | unchanged | no fix beat the margin |

L2-ARCTIC phone error type, per-class recall (RF, test): correct 1.00 -> 0.90, substitution 0.05 -> 0.32, deletion 0.00 -> 0.30, addition 0.00 -> 0.10. Precision falls (deletion 0.50 -> 0.23, addition 0.09), so the gain is a recall / precision trade, not new information. Compared with the best US 3.1 model on that task (SVM, macro-F1 0.326) the post-fix best model is +0.077.

## What worked and what did not

1. **Class weighting + tuned thresholds** fixed the detection weakness the memo diagnosed (SO762 mispronunciation, L2-ARCTIC minority classes). Weights alone mostly raise recall; the tuned threshold is what recovers macro-F1. Additions remain the hardest class (F1 about 0.09).
2. **Prosody features helped at sentence level, not just where the memo expected.** Prosody and fluency were not lagging relative to the ceiling, but pause / speaking-rate / articulation-rate and energy cues improved all four sentence scores (including accuracy and total), so the US 3.1 baseline was missing them. Word accuracy improved slightly; phone-level and classification tasks did not (< 0.01 macro-F1).
3. **Resampling for regression to the mean did not help.** Oversampling low scores (power 0.5) lowers the over-prediction of low-scoring items (phone score < 0.5: mean prediction 1.62 -> 1.44, truth 0.13) but shifts everything down; RMSE worsens (phone RF 0.358 -> 0.408, word RF 1.739 -> 1.985 on validation) and PCC is unchanged. Phone and word score therefore stay at roughly 45-48% of the human ceiling (0.70-0.73) and are the clearest headroom for the deep models.
4. **Non-trivially-weak check.** After the fixes the classification baselines are above chance on every task (mispronunciation balanced accuracy 0.57-0.66, 4-class macro-F1 0.40) but still far from usable: L2 deletion / addition F1 is low and phone / word regression barely beats the mean predictor on RMSE.

## Caveats

- Fixes were selected per model on a 19-speaker SO762 validation set; SO762 classification gains rest on that selection and a single split.
- Threshold weights are tuned for macro-F1; a different operating point (for example high precision for learner feedback) would change precision / recall.
- The regression selection metric was switched from MAE to RMSE during US 3.1 after observing the median-predictor effect.
- Prosody features use librosa pYIN (the Praat-based library has no build for the work machine). The pause / rate features come from the Sprint 2 MFA alignments, so they inherit its boundary accuracy.
