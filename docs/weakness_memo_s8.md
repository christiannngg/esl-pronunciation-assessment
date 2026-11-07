# Weakness memo, Sprint 8 (US 8.2)

Status: checks 1, 2, 3 and the ablation are done. Check 4 (age probe) is optional and not run.
Source tables: `results/tables/generalization/us82_*.csv`. Code: `src/evaluation/report_us82.py`. Sign convention: ablation `delta` is the metric change (RMSE for regression, macro-F1 for classification); `delta_gap` is on the error scale, positive = child/teen relatively worse.

## 1. Target (rule kept as written)

Weakest subgroup: SO762 child/teen. Target head: `phone_mispron_t1.0`, the only Holm-significant Family A gap. Regression-head compression is reported as secondary/exploratory; widening the target would be a logged post-hoc deviation.

## 2. Diagnosis

**Check 1 (is it label noise or range restriction?).** The child/teen compression is real. Matching adult label distributions to the child's leaves the slope gap unchanged. Label noise (Spearman-Brown reliability of the 5-expert consensus) explains only ~6-12% of slope attenuation, against ~40-64% observed. The PCC gap is partly range restriction/noise: for `phone_score` it vanishes after label matching.

**Check 3 (where in the score range).** Child/teen speakers have few low-score items (sent_accuracy <=6: 7% vs 24.5% adult; word_accuracy 0-4: 3% vs 7%). Errors are in the tails, with positive bias at low scores: sent_accuracy <=6 bias +2.63 (child/teen) vs +1.11 (adult); word_accuracy 0-4 bias +6.42 vs +4.60. The model regresses to the mean most where child/teen training data is thinnest.

**Check 2 (ranking vs calibration, target head, test set; `us82_check2_ranking.csv`).** Recall of mispronounced phones at the pooled operating point is far lower for child/teen than adult. Splitting that gap into a ranking part (adult recall minus child recall at the adult false-positive rate) and a calibration part (child recall at adult FPR minus child recall at the pooled threshold):

| System  | Recall adult / child (pooled threshold) | Total gap | Ranking part [95% CI] | Calibration part [95% CI] | AUC adult / child |
| ------- | --------------------------------------- | --------- | --------------------- | ------------------------- | ----------------- |
| FT-base | 0.437 / 0.172                           | 0.264     | 0.092 [-0.015, 0.203] | 0.172 [0.064, 0.268]      | 0.909 / 0.868     |
| MT-full | 0.330 / 0.052                           | 0.278     | 0.092 [-0.007, 0.198] | 0.187 [0.069, 0.288]      | 0.917 / 0.864     |
| MT-GN15 | 0.465 / 0.127                           | 0.338     | 0.057 [-0.047, 0.181] | **0.281 [0.144, 0.372]**  | 0.911 / 0.869     |

For MT-GN15, about 83% of the recall gap is calibration (the pooled threshold sits too high for child/teen scores) and the ranking part is not significant. Ranking is not perfectly equal: AUC is lower for child/teen (-0.04, CI excludes zero), and AP is lower (-0.20), though AP also depends on the lower child/teen positive rate (3.0% vs 5.2%). FT-stab and FT-base are identical on this head (same classification head), so FT-stab is listed once in effect.

## 3. Ablation (MT-GN15 is the demo model)

| Cause tested        | Contrast                         | Target head `phone_mispron_t1.0`: adult / child F1 change, gap change        | Verdict                                                                                                                             |
| ------------------- | -------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Pretrained features | B -> F                           | +0.106 / +0.026, gap +0.080 [+0.050,+0.116], Holm p<0.001                    | Frozen WavLM helps adults ~4x more. Gap also widens on fluency, prosodic, total, phone_score (Holm p<0.05).                         |
| Fine-tuning         | F -> FT-base                     | +0.018 / +0.020, gap -0.002 [-0.025,+0.018]                                  | Fine-tuning helps both groups equally. It does not close or widen the gap.                                                          |
| Multi-task sharing  | FT-base -> MT-GN15               | +0.006 / **-0.024** [-0.038,-0.004], gap +0.030 [+0.009,+0.048], Holm p=0.12 | Child/teen gets worse on the target head. The gap widens, but not Holm-significant.                                                 |
| Multi-task, full    | FT-base -> MT-full (exploratory) | -0.011 / **-0.071** [-0.094,-0.036], gap +0.060                              | Same direction, larger.                                                                                                             |
| Stabilization       | FT-base -> FT-stab (exploratory) | regression heads only; classification head identical (0 change)              | Narrows gap on sent_accuracy (-0.082), sent_fluency (-0.065), sent_total (-0.068, CI touches 0). Not applicable to the target head. |

L2-ARCTIC (exploratory, descriptive): B->F and F->FT-stab both improve phone error pooled; FT-stab->MT-full is flat (+0.001 phone, +0.001 word). Per-L1 differences are not interpreted (4 speakers per L1).

## 4. What this means for US 8.3

- Mechanisms on the target head: (a) pretrained features favour adults (B->F); (b) multi-task sharing hurts child/teen mispronunciation F1 (-0.024 for GN15, -0.071 for full); fine-tuning itself is neutral. Check 2 adds that the harm shows up mainly as miscalibration at a shared threshold (calibration part 0.281 of 0.338), with only a small, non-significant ranking deficit.
- Candidate fixes should therefore be ordered by this evidence: (1) group-aware score/threshold calibration or child-weighted loss for the target head, which attacks the dominant calibration component; (2) loss reweighting or head-specific treatment on the multi-task side, which attacks the shared-representation component. Fine-tuning intensity and FT-stab-style stabilization are not indicated for this head.
- Any group-aware threshold needs the age group at inference, or a speaker-independent proxy. This must be stated as a deployment condition in the 8.3 case study, and selection must happen by speaker-grouped CV, not on test.
- The target-head F1 gap vs MT-GN15 is the number to move. Success is judged by the pre-declared acceptance rules in `docs/sprint8_plan.md`.

## 5. Caveats

- Holm-significant evidence for mechanism (b) is not available (p=0.12); the child F1 drop CI excludes zero but the gap change does not survive Family B correction. Present as suggestive.
- Check 2 covers FT-base, FT-stab, MT-full, MT-GN15 only (B and F have no probability export). Calibration conclusions use a pooled operating point chosen on all test speakers.
- One seed per system in these bundles for the ablation contrasts.

## 6. Update after US 8.3 (2026-11-07)
The calibration reading above did not survive a direct test. Per-age-group decision thresholds, fit on out-of-fold probabilities, do not reduce the macro-F1 gap of the target head (gap 0.124 -> 0.132), and the child/teen thresholds are unstable across folds. The calibration part of the recall gap is a statement about recall at a given false-positive rate; at the F1-optimal operating point the child/teen deficit comes mainly from rarer positives and weaker ranking. The group-aware reweighting and the child-like augmentation of US 8.3 did not help either. See `docs/us83_fix.md`.
