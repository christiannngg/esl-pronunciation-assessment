# Sprint 8 plan: subgroup generalization, diagnosis, one fix (US 8.1 - 8.3)

**Status:** fixed on 2026-11-03, BEFORE any subgroup number was computed. Anything that changes after this point is listed in section 9 with the reason.
Context: `docs/revised_scope.md` (D3, D4, D5), `docs/go_forward_plan.md` (section 2), `docs/midterm_results_report.md`.
**Model freeze: Sun 2026-11-08.** Lab access ends 2026-12-12 (all lab work is done and copied off long before that).

## 1. What was verified in Step 0

- The saved prediction bundles (`results/analysis/us71/*.npz`: `y`, `pred`, `speaker`) for B, F, FT-base and MT-GN15 match the test manifest row for row: for all 9 tasks the speaker sequence and labels equal those rebuilt from `data/processed/features/tables/*_meta.csv.gz` (SO762: rows with split = test, in table order; L2-ARCTIC: test rows of folds 0 to 3, concatenated in that order). Subgroup labels (age group, L1, word / phone identity, utterance id) can therefore be joined by reconstructing the row indices; the bundles do not need to change.
- Test sizes (speakers): SO762 61 adult, 64 child/teen (sentence 1,220 / 1,280 utterances; word 8,701 / 7,266; phone 25,752 / 21,617). L2-ARCTIC 4 speakers per L1 in every L1 (6 L1s, 24 speakers).
- Missing systems: FT-stab and MT-full (SO762 and L2) have no bundles because their job files live on the lab. `report_us71_pairs.py` now has exporters for them (`FT-stab`, `MT-full`); same reproduction check against the saved tables as the existing exporters. Run on the lab: `python -m src.evaluation.report_us71_pairs export --systems FT-stab MT-full`, then copy the npz files back.

## 2. Systems analysed

Single-task reference per task (FT-base; FT-stab for sent_total and l2_phone_error, as carried forward in US 7.1), MT-GN15 (SO762 heads), MT-full (reference). L2-ARCTIC: FT-stab and MT-full. B and F appear for context and for the ablations.

## 3. US 8.1 design (subgroup generalization)

**Subgroups.** Primary: SO762 age group (adult, child/teen). Descriptive only: L1 on L2-ARCTIC (4 speakers per L1, so per-speaker values are shown next to the intervals and no L1 test is run); SO762 gender; utterance length (terciles of duration or unit count); true-score bin; speaker-level proficiency as the accent-strength proxy (SO762: speaker mean human accuracy; L2-ARCTIC: speaker error rate from `l2arctic_error_rates_by_speaker.csv`). The proxy is derived from labels, so it is range-restricted by construction and is read as descriptive.

**Metrics.** Primary metric inside a subgroup is the project's headline metric (RMSE for regression, macro-F1 for classification). Diagnostics, labelled exploratory: PCC, mean bias, slope of prediction on truth, per-group human ceiling (annotator agreement) for regression; AUC, AP and recall at the pooled operating point for classification.

**Statistics.** Speaker-level bootstrap with the settings of `configs/baseline.yaml` (`bootstrap`). Paired within a subgroup for system comparisons. The group gap is `metric(child/teen) - metric(adult)`, resampling speakers within each group independently (RMSE: positive = child/teen worse; macro-F1: negative = child/teen worse).

**Pre-declared families (Holm-Bonferroni within each family, alpha 0.05, both raw and adjusted p reported):**

- Family A, does the gap exist: the age gap of MT-GN15 on the 7 SO762 heads (7 tests).
- Family B, what changes it: change in the age gap for B to F, F to FT-base, FT-base to MT-GN15, on the 7 SO762 heads (21 tests).
- Everything else (L1, gender, length, score bin, proxy, L2 heads, other systems) is exploratory and labelled so.

## 4. US 8.2 design (diagnosis)

Four checks, in this order, each with its own table:

1. **Is the gap real?** Per-group PCC against the per-group human ceiling and against the within-group label spread (range restriction lowers PCC without any model failure).
2. **Prior shift or ranking?** For `phone_mispron_t1.0`: AUC and AP per group against recall at the pooled threshold. Similar AUC with near-zero recall is calibration; lower AUC is ranking (US 4.3 found AUC 0.80 to 0.83 child/teen against 0.89 to 0.90 adult, so some ranking loss is expected).
3. **Bias and compression by group** (mean error, slope, worst score bin), for the regression heads.
4. **Is age decodable from the encoder output?** Linear probe on pooled utterance embeddings, speaker-grouped CV. Only informs whether an adversary has something to remove; it does not prove an adversary helps.

Ablation table: Family B contrasts (representation, fine-tuning, sharing) and the same on the L2 heads as exploratory. Output: `docs/weakness_memo_s8.md`.

**Target selection rule.** Target subgroup is SO762 child/teen (agreed with the user, 2026-11-03). Target heads = heads with a Holm-significant adverse child/teen gap in Family A for MT-GN15. If none qualifies, the target head is `phone_mispron_t1.0` (documented since Sprint 4) and the null is reported with intervals.

## 5. US 8.3 design (fix)

**Base model.** MT-GN15 (the model the demo serves, D6). Everything not named below stays as in US 6.3.

**Candidates, fixed before any run (at most 2 settings each, to limit winner's curse; GradNorm had 2):**

- F0, control: MT-GN15 unchanged, re-run inside the CV so that it is compared like for like.
- F1, group-aware reweighting: sample weights from the inverse frequency of (age group x label), and per-group class weights for mispronunciation. Two strengths.
- F2, child-like augmentation of adult training audio (vocal-tract-length or pitch perturbation, precomputed once to the lab scratch disk because the job has 8 CPU cores). Two strengths. First check what US 5.3 already tried, so nothing is repeated.
- F3, gradient-reversal age adversary: stretch, only if F1 and F2 are done by Fri Nov 6 midday. One setting.

**Selection protocol.** Speaker-grouped 5-fold CV over the 125 train+val speakers, stratified by age group, fold file saved to `data/splits/so762_cv_s8.csv` for audit, one seed per fit. The test set is not read during selection. Selection statistic: mean over target heads of the child/teen metric, relative to F0, from the pooled out-of-fold predictions. A candidate is eligible if no non-target head is worse than F0 by more than `max(0.01, fold sd)` in the overall metric and the adult-group metric on the target heads is not worse by more than that margin. The winner must beat F0 by more than that margin with a paired out-of-fold speaker-bootstrap interval that excludes 0. Ties go to the fewest extra mechanisms (F1, then F2, then F3). If nothing qualifies, no fix is selected: the single best candidate is still evaluated once on test, reported as inconclusive or negative, and MT-GN15 stays.

**Test evaluation (once).** Winner and control, 3 seeds each, trained on the train split exactly as US 6.3 (validation for early stopping). The fix is accepted only if all hold: (i) on at least one target head the child/teen metric improves with the paired 95% interval excluding 0, and none is significantly worse; (ii) no SO762 head is significantly worse overall; (iii) the adult metric on the target heads is not significantly worse. Accepted: the fixed model replaces MT-GN15 for the demo and the final report. Not accepted: MT-GN15 stays and the case study is reported as a negative result with the diagnosis. Before and after reports the child/teen metric, the adult metric and the change in the age gap, each with its interval.

**Compute.** One GPU per person, 20 queued jobs, 2 days at most: the CV grid runs as one sbatch script that loops over configs (about 5 configs x 5 folds, plus 3 final seeds), asking for a realistic time limit.

## 6. Disclosures to carry into the report

Number of candidates screened, the seed count (1 in CV, 3 on test), the small child/teen validation count (9 speakers; hence CV), Holm applies only to Families A and B, L1 and the other slices are descriptive, and the age gap in RMSE is partly a label-spread effect (check 1).

## 7. Outputs

`docs/generalization_report.md`, `results/tables/generalization/`, `docs/weakness_memo_s8.md`, `docs/us83_fix.md` and `docs/us83_fix_summary.md`, `src/evaluation/report_us81.py`, `report_us82.py`, `report_us83.py` with tests, `results/analysis/us8*/`.

## 8. Schedule

| day       | work                                                                      |
| --------- | ------------------------------------------------------------------------- |
| Tue Nov 3 | Step 0 (done); export FT-stab / MT-full on the lab; start the 8.1 builder |
| Wed Nov 4 | 8.1 report                                                                |
| Thu Nov 5 | 8.2 memo and ablations; choose the target; build and submit the CV grid   |
| Fri Nov 6 | CV results and selection; weekly video 1                                  |
| Sat Nov 7 | Final seeds, one test evaluation, 8.3 write-up                            |
| Sun Nov 8 | Freeze; copy checkpoints, predictions and tables off the lab              |

If the CV grid slips, the freeze can move by up to two days: the Sprint 9 app only needs the MT-GN15 interface.

## 9. Changes after this date

- **2026-11-03, AUC / AP moved to US 8.2.** The prediction bundles hold hard predictions only, so the threshold-free diagnostics of check 2 cannot be computed from them. US 8.2 needs the export extended to save probabilities for `phone_mispron_t1.0` (and the other classification heads if wanted). The US 8.1 tables report recall, precision and positive rate instead.
- **2026-11-03, bootstrap implementation.** The 8.1 tables use per-speaker sufficient statistics (`src/evaluation/report_us81.py`) with the same conventions as `metrics.paired_speaker_bootstrap` (speaker resampling, percentile CI, p = 2 min(P(d <= 0), P(d >= 0)), 500 draws, seed from `configs/baseline.yaml`) but a different random stream. US 7.1 intervals are unchanged. The overall metrics are checked against the US 7.1 table before any file is written.
- **2026-11-03, result of Family A, recorded without changing the rule.** Only `phone_mispron_t1.0` has a Holm-significant adverse gap. The regression heads show strong child/teen compression (slope, PCC) that the pre-declared RMSE family does not detect; whether to add them as targets is an open decision (`docs/generalization_report.md`, section 5). Until decided, the target rule in section 4 stands.

- (US 8.2) Probability export limited to FT-base, FT-stab, MT-full, MT-GN15 (B and F lack it). Label-noise argument uses Spearman-Brown reliability. Age probe (check 4) is optional, only if the embedding cache exists on the lab.
- (US 8.2) Check 2 outcome reorders the 8.3 candidate priorities toward calibration-type fixes; selection is still by speaker-grouped CV with the pre-declared target head, so the rule is unchanged.

### US 8.3 precisions, logged on Wed 2026-11-04 BEFORE any 8.3 job was run

- **Implementation of the candidates** (`src/models/fix_s8.py`, `configs/multitask.yaml` variants `s8_*`). F0 = MT-GN15 (gradnorm alpha 1.5, recipe `base`, 7 heads). F1 = age-group reweighting of every head's loss *within each label* (weight = share of the row's age group among training rows with the same label, to the power -strength, rescaled so each label keeps its total weight; regression heads use training-label terciles as labels); the control's class-balanced weights are kept. Strengths 0.5 and 1.0. F2 = pitch+formant-scaled copies (phase-vocoder pitch shift, duration preserved exactly, no time stretch) of adult training utterances, shifts drawn once per utterance from [2,4] or [4,6] semitones upward, swapped in with probability 0.5 in training batches; adult train+val speakers only. F3 not defined or run unless F1 and F2 finish by Fri Nov 6 midday.
- **CV protocol details.** Folds: `data/splits/so762_cv_s8.csv` (125 train+val speakers, 5 folds stratified by age group, seed 8). Inside fold k, 15% of the other speakers (per age group) are `val` for early stopping and the rest `train`; the 125 official test speakers are `excluded` (never read). One seed (0) per fit. Out-of-fold predictions are the argmax predictions of each fit.
- **'Fold sd' made precise:** margin of head h = max(0.01, sd over the 5 folds of F0's per-fold overall error of h). The same margin applies to the adult check on the target head. 'Beats F0' = child/teen error of the target head lower than F0's by more than the margin, with the paired speaker-bootstrap 95% interval (settings of `configs/baseline.yaml`) excluding 0.
- **Tie rule made precise:** qualifying candidates whose child/teen improvement is within the margin of the best are tied; fewest extra mechanisms wins, in the order F1(0.5), F1(1.0), F2(a), F2(b), F3.
- **Reference analysis, not a selectable candidate (decided after 8.2 check 2).** A per-age-group decision threshold on the target head, fit on F0's out-of-fold probabilities, is reported as a reference for how much of the recall gap a calibration-only fix would remove. It needs the learner's age group at inference, which the demo does not have, so it is not eligible for selection. If the user prefers to make it a selectable candidate, that is a further logged change before the CV results are read.
