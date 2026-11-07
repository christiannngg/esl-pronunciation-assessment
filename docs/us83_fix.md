# US 8.3: generalization fix for the weakest subgroup (case study)

**Outcome: negative result. The fix was not accepted. MT-GN15 stays the model of record and the demo model.** The weakness is real, it was diagnosed (US 8.1, 8.2), two kinds of fix were screened by cross-validation, the one eligible candidate was tested once on the held-out test speakers, and it made the target head worse.

Status: complete. Pre-declared design and every choice made before a run: `docs/sprint8_plan.md` sections 5 and 9. Code: `src/models/fix_s8.py`, `src/evaluation/report_us83.py`, `run_83_lab.sh`. Tables: `results/tables/fix/`.

## 1. The weakness
Subgroup SO762 child/teen speakers, target head `phone_mispron_t1.0` (the only Holm-significant adverse age gap in US 8.1). On the test speakers, the control model (MT-GN15 re-run, 3 seeds) scores macro-F1 0.717 for adult speakers and 0.597 for child/teen speakers (error 0.283 vs 0.403, gap 0.120). US 8.2 found: the compression of child/teen scores is real and not label noise; fine-tuning does not change the gap; multi-task sharing costs the child/teen group on this head (-0.024 F1); recall at the shared threshold is far lower for child/teen speakers, mostly through the calibration part of the recall gap.

## 2. What was tried (declared before any run)
All candidates start from MT-GN15 (GradNorm alpha 1.5, 7 heads) and change one thing:
- **F1, group-aware reweighting** (strengths 0.5 and 1.0): the loss of every head is reweighted by age group within each label, keeping the class-balanced weights.
- **F2, child-like augmentation** (shifts of 2-4 and 4-6 semitones): pitch- and formant-shifted copies of adult training utterances (duration preserved exactly, no time stretch), swapped into 50% of adult training samples.
- F3 (age adversary) was a stretch goal and was not run.

## 3. Selection by cross-validation (test speakers never read)
Speaker-grouped 5-fold CV over the 125 train+val speakers, stratified by age group, one fit per candidate and fold (25 fits, no failures). Change in the child/teen error of the target head against the control, pooled out-of-fold, with a paired speaker bootstrap (negative = better; margin 0.017):

| candidate | child/teen change | 95% interval | other heads | eligible | qualifies |
|---|---|---|---|---|---|
| F1 (0.5) | -0.003 | [-0.022, +0.015] | `sent_total` worse by 0.051 (margin 0.033) | no | no |
| F1 (1.0) | +0.015 | [-0.004, +0.040] | `sent_total` worse by 0.070 | no | no |
| F2 (2-4 st) | 0.000 | [-0.023, +0.024] | none | yes | no |
| F2 (4-6 st) | +0.009 | [-0.015, +0.036] | `sent_total` worse by 0.044 | no | no |

No candidate beat the control by more than the margin with an interval excluding 0. By the pre-declared rule, the one eligible candidate (F2 2-4 st) was evaluated once on test, to be reported as inconclusive or negative.

## 4. The one test evaluation (3 seeds each, official split)
Error = 1 - macro-F1 (target head) or RMSE (other heads). Change = fix - control; positive = the fix is worse. 95% paired speaker-bootstrap intervals.

| head | overall | adult | child/teen | gap change (child/teen - adult) |
|---|---|---|---|---|
| **phone_mispron_t1.0 (target)** | **+0.038 [+0.016, +0.061]** | **+0.027 [+0.006, +0.054]** | **+0.062 [+0.021, +0.100]** | +0.035 [-0.017, +0.075] |
| sent_accuracy | -0.009 [-0.025, +0.009] | -0.019 [-0.047, +0.004] | +0.001 [-0.021, +0.025] | +0.021 [-0.010, +0.055] |
| sent_fluency | +0.011 [-0.001, +0.022] | +0.010 | +0.011 | +0.001 |
| sent_prosodic | +0.011 [+0.001, +0.021] | +0.016 [+0.002, +0.029] | +0.007 | -0.009 |
| sent_total | +0.007 [-0.005, +0.019] | +0.003 | +0.011 | +0.008 |
| word_accuracy | -0.008 [-0.022, +0.008] | -0.022 [-0.042, -0.002] | +0.011 | +0.033 [+0.006, +0.062] |
| phone_score | -0.003 [-0.006, 0.000] | -0.006 [-0.011, -0.003] | +0.002 | +0.008 [+0.003, +0.014] |

In F1 terms the target head goes from 0.690 to 0.651 overall, and the child/teen F1 from 0.597 to about 0.535.

**Acceptance rules (all three failed):** (i) the child/teen target-head metric did not improve; it got significantly worse. (ii) Two heads are significantly worse overall: the target head and `sent_prosodic` (+0.011 RMSE). (iii) The adult metric on the target head is significantly worse too. Verdict: not accepted; MT-GN15 stays. Intervals are uncorrected for multiple testing (Family B Holm applies only to the US 8.1/8.2 contrasts); the target-head result is the pre-declared primary and its intervals exclude 0 by a wide margin; the rest is descriptive.

Sanity check of the control: the re-run control (3 seeds, same code path) reproduces the published MT-GN15 within run-to-run variation (target-head error 0.310 vs 0.319; regression heads within about 0.02 RMSE).

## 5. Reference: would a per-age-group threshold have fixed it?
Not selectable (the demo does not know the learner's age), reported to test the calibration reading of US 8.2. Thresholds were fit on F0's out-of-fold probabilities, on other folds only (`us83_threshold_reference.csv`):

| rule | adult error | child/teen error | child/teen change vs default | gap |
|---|---|---|---|---|
| default (0.5) | 0.307 | 0.430 | - | 0.124 |
| one pooled threshold | 0.311 | 0.436 | +0.006 [-0.006, +0.015] | 0.126 |
| per-group thresholds | 0.310 | 0.442 | +0.012 [-0.000, +0.022] | 0.132 |

Per-group thresholds do not reduce the macro-F1 gap, and the chosen child/teen thresholds are unstable across folds (0.20 to 0.61). **This corrects the implication of the US 8.2 memo.** The calibration part of the recall gap is real, but it is a statement about recall at a given false-positive rate. The F1-optimal operating point for child/teen speakers does not move to a lower threshold: positives are rare in that group (about 3% against 5% for adults) and ranking is worse (AP 0.235 against 0.436), so lowering the threshold costs precision about as fast as it gains recall. Equalizing recall is not the same as closing the F1 gap.

## 6. What this means
- **Honest claim for the report:** a subgroup weakness was found, measured, diagnosed, and two inexpensive fix families were tested under a protocol fixed in advance. Neither helped; the best candidate hurt. The weakness is documented as a limitation of the delivered model, with its size and interval.
- **Hypotheses for why (not tested here):** pitch/formant-shifted adult speech is not child speech, and mispronunciation detection depends on fine phone-level acoustic detail, so synthetic copies may add noise; group reweighting up-weights a small number of child/teen positives that the model may fit unevenly. The 8.2 evidence points to scarce child/teen mispronounced-phone examples and weaker ranking, which a loss reweighting cannot create.
- **What would plausibly help (future work, untested):** more real child/teen speech with phone-level labels, or an age-aware component trained on such data.

## 7. Limits
- One seed per CV fit; CV has 58 child/teen speakers, the test set 64.
- Five candidates were screened on CV; none won, so there is no winner's-curse issue, but a different candidate list might have behaved differently.
- The control for the test comparison was re-run (3 seeds) rather than reusing the US 6.3 predictions; both are reported.
- F3 (age adversary) was not tried; the decision-layer threshold analysis is a reference only.

## 8. Reproduce
Folds: the committed `data/splits/so762_cv_s8.csv` (= `fix_s8.make_folds()`); on the lab `bash run_83_lab.sh cv all`, `select`, `reference`, `final`; locally `python -m src.evaluation.report_us83 final` (needs the pushed `results/analysis/us71/S8-*.npz`).
