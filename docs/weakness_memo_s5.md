# Sprint 5 weakness memo (US 5.2): end-to-end fine-tuned WavLM-base-plus vs frozen 4.3

**Status: draft for review.** Based on `results/tables/finetune/us52_report.md` (tags `base` = US 5.1 runs, `base__seen` = seen-speaker diagnostic runs), the curve figures `results/figures/us52/curves_{primary,loss}_base.png`, and `us51_report.md`. US 5.3 (fixes) had not been run when this was written; the plan is in section 6.

Scope: one fine-tuned `wavlm-base-plus` per task (CNN frozen, learned layer-weighted sum, MLP head, encoder lr 3e-5, early stopping on validation, patience 4, max 15 epochs). SO762: 3 seeds on its single fold. L2-ARCTIC: 4 folds, 1 seed each. Reference: frozen WavLM-base-plus from US 4.3.

How the evidence was gathered, and its limits:

- The overfitting criteria C1-C5 and their thresholds were fixed in `configs/finetune.yaml` (`diagnose52`) **before** the diagnostic runs were inspected. They are reported as declared; nothing was retuned after looking (see 1.2 for what they miss).
- Decisions for US 5.3 use **validation** evidence only. Test numbers are used to report outcomes (paired speaker bootstrap from US 5.1), not to choose fixes.
- Validation is 19 speakers on SO762, so single-task differences of about 0.02-0.05 RMSE are inside the noise (`job_val_sd` is 0.002-0.022 across seeds, which does not include the speaker-sampling noise).
- The seen-speaker runs train on 80 % of each training speaker's utterances, so their numbers differ slightly from `base`. They are only compared with each other (seen vs validation in the same tag).
- The train score is a training-subset evaluation in eval mode (no dropout), not the running training loss.

## 1. Is there overfitting? (curves, train-vs-validation gaps, pre-declared flags)

### 1.1 What the pre-declared rule says

Risk is **low for all nine tasks** (high needs 3 criteria, medium 2). Criteria that fired:

- **C1** (relative train-vs-validation gap at the best epoch >= 0.30): sent_accuracy 0.310, sent_fluency 0.360, l2_phone_error 0.315. Close misses: mispronunciation 0.269, phone_score 0.250, sent_total 0.236, sent_prosodic 0.215.
- **C2** (validation loss >= 25 % above its minimum at the last epoch while training loss keeps falling, in >= 50 % of jobs): l2_word_error only (val loss +83 %, train loss -58 % since the best epoch).
- **C3** (best epoch <= 1 in >= 50 % of jobs), **C4** (< 2 % gain over the job's own head-only epoch in >= 50 % of jobs) and **C5** (seen-speaker gap) fired for **no task**.

### 1.2 What the rule misses, and what the curves show (post hoc reading, labelled as such)

The curves tell a stronger story than the flags, for three reasons I can state without changing the criteria:

1. **Early stopping hides the gap.** The gap at the _selected_ checkpoint is moderate, but at the last epoch it is large for the regression tasks (gap_last 0.49-0.64 for the four sentence scores and phone_score; 0.53 for word_accuracy). Training RMSE falls to 0.2-0.4 (sentence) and 0.15 (phone_score) while validation RMSE stays at 0.64-1.0 and 0.31-0.33. word_accuracy's selected epoch is 1.7 on average, and one of three seeds chose the head-only epoch, so for that task early stopping is effectively "barely fine-tune".
2. **For classification, validation loss and validation macro-F1 decouple.** Validation F1 keeps rising slowly to the epoch cap while validation loss climbs steadily after epoch 1-3: mispronunciation 0.41 -> 1.2 (+162 % over the minimum), l2_phone_error about 0.9 -> 2.0-2.3 (+124 %), l2_word_error +83 %. Train F1 at the same time goes to 0.97 / 0.79 / 0.84 against validation 0.69 / 0.54 / 0.72. The model is increasingly over-confident while its argmax decisions still improve. C2 does not fire for mispronunciation and l2_phone_error because it needs training loss to keep falling _after the best epoch_, and for F1-selected checkpoints the best epoch is late (mean 10.7 and 11.8 of 15). That is a blind spot of C2 for F1-selected tasks, not evidence of health.
3. **The runs for mispronunciation and l2_phone_error end at the 15-epoch cap** (mean epochs run 13.3 and 15.0; best epochs of 13-14 in the figures). Validation F1 may not have converged. Whether more epochs help is open and is a separate question from regularisation.

Summary judgement: **overfitting is present in all nine tasks at later epochs, and early stopping is what keeps it from reaching the test metric.** The strongest evidence is for the sentence tasks (C1 on two of four, large late gaps), the two L2-ARCTIC classification tasks and mispronunciation (loss divergence, over-confidence), and l2_word_error (C2). The weakest is word_accuracy, where the fix is "more fine-tuning", not less.

### 1.3 Training instability (visible in the figure, not in the tables)

In the sent_total panel one of three seeds shows a sharp spike at epoch 5 in both training loss (about 0.2 -> 0.95) and validation loss, with train RMSE jumping to about 1.9. It recovered, and the selected epoch (4) was before it, so no headline number was affected. The cause was not diagnosed (candidates: encoder lr too high for that seed after warmup, bf16). It is the reason the lr / grad-clip settings are on the US 5.3 list, and `sent_total`'s low mean `train_loss_drop` (0.08) is a side effect of it.

## 2. Seen vs held-out speakers

Held back 20 % of each training speaker's utterances (never trained on) and scored them as "seen speaker, unseen utterance". Gap = validation minus seen for RMSE (seen minus validation for F1), positive = seen speakers are easier.

| task                    | train subset | seen  | held-out val | rel. gap | 95 % CI (abs. diff) | CI excludes 0 |
| ----------------------- | ------------ | ----- | ------------ | -------- | ------------------- | ------------- |
| sent_accuracy           | 0.537        | 0.918 | 0.969        | +0.052   | [-0.166, 0.249]     | no            |
| sent_fluency            | 0.405        | 0.705 | 0.644        | -0.094   | [-0.158, 0.047]     | no            |
| sent_prosodic           | 0.391        | 0.659 | 0.644        | -0.023   | [-0.117, 0.108]     | no            |
| sent_total              | 0.502        | 0.841 | 0.907        | +0.073   | [-0.084, 0.223]     | no            |
| word_accuracy           | 1.316        | 1.605 | 1.587        | -0.012   | [-0.459, 0.457]     | no            |
| phone_score             | 0.182        | 0.327 | 0.317        | -0.032   | [-0.112, 0.104]     | no            |
| phone_mispron_t1.0 (F1) | 0.974        | 0.728 | 0.687        | +0.057   | [-0.032, 0.155]     | no            |
| l2_phone_error (F1)     | 0.794        | 0.578 | 0.544        | +0.059   | [-0.004, 0.076]     | no            |
| l2_word_error (F1)      | 0.844        | 0.747 | 0.716        | +0.041   | [0.005, 0.057]      | **yes**       |

Reading:

- **The train-vs-validation gap is not a speaker effect.** Training utterances score far better than unseen utterances of the _same_ speakers (sent_accuracy 0.54 vs 0.92; mispronunciation F1 0.97 vs 0.73; l2_phone 0.79 vs 0.58), while those unseen utterances are about as hard as held-out speakers' (rel. gap within +/-0.09, and C5 fired nowhere). The model memorises training _utterances_, not speaker identity.
- The one significant case is l2_word_error, and it is small (+0.030 F1, 4 %); l2_phone_error is borderline (CI [-0.004, 0.076]).
- Power is low: the sentence-task CIs are about +/-0.2 RMSE, so only a speaker effect of roughly 15-25 % could have been detected. "No speaker effect" means "no large effect", not "none".
- **Text overlap.** SO762 held-out sentences never appear in training (0.0 % seen / validation, 1.5 % test), so the SO762 gaps are not a text effect. In L2-ARCTIC 95-98 % of held-out utterance texts also occur in training (all speakers read the same prompts), so the L2 numbers measure generalisation to new _speakers_, not to new sentences. This is a scope statement for the final report.

Consequence for US 5.3: the evidence supports generic regularisation (dropout, weight decay, lower encoder lr, layer freezing, SpecAugment-type masking). It does **not** motivate speaker-invariance augmentation (speed / pitch perturbation) as a first move.

## 3. Where fine-tuning helped, did nothing, or hurt (vs frozen 4.3, same encoder)

Test (paired speaker bootstrap, from US 5.1) and validation (difference of the validation primary metric against the spread over jobs):

| task                    | frozen test | fine-tuned test | delta (95 % CI)         | test verdict | val delta  | job sd | validation flag  |
| ----------------------- | ----------- | --------------- | ----------------------- | ------------ | ---------- | ------ | ---------------- |
| sent_accuracy           | 1.113       | 1.071           | -0.042 [-0.067, -0.020] | helped       | **+0.014** | 0.008  | **worse on val** |
| sent_fluency            | 0.915       | 0.870           | -0.045 [-0.073, -0.024] | helped       | -0.021     | 0.002  | -                |
| sent_prosodic           | 0.901       | 0.902           | +0.001 [-0.017, 0.018]  | unchanged    | -0.035     | 0.018  | -                |
| sent_total              | 1.114       | 1.066           | -0.048 [-0.076, -0.022] | helped       | **+0.013** | 0.008  | **worse on val** |
| word_accuracy           | 1.633       | 1.598           | -0.035 [-0.069, -0.010] | helped       | -0.080     | 0.022  | -                |
| phone_score             | 0.314       | 0.306           | -0.007 [-0.017, 0.000]  | unchanged    | -0.022     | 0.009  | -                |
| phone_mispron_t1.0 (F1) | 0.661       | 0.680           | +0.018 [0.009, 0.028]   | helped       | +0.049     | 0.007  | -                |
| l2_phone_error (F1)     | 0.523       | 0.559           | +0.036 [0.018, 0.055]   | helped       | +0.038     | 0.031  | -                |
| l2_word_error (F1)      | 0.716       | 0.721           | +0.005 [-0.002, 0.011]  | unchanged    | +0.011     | 0.017  | -                |

- **Helped (6):** sent_accuracy, sent_fluency, sent_total, word_accuracy, mispronunciation, l2_phone_error. **Unchanged (3):** sent_prosodic, phone_score (p = 0.07), l2_word_error. **Hurt on test: none.**
- **Validation regressions (2):** sent_accuracy and sent_total are slightly worse than frozen on validation (+0.014 / +0.013 RMSE, about 1.6 job sd) although they are significantly better on test (-0.042 / -0.048). I treat this as validation noise (19 speakers) plus the C1 gap on sent_accuracy, and keep both on the **watch list** for US 5.3: any fix there is judged on validation and must not move test the wrong way, and I will not pick checkpoints from the test result.
- **Low value of full fine-tuning:** sent_prosodic, phone_score and l2_word_error gain nothing on test; word_accuracy gains only 3.3 % in validation over its own head-only epoch (phone_score 7 %, l2_word_error 7 %). They cost 12 trainable layers for no measurable return, so partial freezing is a fair test there.
- **Groups (descriptive, not used to select fixes):** no age group or L1 gets significantly worse. Gains concentrate in child/teen speakers for sentence accuracy / total, word_accuracy and phone_score. L2 phone: Vietnamese +0.079, Mandarin +0.044, Spanish +0.032, Korean +0.006; Arabic and Hindi not significant (Hindi remains the weakest L1 at 0.453). l2_word_error: significant gains only for Mandarin (+0.009) and Vietnamese (+0.014).

### 3.1 A new weakness: low-score compression got worse

Regression to the mean (test), fine-tuned vs frozen:

| task          | system     | slope (pred on true) | PCC   | worst bin: true mean -> predicted |
| ------------- | ---------- | -------------------- | ----- | --------------------------------- |
| word_accuracy | frozen 4.3 | 0.200                | 0.415 | 2.84 -> 8.14                      |
| word_accuracy | fine-tuned | 0.168                | 0.477 | 2.84 -> 8.51                      |
| phone_score   | frozen 4.3 | 0.342                | 0.533 | 0.13 -> 1.31                      |
| phone_score   | fine-tuned | 0.275                | 0.567 | 0.13 -> 1.44                      |

Fine-tuning improves correlation and RMSE but **flattens the predictions further** (slope down 16-20 %, the lowest-score bin predicted even closer to the majority score). For a feedback system this is the costly direction: the words and phones that really are bad are exactly the ones predicted as fine. This was a known weakness in Sprint 4 (4.2) that fine-tuning did not fix and slightly worsened. I do not expect dropout / augmentation to fix it, so US 5.3 will re-measure the slope but not promise a fix (section 6).

## 4. Weakness list (what the reviewer asked to see flagged)

| #   | weakness                                               | evidence                                                                                              | affected                                                 | confidence                      |
| --- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------- |
| W1  | Train-validation overfitting, hidden by early stopping | C1 on 3 tasks; late gaps 0.49-0.64; train metric 0.2-0.4 vs val 0.64-1.0                              | 4 sentence tasks, phone_score, word_accuracy             | high (curves), medium (flags)   |
| W2  | Over-confident classification heads                    | val loss +83 % to +162 % after its minimum, while F1 still rises; train F1 0.79-0.97 vs val 0.54-0.72 | mispronunciation, l2_phone_error, l2_word_error          | high                            |
| W3  | No speaker-memorisation effect found                   | C5 nowhere; seen ~ held-out                                                                           | all                                                      | medium (low power)              |
| W4  | One unstable seed                                      | sent_total spike at epoch 5                                                                           | sent_total                                               | medium (one run, cause unknown) |
| W5  | Possible under-training at the 15-epoch cap            | best epoch 13-14 of 15                                                                                | mispronunciation, l2_phone_error                         | medium                          |
| W6  | Low value of full fine-tuning                          | test unchanged; gain over head-only 3-7 %                                                             | sent_prosodic, phone_score, l2_word_error, word_accuracy | medium                          |
| W7  | Validation-side regression vs frozen                   | +0.014 / +0.013 RMSE vs job sd 0.008                                                                  | sent_accuracy, sent_total                                | low (19 speakers)               |
| W8  | Low-score compression worse than frozen                | slope 0.200 -> 0.168, 0.342 -> 0.275                                                                  | word_accuracy, phone_score                               | high                            |
| W9  | L2 numbers do not test unseen text                     | 95-98 % prompt overlap                                                                                | l2_*                                                     | high (scope note)               |

## 5. Method limits to carry into the final report

- Criteria were declared first and not retuned, but C2 is ill-suited to F1-selected tasks (1.2); I report that instead of adjusting it. A future version should compare the validation-loss minimum with the F1-selected epoch.
- Validation (19 SO762 speakers, 4 speakers per L1 group) cannot resolve differences under about 0.02-0.05; a fix is only credited if it beats the job-to-job spread (`job_val_sd`) and is consistent across seeds.
- L2-ARCTIC has one seed per fold, so there is no seed spread there.
- The seen-speaker runs have 20 % less training data than `base`.
- Mispronunciation uses validation-tuned `cw+thr` for fine-tuned models (no out-of-fold probabilities), unlike the 4.3 OOF age-group weights.

## 6. Plan for US 5.3 (fix only where confirmed, then re-measure)

Every fix is chosen on validation, run with the same seeds and splits, and reported as a before/after table with a paired speaker bootstrap on test and the job spread on validation. Success criteria are set now, before any run:

| weakness                       | fix, in order of trial                                                                                                                                                                                                                                 | target (validation)                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| W1 sentence tasks, phone_score | (a) lower encoder lr / layer-wise lr decay; (b) head dropout 0.2 -> 0.4 and weight decay 0.01 -> 0.1; (c) HF SpecAugment-style time / feature masking with a short mask length, no waveform speed / pitch; (d) partial freezing of the lowest k layers | gap_best < 0.30 and last-epoch val loss <= 1.25x its minimum, **with** the validation metric no worse than 5.1 by more than one job sd |
| W2 classification heads        | (a) lower encoder lr / freeze lower layers; (b) dropout / weight decay; (c) feature masking with phone-safe lengths (units are 3-10 frames); then (d) more epochs (cap 25) only if the head stays calibrated                                           | last-epoch val loss <= 1.5x its minimum, val F1 not below 5.1 minus one job sd                                                         |
| W4 sent_total                  | gradient clip 0.5 and lower peak encoder lr; check all three seeds                                                                                                                                                                                     | no loss spike above 2x the previous epoch in any seed                                                                                  |
| W6                             | partial freezing (k = 6, 9) on sent_prosodic, phone_score, l2_word_error, word_accuracy                                                                                                                                                                | within one job sd of full fine-tuning at lower cost                                                                                    |
| W3                             | speed / pitch augmentation **only if** (a)-(d) leave a confirmed gap, and only on tasks where labels survive (not the fluency / prosodic sentence scores, not durations)                                                                               | not a first move                                                                                                                       |
| W8, W9, W7                     | re-measure slope and low-bin bias; no dedicated fix promised for W8 (a loss that re-weights low scores is a Sprint 6 candidate); W7 re-checked with the new config                                                                                     | report the numbers either way                                                                                                          |

Decision rule per task: keep the stabilized config if it meets its target without losing validation performance; otherwise keep 5.1 and state that the weakness was not fixable by the tried regularisers.
