# Sprint 6 weakness memo (US 6.2): negative transfer and task imbalance in the shared-encoder multi-task models

**Status: draft for review.** Based on `results/tables/multitask/us61_report_mt_base.md` and `us61_report_mt_lr.md` (US 6.1), `us62_report.md` with its csvs, and `results/figures/multitask/us62_gradients_{so762,l2}.png` (US 6.2). US 6.3 (fixes) has not been run; the plan is in section 7.

Scope: two multi-task models, each one shared `wavlm-base-plus` with one head per task. `so762`: 7 heads (4 sentence scores, word_accuracy, phone_score, phone_mispron_t1.0), recipe `base`, 3 seeds. `l2`: 2 heads (l2_phone_error, l2_word_error), recipe `lr`, 4 folds x 3 seeds. Each is compared with the single-task model trained with the **same recipe** (5.1 `base` for `so762`, the US 5.3 `lr` model for `l2`). Every other training setting is identical to Sprint 5.

How the evidence was gathered, and its limits:

- The diagnostic criteria (dominant / starved gradient share, conflict, stopping compromise, lagging) and their thresholds were fixed in `configs/multitask.yaml` (`compare62`) **before** the gradient diagnostics were looked at. They are reported as declared and were not retuned.
- What had already been seen: the US 6.1 test table. The suspicion that `phone_mispron_t1.0` is the hurt head and that one shared stopping epoch is a compromise come from it. They are hypotheses that 6.2 tests, not independent findings.
- **The ablation `word_phone` was added after the first diagnosis** (the block structure of the gradient cosines suggested it), and it replaced `drop_wordacc`. The other two ablations (`phone_pair`, `w_mispron3`) were declared before any diagnosis.
- **Ablation comparisons are on the test set** (paired speaker bootstrap, 3 seeds, same speakers). They are used here to _understand_ a weakness. The choice of a fix in US 6.3 is made on validation (section 7), and the dependence of that choice on these test readings is stated there.
- The `reading` column (recovers / partly / hurts / no effect) in `us62_report.md` is only meaningful for a head with negative transfer, i.e. `phone_mispron_t1.0`. On the other rows it is a mechanical label and is ignored below.
- SO762 has 19 validation speakers, so differences below about 0.02-0.05 RMSE are inside the noise. Confidence intervals are not corrected for the many heads compared.
- Gradients are measured on one layer (the last trainable transformer layer) and are averaged over sampled steps within each epoch.

## 1. Which heads are hurt? (test, matched single-task comparator)

| group | head                        | single-task | multi-task | delta [95 % CI]             | verdict                   |
| ----- | --------------------------- | ----------- | ---------- | --------------------------- | ------------------------- |
| so762 | sent_accuracy               | 1.071       | 1.070      | -0.001 [-0.020, 0.017]      | no clear change           |
| so762 | sent_fluency                | 0.870       | 0.887      | +0.017 [-0.002, 0.035]      | no clear change           |
| so762 | sent_prosodic               | 0.902       | 0.914      | +0.011 [-0.009, 0.031]      | no clear change           |
| so762 | sent_total                  | 1.066       | 1.055      | -0.011 [-0.029, 0.006]      | no clear change           |
| so762 | word_accuracy               | 1.598       | 1.598      | -0.001 [-0.034, 0.033]      | no clear change           |
| so762 | phone_score                 | 0.306       | 0.303      | -0.003 [-0.009, 0.002]      | no clear change           |
| so762 | **phone_mispron_t1.0** (F1) | 0.680       | 0.654      | **-0.026 [-0.045, -0.009]** | **negative transfer**     |
| l2    | l2_phone_error (F1)         | 0.581       | 0.577      | -0.005 [-0.007, -0.002]     | negative transfer (small) |
| l2    | l2_word_error (F1)          | 0.722       | 0.720      | -0.002 [-0.006, 0.001]      | no clear change           |

- **No head shows positive transfer** against its matched single-task model. The apparent gain of `l2_phone_error` over the 5.1 `base` model (+0.013) is the `lr` recipe, not sharing: against the 5.3 `lr` model it is -0.005.
- **Validation and test disagree on several SO762 heads.** `sent_accuracy` / `sent_total` are better on validation by 0.036 / 0.041 RMSE but unchanged on test; `sent_fluency`, `sent_prosodic`, `word_accuracy` are worse on validation (by 0.025-0.066) but unchanged on test. `phone_mispron_t1.0` is within the validation margin although it is significantly worse on test. This is what 19 validation speakers look like; it is also why validation alone would have missed the one real loss.
- The L2 loss (-0.005 macro-F1) is statistically clear (12 jobs) but practically negligible.

## 2. Task imbalance and gradient dominance (last shared transformer layer)

Share of the summed gradient norms of the unweighted head losses (1/K = balanced; K = 7 for SO762, 2 for L2):

| head                   | share     | sd over seeds | x balanced | label               |
| ---------------------- | --------- | ------------- | ---------- | ------------------- |
| sent_accuracy          | 0.082     | 0.035         | 0.58       | balanced            |
| sent_fluency           | 0.254     | 0.152         | 1.78       | balanced            |
| sent_prosodic          | 0.244     | 0.044         | 1.71       | balanced            |
| sent_total             | 0.149     | 0.101         | 1.04       | balanced            |
| word_accuracy          | 0.096     | 0.061         | 0.67       | balanced            |
| phone_score            | 0.130     | 0.056         | 0.91       | balanced            |
| **phone_mispron_t1.0** | **0.045** | 0.014         | **0.32**   | **starved**         |
| l2_phone_error         | 0.354     | 0.052         | 0.71       | balanced (see note) |
| l2_word_error          | 0.646     | 0.052         | 1.29       | balanced (see note) |

- **The four sentence heads together hold 73 % of the gradient norm** (word_accuracy + the two phone heads: 27 %). Their targets are strongly correlated and their gradients are aligned (cosines 0.12-0.45), so they act as one block with 4/7 of the weight.
- **The word / phone heads are almost orthogonal to the sentence heads** (cosine 0.01-0.05) and aligned with each other (0.24-0.32). Nothing conflicts, but nothing is shared across the two levels either.
- **No conflict anywhere**: no pair has a mean cosine below -0.1 (all 21 SO762 pairs and the one L2 pair are positive; the lowest sampled value in any job is 0.000). Gradient-surgery methods (PCGrad and similar) have nothing to fix here.
- **`phone_mispron_t1.0` is the only starved head** (0.32 x balanced, consistent across seeds). It is also the head with the negative transfer.
- **Note on L2:** the word head carries about 65 % of the gradient (about 1.8:1; about 2.2:1 late in training). The declared "dominant" rule (> 2/K) can never fire when K = 2, so the label "balanced" says nothing about L2; I report the ratio and do not change the rule.
- **Learning progress** (validation error at the shared checkpoint / error of the head-only epoch): sentence heads 0.47-0.68, i.e. they gain a lot; word_accuracy 0.997, phone_score 0.947, phone_mispron 0.819, i.e. the word / phone heads barely improve over their frozen-feature starting point inside the shared model. All three are flagged `lagging`.

## 3. One shared checkpoint for all heads

| head               | shared epoch | head's own best epoch | val error shared | val error own best | regret    | flag           |
| ------------------ | ------------ | --------------------- | ---------------- | ------------------ | --------- | -------------- |
| sent_prosodic      | 7.7          | 8.0                   | 0.699            | 0.668              | 4.3 %     | compromise     |
| word_accuracy      | 7.7          | 3.0                   | 1.692            | 1.635              | 3.3 %     | compromise     |
| phone_mispron_t1.0 | 7.7          | 11.0                  | 0.337            | 0.327              | 2.8 %     | compromise     |
| other SO762 heads  | 7.7          | 4.0-6.7               |                  |                    | 0.7-1.5 % | none           |
| l2_word_error      | 13.3         | 7.8                   | 0.282            | 0.278              | 1.4 %     | none (lagging) |

- The regret is an **optimistic upper bound** on what per-head checkpointing could gain (each head's best epoch is chosen on the same validation data it is scored on). For `phone_mispron_t1.0` it is about 0.010 macro-F1 on validation, against a test deficit of 0.026: even under the optimistic bound, stopping accounts for **at most about a third** of the loss (comparison across validation and test, so rough).
- `phone_mispron_t1.0` also has the largest validation-to-training loss ratio of any head (9.6, against 2.1-4.2 for the others): the over-confident classification head of Sprint 5 (W2) is still there. Part of its small gradient share may simply be that it has already fit its training data.

## 4. Ablations: what removes the negative transfer on phone_mispron_t1.0? (SO762, recipe `base`, 3 seeds, test)

| ablation     | change                                                 | phone_mispron_t1.0 macro-F1 | vs full multi-task (0.654) [CI] | vs single-task (0.680) [CI] |
| ------------ | ------------------------------------------------------ | --------------------------- | ------------------------------- | --------------------------- |
| `phone_pair` | only phone_score + phone_mispron share the encoder     | **0.690**                   | **+0.036 [0.015, 0.060]**       | +0.010 [0.000, 0.020]       |
| `word_phone` | word_accuracy + the two phone heads, no sentence heads | **0.686**                   | **+0.032 [0.005, 0.062]**       | +0.007 [-0.009, 0.025]      |
| `w_mispron3` | all 7 heads, `phone_mispron_t1.0` loss weight x3       | 0.653                       | -0.002 [-0.007, 0.005]          | -0.027 [-0.048, -0.008]     |

Reading (the declared rules, applied to the one head with negative transfer):

- **Removing the sentence heads removes the loss.** `phone_pair` and `word_phone` both bring `phone_mispron_t1.0` back to the single-task level (`phone_pair` is borderline above it; I do not claim it exceeds it). Adding `word_accuracy` to the phone pair changes nothing visible (0.686 vs 0.690, not tested directly), so the word head is not the culprit. This points at **interference from the sentence heads**.
- **Making the head heavier does not help.** x3 on its loss weight leaves it at 0.653, still significantly below single-task. The loss is therefore **not simply a question of loss scale or gradient magnitude** (although only one weight, x3, was tried). The gradient share is a symptom, not the cause.
- **A cost on the other phone head:** `phone_score` is slightly worse than single-task in the reduced groups (0.309 / 0.310 vs 0.306; CI excludes 0 but it is about 1 %), while it is not worse in the full model (0.303). The reduced groups trade a little `phone_score` for `phone_mispron_t1.0`.
- **`w_mispron3` moved the other heads** (lower RMSE than the full model on sent_accuracy, sent_prosodic, sent_total, word_accuracy by 0.013-0.017, CIs exclude 0; `sent_total` also below single-task by 0.027 [0.009, 0.046]). I treat this as **exploratory**: many uncorrected comparisons, correlated sentence targets, 3 seeds, and no mechanism I can state. It is not evidence that multi-task helps, and it is rechecked in US 6.3 rather than used.
- `word_accuracy` under `word_phone`: 1.581 vs single-task 1.598, not significant.

## 5. Weakness list (what the reviewer asked to see flagged)

| #   | weakness                                                                                                                                                                                                                                             | evidence                                                                                                                          | affected                                         | confidence                                        |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| M1  | **Negative transfer from the sentence heads onto the phone-level classification head**                                                                                                                                                               | -0.026 macro-F1 [-0.045, -0.009]; recovered when sentence heads are removed (`phone_pair` +0.036, `word_phone` +0.032 vs full MT) | phone_mispron_t1.0                               | medium-high (3 seeds, CIs clear; one probe layer) |
| M2  | **Level imbalance**: 4 aligned sentence heads hold 73 % of the gradient norm; word / phone heads make almost no progress beyond their head-only epoch (error ratio 0.82-1.0 vs 0.47-0.68) and are orthogonal to the sentence block                   | gradient shares, cosines, progress ratios                                                                                         | word_accuracy, phone_score, phone_mispron_t1.0   | medium                                            |
| M3  | Starved gradient share of phone_mispron_t1.0 (0.32 x balanced) that **loss re-weighting x3 does not repair**                                                                                                                                         | shares; `w_mispron3` 0.653                                                                                                        | phone_mispron_t1.0                               | medium (one weight tried)                         |
| M4  | One stopping epoch is a compromise for three heads; explains at most about a third of M1                                                                                                                                                             | regret 2.8-4.3 %; own-best epochs 3-11 vs 7.7                                                                                     | sent_prosodic, word_accuracy, phone_mispron_t1.0 | low-medium (optimistic bound, 19 val speakers)    |
| M5  | **No positive transfer anywhere**; one significant loss, one negligible loss                                                                                                                                                                         | section 1                                                                                                                         | all                                              | medium-high                                       |
| M6  | Over-confident classification head persists inside the shared model (val/train loss ratio 9.6)                                                                                                                                                       | section 3                                                                                                                         | phone_mispron_t1.0                               | medium                                            |
| M7  | L2: word head takes about 65 % of the gradient; `l2_phone_error` -0.005                                                                                                                                                                              | shares, test CI                                                                                                                   | l2_*                                             | medium; negligible size                           |
| M8  | Validation does not predict test for the multi-task comparison: 5 of 7 SO762 heads are flagged better / worse beyond the margin on validation, none of them significant on test, while the one significant test loss is inside the validation margin | section 1                                                                                                                         | SO762 heads                                      | high (19 speakers)                                |
| W8  | **Still open (Sprint 5):** low-score compression of word_accuracy / phone_score. Not re-measured or addressed in Sprint 6                                                                                                                            | US 5.2 slopes 0.168 / 0.275                                                                                                       | word_accuracy, phone_score                       | high (Sprint 5)                                   |

Not found: gradient conflict (no pair below the threshold in any job); a speaker-group effect was not examined in Sprint 6.

## 6. Method limits to carry into the final report

- One probe layer; conflicts lower in the encoder are not seen. Averaged norms / cosines are not the same as per-step statistics.
- Test-set ablations informed the choice of fix below. To limit the leak, the adoption of any fix is decided on **validation** (criteria below), and test numbers are then reported, not used to tune.
- No correction for multiple comparisons; the exploratory side effects of `w_mispron3` are not claims.
- Wall-clock cost was measured on a shared GPU/CPU with two jobs in parallel and is not a fair speed comparison. The robust statement is about encoders: 1 instead of 7 (SO762) and 1 instead of 2 (L2).
- Near-duplicate targets (`phone_mispron_t1.0` = `phone_score < 1.0`; `sent_total` correlated with the other sentence scores) make aligned gradients less informative than they look.

## 7. Plan for US 6.3 (fix, then re-evaluate)

Every fix is evaluated with the same seeds, splits and decision rules as US 6.1. The criteria below are set now, before any run.

| weakness | fix, in order of trial                                                                                                                                                                                                                                                                                             | adoption criterion (validation)                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1, M2   | **(a) Regroup by level**: a sentence-level model (4 heads) and a word / phone-level model (3 heads). The word / phone group is the `word_phone` ablation already trained (3 seeds, tag `mt62_word_phone`); the sentence group is one new job set (3 seeds). Two encoders instead of seven.                         | `phone_mispron_t1.0` validation F1 not worse than single-task by more than the margin, **and** no other head worse than single-task by more than its margin (`max(0.01, job sd)`) |
| M3, M2   | **(b) Uncertainty-weighted loss** (learned per-head log-variance) on the full 7-head model, to document whether automatic re-weighting helps where the manual x3 did not. GradNorm is not planned: the x3 result suggests the problem is not gradient magnitude; it can be added if (b) shows any sign of benefit. | same criterion as (a)                                                                                                                                                             |
| M4       | (c) per-head checkpointing only if (a) and (b) leave a head with regret above the declared 2 %                                                                                                                                                                                                                     | no loss on any other head                                                                                                                                                         |
| M7       | no fix planned for `l2_phone_error` (-0.005); report                                                                                                                                                                                                                                                               | -                                                                                                                                                                                 |
| W8       | unchanged: still open                                                                                                                                                                                                                                                                                              | report as open                                                                                                                                                                    |

Decision rule: adopt the cheapest scheme that meets its criterion on validation; if none does, report the full multi-task model and the per-level split side by side and state that sharing the encoder across levels does not help in this data. Whatever is adopted, the final table reports every head against its single-task model with the paired bootstrap, with the number of encoders and the training cost beside it.
