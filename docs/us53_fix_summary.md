# US 5.3 summary: weakness fixes for the fine-tuned WavLM-base-plus

**Status: complete for the screen of six variants** (`results/tables/finetune/us53_report.md`, `us53_variants.csv`, `us53_selection.csv`, `us53_headline.csv`). Weaknesses and targets: `weakness_memo_s5.md` sections 4 and 6; selection rule: `configs/finetune.yaml` `diagnose53`, fixed before any variant was run and not changed afterwards.

## 1. What was run
Six variants of the 5.1 recipe (`base`), each changing one group of settings, on all nine tasks with the 5.1 seeds and splits (SO762: 3 seeds; L2-ARCTIC: 4 folds x 1 seed):

| variant | change |
|---|---|
| `lr` | encoder lr 3e-5 -> 1.5e-5, layer-wise lr decay 0.85, grad clip 1.0 -> 0.5 |
| `reg` | head dropout 0.2 -> 0.4, weight decay 0.01 -> 0.1 |
| `mask` | HF SpecAugment time masking, 10 % of frames in 2-frame spans |
| `noise` | white noise at 15-35 dB SNR on 50 % of training utterances |
| `freeze6` / `freeze9` | bottom 6 / 9 transformer layers frozen (`freeze9` on the four low-value tasks only) |

Speed / pitch augmentation was not run (5.2 found no speaker-memorisation effect; it would alter the fluency / prosody labels). A variant is chosen per task only if it meets the declared targets (train-validation gap at the best epoch <= 0.30, last-epoch validation loss <= 1.25x / 1.5x its minimum for regression / classification, no training-loss spike >= 2x) **and** its validation metric is not worse than base by more than max(0.01, seed sd). All of this is validation-side; test is reported afterwards and never selects. I also reported the validation score with the same decision rule as the test headline (`val_headline`); it leads to the same selections, so no amendment of the rule was needed.

## 2. Result per task (test, selected variant vs 5.1 base and vs frozen 4.3)

| task | selected | test base -> selected | paired delta vs base (95 % CI) | vs frozen 4.3 before -> after | training-side change |
|---|---|---|---|---|---|
| sent_accuracy | `lr` | 1.071 -> 1.101 | **+0.030 [0.005, 0.063], worse** | better -> no significant change | gap 0.31 -> 0.04; loss ratio 1.07 -> 1.20 |
| sent_fluency | `freeze6` | 0.870 -> 0.861 | -0.009 [-0.041, 0.023], n.s. | better -> better | gap 0.36 -> 0.19 |
| sent_prosodic | base | 0.902 | - | unchanged | already meets the targets |
| sent_total | `lr` | 1.066 -> 1.078 | +0.012 [-0.024, 0.054], n.s. | better -> no significant change | spike 5.35 -> 0.94; gap 0.24 -> -0.04 |
| word_accuracy | base | 1.598 | - | better | already meets the targets |
| phone_score | base | 0.306 | - | unchanged | already meets the targets |
| phone_mispron_t1.0 | base | 0.680 | - | better | **no eligible variant met the targets** |
| l2_phone_error | `lr` | 0.559 -> 0.576 | **+0.017 [0.008, 0.026], better** | better -> better | loss ratio 2.24 -> 1.46; gap 0.315 -> 0.247 |
| l2_word_error | `lr` | 0.721 -> 0.719 | -0.002 [-0.006, 0.003], n.s. | unchanged | loss ratio 1.83 -> 1.14; gap 0.163 -> 0.125 |

The "stabilized" model of a task is the selected variant's checkpoints (`results/models/finetune/<variant>/<task>/`). Where base is selected, the 5.1 checkpoints stay.

**Net effect on test.** Against the frozen 4.3 systems the 5.1 set was better on 6 tasks; the rule-selected set is better on 4 (sent_fluency, word_accuracy, mispronunciation, l2_phone_error) and unchanged on 5. It lost the significant gains on sent_accuracy and sent_total. Over 5.1 base it is significantly better on one task (l2_phone_error), significantly worse on one (sent_accuracy), indistinguishable on three where a variant was selected (sent_fluency, sent_total, l2_word_error) and identical on four where base was kept. It is better calibrated on two classification tasks and cleaner on the training side, but it is **not a better model on test overall**.

## 3. What each weakness turned out to be

| weakness (memo section 4) | outcome |
|---|---|
| W1 train-validation gap, sentence tasks | **Gap can be closed, generalisation did not follow.** `lr`, `mask` and `freeze6` cut the gap (sent_accuracy 0.31 -> 0.04 / 0.10 / 0.13; sent_fluency 0.36 -> 0.08 under `lr`) but test did not improve for any sentence task, and was significantly worse for sent_accuracy under `lr`. The gap was not what limited held-out performance. |
| W2 over-confident classification heads | **Fixed by a lower encoder lr, not by regularisers.** Last-epoch validation loss ratio with `lr`: mispronunciation 2.62 -> 1.43, l2_phone 2.24 -> 1.46, l2_word 1.83 -> 1.14. `reg`, `mask` and `noise` left it at 1.6-2.8. `freeze6` / `freeze9` helped partly (mispronunciation 1.95, l2_word 1.48 / 1.25). Test F1 improved for l2_phone (+0.017) and mispronunciation (+0.016, but `lr` was not eligible there: validation F1 at the best epoch is 0.035 lower, 0.027 lower on `val_headline`), and was unchanged for l2_word. |
| W3 speaker memorisation | Not addressed; no evidence for it (5.2), so no speed / pitch run. |
| W4 unstable seed | **Gone in every variant**, not only under `lr` (max epoch-to-epoch training-loss ratio 5.35 in base vs 0.86-0.98 in all five re-runs of sent_total). It was one seed of one run; I cannot attribute the fix to `lr` specifically. |
| W5 15-epoch cap | **Still open.** l2_phone_error under `lr` again ran to the cap (best epoch 14 of 15). Not tested; a longer schedule on the `lr` variant is the natural follow-up. |
| W6 low value of full fine-tuning | **No cheaper variant was found.** Freezing the bottom layers did not give the required 20 % speed-up, and it costs accuracy: `freeze9` is significantly worse on word_accuracy (+0.048 RMSE) and l2_word_error (-0.008 F1), `freeze6` on l2_word_error (-0.004). Timings are unreliable because the GPU was shared (some `reg` jobs took 2-3x longer than base). |
| W7 validation-side regression | Same pattern in the other direction: **validation cannot rank sentence-task variants.** For sent_accuracy every variant beats base on validation (by 0.006-0.039 RMSE) and none beats it on test (+0.007 to +0.033); `lr` and `reg` differ by 0.001 on validation yet by 0.023 on test. |
| W8 low-score compression | **Not addressed.** Base was kept for word_accuracy and phone_score, so slopes (0.168, 0.275) and the lowest-bin bias are unchanged. No variant targeted it; a loss that re-weights low scores is a Sprint 6 candidate. |
| W9 L2 prompt overlap | Scope note for the final report; nothing to fix. |

## 4. Limits of this evidence
- Validation is 19 speakers (SO762) and 4 speakers per L1 group; with five or six variants per task the selected one is partly the luckiest on validation. The sent_accuracy result is the clearest case.
- About 50 variant-vs-base tests were run on test; a few "significant" differences are expected by chance (the small phone_score effects of `reg` / `mask` / `noise`, 0.003-0.004 RMSE, should not be read as findings).
- L2-ARCTIC has one seed per fold, so its margin is the 0.01 floor.
- Run times were measured on a shared GPU and are not comparable between variants.
- The selected set was determined by the declared rule. Choosing differently for sent_accuracy (for example keeping base) would use the test result to veto a validation-based choice and is not done here; it is a disclosed decision, not a selection.

## 5. Open items / for Sprint 6
- Carry the stabilized set as the fine-tuned reference and keep 5.1 `base` alongside it (the two are identical or statistically indistinguishable on seven of nine tasks).
- Optional follow-up on the two classification tasks that still hit the 15-epoch cap: `lr` with a 25-epoch schedule.
- Weakness W8 (low-score compression) is unaddressed by this sprint's tools.
