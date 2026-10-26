# US 6.1: shared-encoder multi-task model vs single-task models

**Status: code, tests and run scripts written; no GPU job has been run yet.** Nothing in this file reports a result. Results go to `results/tables/multitask/` once the jobs below have run.

Research question 2: *does multi-task learning improve sentence-, word- and phoneme-level prediction compared with independently trained models?*

## 1. What is compared

| | single-task (Sprint 5) | multi-task (Sprint 6) |
|---|---|---|
| encoder | one WavLM-base-plus **per task** (9 encoders) | one WavLM-base-plus **per group** (2 encoders) |
| groups | - | `so762`: sent_accuracy, sent_fluency, sent_prosodic, sent_total, word_accuracy, phone_score, phone_mispron_t1.0 (7 heads). `l2`: l2_phone_error, l2_word_error (2 heads) |
| everything else | recipe of `configs/finetune.yaml` (CNN frozen, learned layer-weighted sum, MLP head, lr 3e-5 encoder / 1e-3 head, head-only first epoch, patience 4, bf16, batch 120 s) | **inherited unchanged** (`configs/multitask.yaml` `inherit: finetune.yaml`; a test asserts it) |
| data | tasks, labels, rows and their order, speaker splits, pooling spans, side features | the same `TaskData` objects, aligned on the utterances |
| comparators | - | `base` = US 5.1 jobs; `stabilized` = the US 5.3 selected variant per task |

Two groups, not one 9-head model: SO762 and L2-ARCTIC differ in speakers, recording conditions, label type (expert scores vs annotated errors), splits (one fixed split vs 4 speaker folds) and L2-ARCTIC prompts overlap 95-98 % between train and test (US 5.2, W9). A joint model would confound task transfer with domain shift. It can be revisited after US 6.3 as an exploratory extension.

## 2. How the multi-task model works (`src/models/multitask.py`)

- **One forward pass per utterance** serves all heads. The pooled units of a table are computed once (sentence tables: mean||std over frames; word / phone tables: mean over the MFA frames) and shared by the heads on that table (`phone_score` and `phone_mispron_t1.0` share the phone units; the four sentence heads share the sentence vector).
- **Per head**: LayerNorm -> its own learned layer-weighted sum (`model.alpha_mode: per_head`, ablation `sharedalpha`) -> + side features -> MLP head. A head's loss never touches another head's parameters (tested).
- **Loss**: per head exactly the single-task loss (MSE on the train-standardised target; balanced class-weighted cross-entropy). Combined by `LossWeighter`: US 6.1 = `fixed`, equal weights **normalised to sum 1**, so the gradient clip and Adam see the same scale as a single-task job. `uncertainty` and `gradnorm` are added in US 6.3 behind the same interface (`combine`, `parameters`).
- **Checkpoint**: one per multi-task job. Early stopping on `train.stop_metric = mean_rel` = mean over heads of `error(epoch) / error(head-only epoch 0)` (error = RMSE or 1 - macro-F1), computed on validation. Scale-free, no test data.
- **Gradient diagnostics for US 6.2** (no extra training cost worth mentioning): every 25 optimiser steps after the head-only epoch, each head's gradient on the parameters of the last transformer layer is measured. Saved per epoch: per-head gradient norm (share of the total) and the cosine similarity between heads, plus the effective loss weights. Only the K x K Gram matrix leaves the GPU.
- **Outputs are in the single-task layout**: `experiments/results/multitask/<tag>/<head>/fold<k>/seed<s>.npz` has the same keys as the Sprint 5 jobs, so the Sprint 5 report code (`pooled_ft`, `headline_variant`, the paired bootstrap) reads them unchanged. Extra keys: `mt_group`, `mt_n_heads`, `mt_group_fit_s`, `mt_head_best_epoch`. Group history: `.../_group_<group>/fold<k>/seed<s>.npz`.

## 3. Protocol (fixed before any multi-task job was run; `configs/multitask.yaml` `compare61`)

1. **Seeds**: SO762 3 seeds (as 5.1); L2-ARCTIC 4 folds x **3 seeds** for both sides. Sprint 5 used one L2 seed per fold, so the single-task comparators get seeds 1, 2 under new tags (`base_l2x`, `<variant>_l2x`; `run_61_local.sh st-l2`); the Sprint 5 files are not touched. Both sides are always restricted to the seeds they share.
2. **Recipe screen**: the single-task models got a screen of six recipes in US 5.3; the multi-task model gets the same kind of small, declared screen, `base` and `lr` (tags `mt_base`, `mt_lr`). Rule: lowest mean over heads of (multi-task validation error / single-task `base` validation error); `lr` goes forward only if it wins by more than 0.01.
3. **Comparison per head** (test, after all choices): paired speaker bootstrap of multi-task minus single-task (500 resamples, 95 % CI, two-sided p), verdict `better` / `worse` only when the interval excludes 0.
4. **Validation side**: multi-task is flagged `worse` on validation when it is worse by more than `max(0.01, job-to-job sd of the single-task validation metric)` (the 5.3 margin). Validation has 19 SO762 speakers; differences below the margin are not credited.
5. **Classification decision rule** (plain class-weighted argmax vs validation-tuned `cw+thr`) is chosen on validation for both sides, exactly as in the US 5.1 report.
6. Test numbers never select anything.

## 4. Order of use

```bash
bash run_61_local.sh test         # 1. unit tests (CPU)
bash run_61_local.sh dry          # 2. real tables: heads agree on utterances and splits? (no GPU)
bash run_61_local.sh smoke        # 3. 2-epoch jobs of both groups: speed and memory
bash run_61_local.sh st-l2        # 4. extra single-task L2 seeds (comparators)
bash run_61_local.sh train base   # 5. multi-task, recipe base
bash run_61_local.sh train lr     # 6. multi-task, recipe lr
bash run_61_local.sh screen       # 7. which recipe goes forward
bash run_61_local.sh report mt_base   # 8. comparison table (use the tag the screen chose)
```

Running several jobs on the GPU at once: `bash run_61_local.sh --parallel N <command>` (works with `smoke`, `st-l2`, `train base|lr`). One job = one (group, seed) for `train`, one (variant, task, seed) for `st-l2`; jobs write different files and are seeded, so results equal a sequential run. Choose N with `bash run_61_local.sh --parallel 3 smoke` (prints wall time, peak GPU memory and per-job s/epoch; compare with the single `smoke`): N x peak GPU GB must fit in the card, and the total time must drop. Logs: `logs/61_<job>.log`. Never start two processes on the same (tag, group, seed).

`report` writes `results/tables/multitask/us61_report_<tag>.md` (+ `us61_headline_<tag>.csv`, `us61_jobs_<tag>.csv`, `us61_efficiency_<tag>.csv`).

## 5. Known limits, stated before the results

- **Near-duplicate targets.** `phone_mispron_t1.0` is `phone_score < 1.0` (the same labels, thresholded) and `sent_total` is strongly correlated with the other sentence scores. Heads on such targets can make joint training look better than it is; US 6.2 separates this from genuine transfer (leave-one-head-out).
- **One checkpoint per multi-task job.** Single-task jobs each stop at their own best epoch; a multi-task job stops once for all heads. The report lists each head's own best epoch (`mt_head_best_epoch`) so a mismatch is visible. Per-head checkpointing is *not* implemented in 6.1: it would need predictions at every epoch or one stored state per head; it is a candidate for US 6.3 if 6.2 shows the single stopping epoch hurts specific heads.
- **Validation power.** 19 SO762 validation speakers (4 per L1 group in L2-ARCTIC): a difference smaller than about 0.02-0.05 RMSE is inside the noise.
- **Compute times** are wall clock on a possibly shared GPU: indicative only.
- **Open weakness not addressed in Sprint 6**: W8 (low-score compression of `word_accuracy` / `phone_score`, US 5.2). Reported as still open.

## 6. What the tests cover (`tests/test_multitask.py`, `tests/test_report_us61.py`)

Loss-weight maths; cosine / Gram helpers; config inheritance (recipe identical to Sprint 5, 9 tasks each in exactly one group); heads agree on utterances, rows and splits (and refuse when they do not); the multi-task forward pass equals the single-task model head by head; gradient routing; the gradient Gram equals explicit gradients and does not consume the graph; a full tiny-model job writes single-task-compatible files and is deterministic; **a multi-task job with one head reproduces the single-task job** (same best epoch, predictions within 2e-4); report: validation margin directions, seed matching, recipe rule, full `build()` on synthetic jobs, with and without the US 5.3 comparator.
