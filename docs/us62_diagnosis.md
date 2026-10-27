# US 6.2: diagnosis of the multi-task models

**Status: code, tests and run script written; the diagnosis has not been run yet.** Nothing in this file reports a result. Results go to `results/tables/multitask/us62_report.md` once `bash run_62_local.sh analyse` has run.

Research question 2 follow-up: *where does sharing the encoder help or hurt, and why?* US 6.1 found no significant gain on 8 of 9 heads and one significant loss (`phone_mispron_t1.0`, macro-F1 0.680 -> 0.654, CI [-0.045, -0.009]). US 6.2 looks for the reason.

## 1. What is checked (task text -> diagnostic)

| US 6.2 asks for | diagnostic | data it uses |
|---|---|---|
| per-task MT vs ST comparison | transfer table with the **matched** comparator (single-task model with the same recipe: `base` for `mt_base`, the US 5.3 `lr` model for `mt_lr`) | the US 6.1 headline csvs |
| negative transfer / task imbalance | which heads are significantly worse; unit counts per head; progress of each head relative to its own head-only epoch ("lagging") | job files (`n_train`, `hist_val_primary`) and group file (`ref_err`) |
| gradient dominance | each head's share of the summed gradient norms on the last shared transformer layer (1/K = balanced); labelled dominant / starved | group files (`hist_gnorm`, `hist_gn`) |
| gradient conflict | mean gradient cosine per pair, fraction of jobs below the threshold; approximate cosine of each head with the sum of the others | group files (`hist_gcos`) |
| the effect of the loss weighting | effective share (weight x norm); one ablation with a changed fixed weight (`w_mispron3`) | group files (`hist_w`); ablation jobs |
| (added) one checkpoint for all heads | validation error at the shared checkpoint vs each head's own best epoch ("stopping compromise") | job files (`hist_val_primary`, `mt_head_best_epoch`) |

No new training is needed for the first run: the gradient diagnostics were logged during the US 6.1 jobs (every 25 optimiser steps after the head-only epoch).

## 2. Criteria, fixed before the gradient diagnostics were looked at (`configs/multitask.yaml` `compare62`)

| name | rule |
|---|---|
| negative transfer | test verdict `worse` against the matched single-task model (paired speaker bootstrap, 95 % CI excludes 0) |
| dominant / starved | gradient-norm share > 2/K, or < 1/(2K) |
| conflict | mean gradient cosine < -0.1 in at least 67 % of the jobs (seeds x folds) |
| aligned (reported only) | mean cosine > 0.3 |
| stopping compromise | validation error at the shared checkpoint is at least 2 % above the head's own best epoch |
| lagging | error / error(head-only epoch) at the shared checkpoint is worse than the head average by more than 0.02 |

What was seen before the criteria were set: the US 6.1 table. The suspicion that one stopping epoch is a compromise and that `phone_mispron_t1.0` is the problem head come from it, so they are **hypotheses**, not findings. The thresholds were chosen without any gradient number in view.

## 3. Ablations (GPU, SO762 only, recipe `base`, 3 seeds each)

| variant | change | what a recovery would mean |
|---|---|---|
| `phone_pair` | only `phone_score` and `phone_mispron_t1.0` share the encoder | interference from the sentence / word heads |
| `w_mispron3` | all 7 heads, `phone_mispron_t1.0` loss weighted x3 (then normalised to sum 1) | under-weighting of that head (task imbalance) |
| `drop_wordacc` | the 6 heads without `word_accuracy` | the word-level head drags the shared encoder |

Reading rule (declared): an ablation **recovers** a negatively transferred head if the head is no longer significantly worse than its single-task comparator *and* significantly better than in the full multi-task model; **partly** if only the second holds; **no effect** / **hurts** otherwise. Ablations change the number of heads, hence the stopping criterion (mean over the heads present); that is part of the intervention and is not corrected.

## 4. Order of use

```bash
bash run_62_local.sh test                       # 1. unit tests (CPU)
python -m src.models.multitask dry-run --variant phone_pair   # 2. optional: the ablation groups build on the real tables (also drop_wordacc)
bash run_62_local.sh analyse                    # 3. diagnosis from existing results (no GPU); read results/tables/multitask/us62_report.md
bash run_62_local.sh --parallel 2 ablate all    # 4. only after reading 3: the three ablations (9 jobs)
bash run_62_local.sh analyse                    # 5. again: picks up every finished ablation
```

Needs the US 6.1 headline csvs for the carried-forward recipes (`report_us61 --tag mt_base --groups so762`, `report_us61 --tag mt_lr --variant lr --groups l2`), which exist already.

## 5. Limits, stated before the results

- **One probe layer.** Gradients are measured on the last trainable transformer layer. Conflicts that exist only lower in the encoder are not seen; a consistent picture over seeds is the evidence, not one number.
- **Averages of products.** Cosines and norms are averaged per epoch; `cos with sum of others` is rebuilt from the averages and is approximate. For two heads (L2) it is the cosine itself.
- **Stopping regret is optimistic.** Each head's best epoch is chosen on the same 19 validation speakers it is scored on; it bounds, not predicts, what per-head checkpointing would gain.
- **Near-duplicate targets.** `phone_mispron_t1.0` is `phone_score < 1.0`; `sent_total` correlates with the other sentence scores. Aligned gradients between such heads are not independent evidence of transfer.
- **Validation power.** 19 SO762 validation speakers; L2-ARCTIC has 4 folds x 3 seeds, so L2 conclusions are better supported than SO762 ones.
- **Not addressed in Sprint 6:** W8 (low-score compression of `word_accuracy` / `phone_score`); still open.

## 6. What the tests cover (`tests/test_report_us62.py`)

Job-level gradient averaging (epochs without samples ignored, step weighting, loss weights); dominance thresholds at the boundary; conflict rule (mean below threshold in enough jobs, not one noisy job); approximate alignment against explicit vectors; stopping regret for regression and classification (primary = -macro-F1); matched comparator choice; ablation reading rule; padding of unequal epoch counts; planted effects (dominant head, conflicting pair, stopping compromise) recovered from synthetic group and job files; unfinished jobs skipped and changed head lists rejected; the transfer table reads the right comparator columns; the report contains every section; the config declares every criterion and every ablation variant loads with the intended head list and weights.
