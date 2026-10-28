# US 6.3: fixing negative transfer and task imbalance in the SO762 multi-task model

**Status: code, tests and run script written; no GPU job of US 6.3 has been run yet.** Nothing in this file reports a result. Results go to `results/tables/multitask/us63_report.md` once the jobs below have run. The diagnosis this builds on is `docs/weakness_memo_s6.md` (US 6.2).

## 1. What is fixed, and what is not

| weakness (memo s6) | targeted by | status |
|---|---|---|
| M1 negative transfer of the sentence heads onto `phone_mispron_t1.0` (-0.026 F1) | regroup by level; uncertainty weights; GradNorm | **target** |
| M2 / M3 level imbalance, starved head (x3 weight did not help) | same | **target** |
| M4 one shared stopping epoch (at most a third of M1) | per-head checkpointing, **only if** a candidate keeps regret >= 2 % (section 4 of the report shows it) | conditional, not implemented |
| M7 `l2_phone_error` -0.005 | nothing | negligible, reported; the L2 group is not rerun |
| W8 low-score compression (Sprint 5) | nothing | **open**, reported as open |

## 2. Candidates (SO762, recipe `base`, 3 seeds, same splits as US 6.1)

| name | what | encoders | tag | new jobs |
|---|---|---|---|---|
| `full` | 7 heads, fixed equal weights (US 6.1) | 1 | `mt_base` | no |
| `regroup` | sentence block (4 heads) + word / phone block (3 heads) | 2 | `mt63_sent4` + `mt62_word_phone` | sentence block only |
| `uncertainty` | 7 heads, learned per-head log-variance (Kendall et al. 2018) | 1 | `mt63_uncertainty` | yes |
| `gradnorm_a05` | 7 heads, GradNorm, alpha 0.5 | 1 | `mt63_gradnorm_a05` | yes |
| `gradnorm_a15` | 7 heads, GradNorm, alpha 1.5 | 1 | `mt63_gradnorm_a15` | yes |

Expectations written down before running: **regroup** is the likeliest winner (the US 6.2 ablations removed the loss when the sentence heads were removed). **uncertainty** and **GradNorm** are expected NOT to help: there is no gradient conflict, a x3 weight on the target head did not help, and uncertainty weighting gives more weight to heads whose training loss is low, which is the over-fit target head. They are run because the story names them and because a clean negative result is a result.

## 3. Loss schemes (`src/models/multitask.py`)

All schemes hand the model a combination normalised to sum 1 (as `fixed`), so the gradient clip and Adam see the scale of a single-task job; only relative weights change.

- **uncertainty**: weight c_k exp(-s_k), c = 1/2 (regression) or 1 (classification); s_k (init 0, lr `train.loss.lr`) trained on sum_k [c_k exp(-s_k) L_k + s_k/2] with the L_k detached, so that term moves only s. Note the starting point: the classification target head begins with twice the weight of a regression head.
- **GradNorm** (first order, targets held constant): weights w_k (mean 1, floor 0.05, Adam lr 0.025) move so that G_k = w_k ||grad L_k|| on the last shared transformer layer approaches mean(G) * r_k^alpha, r_k = relative inverse training rate (EMA of the head loss over its mean in the second half of the head-only epoch, normalised over heads). Updated every optimiser step after the head-only epoch (`train.loss.update_every`). It uses the same probe as the US 6.2 diagnostics (`head_gram`); when a diagnostic step and an update step coincide the gradient is computed once.
- Applied weights per epoch are stored in the group history (`hist_w`) and reported.

## 4. Adoption rule (fixed in `configs/multitask.yaml` `compare63` before any US 6.3 job ran)

Validation only. A candidate passes if (1) the target head's validation F1 is not worse than single-task by more than its margin `max(0.01, job sd)`, (2) no other head is, (3) the target head's validation F1 is at least the full model's. (3) was added at the start of US 6.3 because (1) is already met by the full model on validation (0.007 worse against a margin of 0.010), so (1) alone cannot tell a fix from no fix. Among passing candidates the lowest mean over heads of (validation error / single-task validation error) wins; candidates within 0.01 of it are tied and the fewest encoders wins. If none passes, the full model and the per-level split are reported side by side and the conclusion is that sharing the encoder across levels does not help in this data. Test numbers (paired speaker bootstrap against single-task and against the full model) are reported for every candidate and never select. Criterion (2) is the memo's own and is strict: on 19 validation speakers several heads are flagged `worse` on validation without any test change (M8), so it can fail every candidate; if that happens it is reported, not relaxed.

## 5. Order of use (lab GPU)

```bash
bash run_63_local.sh test                 # 1. unit tests (CPU, seconds)  -- includes the torch tests that could not be run when the code was written
bash run_63_local.sh status               # 2. mt_base (US 6.1), mt62_word_phone (US 6.2) and the baselines must be finished
bash run_62_local.sh ablate word_phone    #    only if status shows mt62_word_phone missing
bash run_63_local.sh smoke                # 3. 2-epoch jobs of uncertainty and gradnorm_a05: speed, memory, weights move (tags mt63_smoke_*, delete afterwards)
bash run_63_local.sh --parallel 2 train all   # 4. sent4, uncertainty, gradnorm_a05, gradnorm_a15, 3 seeds each (12 jobs)
bash run_63_local.sh report               # 5. results/tables/multitask/us63_report.md (+ us63_*.csv)
```

GradNorm computes every head's gradient on the last layer at each step (plus the K partial backward passes), so its jobs are slower than the full model; if the smoke run shows it is too slow, raise `train.loss.update_every` in the config **before** training (it is part of the declared recipe, so do not change it afterwards).

## 6. Known limits, stated before the results

- The idea of regrouping comes from the US 6.2 ablations, which were read on test; adoption is on validation and regrouping gets no credit for having been seen first.
- Five candidates are screened on 19 validation speakers: the adopted one is partly the luckiest (winner's curse). All are reported.
- One probe layer for GradNorm and for the gradient shares.
- Three seeds; the target-head effect is 0.026 F1.
- `mt62_word_phone` was trained in US 6.2 with the same recipe and seeds; it is reused, not retrained.
- W8 and M7 are not addressed.
