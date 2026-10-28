# US 6.3 summary: fixing negative transfer in the SO762 multi-task model

**Status: complete for the five declared candidates** (`results/tables/multitask/us63_report.md`, `us63_heads.csv`, `us63_candidates.csv`, `us63_weights.csv`, `us63_regret.csv`, `us63_cost.csv`). Weakness and plan: `weakness_memo_s6.md` (sections 5 and 7); protocol and adoption rule: `configs/multitask.yaml` `compare63` and `docs/us63_fix.md`, both fixed before any US 6.3 job was run and not changed afterwards.

## 1. What was run

SO762 only, recipe `base`, 3 seeds, same splits as US 6.1. Five candidates, each compared per head with the same-recipe single-task model (US 5.1 `base`) and with the full 7-head multi-task model (US 6.1):

| candidate | change | encoders |
|---|---|---|
| `full` | 7 heads, fixed equal weights (US 6.1) | 1 |
| `regroup` | sentence block (4 heads, new) + word / phone block (3 heads, US 6.2 `word_phone`) | 2 |
| `uncertainty` | learned per-head log-variance weights | 1 |
| `gradnorm_a05` / `gradnorm_a15` | GradNorm (first order, last-layer probe), alpha 0.5 / 1.5 | 1 |

Not run, by decision: a loss that re-weights low scores (W8), a fix for `l2_phone_error` (M7, -0.005), per-head checkpointing (see section 4).

## 2. Result under the declared rule: no candidate is adopted

Adoption is validation only: (1) the target head `phone_mispron_t1.0` is not worse than single-task by more than its margin, (2) no other head is, (3) the target head is at least as good as in the full model.

| candidate | (1) target ok | (2) no other head worse | (3) target >= full | mean rel. val. error | passes |
|---|---|---|---|---|---|
| full | yes | **no** (sent_fluency, sent_prosodic, word_accuracy) | yes | 1.015 | no |
| regroup | **no** (0.028 worse, margin 0.010) | **no** (sent_fluency, sent_prosodic) | **no** | 1.020 | no |
| uncertainty | **no** (0.010 worse, margin 0.010) | **no** (4 heads) | **no** | 1.020 | no |
| gradnorm_a05 | yes | **no** (the same three as `full`) | yes | 1.009 | no |
| gradnorm_a15 | yes | **no** (the same three as `full`) | yes | **1.005** | no |

Criterion (2) fails for every candidate, **including the full model, on the same three heads that are unchanged on test** (M8: 19 validation speakers). By the declared rule the outcome is "none adopted: report the full model and the per-level split side by side". This is the result of the rule and it is not relaxed here.

## 3. What test shows (reported for every candidate, never used to select)

Target head `phone_mispron_t1.0`, test macro-F1 (single-task 0.680):

| candidate | test F1 | vs single-task [95 % CI] | vs full MT | other heads worse than single-task on test |
|---|---|---|---|---|
| full | 0.654 | **-0.026 [-0.045, -0.009] worse** | - | 0 |
| regroup | 0.686 | +0.007 [-0.009, 0.025] | +0.032 better | **4** (sent_fluency +0.047, sent_prosodic +0.037, sent_total +0.021, phone_score +0.004) |
| uncertainty | 0.662 | **-0.018 [-0.035, -0.005] worse** | +0.007 better | 0 |
| gradnorm_a05 | 0.683 | +0.003 [-0.008, 0.014] | +0.029 better | 0 (sent_total better by 0.022 [0.002, 0.044]) |
| gradnorm_a15 | 0.681 | +0.001 [-0.012, 0.012] | +0.027 better | 0 |

- **regroup removes the loss and creates new ones.** The target head recovers (confirming the US 6.2 ablation), but three sentence heads and `phone_score` become significantly worse than single-task. The sentence block trained alone is worse than the same heads inside the full model (sent_fluency +0.030, sent_prosodic +0.026, sent_total +0.032 vs full), so the word / phone heads are not pure interference for the sentence heads. Two encoders, one weakness traded for four.
- **uncertainty does not fix it.** Its learned weights barely left their starting point (target head 1.82x balanced against a start of 2.0x, the other heads 0.79-0.91x), so it is in effect the x2 re-weighting that US 6.2 already found insufficient. The prediction written down before the run (no help) held.
- **GradNorm brings the target head back to the single-task level without a loss elsewhere.** Neither alpha leaves any head significantly worse than single-task on test. The target head is not better than single-task either: the CIs include 0 on both sides. The one significant gain (`gradnorm_a05`, sent_total) is one of 35 head-candidate comparisons without correction for multiplicity and is not read as positive transfer.
- What GradNorm did to the weights (mean over the epochs after the head-only epoch, x balanced): `gradnorm_a05` target head 3.09, sent_fluency 0.29, sent_prosodic 0.23, sent_total 0.57, sent_accuracy 1.20, word_accuracy 0.93, phone_score 0.71; `gradnorm_a15` target head 2.04, word_accuracy 1.52, phone_score 1.23, sent_fluency 0.30, sent_prosodic 0.24. It moves weight from the two sentence heads that held the largest gradient shares (M2) to the starved target head, far more than the manual x3 of US 6.2 (which left the head at 0.653). That x3 did not help while GradNorm did suggests that **which** heads are down-weighted matters, not only the target head's own weight; this is a hypothesis, not tested (no ablation separates the two).
- The effective share (weight x gradient norm) table in the revised report shows how balanced the encoder's gradient actually became. The first version of the report only showed the unweighted share, which cannot change when weights change; the column was added after the run and the report regenerated.

## 4. Per-head checkpointing (plan item d)

Declared trigger: a head keeps regret >= 2 % after (a)-(c). Regret of the shared checkpoint against each head's own best epoch (optimistic bound): `gradnorm_a15` none (largest `word_accuracy` 2.0 %, just below), `gradnorm_a05` only `word_accuracy` (2.2 %), full 3 heads (2.8-4.3 %), regroup 2 heads, uncertainty 3 heads. The target head has regret 0.0 % under `gradnorm_a05` and 1.3 % under `gradnorm_a15`, against 2.8 % in the full model. **Not implemented**: the one head over the trigger under GradNorm (`word_accuracy`) is unchanged on test, the number is an upper bound, and the machinery would be new. The trigger is stated as technically met for `gradnorm_a05`.

## 5. Weakness -> fix, for the reviewer

| weakness (memo s6) | outcome |
|---|---|
| **M1** negative transfer from the sentence heads onto `phone_mispron_t1.0` (-0.026 F1) | **Removed by GradNorm** (test: +0.003 / +0.001 vs single-task, CIs include 0, both better than the full model by about 0.03) and by regrouping, but regrouping costs four heads. Uncertainty weighting does not remove it. **Not adopted under the declared validation rule** (criterion 2 fails for every candidate, the full model included). |
| **M2 / M3** level imbalance; starved head that x3 did not repair | GradNorm re-weights the heads by a large factor and the target head recovers. Whether the effective gradient shares are balanced: see the effective-share table in the report. |
| **M4** one shared stopping epoch | Smaller under GradNorm (regret 0.0-2.2 %); per-head checkpointing not implemented (section 4). |
| **M7** `l2_phone_error` -0.005 | Not addressed, negligible. |
| **W8** low-score compression (Sprint 5) | **Still open.** No loss that re-weights low scores was tried. |

**Answer to the research question for the SO762 group.** With GradNorm the shared-encoder model matches the single-task models on all seven heads on test (no head significantly worse, no clear gain) with one encoder instead of seven. There is no evidence that joint learning improves the heads. The case for it is parity at a smaller model, not accuracy.

## 6. Disclosures and limits

- **A post hoc reading, labelled as such.** Under the declared rule nothing is adopted. Criterion (2) is the memo's own and fails identically for the full model, so it does not discriminate between candidates. A variant of it measured against the full model rather than against single-task (no head worse than in the full model by more than its margin) would let `gradnorm_a05` and `gradnorm_a15` pass, and `gradnorm_a15` would then be adopted (lowest mean relative validation error, 1.005). This amendment was thought of after the results were seen and is **not applied**; if the writeup carries GradNorm forward, it is as the best-supported candidate, not as a rule-adopted fix. The decision on how to present it is the researcher's.
- The regroup idea came from US 6.2 ablations read on test; it was given no credit for it.
- Five candidates on 19 validation speakers: winner's curse; all reported. Test CIs are not corrected for multiplicity (35 head-candidate comparisons).
- Three seeds; the target-head effect is 0.026 F1 and the GradNorm result is "within noise of single-task", not "equal".
- GradNorm is the first-order form with the last shared transformer layer as the probe, updated every step; alpha only took two values.
- The cost table (minutes per seed) is not comparable between candidates: jobs ran three at a time on a shared GPU, and the 6.1 `full` figure was measured under different load. Only the number of encoders is a reliable cost statement.
- The `uncertainty` candidate began with the target head at twice the weight of a regression head by construction (Kendall's c = 1 vs 1/2).

## 7. For Sprint 7

- Carry the full model (`mt_base`), `gradnorm_a15` (and optionally `gradnorm_a05`) and the per-level split into the midterm comparison, with the disclosure in section 6.
- W8 stays on the open list, together with the 15-epoch cap of Sprint 5 (not revisited in Sprint 6).
- The L2 group was not touched in 6.3.
