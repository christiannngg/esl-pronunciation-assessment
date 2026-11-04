# Weakness memo, Sprint 8 (US 8.2)

Status: checks 1, 3 and the ablation are done. Check 2 (ranking vs calibration) awaits the lab probability export. Check 4 (age probe) is optional.
Source tables: `results/tables/generalization/us82_*.csv`. Code: `src/evaluation/report_us82.py`. Sign convention: ablation `delta` is the metric change (RMSE for regression, macro-F1 for classification); `delta_gap` is on the error scale, positive = child/teen relatively worse.

## 1. Target (rule kept as written)

Weakest subgroup: SO762 child/teen. Target head: `phone_mispron_t1.0`, the only Holm-significant Family A gap. Regression-head compression is reported as secondary/exploratory; widening the target would be a logged post-hoc deviation.

## 2. Diagnosis

**Check 1 (is it label noise or range restriction?).** The child/teen compression is real. Matching adult label distributions to the child's leaves the slope gap unchanged. Label noise (Spearman-Brown reliability of the 5-expert consensus) explains only ~6-12% of slope attenuation, against ~40-64% observed. The PCC gap is partly range restriction/noise: for `phone_score` it vanishes after label matching.

**Check 3 (where in the score range).** Child/teen speakers have few low-score items (sent_accuracy <=6: 7% vs 24.5% adult; word_accuracy 0-4: 3% vs 7%). Errors are in the tails, with positive bias at low scores: sent_accuracy <=6 bias +2.63 (child/teen) vs +1.11 (adult); word_accuracy 0-4 bias +6.42 vs +4.60. The model regresses to the mean most where child/teen training data is thinnest.

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

- Two mechanisms on the target head: (a) pretrained features favour adults (B->F); (b) multi-task sharing hurts child/teen mispronunciation F1 (-0.024 for GN15, -0.071 for full). Fine-tuning itself is neutral.
- Candidate fixes should therefore act on the shared-representation/multi-task side of the target head (e.g. child-aware loss weighting or head-specific treatment), not on fine-tuning intensity. FT-stab-style stabilization is a regression-head effect and is secondary.
- The target-head F1 gap vs MT-GN15 is the number to move. Success is judged by the pre-declared acceptance rules in `docs/sprint8_plan.md`.

## 5. Caveats

- Holm-significant evidence for mechanism (b) is not available (p=0.12); the child F1 drop CI excludes zero but the gap change does not survive Family B correction. Present as suggestive.
- Hard predictions only so far: no AUC/AP, so ranking vs calibration (check 2) is not yet separated.
- One seed per system in these bundles for the ablation contrasts.
