# Generalization report (US 8.1): per-background results of the carried systems

**Status:** draft for review, 2026-11-03. Aggregation only: nothing was retrained. Every number comes from the saved test predictions (`results/analysis/us71/*.npz`) through `python -m src.evaluation.report_us81`; tables are in `results/tables/generalization/`. Design and test families were fixed before the numbers existed (`docs/sprint8_plan.md`, sections 3-4).

**What "unseen speakers" means here (D3).** The SO762 test speakers (125) and the L2-ARCTIC test folds (24 speakers) were never used for training or selection, so the test set already is the unseen-speaker evaluation. This report breaks it down by age group (SO762) and L1 (L2-ARCTIC).

## 1. What the evidence says

1. **The gain over the baseline holds in every L1 and for adults, but not equally for child/teen speakers.** All 24 baseline-to-carried comparisons by L1 are significantly better. By age group, 22 of 28 are better; the six that are not are all child/teen speakers on `sent_fluency`, `sent_prosodic` and `sent_total` (two carried systems each). On those heads adults improve by 0.17-0.19 RMSE; child/teen speakers change by -0.05 to +0.01, with intervals that include 0 (table 4).
2. **Pre-declared Family A (is there an age gap? 7 tests, Holm): one head.** Only `phone_mispron_t1.0` shows an adverse child/teen gap (macro-F1 0.584 against 0.709, Holm p < 0.01). The other six RMSE gaps are not significant (table 2). The mispronunciation gap is a recall gap: 0.127 against 0.465 at the pooled operating point.
3. **RMSE hides a large difference on the regression heads (exploratory).** Child/teen RMSE is not worse on five of six regression heads, but the predictions follow the true score much more weakly: sentence-level slope of prediction on truth is 0.24-0.35 for child/teen speakers against 0.61-0.70 for adults, PCC 0.44-0.55 against 0.82-0.89 (table 3). Part of the PCC difference is a narrower label spread (SD 1.1-1.3 against 1.7-1.8), but slope is not reduced mechanically by a narrower label range, so compression looks real. This is tested properly in US 8.2 (check 1, against the human ceiling).
4. **Pre-declared Family B (what changes the gap? 21 tests, Holm): the gap appears with the pretrained representation and nothing later removes it.** B to F widens the gap significantly on five heads (`sent_fluency`, `sent_prosodic`, `sent_total`, `phone_score`, `phone_mispron_t1.0`). F to FT-base and FT-base to MT-GN15 change no gap significantly after Holm. The one near-miss is `phone_mispron_t1.0` under MT-GN15 (+0.030 [+0.009, +0.048], raw p 0.008, Holm p 0.12): not significant after correction (table 5). Multi-task sharing is therefore not shown to cause the gap.
5. **L1 (descriptive).** Hindi is again the weakest L1 on phone error (0.463 for FT-stab) and Vietnamese the strongest (0.608). With 4 speakers per L1 this is a pattern, not a test (table 6).

## 2. Tables

Metric: RMSE (lower is better), macro-F1 for `phone_mispron_t1.0` (higher is better). Test set, seed-averaged predictions.

**Table 1. Adult / child-teen by system.**

| task                    | B             | FT-base       | MT-GN15       |
| ----------------------- | ------------- | ------------- | ------------- |
| sent_accuracy           | 1.282 / 1.142 | 1.071 / 1.071 | 1.080 / 1.048 |
| sent_fluency            | 0.963 / 0.951 | 0.784 / 0.945 | 0.789 / 0.959 |
| sent_prosodic           | 1.039 / 0.955 | 0.856 / 0.944 | 0.861 / 0.954 |
| sent_total              | 1.181 / 1.179 | 0.989 / 1.134 | 0.992 / 1.123 |
| word_accuracy           | 1.830 / 1.499 | 1.724 / 1.433 | 1.721 / 1.458 |
| phone_score             | 0.390 / 0.301 | 0.332 / 0.273 | 0.330 / 0.273 |
| phone_mispron_t1.0 (F1) | 0.579 / 0.562 | 0.703 / 0.608 | 0.709 / 0.584 |

Test speakers: 61 adult, 64 child/teen.

**Table 2. Family A: adverse age gap of MT-GN15** (error(child/teen) - error(adult), error = RMSE or 1 - macro-F1; positive = child/teen worse; speaker bootstrap, 500 draws; "0/500" = no draw on the other side of 0).

| task                   | gap [95% CI]                | raw p | Holm p | reading              |
| ---------------------- | --------------------------- | ----- | ------ | -------------------- |
| sent_accuracy          | -0.032 [-0.254, +0.176]     | 0.77  | 1.00   | no significant gap   |
| sent_fluency           | +0.170 [-0.139, +0.506]     | 0.43  | 1.00   | no significant gap   |
| sent_prosodic          | +0.093 [-0.224, +0.428]     | 0.68  | 1.00   | no significant gap   |
| sent_total             | +0.131 [-0.155, +0.411]     | 0.50  | 1.00   | no significant gap   |
| word_accuracy          | -0.263 [-0.671, +0.114]     | 0.18  | 1.00   | no significant gap   |
| phone_score            | -0.057 [-0.152, +0.032]     | 0.20  | 1.00   | no significant gap   |
| **phone_mispron_t1.0** | **+0.125 [+0.083, +0.173]** | 0/500 | < 0.01 | **child/teen worse** |

**Table 3. Diagnostics of MT-GN15, adult / child-teen (exploratory).**

| task          | PCC           | slope of prediction on truth | label SD    |
| ------------- | ------------- | ---------------------------- | ----------- |
| sent_accuracy | 0.817 / 0.436 | 0.613 / 0.237                | 1.83 / 1.15 |
| sent_fluency  | 0.886 / 0.552 | 0.704 / 0.353                | 1.67 / 1.14 |
| sent_prosodic | 0.868 / 0.553 | 0.651 / 0.333                | 1.68 / 1.14 |
| sent_total    | 0.846 / 0.456 | 0.636 / 0.235                | 1.78 / 1.25 |
| word_accuracy | 0.513 / 0.309 | 0.316 / 0.126                | 1.98 / 1.52 |
| phone_score   | 0.592 / 0.497 | 0.366 / 0.214                | 0.41 / 0.31 |

`phone_mispron_t1.0`: recall 0.465 / 0.127, precision 0.434 / 0.343, positive rate 5.2% / 3.0%.

**Table 4. Baseline to carried system within each age group** (delta RMSE or F1 with 95% CI; negative RMSE / positive F1 is better).

| task                    | adult, B to FT-base     | child/teen, B to FT-base | child/teen, B to MT-GN15 |
| ----------------------- | ----------------------- | ------------------------ | ------------------------ |
| sent_fluency            | -0.179 [-0.242, -0.124] | -0.006 [-0.041, +0.017]  | +0.007 [-0.044, +0.050]  |
| sent_prosodic           | -0.184 [-0.256, -0.116] | -0.011 [-0.075, +0.033]  | -0.001 [-0.065, +0.050]  |
| sent_total              | -0.191 [-0.278, -0.108] | -0.046 [-0.124, +0.018]  | -0.056 [-0.140, +0.009]  |
| sent_accuracy           | -0.211 [-0.306, -0.115] | -0.071 [-0.140, -0.008]  | -0.095 [-0.171, -0.034]  |
| word_accuracy           | -0.106 [-0.148, -0.067] | -0.066 [-0.103, -0.033]  | -0.041 [-0.078, -0.002]  |
| phone_score             | -0.058 [-0.070, -0.047] | -0.028 [-0.037, -0.018]  | -0.028 [-0.037, -0.019]  |
| phone_mispron_t1.0 (F1) | +0.124 [+0.102, +0.147] | +0.046 [+0.030, +0.060]  | +0.022 [+0.004, +0.042]  |

Every child/teen gain is smaller than the adult gain on the same head.

**Table 5. Family B: change of the adverse gap** (significant after Holm: B to F on five heads).

| task               | B to F                              | F to FT-base                        | FT-base to MT-GN15                  |
| ------------------ | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| sent_accuracy      | +0.149 [+0.039, +0.258] (Holm 0.06) | -0.010 [-0.061, +0.042]             | -0.032 [-0.074, +0.008]             |
| sent_fluency       | **+0.165 [+0.092, +0.246]**         | +0.007 [-0.039, +0.057]             | +0.008 [-0.023, +0.045]             |
| sent_prosodic      | **+0.199 [+0.125, +0.285]**         | -0.026 [-0.070, +0.009]             | +0.005 [-0.035, +0.046]             |
| sent_total         | **+0.179 [+0.071, +0.285]**         | -0.033 [-0.088, +0.018]             | -0.013 [-0.054, +0.029]             |
| word_accuracy      | +0.079 [+0.019, +0.136] (Holm 0.17) | -0.039 [-0.094, +0.012]             | +0.027 [-0.024, +0.082]             |
| phone_score        | **+0.044 [+0.023, +0.068]**         | -0.014 [-0.030, -0.000] (Holm 0.47) | +0.002 [-0.005, +0.010]             |
| phone_mispron_t1.0 | **+0.080 [+0.050, +0.116]**         | -0.002 [-0.025, +0.018]             | +0.030 [+0.009, +0.048] (Holm 0.12) |

**Table 6. L1 on L2-ARCTIC, macro-F1** (FT-stab / MT-full; baseline B for phone error in the last column). 4 speakers per L1: read as a pattern only.

| L1         | phone error   | word error    | phone error, B | phone error, per-speaker range (FT-stab) |
| ---------- | ------------- | ------------- | -------------- | ---------------------------------------- |
| Arabic     | 0.546 / 0.552 | 0.695 / 0.698 | 0.344          | 0.519-0.564                              |
| Hindi      | 0.463 / 0.468 | 0.671 / 0.672 | 0.365          | 0.421-0.495                              |
| Korean     | 0.543 / 0.549 | 0.695 / 0.698 | 0.409          | 0.377-0.558                              |
| Mandarin   | 0.543 / 0.550 | 0.714 / 0.714 | 0.412          | 0.487-0.581                              |
| Spanish    | 0.562 / 0.558 | 0.711 / 0.715 | 0.414          | 0.475-0.611                              |
| Vietnamese | 0.608 / 0.598 | 0.772 / 0.770 | 0.410          | 0.584-0.646                              |

## 3. Other slices (exploratory, in `us81_slices.csv`)

- **Gender (SO762):** MT-GN15 RMSE is higher for male speakers on `sent_fluency` (0.984 against 0.741), `sent_prosodic` (1.004 against 0.788) and `sent_total` (1.137 against 0.967). Gender and age group are probably entangled in this test set, so this is flagged for US 8.2 (age x gender), not interpreted here.
- **Utterance length, true-score bin and proficiency tercile:** RMSE rises with length and is much higher for the low-proficiency tercile; both are tied to label spread by construction (the proficiency proxy is built from labels), and the score bins condition on the truth, so they show regression to the mean (for example `sent_accuracy`, scores <= 6: bias +1.47; scores 9-10: bias -0.61) and nothing else. They are in the CSV and not interpreted.
- **L2-ARCTIC length and proficiency:** differences are within 0.02-0.05 macro-F1 (the proficiency proxy is the phone error rate itself, so more errors means easier macro-F1).

## 4. Limits and disclosures

- **Primary metric.** Family A used RMSE and macro-F1 as fixed in the plan. For the regression heads RMSE is a weak lens on this question (finding 3), and no family was changed after seeing the numbers. Using PCC or slope as a test statistic would be a new, post-hoc family and would be logged as a deviation.
- **Multiplicity.** Holm applies within Family A (7 tests) and Family B (21 tests) only. Everything else, including the six baseline comparisons in table 4 and all slices, is unadjusted and exploratory.
- **Bootstrap.** Speaker-level, 500 draws (`configs/baseline.yaml`), so p-values resolve to about 0.002 and "0/500" means no draw on the other side. The 8.1 code computes metrics from per-speaker statistics (same CI and p conventions as `metrics.paired_speaker_bootstrap`, a different random stream), so intervals differ slightly from the US 7.1 ones; the overall metrics reproduce the US 7.1 table (checked automatically before any file is written).
- **L1 intervals** come from resampling 4 speakers and are unreliable; the CSV carries them, the per-speaker values (`us81_speakers.csv`) are the honest view.
- **Seeds.** SO762 FT and MT predictions are 3-seed averages, L2 FT-stab is 1 seed per fold, B and F are single fits: seed variance is not in the intervals.
- **No AUC / AP.** The bundles hold hard predictions only. Threshold-free diagnostics need probabilities and move to US 8.2 (see `docs/sprint8_plan.md`, section 9).
- **Children's label distribution differs** (lower positive rate, narrower spread), so group differences in RMSE and macro-F1 mix model behaviour with label statistics; US 8.2 separates them.

## 5. What this means for US 8.2 and 8.3

- Under the rule fixed in the plan, the **target head is `phone_mispron_t1.0`** (the only Holm-significant adverse gap), which is also the documented fallback.
- The regression heads show a different problem the primary family could not see: **child/teen predictions are strongly compressed** (slope 0.24-0.35 against 0.61-0.70). Whether that counts as a target is a decision for you, not something I changed: keep the rule as written and measure compression as a secondary outcome of the fix, or add the heads to the target set as a logged post-hoc deviation.
- Table 4 gives the case study its starting claim: **the deep model's headline gain does not transfer to child/teen speakers on fluency, prosody and total score.**

## 6. Reproduction

`python -m src.evaluation.report_us81` (about 2 seconds, numpy and pandas only); tests: `python -m pytest tests/test_report_us81.py` (13 pass). Outputs: `us81_slices.csv`, `us81_pairs.csv` (baseline and carried comparisons per group), `us81_age_gap.csv` (family A), `us81_gap_change.csv` (family B), `us81_speakers.csv`, `us81_slice_definitions.json`. Bundles for FT-stab and MT-full come from `python -m src.evaluation.report_us71_pairs export --systems FT-stab MT-full` on the lab.
