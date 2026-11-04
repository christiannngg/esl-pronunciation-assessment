# Go-forward plan, Sprints 8-10 and the final deliverables (US 7.2)

**Status:** draft for review, 2026-10-29. Scope decisions: `docs/revised_scope.md`. Results: `docs/midterm_results_report.md`.

## 0. Assumptions to confirm

| #   | assumption                                                                                                                              | why it matters                                                                                |
| --- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| A1  | The final report is **3-4 pages** in conference format with 8-12 citations (rubric checklist). The course schedule says 6-12 pages.     | Decides how much material is cut and how many figures fit                                     |
| A2  | The final presentation is somewhere in weeks 14-15 (Dec 7-18); the exact slot is unknown, so the plan is ready by **Dec 4**             | A slot in the first days of week 14 leaves no slack otherwise                                 |
| A3  | The lab account stays active until after the last planned GPU run. The student guide says access ends on the date in the welcome email. | All GPU runs must finish, and models, predictions and tables must be copied, before that date |
| A4  | Forced alignment can run inside the app backend                                                                                         | Phone and word feedback depends on it (Sprint 9)                                              |
| A5  | Weekly videos start in week 9 (Nov 2-6) and are 5 minutes each                                                                          | Seven videos if they run through week 15                                                      |

## 1. Calendar (today is Thu Oct 29; final report Mon Dec 14)

Week numbers follow the course schedule (week 6 ended Fri Oct 16).

| dates           | course week | work                                                                                                 | exit milestone                                                         |
| --------------- | ----------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Oct 28 - Nov 1  | 8           | **Sprint 7:** midterm report (done), scope review (this document)                                    | Revised scope and plan agreed                                          |
| Nov 2 - Nov 8   | 9           | **Sprint 8:** subgroup generalisation, ablations, one subgroup fix                                   | **Model freeze Sat Nov 8**; weekly video #1 (Fri Nov 6)                |
| Nov 9 - Nov 15  | 10          | **Sprint 9:** FastAPI + React demo on the frozen model; start the user sessions                      | Working end-to-end demo; video #2 (Nov 13)                             |
| Nov 16 - Nov 22 | 11          | **Sprint 10:** final tables and figures, significance, weakness case studies, finish user validation | **Results freeze Sun Nov 22**; video #3 (Nov 20)                       |
| Nov 23 - Nov 29 | 12          | Buffer + first full report draft (Thanksgiving Nov 26)                                               | Draft of every report section; video #4 (Nov 27, short week)           |
| Nov 30 - Dec 4  | 13          | Report revision, slides, demo rehearsal                                                              | **Report draft complete and slides ready Fri Dec 4**; video #5 (Dec 4) |
| Dec 5 - Dec 13  | 14          | Presentation window opens Dec 7; rehearse, final report polish, repo cleanup                         | **Repo and report frozen Sun Dec 13**; video #6 (Dec 11)               |
| Mon Dec 14      | 15          | **Final report due**                                                                                 | Submit                                                                 |
| Dec 14 - Dec 18 | 15          | Presentation if scheduled late                                                                       |                                                                        |

Slack is about three weeks (Nov 23 - Dec 13). It is spent in this order if something slips: reruns of a failed experiment, then the report, then rehearsal. It is **not** spent on new model work after Nov 8.

## 2. Sprint 8 (Nov 2-8): generalisation and ablations

## US 8.1: subgroup generalisation (carried systems only)

- Break down test results of the carried systems (single-task reference, `MT-GN15`, `MT-full`; L2: stabilized single-task and multi-task) by SO762 age group and L2-ARCTIC L1, with speaker-level intervals.
- No new training; reads the saved predictions (the export scripts from US 7.1 already show how).
- Output: `docs/generalization_report.md`, tables in `results/tables/generalization/`.

## US 8.2: diagnose and ablate

- Candidate subgroup weaknesses from earlier memos: child/teen word accuracy and mispronunciation; Hindi and Arabic on L2 phone error.
- Ablations reuse existing runs: frozen vs fine-tuned (US 4.3 vs 5.1) and single- vs multi-task (US 5.1 vs 6.1 / 6.3). Add a table of which differences are significant.
- Output: weakness memo with the ablation table.

## US 8.3: one fix for the weakest subgroup

- First tries: group-aware loss reweighting or targeted augmentation; speaker-adversarial training only if time remains.
- Selection on validation, test reported afterwards (same rules as every earlier fix). Report before/after with the speaker bootstrap.
- Output: this is the headline "found a weakness, fixed it" case study.

**Optional side experiments (each capped at one day, include only if clean):** the 25-epoch `lr` run for the two classification tasks that hit the cap, and a low-score loss for word accuracy and phone score.

**Model freeze Nov 8:** after this, no retraining; the demo and the report depend on these checkpoints. Copy checkpoints, predictions and tables off the lab (see A3) the same day.

## 3. Sprint 9 (Nov 9-15): app and user validation

- **First 1-2 days:** confirm forced alignment runs in the backend (A4) and check inference speed for `MT-GN15`. If alignment cannot run, fall back to prerecorded examples with an interactive front end.
- **US 9.1:** FastAPI backend serving `MT-GN15` plus the L2 model; React front end: show sentence, record, upload, display sentence / word / phone feedback.
- **US 9.2:** about 5 ESL speakers or peers, structured questions (clarity, usefulness, trust in the scores); recruit during this week, run sessions during this week and the next.
- Output: working demo, a recorded walkthrough usable as the presentation fallback.

## 4. Sprint 10 (Nov 16-22): final results package

- **US 10.1:** final tables and figures; the significance tests (most already exist from US 7.1); one consolidated "weakness -> fix" section (the ledger in the midterm report is the starting point).
- **US 10.2 (part):** repo documentation, tests, reproducibility notes; code freeze on Nov 29.
- Finish the user-validation summary.
- **Results freeze Nov 22:** after this date, numbers in the report do not change.

## 5. Final deliverables

**Final report (Mon Dec 14).** Page budget under A1 (3-4 pages, conference format): abstract and introduction (about 0.5 page), related work with 8-12 citations (0.5), method (0.75), experiments and results (1.25, built from midterm Tables 3.1 and 3.2), weakness -> fix case studies (0.5), discussion and limits (0.25), references. If the limit is 6-12 pages instead, the same material expands in the method, the ablations and the subgroup results. Reuse from `docs/midterm_results_report.md`: the headline table, the paired-comparison table, the GradNorm disclosure, the weakness ledger and the limits section.

**Final presentation (8-10 minutes, weeks 14-15).** Slides built from the same tables; a live demo (with the Sprint 9 recording as a fallback); rehearse for timing at least twice by Dec 10. Prepare answers for: what weaknesses did you find and how did you fix them; why is multi-task only at parity; how do the models treat child and non-Mandarin speakers; why not more heads or a bigger encoder.

**Weekly progress video (5 minutes, from week 9).** Record each Friday from that week's sprint summary: what was done, one result, one problem, next week's plan. Reuse the figures from the report. Budget about 30-45 minutes per video. Week 12 (Thanksgiving) can be short.

## 6. Risks and mitigations

| risk                                                                                                 | likelihood | mitigation                                                                                                     |
| ---------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| Lab access ends before the last run, or the GPU queue blocks (one GPU per person, jobs up to 2 days) | medium     | Check A3 now; submit Sprint 8 jobs early in the week; copy outputs after each sprint                           |
| Forced alignment does not run in the backend                                                         | medium     | Check on day 1 of Sprint 9; fall back to prerecorded demo                                                      |
| Sprint 8 finds no clear subgroup weakness                                                            | low        | Report the null with intervals; fall back to the child/teen mispronunciation case (known, documented)          |
| The subgroup fix does not help                                                                       | medium     | Report it as a negative result with the diagnosis; this still satisfies "found, tried, reported"               |
| Report length mismatch (A1)                                                                          | low        | Confirm now; the material is the same either way                                                               |
| User validation too small to say anything                                                            | medium     | Frame as qualitative signal only (as the proposal does)                                                        |
| Time lost to a late-term crunch in other courses or work                                             | unknown    | Results freeze on Nov 22 and report draft on Nov 29 give two cushions; do not start new model work after Nov 8 |

## 7. Immediate next steps (this week)

1. Review and agree D1-D9 in `revised_scope.md` (or tell me what to change).
2. Answer A1-A4.
3. Optionally send the instructor a short note with D1-D4 (I can draft it). The midterm checkpoint is done, so this is courtesy only.
4. Commit and push the Sprint 7 files.
5. Start Sprint 8 once the scope is agreed: first the subgroup breakdown script (reads saved predictions; runs on the Mac and the lab as in US 7.1).
