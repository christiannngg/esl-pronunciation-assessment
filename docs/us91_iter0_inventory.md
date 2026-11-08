# Sprint 9, iteration 0: what inference needs (gate G0)

Date: 2026-11-07. Plan: `docs/sprint9_plan.md`. Nothing was run on the model yet; this is a code and file inventory.

## 1. What the checkpoint contains

`results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed{0,1,2}.pt` (193 MB each, fp16). Written by `multitask.run_group` as `best_state`: the shared WavLM-base-plus encoder, 7 heads and the per-head layer weights (`alpha`).
It does NOT contain: target mean/std, side-feature mean/std, class lists, or the decision rule. These must be rebuilt (section 3).

## 2. Model structure to rebuild (`build_mt_model_class`)

- Encoder `microsoft/wavlm-base-plus` (`configs/frozen.yaml`, revision null), hidden states of layers 0-12, CNN front end frozen. Pin the revision hash in the export.
- Pooling: sentence heads = mean||std over all frames (dim 1536); word and phone heads = mean over the MFA frame interval (dim 768).
- Per head: LayerNorm -> learned softmax layer-weighted sum -> concat side features -> MLP (hidden 256).
- Waveform: 16 kHz mono, per-utterance normalisation as in `finetune.normalize_wave`; silence is not trimmed.
- Heads: sent_accuracy, sent_fluency, sent_prosodic, sent_total (regression); word_accuracy, phone_score (regression); phone_mispron_t1.0 (classification).

## 3. Gaps found (these block G0)

| #   | gap                                                                                                                                                                                                                                                                                                                                                                                                                    | why it matters                                                                                                                | how to close                                                                                                                                       |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| G-a | **Side features are required inputs.** Sentence: dur, n_words, n_phones. Word: dur, n_phones, phone_dur_mean, phone_dur_std, n_chars, word_pos. Phone: dur, log_dur, has_prev, has_next, word_initial, word_final, n_phones_in_word, is_vowel + canonical-phone one-hot (47 columns). They come from `src/features/aggregate.py` (MFA intervals + canonical text) and are standardised with the TRAINING-row mean/std. | The app must compute exactly these from the live alignment and the prompt's canonical phones.                                 | Extract the column definitions into a serving function; test it against the saved `*_side_X.npy` rows (must match exactly for the same alignment). |
| G-b | **Standardisation constants are not saved.** Regression target mean/std (`mu`, `sd`) and side mean/std are computed from the fold-0 training rows at train time.                                                                                                                                                                                                                                                       | Without them, outputs cannot be turned back into scores.                                                                      | A one-off export script on the Mac recomputes them with the same code path and writes `serving/stats.json`; verify against saved predictions.      |
| G-c | **Classification decision rule is not in the checkpoint.** `phone_mispron_t1.0` uses either plain class-weighted argmax or validation-tuned `cw+thr` decision weights, whichever scored better on validation, on SEED-AVERAGED probabilities (`report_us51.pooled_ft`).                                                                                                                                                | The flag shown to learners depends on it.                                                                                     | Recover the chosen rule and weights from the report code and tables; store in `stats.json`.                                                        |
| G-d | **The saved prediction bundles are 3-seed averages**, and the per-seed job files with validation scores (`experiments/results/multitask/mt63_gradnorm_a15`) are on the lab, not on the Mac.                                                                                                                                                                                                                            | (1) The seed cannot be chosen by validation from local files. (2) A single seed will not match the saved predictions exactly. | (1) Copy the job files from the lab, or recompute validation scores from the three checkpoints on the Mac. (2) Change gate G1 (section 4).         |

## 4. Change to gate G1 (parity), decided now

- Exact check: run all three seeds offline, average, and compare with `results/analysis/us71/MT-GN15__*.npz` (2,500 test utterances, 47,369 phones). This must match within fp16/fp32 noise.
- Serving check: the single chosen seed against the same bundles. Expect a difference of about the seed noise (about 0.02 RMSE on regression heads); report it, do not hide it. Mispronunciation flags: report agreement and F1 against the labels, not a 95% agreement target (a single seed's flags will differ from the average).

## 5. G0 verdict

Not yet passed. Everything needed is located in the repo, but G-a to G-d must be closed in iteration 1 before app code. None of them needs new training. The seed choice is open until G-d (1) is done; default stays one seed.

## 6. Update 2026-11-07 (evening): seed choice, gap G-d (1) closed

Per-seed job files copied from the lab (`experiments/results/multitask/mt63_gradnorm_a15`). Validation error (RMSE; 1 - macro-F1 for phone_mispron_t1.0), fold 0:

| seed | sent_acc | sent_flu | sent_pros | sent_total | word_acc | phone_score | phone_mispron | mean relative error* |
| ---- | -------- | -------- | --------- | ---------- | -------- | ----------- | ------------- | -------------------- |
| 0    | 0.948    | 0.698    | 0.687     | 0.881      | 1.668    | 0.321       | 0.304         | 0.9977               |
| 1    | 0.961    | 0.683    | 0.681     | 0.878      | 1.672    | 0.331       | 0.306         | 1.0003               |
| 2    | 0.949    | 0.687    | 0.676     | 0.875      | 1.673    | 0.323       | 0.323         | 1.0020               |

*Each head's error divided by its mean over the 3 seeds, then averaged over heads. Differences are about 0.5%, well inside seed noise (about 0.01-0.02).
Decision: **serve seed 0.** It is best or tied on 4 of 7 heads, and best on the mispronunciation head, the one learners see most. The stored `mean_rel` (seed 2 lowest, 0.665) is not used: it divides by each seed's own head-only epoch-0 error (1.51, 1.29, 1.51), so it is not comparable across seeds.
G-c: the job files store `val_proba`, `val_pred` and `classes`, so the decision rule is recoverable by comparing argmax(val_proba) with val_pred (iteration 1).
