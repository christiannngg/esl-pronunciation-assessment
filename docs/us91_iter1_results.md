# Sprint 9, iteration 1: results (gates G1, G1b and the alignment decision)

Date: 2026-11-07. Plan: `docs/sprint9_plan.md`. Inventory: `docs/us91_iter0_inventory.md`. Code: `src/serving/` (`model.py`, `side_features.py`, `aligner.py`, `parity.py`, `time_mfa.py`, `align_sensitivity.py`). Tables: `results/tables/serving/`.
Machine: the researcher's Mac (Apple Silicon), CPU, fp32, PyTorch 2.14, MFA 3.4.2. Served model: MT-GN15, seed 0.

## 1. Gate G1: parity with training predictions (stored alignment)

Reference: the per-seed job file of seed 0 (`test_pred` / `test_proba`). 40 test utterances (evenly spaced), 271 words, 831 phones.

| head               | n   | flag agreement | max abs diff        | mean abs diff | tolerance (set before the run) | result              |
| ------------------ | --- | -------------- | ------------------- | ------------- | ------------------------------ | ------------------- |
| sent_accuracy      | 40  |                | 0.039               | 0.0054        | mean <= 0.02, max <= 0.10      | pass                |
| sent_fluency       | 40  |                | 0.014               | 0.0045        | same                           | pass                |
| sent_prosodic      | 40  |                | 0.017               | 0.0046        | same                           | pass                |
| sent_total         | 40  |                | 0.024               | 0.0065        | same                           | pass                |
| word_accuracy      | 271 |                | **0.115**           | 0.0060        | same                           | **max above limit** |
| phone_score        | 831 |                | 0.031               | 0.0011        | same                           | pass                |
| phone_mispron_t1.0 | 831 | **100 %**      | 0.037 (probability) | 0.0004        | agreement >= 99 %, max <= 0.05 | pass                |

**Verdict: pass, with one documented exception.** The largest word-accuracy difference over 271 words is 0.115 on a 0-10 scale, against a limit of 0.10 (the head's target standard deviation is 1.98, so 0.06 sd). The mean difference is 0.006 and all 831 mispronunciation flags agree. The cause was not isolated. It is consistent with fp32 here against bf16 autocast on the GPU during training, but that was not tested. The limit is not changed; the exception is reported as is.

## 2. Gate G1b: latency (CPU, batch size 1)

- Model forward (encoder + all 7 heads), 40 utterances: median 0.09 s, p90 0.11 s. Model load: about 3 s.
- MFA `align_one`, one utterance: 3.4-3.7 s warm. The first call after a long pause was 4.4 s in the repeat runs and 18.6 s in the very first run (model/database initialisation), so the backend must align one dummy clip at startup.
- Expected end-to-end per recording: about 4 s plus audio decoding (to be measured through the API in iteration 2). Budget 10 s: met. Alignment is about 97 % of the time; the model is not the bottleneck, and no GPU is needed.

## 3. Alignment: live `mfa align_one` against the training alignment (30 SO762 utterances)

Training used batch `mfa align` (about 20 utterances per speaker, with speaker adaptation); the app aligns one recording. All 30 live alignments passed the structural checks (same ingest as training).

- Boundary differences per utterance (largest phone boundary): <= 30 ms in 12 clips, 30-100 ms in 6, > 100 ms in 12. The per-clip median difference is 0 ms in every clip but one.
- 41 of 500 phones differ by more than 100 ms, in 12 clips. In 11 of them it is one local patch of 2-5 neighbouring phones (a long vowel with a chain of 30 ms phones, then boundaries re-sync). One clip (`096490020`) is shifted by 0.6-1.0 s over most of the utterance; the training alignment of that clip is also odd ("IT'S" lasts 1.46 s), so neither alignment is clearly right.

## 4. What the alignment differences do to the model outputs

Same audio, same model, stored alignment against live alignment (`align_sensitivity.csv`).

| subset                  | clips | mean abs change, sentence total | word accuracy | phone score | mispronunciation flags changed |
| ----------------------- | ----- | ------------------------------- | ------------- | ----------- | ------------------------------ |
| boundary diff <= 100 ms | 18    | 0.000                           | 0.003         | 0.002       | 0 / 288                        |
| boundary diff > 100 ms  | 12    | 0.000                           | 0.273         | 0.049       | 7 / 212                        |
| all                     | 30    | 0.000                           | 0.111         | 0.021       | 7 / 500                        |

- **Sentence scores do not depend on the alignment** (sentence heads pool the whole utterance and their side features come from the audio length and the prompt).
- All 7 flipped flags come from the single clip `096490020` (word accuracy changes by 2.0 on average; 7 of its 16 phones flip). Without it: 0 flips in 484 phones, the next largest average word change is 0.72 (`000480108`) and the rest are <= 0.14.
- A simple detector was tried on the live alignments (runs of phones at the 30 ms minimum duration). It flags `000480108` and `007390019` but **not** the worst clip `096490020`. No detector is adopted. A better candidate (not tested): MFA's per-utterance alignment likelihood.

## 5. Decision

Use live `mfa align_one` in the backend (warmed up at startup, requests serialised), with these consequences for the app:

1. Sentence-level feedback is reliable with respect to alignment; word and phone feedback is correct in 29 of 30 clips up to small shifts and unreliable in about 1 of 30. The interface (iteration 4) states that word and phone feedback is approximate and uses wording such as "might need practice".
2. Alignment failures that fail the structural checks return an error ("could not align your recording"), never scores.
3. The prerecorded-examples fallback is not needed. The CTC fallback is not needed.
4. Open item: a confidence signal for the alignment (MFA likelihood). Only if time remains.
   Limits: 30 clips of one corpus (read speech, native and non-native), one sampling pattern, SO762 sentences rather than the new prompt bank; the app recordings (browser microphone) may align differently.
