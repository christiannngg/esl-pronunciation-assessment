# Sprint 9 plan: web app and user validation (US 9.1, 9.2)

**Status:** draft for review, 2026-11-07, written before any Sprint 9 code. Context: `docs/go_forward_plan.md` (section 3), `docs/revised_scope.md` (D6, D7, D9), `docs/us83_fix_summary.md` (MT-GN15 stays the model of record).

## 0. Decisions already made

| #   | decision                                                                                                                  | source              |
| --- | ------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| S1  | The app runs **locally on the Mac** (FastAPI + React on localhost, CPU or MPS). No hosting.                               | answered 2026-11-07 |
| S2  | The app serves **MT-GN15 only** (7 SO762 heads). The L2-ARCTIC model is a stretch (iteration 7).                          | answered 2026-11-07 |
| S3  | User validation: **about 5 peers, in person or by call, no audio kept**. Only ratings and notes are stored.               | answered 2026-11-07 |
| S4  | **No model work.** The checkpoint is frozen (Nov 8). The app must reproduce the saved test predictions, not improve them. | D9                  |

## 1. What the app has to do, and the three constraints that shape it

The roadmap says "record a sentence, get sentence / word / phone feedback". Reading the code and docs gives three constraints that the plan is built around.

1. **The model is text-dependent and alignment-dependent.** Word and phone heads pool encoder frames over MFA intervals, and the labels assume a known canonical pronunciation. So the app must be **prompted-reading**: the learner reads a sentence from a fixed bank, and the backend knows its words and canonical phones. Free speech is out of scope.
2. **Alignment at inference is the largest technical risk (assumption A4).** Training used Montreal Forced Aligner 3.4.2 with the `english_us_arpa` model and a per-(word, pronunciation) dictionary (`src/preprocessing/alignment.py`). MFA lives in a separate conda env and is slow to start per call. This decides the app's architecture, so it is tested first.
3. **The checkpoint may not be self-contained.** `multitask.py` uses "side features" and train-standardised regression targets. Serving needs the standardisation mean/std, the head layout and whatever side features are computed at inference. All of this is inventoried in iteration 0.

**Is CPU-only on the Mac efficient?** Probably yes for the model, but this is an estimate, not a measurement. The encoder is `wavlm_base_plus` (about 95M parameters, one forward pass for all 7 heads). That is usually around a second or less for a short sentence on a recent Apple CPU. I could not benchmark it (my shell is a Linux VM, not your Mac), so iteration 1 measures it. Two practical points: the checkpoint is stored in fp16, so convert it to fp32 on load for CPU (fp16 matmul on CPU is slow or unsupported in places), and **alignment, not the model, is the likely latency bottleneck**.

## 2. Iteration plan

Each iteration ends with something that runs and a gate that must pass before the next one starts. If a gate fails, the fallback is named in advance.

Working dates follow the go-forward plan: Sprint 9 is Nov 9-15, with sessions running to Nov 22.

### Iteration 0: freeze and inventory (Sat-Sun Nov 7-8, about half a day)

- Pick the serving checkpoint among `results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed{0,1,2}.pt`. Default: **one seed, chosen by validation `mean_rel`** (serving a 3-seed ensemble triples latency for a small gain). Record the choice and why in the plan's decision log.
- Inventory what inference needs: encoder key and revision (`configs/frozen.yaml`), head names and kinds, target standardisation constants, side features (what they are and whether they can be computed from audio alone), the classification threshold for `phone_mispron_t1.0`.
- Copy the checkpoint, configs and prediction bundles off the lab if any are still only there (lab access ends Dec 12).
- **Gate G0:** a written list of everything inference needs, with each item located in the repo. If side features need data that only exists at training time, stop and decide before iteration 1.

### Iteration 1: inference core, no web (Mon-Tue Nov 9-10)

Deliverable: `src/serving/` with a function `assess(wav_path, prompt_id) -> dict` and a command-line test harness.

- Audio: decode browser audio (webm/opus or wav) to 16 kHz mono with ffmpeg (already on the Mac). Same normalisation as `src/preprocessing/audio.py`; no silence trimming.
- **Prompt bank:** about 15 short sentences with canonical ARPAbet pronunciations built the same way as the SO762 canonical entries (`so762_canonical`, `build_so762_dictionary`). Prefer sentences whose words are in SO762 so the model has seen the style.
- **Alignment spike (timeboxed to 1 day), in this order:**
    1. MFA called from the backend through the same conda env and dictionary scheme as training (exact parity, but check seconds per call; try keeping a warm process or `mfa align_one`).
    2. If too slow or too fragile: torchaudio CTC forced alignment with a phone-level model, accepting a train/serve mismatch that must be **disclosed and measured** (G1 below).
    3. If neither works by end of Tuesday: **prerecorded-examples demo** (the proposal's own mitigation), with a live-record mode that only shows sentence-level scores. Word and phone views use prerecorded clips.
- **Gate G1 (parity):** run `assess` on 50 test utterances and compare with the saved predictions (`results/analysis/us71/S8-*.npz`). Pass criteria, fixed now: sentence scores within 0.1 of the saved values, and phone mispronunciation flags agree on at least 95% of phones. If the alignment fallback (2) is used, report the larger gap instead of hiding it.
- **Gate G1b (latency):** median end-to-end time for a 5 s utterance on the Mac, measured. Budget: 10 s or less. Report encoder time and alignment time separately.

### Iteration 2: FastAPI backend (Tue-Wed Nov 10-11)

- Endpoints: `GET /health`, `GET /prompts`, `POST /assess` (multipart audio + `prompt_id`). The model is loaded once at startup and requests are serialised with a lock.
- Response schema (pydantic): four sentence scores, per-word accuracy with time spans, per-phone score and mispronunciation flag with spans, plus a `warnings` list (for example "clip too short", "audio clipped", "possible alignment failure").
- Privacy by design (S3): audio is processed in memory or a temp file deleted in a `finally` block; nothing is written to the log. Add a test that asserts no audio file remains.
- Error handling: empty or tiny audio, over-long audio, unsupported format, unknown prompt, alignment failure. Each has a clear status code and message.
- **Gate G2:** pytest passes for the unit tests and the API integration tests (FastAPI `TestClient` with fixture clips from the test set and a golden-output file). `uvicorn` starts from a single command.

### Iteration 3: React front end, minimum viable (Wed-Thu Nov 11-12)

- Vite + React + TypeScript in `app/web/`. Screens: pick or show a prompt, record (MediaRecorder; `localhost` counts as a secure context so the mic prompt works), upload, loading state, results.
- Results view v1: sentence score bars, the sentence with each word coloured by accuracy, click a word to see its phones.
- **Gate G3:** one full pass, record to results, works on the Mac in Chrome, plus a Playwright smoke test with a recorded fixture clip standing in for the mic.

### Iteration 4: feedback clarity and honesty (Thu Nov 12)

This is where the model's known weaknesses become part of the design instead of a footnote in the report.

- Turn raw scores into learner-friendly bands and labels, with the scale shown. Do not show false precision.
- Show a short, persistent note: scores are estimates for practice, not a grade.
- Reflect documented limits: word accuracy compresses low scores toward the mean (weakness W8), and mispronunciation detection is weaker for child/teen speakers (US 8.1-8.3). Do not show confident "you mispronounced X" language for single phones; use "might need practice".
- Pilot it on yourself plus one friend. Fix anything confusing before the real sessions.
- **Gate G4 (app freeze v0.9, end of Nov 12):** no feature changes after this point except bug fixes, so every participant sees the same app.

### Iteration 5: user validation, US 9.2 (Thu Nov 12 - Sun Nov 22)

- **Protocol (one page, written before recruiting):** 15-minute sessions. Each participant reads 4 prompts, then looks at the feedback. One planted-error probe: they read one sentence normally and one with a deliberate mispronunciation of a chosen word, to see whether the app's feedback visibly changes.
- **Instrument:** 5-point ratings on clarity ("I understood what the feedback told me"), usefulness ("I know what to practise next"), trust ("the scores seemed right to me"), plus one comprehension check (they say in their own words what the app asked them to fix) and 3 open questions. Store anonymous IDs (P1-P5) only.
- **Consent:** short verbal script read at the start: practice app, no audio saved, ratings and notes only, can stop at any time. Include it in the appendix of the summary.
- **Recruiting:** ask on Nov 9-10 so sessions can start Nov 13. Aim for 5; if only 3 are reachable, say so and keep the claim qualitative.
- **Deliverable:** `docs/us92_validation.md`: protocol, results table (ratings per participant, no averages over 5 presented as statistics), quotes, issues found, what you would change. State plainly what this does **not** show: it is not an accuracy test against human raters, the peers may not be ESL speakers, and n is tiny.

### Iteration 6: hardening and documentation (Fri-Sun Nov 13-15, may spill to Nov 22)

- Pin new dependencies in `requirements.txt` and `environment.yml` (add what `fastapi` needs: `python-multipart`, `pydantic`, `httpx` for tests; the MFA environment is documented separately).
- One-command start script for backend and front end; README section "run the demo"; `docs/us91_app.md` covering architecture, schema, the alignment choice, measured latency and the parity result.
- Record a 2-3 minute walkthrough of the app. It doubles as the **weekly video #2 material (Fri Nov 13)** and as the fallback if the live demo fails at the presentation.
- **Gate G5:** a fresh clone follows the README and gets a working demo on the Mac.

### Iteration 7: stretch (only if G5 passes with time left)

Priority order: (a) L2-ARCTIC phone/word error view, which needs the L2 checkpoint and its own prompts; (b) 3-seed ensemble toggle; (c) per-phone audio playback of the learner's own segment (kept in the browser only).

## 3. Calendar

| date                    | focus                                           | exit              |
| ----------------------- | ----------------------------------------------- | ----------------- |
| Sat-Sun Nov 7-8         | Iteration 0; send recruiting messages           | G0                |
| Mon-Tue Nov 9-10        | Iteration 1 (alignment decision on Tue evening) | G1, G1b           |
| Tue-Wed Nov 10-11       | Iteration 2                                     | G2                |
| Wed-Thu Nov 11-12       | Iterations 3, 4                                 | G3, G4 (app v0.9) |
| Thu Nov 12 - Sun Nov 22 | Iteration 5 sessions (target 5)                 | summary draft     |
| Fri Nov 13              | Weekly video #2 with the walkthrough            |                   |
| Fri-Sun Nov 13-15       | Iteration 6                                     | G5                |
| Sun Nov 22              | Results freeze: validation summary final        |                   |

## 4. Risks

| risk                                                | likelihood | mitigation                                                                         |
| --------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------- |
| MFA too slow or fragile at inference                | medium     | Tuesday decision point; CTC fallback with disclosed mismatch; prerecorded fallback |
| Checkpoint needs training-only side features        | low-medium | Found in iteration 0, before any app code                                          |
| Live predictions differ from saved test predictions | medium     | G1 parity test with pre-set tolerances; report the gap if there is one             |
| Browser mic or codec problems                       | low        | Convert with ffmpeg on the server; Playwright test uses a fixture clip             |
| Fewer than 5 participants                           | medium     | Recruit early; report n honestly; heuristic walkthrough as a supplement            |
| Peers are native speakers, not ESL                  | medium     | Say so in the summary; ask for at least 2 non-native speakers                      |
| App polish eats the schedule                        | medium     | Gate G4 freeze; stretch work only after G5                                         |

## 5. Mapping to the rubric

Testing and validation: parity test, API tests, e2e smoke test. Reproducibility: pinned dependencies, one-command start, README. Research standards: privacy handling and the honest limits statement. Presentation: interactive demo with a recorded fallback (interactive elements and Q&A).

## 6. Open items

1. Which of the three seeds, or confirmation that one seed is acceptable (iteration 0).
2. Sentence choice for the prompt bank (iteration 1). I will propose 15 and you approve.
3. Whether any recruits are non-native English speakers (iteration 5).
