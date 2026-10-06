# US 2.2 — Preprocessing and speaker-independent splits

## How to reproduce
```
python -m src.preprocessing.pipeline audio      # 16 kHz mono, DC removal, peak norm (gain cap 20 dB), QC flags
python -m src.preprocessing.pipeline splits     # speaker lists -> data/splits/
python -m src.preprocessing.pipeline mfa-prep   # MFA corpora + SO762 dictionary + phone-set check
python -m src.preprocessing.pipeline mfa-run    # needs MFA (esl-pronunciation conda env)
python -m src.preprocessing.pipeline ingest     # tidy tables + results/tables/preprocessing/alignment_report.json
pytest tests                                    # 39 tests
```
Settings live in `configs/preprocess.yaml` (seed 42, val fraction 0.15, MFA beam 100 / retry 400).

## Audio
- SO762: 5,000 files; L2-ARCTIC: 3,599 manually annotated utterances (the arctic corpus; the 22 suitcase files are excluded).
- 0 read errors; 0 clipped / low-level / too-short / empty flags. Gain applied: 0–20 dB (SO762 median ≈ 6.7 dB, L2-ARCTIC ≈ 4 dB).
- Silence is never trimmed (alignment boundaries and pause/fluency cues depend on it).

## Splits (`data/splits/`)
| | train | val | test |
|---|---|---|---|
| SO762 speakers | 106 | 19 | 125 (official test, unchanged) |
| SO762 utterances | 2,120 | 380 | 2,500 |

- Val = stratified by age group × gender from the official train speakers; among 200 seeded draws the one with the smallest max KS statistic vs the remaining train speakers on the four sentence scores was kept (0.040; test was never consulted).
- L2-ARCTIC: 24 speakers, 4 grouped folds of 6 (one per L1, 3F + 3M). Outer fold k: test = k, val = (k+1) mod 4, train = the other two.
- Checks verified against the raw files: SO762 test speakers equal the raw `test/spk2utt`; train+val equal the raw `train/spk2utt`; all splits disjoint. Speaker lists, `so762_speakers.csv`, `l2arctic_folds.csv`, `splits_metadata.json` (seed, counts, hashes) are saved.

## Alignment
**SO762 (no timestamps shipped → MFA 3.4.2, `english_us_arpa` acoustic model).**
- Each (word, canonical pronunciation) pair is its own dictionary token, so MFA aligns the pronunciation the corpus intended. A first attempt with ordinary multi-pronunciation entries let MFA pick its own variants and 53% of utterances were rejected by the structural check; that was a pipeline bug, not a data property.
- Result: 5,000 / 5,000 utterances pass structural checks (word count, word text, phones per word equal canonical, ordered non-overlapping intervals, end ≤ audio duration). 31,816 word rows, 94,445 phone rows, equal to the manifest counts; no null values.
- Caveat: "structurally valid" is not "accurate". Phone durations: median 110 ms, 3.2% ≤ 30 ms, 1.5% > 500 ms. Phones longer than 500 ms score lower (mean 1.63 vs 1.86 overall), so some long phones are probably absorbing pauses or mispronounced material. Treat boundaries of low-scoring phones with caution.
- Two bare vowels (IH, UH) in `scores.json` were mapped to IH0 / UH0 in the MFA dictionary only.

**L2-ARCTIC (manual-annotation TextGrids already carry boundaries).** MFA was run only as a consistency check vs the provided forced alignments (made with MFA v1.0 and the CMU dictionary):
- Word boundaries, 3,575 utterances with matching word sequences: median 10 ms difference, 61% within 20 ms, 92% within 50 ms.
- Phone starts, 1,796 utterances with identical phone sequences: median 10 ms, 77% within 20 ms, 96% within 50 ms. The other 1,803 utterances differ in phone choice (AO R vs ER, AA vs AO, …) because the dictionaries differ; this subset is therefore biased toward easier utterances, so the word-level figure is the more representative one.
- Annotators left about 90% of boundaries within 1 ms of the provided forced alignment (3,380 utterances compared; 219 had different phone counts). The manual annotation is therefore not independent boundary ground truth.
- Labels: 146,976 phone rows; 376 repaired typos/variants; 0 unmapped canonical phones.

## Decisions and open items
- `*` accented-variant tags are counted as errors with an `is_deviation` flag (`count_deviation_as_error` switch in `labels.clean_l2arctic_phones`).
- AX is mapped to AH (assumption; AX is a reduced vowel).
- SO762 phone scores are kept as the continuous 0–2 value. Binarisation threshold is a modelling choice for Sprint 3+; default < 1.0 = mispronounced (alternatives < 0.5, < 1.5).
- 4 folds for L2-ARCTIC is my assumption; the structure (2F+2M per L1) only supports 4 gender-balanced folds without regrouping.
- Not done: no human-verified boundary accuracy for SO762; L2-ARCTIC val/test use the annotators' labels only for the annotated subset (3,599 utterances).
