# EDA Summary: Sprint 2, US 2.1

**Notebook:** `notebooks/01_eda.ipynb` (runs end to end in about 20 s) · **Figures:** `results/figures/eda/` · **Tables:** `results/tables/eda/` (full register: `imbalance_edge_case_register.csv`, headline numbers: `key_stats.json`)

Loaders and helpers: `src/data/loaders.py`, `src/data/eda_utils.py` (parser tests: `tests/test_loaders.py`).

## 1. Dataset structure

| SpeechOcean762 | L2-ARCTIC                                                  |
|----------------|------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Speakers       | 250 (125 train / 125 test, no overlap, 20 utterances each) | 24 (4 per L1: Arabic, Hindi, Korean, Mandarin, Spanish, Vietnamese; 2 F + 2 M each) |
| L1             | all Mandarin                                               | 6 L1s |
| Age            | 122 child/teen (6-15) + 128 adult (19-43)                  | adults |
| Audio          | 5,000 utterances, 5.6 h, 16 kHz                            | 26,867 scripted wavs, 27.1 h, 44.1 kHz, plus 22 spontaneous "suitcase" narratives (26 min) |
| Labels         | sentence (accuracy, fluency, prosodic, completeness, total), word (accuracy, stress, total), phoneme (0-2), 5 experts | phone-level error tags (substitution / deletion / addition) on **3,599 scripted utterances (13.4%)** + 22 suitcase files; forced alignment for every scripted file; **no sentence or word scores** |

L2-ARCTIC totals reproduce the README exactly (scripted 14,098 S / 3,420 D / 1,092 A; suitcase 1,673 / 456 / 90). Headline numbers were also recomputed independently from the raw files and matched.

## 2. Class imbalance

## SpeechOcean762

- Completeness is 10 for 99.5% of utterances, so it is not a usable target.
- Word stress is wrong for only 0.9% of words (287 of 31,816), and expert agreement on it is weak (ICC 0.13).
- Phone level: 92.1% of individual expert labels are 2 (correct). The consensus score is below 0.5 for 3.6% of tokens, below 1.0 for 4.8%, and below 2.0 for 19.3%. The binarisation threshold changes prevalence several-fold and must be fixed before Sprint 3.
- Sentence and word scores are left-skewed (modal sentence total score 8; 88.8% of words score 10).

## L2-ARCTIC

- 84.5% of annotated phone events are correct. Among errors: 75.8% substitutions, 18.4% deletions, 5.9% additions (4-class imbalance ratio 93:1).
- Error support is uneven across phonemes: five phonemes (Z, DH, R, D, AH) hold 43% of substitution + deletion events; DH and Z have error rates above 50%; M and OY have fewer than 50 error events.
- 96.4% of annotated utterances contain at least one error, so utterance-level "has error" is not a useful target. Word-level "has error" (41.6% positive) is far more balanced.

## 3. Edge cases and risks

- **Between-speaker variance is large** (70-78% of score variance in SpeechOcean762), so splits must be speaker-independent.
- **Human agreement caps achievable performance**: sentence-level leave-one-out r about 0.77-0.80, phone-level Fleiss kappa 0.46. Rater severity differs systematically (two of the five raters score noticeably lower, by up to about 2 points on fluency).
- **L1 matters in L2-ARCTIC**: errors per 100 phones range from 9.4 (Korean) to 23.4 (Vietnamese); deletions are 32% of Vietnamese errors but 10% of Arabic. Speaker-level rates range 6.8-26.5. Aggregate metrics will hide weak groups.
- **Only 24 L2-ARCTIC speakers**: a fixed speaker-held-out test set of a few speakers is high-variance.
- **Prompt text is shared across speakers** (99 of the 300 annotated L2-ARCTIC prompts are annotated for all 24 speakers; 119 are annotated only for one L1's 4 speakers). Text cannot be disjoint under a speaker split. In SpeechOcean762 only 52 test utterances (2.1%) reuse a train prompt.
- **Annotation tag noise in L2-ARCTIC**: 9,105 of 20,829 error tags contain stray whitespace, 182 use upper-case s/d/a, 6 canonical labels are typos (`D_`, `ER)`, `V``, `W``, `Y_`, `Z_`), 3,675 phone intervals are empty. `parse_phone_label` normalises these.
- **Policy decisions needed for tags**: 1,744 substitutions carry `*` (accented variant, 11.1%; 1,471 or 10.4% of scripted substitutions, concentrated in Arabic and Hindi speakers) and 262 are tagged `err`.
- **Train/test shift in SpeechOcean762 is small**: speaker-level differences by age group are not significant (adult p = 0.26, child/teen p = 0.27), but report child/teen and adult results separately because the groups differ in prompt length, speaking rate and score spread.
- **Different label schemes**: SpeechOcean762 has scores and L2-ARCTIC has error types, so they cannot share a joint label set. Phoneme difficulty for Mandarin speakers is only moderately correlated across the datasets (Spearman 0.50 over 36 phonemes).
- **Audio**: SpeechOcean762 is 16 kHz; L2-ARCTIC is 44.1 kHz and needs resampling. One SpeechOcean762 clip is 20.4 s (longest test clip 12.2 s).

## 4. Recommended modeling and splitting strategy (input to US 2.2)

1. **SpeechOcean762**: keep the official test set unchanged. Build validation from train speakers only (about 19 speakers), stratified by age group × gender × speaker-mean score.
2. **L2-ARCTIC**: speaker-grouped folds stratified by L1 and gender instead of one tiny hold-out; supervised error-type work uses the annotated subset only; suitcase files are excluded from training and validation.
3. **Targets**: viable regression targets are sentence accuracy / fluency / prosodic / total and word accuracy (SpeechOcean762), plus phone score. Viable classification targets are L2-ARCTIC phone error type and derived word-level "has error". Leave completeness and word stress out of the modeling targets (they are not among the six proposed heads); avoid utterance-level "has error".
4. **Imbalance handling** (class weights or resampling) happens inside training splits only. Report macro-F1 / balanced accuracy with per-class support.
5. **Reporting rules for Sprint 3 onward**: per age group, per L1, and relative to the human-agreement ceiling.
6. **Preprocessing decisions for US 2.2**: `*`/`err` tag policy, typo-label mapping, a common 39-phone inventory, resampling to 16 kHz, attachment rule for addition intervals that fall between words.

Scope note: the proposal's six heads are overall pronunciation, fluency, prosody, word-level accuracy, phoneme correctness and error type. Completeness and word stress are not among them, so leaving them out of modeling needs no scope change (the columns stay in the manifests). The head to watch at the Sprint 7 scope review is error type, which only L2-ARCTIC supports.
