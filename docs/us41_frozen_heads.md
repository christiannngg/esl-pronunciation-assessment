# US 4.1: task heads on frozen embeddings and the benchmark against Sprint 3

Code: `src/models/frozen_heads.py` (training) · `src/evaluation/report_frozen.py` (tables) · `experiments/scripts/export_side_features.py` (side features) ·
config: `configs/frozen.yaml` → `heads:` · tests: `tests/test_frozen_heads.py` · Colab: `notebooks/03_frozen_extraction_colab.ipynb`, sections 10–14.
Embeddings come from `docs/us41_frozen_extraction.md`.

## What is trained

Same nine tasks as the Sprint 3 headline (4 sentence scores, word accuracy, phone score, mispronounced-phone at threshold 1.0, L2-ARCTIC phone error class, L2-ARCTIC word error),
same rows, same splits (SO762 train/val/test speakers from `data/splits/so762_speakers.csv`; L2-ARCTIC 4 grouped folds, test = k, val = (k+1) % 4).
Row order is the Sprint 3 meta order; the loader asserts that embedding rows and side-feature rows equal the meta row count.

| setting (`--mode`)     | input to the head                                                 | purpose                                                                                          |
|------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `emb+side` (main)      | frozen embedding of one layer + the Sprint 3 non-acoustic columns | fair comparison: the MFCC baselines also saw canonical phone identity, durations, word position  |
| `emb`                  | embedding only                                                    | ablation: how much do the side columns add?                                                      |
| `side`                 | side columns only                                                 | ablation: what the heads can do with no acoustic embedding at all                                |

Side columns (exported from the Sprint 3 tables by name, no MFCC statistics): sentence `dur, n_words, n_phones`; SO762 word `dur, n_phones, phone_dur_mean/std, n_chars, word_pos`;
SO762 phone `dur, log_dur, has_prev/next, word_initial/final, n_phones_in_word, is_vowel` + 39-d canonical phone one-hot; L2 word `dur, n_canon_phones, n_chars`;
L2 phone `has_prev/next, is_vowel` + anchor-phone one-hot. **The annotated phone duration never reaches the L2 phone heads** (exporter asserts it), same leakage control as Sprint 3.

Heads: `linear` (single linear layer = linear probe) and `mlp` (one hidden layer of 512, GELU, dropout). AdamW, early stopping on validation loss (patience 5, max 40 epochs),
inputs standardised with training-row statistics only, regression on the standardised target.

## Protocol (identical in spirit to Sprint 3)

* Hyper-parameters (weight decay / dropout / class weights) and the early-stopping epoch are chosen on **validation only** (regression RMSE, classification macro-F1).
* Test predictions are written at fit time; the report reads only the validation-selected config.
* `search` stage: every grid point, one seed. `final` stage: the selected config per (task, head, class-weight group) re-fitted with seeds 0,1,2; the seed-averaged prediction / probability is the system output; the seed spread of the validation score is reported.
* Classification variants: `plain` (no class weights, the US 3.1 analogue), `cw` (balanced class weights in the loss), `cw+thr` (cw plus validation-tuned decision weights, the US 3.3 recipe, cross-fitted for the validation estimate).
  The variant shown as the head's result is chosen on validation.
* Test CIs: 95 % speaker-level bootstrap (500 draws, seed 0). Deltas against Sprint 3: paired speaker bootstrap on identical test rows (asserted equal).
* References: **US 3.1** = best Sprint 3 model per task by validation score; **US 3.3** = best model after the validated fix (`results/tables/baseline/us33_before_after.csv`).

## Running it

```bash
# on the Mac, once
python experiments/scripts/export_side_features.py
python experiments/scripts/pack_colab_data.py --side           # dist/esl_side_pack.tar (about 40 MB) -> Drive

# Colab (notebook sections 10-14): per encoder
python -m src.models.frozen_heads run --encoder wavlm_base_plus --stage search
python -m src.models.frozen_heads run --encoder wavlm_base_plus --stage final
# other layers / ablations:   --layer 8     --layer 6-9     --mode emb     --mode side

# on the Mac, with experiments/results/frozen/ copied back from Drive
python -m src.evaluation.report_frozen --encoders wavlm_base_plus hubert_base wav2vec2_base
python -m src.evaluation.report_frozen --encoders wavlm_base_plus --mode emb      # ablation table
```

Outputs: `results/tables/frozen/<encoder>_L<layer>_<mode>_results.{csv,md}` and `frozen_vs_baseline_L<layer>_<mode>.{csv,md}`.

## Things to keep in mind when reading the numbers

* Layer 12 is a **first benchmark**, not a claim: for pretrain-only checkpoints the last layer is often not the best one. The per-layer sweep is a US 4.2 diagnostic and a US 4.3 fix.
* The US 3.3 sentence-level references use pitch / energy / pause features (`pros`); the frozen heads here do not, so a gap on sentence tasks may be a prosody gap and not a representation gap. US 4.2 separates the two.
* The embedding of a very short phone inherits context from its neighbours (25 ms receptive field), and L2-ARCTIC phones use a fixed 150 ms window by design.
* Early stopping uses the validation split, which is also used for selection; the test split is touched once.
