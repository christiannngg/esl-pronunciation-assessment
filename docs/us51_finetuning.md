# US 5.1 - end-to-end fine-tuning (Sprint 5)

Code: `src/models/finetune.py`, `configs/finetune.yaml`, `src/evaluation/report_us51.py`, `run_51_local.sh`, `tests/test_finetune.py`.

## Run order (lab GPU)

```bash
git pull
bash run_51_local.sh verify     # pooling in the fine-tuning code == cached Sprint 4 embeddings (must print OK)
python -m pytest tests/test_finetune.py   # tiny-model tests (torch + transformers)
bash run_51_local.sh smoke      # 2 epochs, 1 sentence + 1 phone job: check s/epoch and GPU memory
bash run_51_local.sh train      # 9 tasks x folds x seeds, resumable (tag base)
bash run_51_local.sh report     # results/tables/finetune/us51_report.md
```

Jobs: SO762 tasks 1 fold x 3 seeds (7 tasks = 21 jobs); L2-ARCTIC tasks 4 folds x 1 seed (2 tasks = 8 jobs). Outputs land in
`experiments/results/finetune/<tag>/<task>/fold<k>/seed<s>.npz` (predictions, per-epoch history, layer weights) and
`results/models/finetune/<tag>/...pt` (best checkpoint, fp16). Both folders are git-ignored.

## What is held fixed relative to Sprint 4

Tasks, rows and row order, splits, pooling units and spans, side features (standardised on train rows), loss (MSE / balanced class weights),
decision weights (US 3.3 `cw+thr`, tuned on pooled validation), metrics, speaker-level paired bootstrap. The encoder is WavLM-base-plus, the
strongest frozen encoder on 7 of 9 tasks in US 4.3. One encoder is fine-tuned per task (the comparator for Sprint 6's multi-task question).

## Recipe (US 5.1 = plain; US 5.3 changes one group of knobs at a time)

- CNN front end frozen; transformer + projection trained. Learned softmax weights over hidden states 0-12 of per-layer LayerNorm'd pooled vectors
  (the weight logits get their own larger learning rate: US 4.3 found them stuck near uniform at the head rate).
- Head-only for the first epoch (encoder in eval mode, no graph), then linear warm-up of the encoder lr and linear decay. AdamW, grad clip 1.0, bf16 autocast.
- Early stopping on the validation primary metric (RMSE / macro-F1) with patience 4; best checkpoint kept.
- Checkpoint dropouts kept; layerdrop and HF SpecAugment masking switched OFF (they are US 5.3 candidates).

## Caveats to carry into the weakness memo

- The first conv layer of these checkpoints is a GroupNorm over time, so padding changes the output. Training/early-stopping evaluation uses
  length-bucketed batches (<= 5 % length spread); every headline prediction (val / test / train subsample) uses batch size 1, as the frozen extraction did.
- Validation (19 speakers on SO762) drives early stopping and the decision weights; it cannot rank candidates closer than about 0.05 RMSE (US 4.2).
- No out-of-fold probabilities exist for fine-tuned models, so mispronunciation decision weights are the validation-tuned `cw+thr` rule, not the US 4.3
  age-group OOF weights. The paired comparison is against the 4.3 headline system; the matched control is the 4.3 `cw+thr` row.
- Seeds: SO762 averages 3 seeds (as US 4.1); L2-ARCTIC uses one seed per fold, so its seed variance is not measured.
- Not yet verified on a GPU: everything under the torch tests and `verify` was written without access to torch on the development machine.
