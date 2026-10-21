# US 5.3 - weakness fixes for the fine-tuned encoder (Sprint 5)

Code: `configs/finetune.yaml` (`variants`, `diagnose53`, `train.augment`), `src/models/finetune.py` (`add_noise`), `src/evaluation/report_us53.py`,
`run_53_local.sh`, `tests/test_report_us53.py`. Plan and targets: `docs/weakness_memo_s5.md` section 6 (declared before any variant was run).

## Variants (one knob group each, relative to the 5.1 recipe `base`)

| variant               | change                                                                                                    | targets weakness   |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ------------------ |
| `lr`                  | encoder lr 3e-5 -> 1.5e-5, layer-wise decay 0.85 (layer 1 gets 0.17x), grad clip 1.0 -> 0.5               | W1, W2, W4 (spike) |
| `reg`                 | head dropout 0.2 -> 0.4, weight decay 0.01 -> 0.1 (head and encoder)                                      | W1, W2             |
| `mask`                | HF SpecAugment time masking, 10 % of frames in 2-frame spans (phones are 3-10 frames), no feature masking | W1, W2             |
| `noise`               | white noise at 15-35 dB SNR on 50 % of training utterances (timing, spans, durations, labels untouched)   | W1, W2             |
| `freeze6` / `freeze9` | bottom 6 / 9 transformer layers stay pretrained (`freeze9`: W6 tasks only)                                | W1, W2, W6         |

Speed / pitch augmentation is not run: 5.2 found no speaker-memorisation effect (C5), and speed / pitch change the fluency / prosody labels. It is the last resort
if the variants above leave a confirmed gap.

## Run order (lab GPU)

```bash
git pull
python -m pytest tests/test_finetune.py tests/test_report_us53.py     # incl. the torch tests for masking / noise / freezing
bash run_53_local.sh smoke        # 2 epochs per variant on 2 tasks
bash run_53_local.sh train lr     # one variant (~1 h); or: train all  (resumable, several hours)
bash run_53_local.sh report       # results/tables/finetune/us53_report.md
```

Jobs of variant `v` land in `experiments/results/finetune/v/...` and checkpoints in `results/models/finetune/v/...` (git-ignored). The report can be run after any
subset of variants; variants with missing jobs are not eligible.

## Selection rule (validation-side only)

Per task: **target met** = train-validation gap at the best epoch <= 0.30, last-epoch validation loss <= 1.25x (regression) / 1.5x (classification) its minimum, no training-loss
spike >= 2x. **Eligible** = validation metric not worse than base by more than max(0.01, seed sd of base). If base already meets the targets it is kept (W6 tasks: the cheapest eligible
frozen-bottom variant that is >= 20 % faster). Otherwise the eligible, target-meeting variant with the best validation metric; if there is none, base is kept and the report says the
weakness was not fixed. Test numbers (paired bootstrap vs base and vs frozen 4.3) are printed for every variant but never select. The selected variant per task is the "stabilized model";
its checkpoints are the ones under that variant's tag.

## Caveats

- Validation is 19 speakers; with 5-6 variants a winner is partly luck (winner's curse), which is why the margin and the target are required and why test deltas of all variants are shown.
- L2-ARCTIC has one seed per fold, so its margin is the 0.01 floor.
- Mispronunciation / L2 phone may still be improving at the 15-epoch cap (W5); a longer schedule is a follow-up on the selected variant, not part of this screen.
