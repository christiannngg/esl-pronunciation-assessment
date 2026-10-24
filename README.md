# Deep Learning for Multi-Level ESL Pronunciation Assessment and Feedback

A research project comparing traditional acoustic-feature machine learning,
frozen pretrained speech representations (wav2vec 2.0 / HuBERT / WavLM), and
a fine-tuned multi-task deep learning model for automated, multi-level ESL
pronunciation assessment — plus a demo web app.

**Research questions:**

1. Do pretrained deep speech representations improve multi-level ESL
   pronunciation assessment compared with traditional acoustic-feature
   approaches?
2. Does multi-task learning improve sentence-, word-, and phoneme-level
   prediction compared with independently trained models?
3. How well do these models generalize to speakers not seen during training?

Datasets: [SpeechOcean762](https://arxiv.org/pdf/2104.01378) and
[L2-ARCTIC](https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.pdf).
See `data/README.md` for acquisition and licensing details.

---

## Repository Structure

```txt
esl-pronunciation-assessment/
├── data/                # raw/ processed/ external/ — see data/README.md
├── src/                 # data/ models/ experiments/ evaluation/ visualization/ utils/
├── notebooks/           # exploration & experiment notebooks
├── experiments/         # configs/ scripts/ results/
├── tests/                # unit/integration tests
├── docs/                # methodology, results, technical report
├── results/              # figures/ tables/ models/ analysis/
├── presentation/         # slides + demo materials
├── requirements.txt      # pip deps (works on Colab too)
└── environment.yml       # conda deps (school cluster / local Mac)
```

---

## Environment Setup

This project runs across **three environments**: Google Colab, the school
GPU cluster, and a local Mac. Use whichever install method fits the
environment — both are kept in sync.

### Option A — pip (any environment, including Colab)

```bash
python -m venv .venv
source .venv/bin/activate          # on Colab, skip this — just pip install directly
pip install -r requirements.txt
```

### Option B — conda (recommended for the school cluster and local Mac)

```bash
conda env create -f environment.yml
conda activate esl-pronunciation
```

> The conda environment additionally installs `ffmpeg` and the
> **Montreal Forced Aligner** (used for phoneme/word alignment in Sprint 2),
> which aren't reliably available via pip.

### GPU note (school cluster)

If `torch.cuda.is_available()` is `False` after installing, your CUDA driver
version likely doesn't match the default PyTorch build. Reinstall with the
matching wheel, e.g.:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu121
```

### macOS note — OpenMP crash

On macOS with conda, you may see:

```txt
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This happens because conda-forge packages (numpy, scikit-learn) and the
`torch` pip wheel each bundle their own OpenMP runtime. Fix:

```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate && conda activate esl-pronunciation
```

This only appears to affect the local macOS/conda setup — not expected on
Colab or the school cluster, but re-check there too.

---

## Experiment Tracking (Weights & Biases)

We use **W&B** so runs from Colab, the school cluster, and the local Mac all
land in one shared dashboard.

1. Create a free account at [wandb.ai](https://wandb.ai) if you don't have one.
2. On **each machine**, run once:

   ```bash
   wandb login
   ```

3. Use the shared helper in `src/utils/tracking.py` to start runs — it
   auto-tags which environment a run came from:

    ```python
    from src.utils.tracking import init_run

    run = init_run(job_type="baseline-train", config={"model": "svm"})
    run.log({"val_mae": 0.42})
    run.finish()
    ```

All runs are logged under the W&B project **`esl-pronunciation-assessment`**.

---

## Quickstart / Smoke Test

After setting up the environment **on each machine**, verify everything works:

```bash
python experiments/scripts/smoke_test.py
```

This checks that core packages import, reports whether a GPU is visible to
PyTorch, and logs a test run to W&B. Run it on Colab, the school cluster, and
your local Mac before starting Sprint 1's data acquisition (US 1.2).

---

## Project Status

Sprints 1–5 of 10 are complete (midterm checkpoint). Each model sprint ends with a documented weakness diagnosis and fix.

| Sprint | Scope                                                        | Status  | Write-up                                                                          |
| ------ | ------------------------------------------------------------ | ------- | --------------------------------------------------------------------------------- |
| 1      | Environment, dataset acquisition                             | Done    | `data/README.md`, `data/manifest.md`                                              |
| 2      | EDA, preprocessing, speaker-independent splits               | Done    | `docs/eda_summary.md`, `docs/preprocessing_summary.md`                            |
| 3      | MFCC baselines, weakness diagnosis and fix                   | Done    | `docs/weakness_memo_s3.md`, `docs/us33_fix_summary.md`                            |
| 4      | Frozen WavLM / HuBERT / wav2vec 2.0 heads, diagnosis and fix | Done    | `docs/us41_frozen_heads.md`, `docs/weakness_memo_s4.md`                           |
| 5      | End-to-end fine-tuned WavLM, diagnosis and fix               | Done    | `docs/us51_finetuning.md`, `docs/weakness_memo_s5.md`, `docs/us53_fix_summary.md` |
| 6      | Multi-task model (shared encoder, per-task heads)            | Next    |                                                                                   |
| 7      | Midterm evaluation and scope review                          | Planned |                                                                                   |
| 8      | Generalization and ablations (unseen speakers, by L1)        | Planned |                                                                                   |
| 9      | Web app (FastAPI + React) and user validation                | Planned |                                                                                   |
| 10     | Final evaluation, report, presentation                       | Planned |                                                                                   |

### Results so far (test set)

Test scores are for the best model at each stage. Macro-F1 (higher is better):

| Task                            | MFCC baseline (S3) | After S3 fix | Frozen WavLM (S4) | Fine-tuned (S5.1) | Stabilized (S5.3) |
| ------------------------------- | ------------------ | ------------ | ----------------- | ----------------- | ----------------- |
| L2-ARCTIC phone error type      | 0.326              | 0.403        | 0.523             | 0.559             | 0.576             |
| SpeechOcean762 mispronunciation | 0.522              | 0.574        | 0.661             | 0.680             | 0.680             |

Full tables are in `results/tables/` (`baseline/`, `frozen/`, `finetune/`). All choices are made on validation speakers. Test is scored afterwards, with paired speaker-level bootstrap confidence intervals.

### Weakness -> fix case studies

1. **Sprint 3, class imbalance.** The MFCC baseline labelled almost every phone "correct" (addition recall 0.00). Class weights and tuned decision thresholds raised L2 phone error macro-F1 from 0.254 to 0.403 for the random forest, which is the model that fix applied to.
2. **Sprint 4, wrong encoder layer.** wav2vec 2.0's last layer was poor on L2-ARCTIC. Re-selecting the layer gained +0.113 macro-F1 on phone error and made the frozen comparison fair.
3. **Sprint 5, overfitting.** Early stopping hid large train-validation gaps. Six fix variants were tested. A lower encoder learning rate fixed over-confident classifier heads (L2 phone error +0.017, significant). On the sentence-score tasks the gap closed, but test performance did not improve.

### Known open issues

- Fine-tuning flattens predictions for the lowest-scoring words and phones, which hurts feedback quality. It is not yet addressed; a low-score re-weighted loss is planned for Sprint 6.
- Two classification tasks still hit the 15-epoch cap.
- L2-ARCTIC prompts overlap across speakers, so those results measure generalization to new speakers, not new sentences.

### Reproducing results

- Setup: see Environment Setup above and `environment.yml`.
- Preprocessing: `python -m src.preprocessing.pipeline {audio,splits,mfa-prep,mfa-run,ingest}`.
- Frozen encoders (Sprint 4): `run_43_local.sh`, `run_43_layers.sh`.
- Fine-tuning (Sprint 5): `run_51_local.sh`, `run_52_local.sh`, `run_53_local.sh`. Each script's header comment lists its usage.
- Reports: `python -m src.evaluation.report_us33` (and `report_us43`, `report_us51`, `report_us53`).
- Tests: `pytest tests`.

See `docs/methodology.md` (added in later sprints) for the full experimental
design, and the project roadmap for the complete 10-sprint plan.
