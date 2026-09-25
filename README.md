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

```
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
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

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

- [x] US 1.1 — Reproducible dev environment
- [ ] US 1.2 — SpeechOcean762 + L2-ARCTIC downloaded and organized

See `docs/methodology.md` (added in later sprints) for the full experimental
design, and the project roadmap for the complete 10-sprint plan.
