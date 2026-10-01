# Data

This directory holds all project datasets. **Raw data is not committed to
git** (see `.gitignore`) — this file documents how to obtain it and where
it actually lives.

## Storage location

Both datasets are stored in a shared Google Drive folder so they're
accessible from Colab (mounted automatically) and from local/cluster
machines without downloading twice:

Google Drive: MyDrive/esl-pronunciation-data/raw/
├── speechocean762/
└── l2arctic/

- **On Colab**: `src/utils/paths.py` mounts Drive automatically.
- **On your local Mac or the school cluster**: set the `DATA_ROOT`
  environment variable to wherever that Drive folder is synced locally,
  e.g. in your shell profile:

```bash
  export DATA_ROOT="/Users/chirstiangonzalez/Google Drive/My Drive/esl-pronunciation-data"
```

  If `DATA_ROOT` isn't set, code falls back to the repo-local `data/`
  directory — fine for quick tests, not for the full datasets.

## Datasets

### SpeechOcean762

- **Description**: 5,000 English utterances from 250 non-native (Mandarin
  L1) speakers, with expert sentence/word/phoneme-level pronunciation
  annotations (accuracy, completeness, fluency, prosody).
- **License**: **CC BY 4.0** — free for commercial and non-commercial use,
  attribution required.
- **Source**: [OpenSLR resource 101](https://www.openslr.org/101/) /
  [GitHub (jimbozhang/speechocean762)](https://github.com/jimbozhang/speechocean762)
- **Access**: No approval needed —
  `git clone https://github.com/jimbozhang/speechocean762.git`
- **Citation**:

```bibtex
  @inproceedings{zhang2021speechocean762,
    title={speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment},
    author={Zhang, Junbo and Zhang, Zhiwen and Wang, Yongqing and Yan, Zhiyong and Song, Qiong and Huang, Yukai and Li, Ke and Povey, Daniel and Wang, Yujun},
    booktitle={Proc. Interspeech 2021},
    year={2021}
  }
```

### L2-ARCTIC

- **Description**: 26,867 utterances (27.1 hours) from 24 non-native
  English speakers across 6 L1 backgrounds (Arabic, Mandarin, Hindi,
  Korean, Spanish, Vietnamese), with forced-aligned phoneme transcriptions
  and 3,599 manually annotated utterances tagging substitution/deletion/
  addition errors.
- **License**: **CC BY-NC 4.0 — non-commercial use only.** This is more
  restrictive than SpeechOcean762's license — worth remembering if the
  demo web app (Sprint 9) is ever taken beyond coursework/portfolio use.
- **Source**: [psi.engr.tamu.edu/l2-arctic-corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- **Access**: Requires filling out a form (name, email, affiliation) on
  the site above; an automated email with a Google Drive download link
  follows within ~10 minutes. No public/instant download exists.
- **Citation**:

```bibtex
  @inproceedings{zhao2018l2arctic,
    author={Guanlong {Zhao} and Sinem {Sonsaat} and Alif {Silpachai} and Ivana {Lucic} and Evgeny {Chukharev-Hudilainen} and John {Levis} and Ricardo {Gutierrez-Osuna}},
    title={L2-ARCTIC: A Non-native English Speech Corpus},
    year=2018,
    booktitle={Proc. Interspeech},
    pages={2783-2787},
    doi={10.21437/Interspeech.2018-1110}
  }
```

## Directory structure per dataset

**SpeechOcean762** (Kaldi-style):

speechocean762/
├── scores.json # averaged/median expert scores per utterance
├── scores-detail.json # all 5 experts' individual scores
├── train/ # spk2age, spk2gender, spk2utt, text, utt2spk, wav.scp
├── test/ # (same files as train/)
└── WAVE/SPEAKER####/*.WAV

**L2-ARCTIC** (per-speaker):

l2arctic/
├── README.md, LICENSE, PROMPTS
├── <SPEAKER_CODE>/
│ ├── wav/ # 44.1kHz audio
│ ├── transcript/ # orthographic transcriptions (.txt)
│ ├── textgrid/ # forced-aligned phoneme transcriptions
│ └── annotation/ # manual error annotations (.TextGrid), ~150/speaker
└── suitcase_corpus/ # spontaneous speech, 22/24 speakers (added v5.0)

## Verification & manifest

After downloading both datasets into the locations above:

```bash
python src/data/verify_integrity.py   # checks file counts against known-good totals
python src/data/build_manifest.py     # writes data/manifest.json + data/manifest.md
```
