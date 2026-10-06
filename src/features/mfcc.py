"""
US 3.1 - frame-level MFCC (+ delta, delta-delta) features, cached per utterance.

    python -m src.features.mfcc                 # all SO762 + annotated L2-ARCTIC utterances
    python -m src.features.mfcc --limit 20      # smoke test

Settings (configs/baseline.yaml -> ``features``): 13 MFCC (c0..c12), 25 ms window, 10 ms hop, 40 mel bands,
20-7600 Hz, delta/delta-delta over +-4 frames, per-utterance CMVN. Output: float32 array (T, 39), frame i is
centred at ``i * HOP / SR`` seconds. Input is the Sprint 2 audio (16 kHz mono, DC removed, peak normalised,
silence never trimmed).

Note: CMVN removes absolute level (including c0), so raw energy is *not* available to the MFCC baseline;
energy / pitch / pause features are the US 3.3 feature-set variants.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.utils.paths import get_data_root

SR = 16000
N_MFCC = 13
N_FFT = 400          # 25 ms
HOP = 160            # 10 ms
N_MELS = 40
FMIN, FMAX = 20, 7600
HOP_S = HOP / SR
DIM = 3 * N_MFCC


def feature_root() -> Path:
    return get_data_root() / "processed" / "features"


def audio_path(dataset: str, utt_id: str, speaker: str | None = None) -> Path:
    base = get_data_root() / "processed" / "audio" / dataset
    return base / f"{utt_id}.wav" if dataset == "so762" else base / speaker / f"{utt_id}.wav"


def mfcc_path(dataset: str, utt_id: str, speaker: str | None = None) -> Path:
    base = feature_root() / "mfcc" / dataset
    return base / f"{utt_id}.npy" if dataset == "so762" else base / speaker / f"{utt_id}.npy"


def mfcc_from_signal(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """(T, 39) CMVN-normalised [MFCC, delta, delta-delta]."""
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr}")
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                             n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=True)
    t = m.shape[1]
    width = min(9, t if t % 2 == 1 else t - 1)       # delta window must be odd and <= T
    if width >= 3:
        d1 = librosa.feature.delta(m, width=width, order=1, mode="nearest")
        d2 = librosa.feature.delta(m, width=width, order=2, mode="nearest")
    else:
        d1 = d2 = np.zeros_like(m)
    f = np.vstack([m, d1, d2]).T
    f = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
    return f.astype(np.float32)


def extract_file(args) -> tuple:
    dataset, utt_id, speaker = args
    out = mfcc_path(dataset, utt_id, speaker)
    if out.exists():
        return utt_id, "cached"
    try:
        y, sr = sf.read(audio_path(dataset, utt_id, speaker), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, mfcc_from_signal(y, sr))
        return utt_id, "ok"
    except Exception as e:  # reported, never silently dropped
        return utt_id, f"error: {type(e).__name__}: {e}"


def load_mfcc(dataset: str, utt_id: str, speaker: str | None = None) -> np.ndarray:
    return np.load(mfcc_path(dataset, utt_id, speaker))


def seg_frames(n_frames: int, start: float, end: float) -> tuple[int, int]:
    """Frame index range [i0, i1) covering [start, end] s; always at least one frame, clamped to the signal."""
    i0 = int(np.floor(start / HOP_S))
    i1 = int(np.ceil(end / HOP_S))
    i0 = min(max(i0, 0), n_frames - 1)
    i1 = min(max(i1, i0 + 1), n_frames)
    return i0, i1


def utterance_table(limit: int | None = None) -> pd.DataFrame:
    """dataset, utt_id, speaker for every utterance that needs features."""
    from src.data.loaders import load_so762
    so = load_so762(with_audio_info=False).utts[["utt_id", "speaker"]].assign(dataset="so762")
    ph = pd.read_csv(get_data_root() / "processed" / "alignments" / "l2arctic_phones_clean.csv.gz",
                     usecols=["utt_id", "speaker", "corpus"])
    ph = ph[ph["corpus"] == "arctic"][["utt_id", "speaker"]].drop_duplicates()
    ph = ph.assign(dataset="l2arctic")      # scripted, annotated utterances only (suitcase narratives excluded)
    if limit:
        so, ph = so.head(limit), ph.head(limit)
    return pd.concat([so, ph], ignore_index=True)[["dataset", "utt_id", "speaker"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    todo = utterance_table(a.limit)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(extract_file, list(todo.itertuples(index=False, name=None)), chunksize=32))
    status = pd.Series([s.split(":")[0] for _, s in res]).value_counts().to_dict()
    errs = [r for r in res if r[1].startswith("error")]
    print(f"[mfcc] {len(res)} utterances: {status}")
    for r in errs[:10]:
        print("  ", r)
    if errs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
