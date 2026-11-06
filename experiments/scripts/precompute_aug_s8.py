"""US 8.3 / F2: child-like copies of the ADULT training utterances (train + val speakers only; the official test speakers are never touched).

Each copy is the original utterance shifted UP in pitch with the formants scaled by the same factor (a phase-vocoder pitch shift, i.e. resampling plus
time-stretch back to the original length), which is a standard vocal-tract-length / F0 child-speech approximation. Duration is preserved (and the output is
padded / trimmed to the exact original sample count), so the phone and word timings, and the fluency / prosody labels, stay valid. NO time stretching.

One shift per utterance, drawn from the bank's semitone range with a seed derived from the utterance id, so a rerun gives the same files. Resumable.

    python experiments/scripts/precompute_aug_s8.py [--banks shift_2_4 shift_4_6] [--limit 20] [--cores 8]
Output: $S8_AUG_ROOT (default results/aug_s8) / <bank> / <utt_id>.npy   (float32, normalised exactly like the originals)
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SR = 16000


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros / trim so that ``len(y) == n`` (the spans of the original utterance must stay valid)."""
    y = np.asarray(y, np.float32)
    if len(y) >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y), np.float32)])


def semitones_for(utt_id: str, bank: str, lo: float, hi: float) -> float:
    rng = np.random.default_rng(zlib.crc32(f"{bank}|{utt_id}".encode()))
    return float(rng.uniform(lo, hi))


def pitch_shift(y: np.ndarray, n_steps: float) -> np.ndarray:
    try:
        import librosa
        return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
    except ImportError:
        pass
    try:
        import torch
        import torchaudio.functional as AF
        return AF.pitch_shift(torch.from_numpy(y)[None], SR, n_steps)[0].numpy()
    except ImportError as e:
        raise SystemExit("neither librosa nor torchaudio is installed: `pip install librosa` in the lab venv") from e


_G: dict = {}


def _init(do_norm: bool, root: str, banks: dict):
    _G.update(do_norm=do_norm, root=Path(root), banks=banks)


def _work(job):
    uid, spk, bank = job
    from src.features import ssl_embeddings as se
    from src.models.finetune import normalize_wave
    out = _G["root"] / bank / f"{uid}.npy"
    if out.exists():
        return 0
    lo, hi = _G["banks"][bank]
    w = se.load_wav("so762", uid, spk)
    shifted = fit_length(pitch_shift(w.astype(np.float32), semitones_for(uid, bank, lo, hi)), len(w))
    arr = normalize_wave(shifted, _G["do_norm"])
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, out)
    return 1


def main(argv=None) -> None:
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--banks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N adult utterances (smoke test)")
    ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    from src.models import finetune as FT
    from src.models import frozen_heads as FH
    from src.models import fix_s8 as FS
    from src.models import multitask as MT
    from src.models.baselines import build_tasks, load_config as load_baseline_config
    cfg = MT.load_config("s8_f0")
    banks = {k: tuple(v) for k, v in yaml.safe_load((REPO / "configs" / "multitask.yaml").read_text())["s8"]["aug_banks"].items()}
    names = a.banks or list(banks)
    root = Path(os.environ.get("S8_AUG_ROOT", REPO / "results" / "aug_s8"))
    folds = pd.read_csv(FS.CV_CSV, dtype={"speaker": str})
    adult = set(folds.loc[folds["age_group"] == "adult", "speaker"])
    _, meta = FH.load_labels(build_tasks(load_baseline_config())["sent_accuracy"])          # one row per utterance
    utts = meta.loc[meta["speaker"].isin(adult), ["utt_id", "speaker"]].drop_duplicates()
    assert not set(utts["speaker"]) - adult
    if a.limit:
        utts = utts.head(a.limit)
    jobs = [(str(u), str(s), b) for b in names for u, s in zip(utts["utt_id"], utts["speaker"])]
    for b in names:
        (root / b).mkdir(parents=True, exist_ok=True)
    print(f"{len(utts)} adult utterances x {len(names)} banks = {len(jobs)} files -> {root} ({a.cores} cores)", flush=True)
    do_norm = FT.peek_normalize(cfg)
    with Pool(a.cores, initializer=_init, initargs=(do_norm, str(root), banks)) as pool:
        done = sum(pool.imap_unordered(_work, jobs, chunksize=8))
    print(f"wrote {done} new files ({len(jobs) - done} existed)")
    for b in names:
        n = len(list((root / b).glob("*.npy")))
        print(f"  {b}: {n} files")


if __name__ == "__main__":
    main()
