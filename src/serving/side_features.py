"""
Sprint 9: the non-acoustic "side" inputs of the SO762 heads, computed from a forced alignment + the prompt, exactly as the training tables were built.

Source of truth: ``src/features/aggregate.py::build_so762`` (columns listed in ``experiments/scripts/export_side_features.py::SIDE_COLUMNS``).
This module copies the formulas so the app does not need librosa or the Sprint 3 tables. ``tests/test_serving_side_features.py`` checks it against the
saved ``so762_{sentence,word,phone}_side_X.npy`` rows (same alignment in, same numbers out).

Input ``words``: one dict per word, in order:
    {"word": "CALL", "start": 0.91, "end": 1.20, "phones": [{"phone": "K", "start": 0.91, "end": 1.0}, ...]}   (seconds; ARPAbet without stress; silence not included)
Standardisation (training mean / std) is NOT applied here; it lives in serving/stats.json and is applied by the model wrapper.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing.labels import PHONE_TO_ID, VOWELS

SR = 16000
HOP = 160                       # src/features/mfcc.py: 10 ms hop, librosa center=True
HOP_S = HOP / SR
N_PHONES = len(PHONE_TO_ID)

SENTENCE_COLUMNS = ["dur", "n_words", "n_phones"]
WORD_COLUMNS = ["dur", "n_phones", "phone_dur_mean", "phone_dur_std", "n_chars", "word_pos"]
PHONE_COLUMNS = (["dur", "log_dur", "has_prev", "has_next", "word_initial", "word_final", "n_phones_in_word", "is_vowel"]
                 + [f"ph_{p}" for p in sorted(PHONE_TO_ID, key=PHONE_TO_ID.get)])


def mfcc_n_frames(n_samples: int) -> int:
    """Number of librosa MFCC frames (center=True) of a signal: ``1 + n // hop``; the training ``dur`` is this times 10 ms."""
    return 1 + int(n_samples) // HOP


def sentence_side(n_samples: int, n_words: int, n_phones: int) -> np.ndarray:
    """(1, 3): dur (s, from the MFCC frame count), number of words, number of phones of the prompt's canonical pronunciation."""
    return np.array([[mfcc_n_frames(n_samples) * HOP_S, n_words, n_phones]], np.float32)


def word_side(words: list[dict]) -> np.ndarray:
    """(n_words, 6)."""
    nw = len(words)
    rows = []
    for i, w in enumerate(words):
        d = np.array([p["end"] - p["start"] for p in w["phones"]], float)
        rows.append([w["end"] - w["start"], len(w["phones"]), d.mean(), d.std(), len(w["word"]), i / max(nw - 1, 1)])
    return np.array(rows, np.float32).reshape(nw, len(WORD_COLUMNS))


def phone_side(words: list[dict]) -> np.ndarray:
    """(n_phones, 47), rows ordered by word then phone, as the training table."""
    flat = [(p, len(w["phones"]), j) for w in words for j, p in enumerate(w["phones"])]
    n = len(flat)
    out = np.zeros((n, len(PHONE_COLUMNS)), np.float32)
    for i, (p, n_in_word, j) in enumerate(flat):
        dur = p["end"] - p["start"]
        out[i, :8] = [dur, np.log(max(dur, 1e-3)), float(i > 0), float(i < n - 1), float(j == 0), float(j == n_in_word - 1), n_in_word,
                      float(p["phone"] in VOWELS)]
        if p["phone"] in PHONE_TO_ID:
            out[i, 8 + PHONE_TO_ID[p["phone"]]] = 1.0
    return out
