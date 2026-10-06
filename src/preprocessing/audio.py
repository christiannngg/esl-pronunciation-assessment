"""
Audio preprocessing for US 2.2.

Every utterance is converted to 16 kHz, mono, float32 -> 16-bit PCM WAV with
DC offset removed and gentle peak normalisation. We deliberately do NOT trim
silence: forced-alignment boundaries and pause/fluency cues depend on it.

Design choices
--------------
* Peak normalisation to ``target_peak`` with gain capped at ``max_gain_db`` so
  that near-silent / noise-only files are not amplified into loud noise.
* Quality-control flags are recorded (not used to drop data silently):
  clipping, very low level, very short, original-rate mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000
TARGET_PEAK = 0.95          # linear full-scale fraction after normalisation
MAX_GAIN_DB = 20.0          # never boost more than this
CLIP_THRESHOLD = 0.999      # |x| >= this counts as a clipped sample
LOW_PEAK = 0.01             # original peak below this => "low_level" flag
MIN_DURATION_S = 0.3


@dataclass
class AudioQC:
    orig_sr: int
    orig_channels: int
    orig_duration_s: float
    duration_s: float
    orig_peak: float
    gain_db: float
    clip_fraction: float
    rms_dbfs: float
    flag_clipped: bool
    flag_low_level: bool
    flag_too_short: bool
    flag_empty: bool

    def as_dict(self) -> dict:
        return asdict(self)


def to_mono(x: np.ndarray) -> np.ndarray:
    """(n,) or (n, ch) -> (n,) by channel averaging."""
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Polyphase resampling with an anti-aliasing filter (exact rational ratio)."""
    if sr == target_sr:
        return x
    g = gcd(int(sr), int(target_sr))
    return resample_poly(x, target_sr // g, sr // g).astype(np.float32)


def normalise(x: np.ndarray, target_peak: float = TARGET_PEAK,
              max_gain_db: float = MAX_GAIN_DB) -> tuple:
    """DC removal + peak normalisation with capped gain. Returns (y, gain_db)."""
    if x.size == 0:
        return x, 0.0
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak < 1e-9:
        return x, 0.0
    gain = target_peak / peak
    gain_db = float(np.clip(20 * np.log10(gain), -60.0, max_gain_db))
    y = x * (10 ** (gain_db / 20))
    return np.clip(y, -1.0, 1.0).astype(np.float32), gain_db


def process_array(x: np.ndarray, sr: int, **kw) -> tuple:
    """Full pipeline on an in-memory signal. Returns (y_float32_16k_mono, AudioQC)."""
    ch = 1 if x.ndim == 1 else x.shape[1]
    orig_duration = len(x) / sr if sr else 0.0
    x = to_mono(x.astype(np.float32))
    orig_peak = float(np.max(np.abs(x))) if x.size else 0.0
    clip_fraction = float(np.mean(np.abs(x) >= CLIP_THRESHOLD)) if x.size else 0.0
    y = resample(x, sr)
    y, gain_db = normalise(y, **kw)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    qc = AudioQC(
        orig_sr=int(sr), orig_channels=int(ch), orig_duration_s=float(orig_duration),
        duration_s=len(y) / TARGET_SR, orig_peak=orig_peak, gain_db=gain_db,
        clip_fraction=clip_fraction,
        rms_dbfs=float(20 * np.log10(rms)) if rms > 0 else -120.0,
        flag_clipped=clip_fraction > 1e-3,
        flag_low_level=orig_peak < LOW_PEAK,
        flag_too_short=(len(y) / TARGET_SR) < MIN_DURATION_S,
        flag_empty=y.size == 0,
    )
    return y, qc


def process_file(src: Path, dst: Optional[Path] = None, overwrite: bool = False, **kw) -> AudioQC:
    """Read ``src``, process, and (if ``dst``) write 16-bit PCM WAV. Returns QC row."""
    x, sr = sf.read(str(src), dtype="float32", always_2d=False)
    y, qc = process_array(x, sr, **kw)
    if dst is not None and (overwrite or not Path(dst).exists()):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), y, TARGET_SR, subtype="PCM_16")
    return qc
