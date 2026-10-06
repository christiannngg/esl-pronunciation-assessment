"""
US 3.3 feature-set variant: pitch, energy and pause / rate features ("prosody extras").

    python -m src.features.prosody frames     # per-utterance frame tracks (cached, resumable)
    python -m src.features.prosody tables     # per-unit extras aligned row-for-row with the US 3.1 tables

Frame track (T, 3), 10 ms hop, frame i centred at i*0.01 s (same grid as the MFCCs):
  f0_st   F0 in semitones relative to the utterance's median voiced F0 (speaker / child-adult normalised), 0 where unvoiced
  voiced  1 if pYIN says voiced
  e_db    frame RMS energy in dB relative to the utterance maximum (<= 0)
Unit extras (10 dims, ``pros_stats``): voiced ratio, F0 mean / std / range / slope, energy mean / std / min / max / slope.
Pause and rate features are added at sentence level (and the pauses before / after a word or phone) from the Sprint 2 alignments.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.features.aggregate import load_table, save_table, tables_dir
from src.features.mfcc import HOP, HOP_S, SR, audio_path, feature_root, seg_frames, utterance_table
from src.utils.paths import get_data_root

PROS_NAMES = ["voiced_ratio", "f0_mean", "f0_std", "f0_range", "f0_slope", "e_mean", "e_std", "e_min", "e_max", "e_slope"]


def track_path(dataset, utt_id, speaker=None) -> Path:
    base = feature_root() / "prosody" / dataset
    return base / f"{utt_id}.npy" if dataset == "so762" else base / speaker / f"{utt_id}.npy"


def frame_track(y: np.ndarray) -> np.ndarray:
    f0, voiced, _ = librosa.pyin(y, fmin=75, fmax=500, sr=SR, frame_length=800, hop_length=HOP, center=True,
                                 fill_na=np.nan, n_thresholds=50, resolution=0.2)
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP, center=True)[0]
    e_db = 20 * np.log10(np.maximum(rms, 1e-5))
    e_db = e_db - e_db.max()
    n = min(len(f0), len(e_db))
    f0, voiced, e_db = f0[:n], voiced[:n].astype(bool) & np.isfinite(f0[:n]), e_db[:n]
    med = np.median(f0[voiced]) if voiced.sum() >= 5 else 150.0
    st = np.where(voiced, 12 * np.log2(np.where(voiced, f0, med) / med), 0.0)
    return np.stack([st, voiced.astype(float), e_db], 1).astype(np.float32)


def _extract(args):
    dataset, utt_id, speaker = args
    out = track_path(dataset, utt_id, speaker)
    if out.exists():
        return "cached"
    try:
        y, sr = sf.read(audio_path(dataset, utt_id, speaker), dtype="float32")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, frame_track(y))
        return "ok"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def load_track(dataset, utt_id, speaker=None) -> np.ndarray:
    return np.load(track_path(dataset, utt_id, speaker))


def _slope(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    t = np.arange(len(x)) * HOP_S
    return float(np.polyfit(t, x, 1)[0])


def pros_stats(T: np.ndarray, start: float, end: float) -> np.ndarray:
    i0, i1 = seg_frames(len(T), start, end)
    x = T[i0:i1]
    v = x[:, 1] > 0
    f = x[v, 0]
    e = x[:, 2]
    return np.array([v.mean(), f.mean() if len(f) else 0.0, f.std() if len(f) > 1 else 0.0,
                     np.ptp(f) if len(f) > 1 else 0.0, _slope(f) if len(f) >= 3 else 0.0,
                     e.mean(), e.std(), e.min(), e.max(), _slope(e)], np.float32)


# --------------------------------------------------------------------------- #
def build_frames(limit=None, jobs=4):
    todo = utterance_table(limit)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(_extract, list(todo.itertuples(index=False, name=None)), chunksize=16))
    errs = [r for r in res if r.startswith("error")]
    print(f"[prosody] {len(res)} utterances; errors={len(errs)}", errs[:3])
    return len(errs)


def _gap_features(starts, ends, i):
    gp = max(starts[i] - ends[i - 1], 0.0) if i > 0 else 0.0
    gn = max(starts[i + 1] - ends[i], 0.0) if i < len(starts) - 1 else 0.0
    return gp, gn


def build_tables():
    ali = get_data_root() / "processed" / "alignments"
    words = pd.read_csv(ali / "so762_words.csv.gz", dtype={"utt_id": str}).sort_values(["utt_id", "word_idx"])
    phones = pd.read_csv(ali / "so762_phones.csv.gz", dtype={"utt_id": str}).sort_values(["utt_id", "word_idx", "phone_idx"])
    W, P = dict(tuple(words.groupby("utt_id"))), dict(tuple(phones.groupby("utt_id")))

    # ---- SO762 sentence
    X, meta, _ = load_table("so762_sentence")
    rows = []
    for u in meta["utt_id"]:
        T = load_track("so762", u)
        w, p = W[u], P[u]
        dur = len(T) * HOP_S
        s0, s1 = float(w["start"].min()), float(w["end"].max())
        gaps = np.maximum(w["start"].to_numpy()[1:] - w["end"].to_numpy()[:-1], 0.0)
        pause_t = float(gaps[gaps >= 0.1].sum())
        speech = max(s1 - s0, 1e-3)
        pd_ = (p["end"] - p["start"]).to_numpy()
        wd = (w["end"] - w["start"]).to_numpy()
        rows.append(np.concatenate([pros_stats(T, s0, s1),
                                    [s0, max(dur - s1, 0.0), float((gaps >= 0.1).sum()), float((gaps >= 0.3).sum()), pause_t,
                                     pause_t / speech, len(p) / speech, len(p) / max(speech - pause_t, 1e-3),
                                     pd_.mean(), pd_.std(), wd.mean(), wd.std()]]))
    names = ["pros_" + n for n in PROS_NAMES] + ["lead_sil", "trail_sil", "n_pause_100ms", "n_pause_300ms", "pause_total",
                                                  "pause_ratio", "speech_rate", "articulation_rate", "phone_dur_mean",
                                                  "phone_dur_std", "word_dur_mean", "word_dur_std"]
    save_table("so762_sentence_pros", np.vstack(rows), names, meta)

    # ---- SO762 word / phone
    for tab, df, key in (("so762_word", W, "word"), ("so762_phone", P, "phone")):
        X, meta, _ = load_table(tab)
        lookup = {u: g for u, g in df.items()}
        rows = []
        cache_u, T, G = None, None, None
        if key == "word":
            for r in meta[["utt_id", "word_idx"]].itertuples(index=False):
                T = load_track("so762", r.utt_id) if r.utt_id != cache_u else T
                g = lookup[r.utt_id]; cache_u = r.utt_id
                st, en = g["start"].to_numpy(), g["end"].to_numpy()
                i = int(np.flatnonzero(g["word_idx"].to_numpy() == r.word_idx)[0])
                gp, gn = _gap_features(st, en, i)
                rows.append(np.concatenate([pros_stats(T, st[i], en[i]), [gp, gn, float(gp >= 0.1), float(gn >= 0.1)]]))
            names = ["pros_" + n for n in PROS_NAMES] + ["gap_before", "gap_after", "pause_before", "pause_after"]
        else:
            for u, g in meta.groupby("utt_id", sort=False):
                T = load_track("so762", u)
                pp = lookup[u].set_index(["word_idx", "phone_idx"])
                keys = list(zip(g["word_idx"], g["phone_idx"]))
                st = np.array([pp.loc[k, "start"] for k in keys]); en = np.array([pp.loc[k, "end"] for k in keys])
                ps = np.vstack([pros_stats(T, a, b) for a, b in zip(st, en)])
                for i in range(len(keys)):
                    prv = ps[i - 1] if i > 0 else np.zeros(10, np.float32)
                    nxt = ps[i + 1] if i < len(keys) - 1 else np.zeros(10, np.float32)
                    gp, gn = _gap_features(st, en, i)
                    rows.append(np.concatenate([ps[i], [prv[1], prv[5], nxt[1], nxt[5], gp, gn]]))
            names = ["pros_" + n for n in PROS_NAMES] + ["prev_f0_mean", "prev_e_mean", "next_f0_mean", "next_e_mean", "gap_before", "gap_after"]
        save_table(f"{tab}_pros", np.vstack(rows), names, meta)

    # ---- L2-ARCTIC (fixed-window pitch / energy; no pause features here)
    ph = pd.read_csv(ali / "l2arctic_phones_clean.csv.gz", dtype={"utt_id": str, "speaker": str})
    ph = ph[ph["corpus"] == "arctic"].sort_values(["speaker", "utt_id", "phone_idx"])
    from src.features.aggregate import fixed_window
    X, meta, _ = load_table("l2_phone")
    rows = []
    byu = {u: g for u, g in ph.groupby("utt_id", sort=False)}
    for (u, spk), g in meta.groupby(["utt_id", "speaker"], sort=False):
        T = load_track("l2arctic", u, spk)
        U = byu[u].reset_index(drop=True)
        allp = np.vstack([pros_stats(T, *fixed_window(a, b)) for a, b in zip(U["start"], U["end"])])
        pos = {int(k): i for i, k in enumerate(U["phone_idx"])}
        for k in g["phone_idx"]:
            i = pos[int(k)]
            prv = allp[i - 1] if i > 0 else np.zeros(10, np.float32)
            nxt = allp[i + 1] if i < len(U) - 1 else np.zeros(10, np.float32)
            rows.append(np.concatenate([allp[i], [prv[1], prv[5], nxt[1], nxt[5]]]))
    save_table("l2_phone_pros", np.vstack(rows),
               ["pros_" + n for n in PROS_NAMES] + ["prev_f0_mean", "prev_e_mean", "next_f0_mean", "next_e_mean"], meta)

    X, meta, _ = load_table("l2_word")
    rows = []
    for (u, spk), g in meta.groupby(["utt_id", "speaker"], sort=False):
        T = load_track("l2arctic", u, spk)
        U = byu[u]
        slot = U[U["error_class"].isin(["correct", "substitution", "deletion"])]
        for wi in g["word_idx"]:
            s = slot[slot["word_idx"] == wi]
            rows.append(pros_stats(T, s["start"].min(), s["end"].max()))
    save_table("l2_word_pros", np.vstack(rows), ["pros_" + n for n in PROS_NAMES], meta)
    print("tables written")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "frames"
    if what == "frames":
        build_frames()
    else:
        build_tables()
