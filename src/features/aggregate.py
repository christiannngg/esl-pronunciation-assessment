"""
US 3.1 - segment-level aggregation of frame MFCCs into one feature row per unit.

    python -m src.features.aggregate            # needs the cached frame features (src.features.mfcc)

Units and feature groups (names are saved so ablations can select columns by prefix):
  so762_sentence   utterance: mean/std/min/max of the 39 frame dims over the whole file      (156) + dur, n_words, n_phones
  so762_word       word span (MFA): the same statistics (156) + dur, n_phones, mean/std phone dur, n_chars, position
  so762_phone      phone interval (MFA): own stats (156) + dur + prev/next phone means (78) + word position + canonical phone one-hot (39)
  l2_phone         annotated unit (correct / substitution / deletion / addition, silence excluded):
                   stats over a FIXED window (+-l2_window_half_s around the unit midpoint) + prev/next row windows + anchor phone one-hot
  l2_word          word span from non-addition units: stats (156) + dur + n canonical phones + n_chars

L2-ARCTIC leakage controls (the annotation defines the units, so some cues would give the label away):
  * a deleted phone has no audio: its interval is only a marker, so every unit uses the same fixed window around
    its midpoint ("window around the expected boundary"), and the annotated interval duration is NOT a feature
    (kept in meta as ``ann_dur`` for a leakage ablation);
  * an addition has no canonical phone: its anchor phone is the canonical phone of the FOLLOWING slot unit (the
    attach-to-following-word rule), so a missing canonical phone does not reveal the class;
  * additions in pauses (no word) are attached to the following word for word-level labels.
SO762 boundaries come from forced alignment of the canonical pronunciation (label-blind), so durations are allowed there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.mfcc import DIM, HOP_S, load_mfcc, seg_frames
from src.preprocessing.labels import ARPABET39, PHONE_TO_ID, VOWELS
from src.utils.paths import get_data_root

WIN_HALF_S = 0.075
N_PHONES = len(PHONE_TO_ID)
_DIMS = [f"c{i}" for i in range(13)] + [f"d{i}" for i in range(13)] + [f"dd{i}" for i in range(13)]
STAT_NAMES = [f"{s}_{d}" for s in ("mean", "std", "min", "max") for d in _DIMS]
PHONE_NAMES = [f"ph_{p}" for p in sorted(PHONE_TO_ID, key=PHONE_TO_ID.get)]


def tables_dir() -> Path:
    return get_data_root() / "processed" / "features" / "tables"


def seg_stats(F: np.ndarray, start: float, end: float) -> np.ndarray:
    i0, i1 = seg_frames(len(F), start, end)
    x = F[i0:i1]
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])


def fixed_window(start: float, end: float, half: float = WIN_HALF_S) -> tuple[float, float]:
    mid = 0.5 * (start + end)
    return mid - half, mid + half


def onehot(phone) -> np.ndarray:
    v = np.zeros(N_PHONES, np.float32)
    if phone in PHONE_TO_ID:
        v[PHONE_TO_ID[phone]] = 1.0
    return v


def save_table(name: str, X: np.ndarray, names: list[str], meta: pd.DataFrame) -> None:
    d = tables_dir()
    d.mkdir(parents=True, exist_ok=True)
    assert X.shape == (len(meta), len(names)), (X.shape, len(meta), len(names))
    assert np.isfinite(X).all(), f"{name}: non-finite feature values"
    np.save(d / f"{name}_X.npy", X.astype(np.float32))
    meta.to_csv(d / f"{name}_meta.csv.gz", index=False)
    (d / f"{name}_names.json").write_text(json.dumps(names))


def load_table(name: str) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    d = tables_dir()
    X = np.load(d / f"{name}_X.npy")
    meta = pd.read_csv(d / f"{name}_meta.csv.gz", dtype={"utt_id": str, "speaker": str})
    names = json.loads((d / f"{name}_names.json").read_text())
    return X, meta, names


# --------------------------------------------------------------------------- #
# SpeechOcean762
# --------------------------------------------------------------------------- #
def build_so762(root: Path | None = None) -> dict:
    from src.data.loaders import load_so762
    ali = get_data_root() / "processed" / "alignments"
    utts = load_so762(with_audio_info=False).utts
    spk_cols = utts[["utt_id", "speaker", "split", "gender", "age_group"]]
    words = pd.read_csv(ali / "so762_words.csv.gz", dtype={"utt_id": str}).sort_values(["utt_id", "word_idx"])
    phones = pd.read_csv(ali / "so762_phones.csv.gz", dtype={"utt_id": str}).sort_values(["utt_id", "word_idx", "phone_idx"])
    assert set(phones["phone"]) <= ARPABET39, set(phones["phone"]) - ARPABET39
    w_by = dict(tuple(words.groupby("utt_id")))
    p_by = dict(tuple(phones.groupby("utt_id")))

    # sentence
    rows, meta = [], []
    for r in utts.itertuples():
        F = load_mfcc("so762", r.utt_id)
        dur = len(F) * HOP_S
        rows.append(np.concatenate([seg_stats(F, 0.0, dur), [dur, r.n_words, r.n_phones]]))
        meta.append((r.utt_id, r.accuracy, r.fluency, r.prosodic, r.total))
    m = pd.DataFrame(meta, columns=["utt_id", "y_accuracy", "y_fluency", "y_prosodic", "y_total"]).merge(spk_cols, on="utt_id")
    save_table("so762_sentence", np.vstack(rows), STAT_NAMES + ["dur", "n_words", "n_phones"], m)

    # word + phone
    wrows, wmeta, prows, pmeta = [], [], [], []
    for utt_id, W in w_by.items():
        F = load_mfcc("so762", utt_id)
        P = p_by[utt_id]
        nw = len(W)
        pdur = (P["end"] - P["start"]).to_numpy()
        stats = np.vstack([seg_stats(F, s, e) for s, e in zip(P["start"], P["end"])])
        means = stats[:, :DIM]
        nph_word = P.groupby("word_idx")["phone_idx"].transform("size").to_numpy()
        for i, p in enumerate(P.itertuples()):
            prev = means[i - 1] if i > 0 else np.zeros(DIM, np.float32)
            nxt = means[i + 1] if i < len(P) - 1 else np.zeros(DIM, np.float32)
            n_in_word = nph_word[i]
            prows.append(np.concatenate([stats[i], [pdur[i], np.log(max(pdur[i], 1e-3))], prev, nxt,
                                         [float(i > 0), float(i < len(P) - 1), float(p.phone_idx == 0),
                                          float(p.phone_idx == n_in_word - 1), n_in_word,
                                          float(p.phone in VOWELS)], onehot(p.phone)]))
            pmeta.append((utt_id, p.word_idx, p.phone_idx, p.phone, p.phone_score))
        for w in W.itertuples():
            Pw = P[P["word_idx"] == w.word_idx]
            d = (Pw["end"] - Pw["start"]).to_numpy()
            wrows.append(np.concatenate([seg_stats(F, w.start, w.end),
                                         [w.end - w.start, len(Pw), d.mean(), d.std(), len(w.word), w.word_idx / max(nw - 1, 1)]]))
            wmeta.append((utt_id, w.word_idx, w.word, w.word_accuracy))
    wm = pd.DataFrame(wmeta, columns=["utt_id", "word_idx", "word", "y_word_accuracy"]).merge(spk_cols, on="utt_id", how="left")
    save_table("so762_word", np.vstack(wrows),
               STAT_NAMES + ["dur", "n_phones", "phone_dur_mean", "phone_dur_std", "n_chars", "word_pos"], wm)
    pm = pd.DataFrame(pmeta, columns=["utt_id", "word_idx", "phone_idx", "phone", "y_phone_score"]).merge(spk_cols, on="utt_id", how="left")
    pnames = (["own_" + s for s in STAT_NAMES] + ["dur", "log_dur"] + ["prev_" + d for d in _DIMS] + ["next_" + d for d in _DIMS]
              + ["has_prev", "has_next", "word_initial", "word_final", "n_phones_in_word", "is_vowel"] + PHONE_NAMES)
    save_table("so762_phone", np.vstack(prows), pnames, pm)
    return {"sentence": len(m), "word": len(wm), "phone": len(pm)}


# --------------------------------------------------------------------------- #
# L2-ARCTIC
# --------------------------------------------------------------------------- #
def _anchor_phones(canon: list) -> list:
    """Canonical phone for every row; rows without one (additions, silence) take the next slot unit's (else previous)."""
    out = list(canon)
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is not None:
            nxt = out[i]
        else:
            out[i] = nxt
    prv = None
    for i in range(len(out)):
        if canon[i] is not None:
            prv = canon[i]
        elif out[i] is None:
            out[i] = prv
    return out


def _attach_words(word_idx: np.ndarray, error_class: np.ndarray) -> np.ndarray:
    """Word index per row; additions that sit in a pause take the following word (else the preceding one)."""
    w = np.array(word_idx, dtype=float)
    n = len(w)
    for i in range(n):
        if error_class[i] == "addition" and np.isnan(w[i]):
            j = next((k for k in range(i + 1, n) if not np.isnan(word_idx[k])), None)
            if j is None:
                j = next((k for k in range(i - 1, -1, -1) if not np.isnan(word_idx[k])), None)
            if j is not None:
                w[i] = word_idx[j]
    return w


def build_l2arctic(half: float = WIN_HALF_S) -> dict:
    ali = get_data_root() / "processed" / "alignments"
    ph = pd.read_csv(ali / "l2arctic_phones_clean.csv.gz", dtype={"utt_id": str, "speaker": str})
    ph = ph[ph["corpus"] == "arctic"]          # suitcase narratives are excluded from training and evaluation
    assert not ph["unmapped"].any()
    folds = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "splits" / "l2arctic_folds.csv", dtype={"speaker": str})
    fold_of, gender_of = dict(zip(folds["speaker"], folds["fold"])), dict(zip(folds["speaker"], folds["gender"]))
    ph = ph.sort_values(["speaker", "utt_id", "phone_idx"]).reset_index(drop=True)

    prows, pmeta, wrows, wmeta = [], [], [], []
    for (spk, utt_id), U in ph.groupby(["speaker", "utt_id"], sort=False):
        F = load_mfcc("l2arctic", utt_id, spk)
        starts, ends = U["start"].to_numpy(), U["end"].to_numpy()
        ec = U["error_class"].to_numpy()
        win = np.vstack([seg_stats(F, *fixed_window(s, e, half)) for s, e in zip(starts, ends)])
        means = win[:, :DIM]
        anchor = _anchor_phones(list(U["cpl_base"].where(U["cpl_base"].notna(), None)))
        wid = _attach_words(U["word_idx"].to_numpy(dtype=float), ec)
        n = len(U)
        for i in range(n):
            if ec[i] == "silence":
                continue
            prev = means[i - 1] if i > 0 else np.zeros(DIM, np.float32)
            nxt = means[i + 1] if i < n - 1 else np.zeros(DIM, np.float32)
            a = anchor[i]
            prows.append(np.concatenate([win[i], prev, nxt, [float(i > 0), float(i < n - 1), float(a in VOWELS)], onehot(a)]))
            pmeta.append((utt_id, spk, U["l1"].iat[i], fold_of[spk], gender_of[spk], U["phone_idx"].iat[i], a, ec[i],
                          float(ends[i] - starts[i]), bool(U["is_deviation"].iat[i]), bool(U["is_err"].iat[i])))
        # words: span from non-addition, non-silence rows; label = any error among the word's rows (incl. attached additions)
        for wi, idx in pd.Series(np.arange(n)).groupby(wid):
            idx = idx.to_numpy()
            slot = [k for k in idx if ec[k] in ("correct", "substitution", "deletion")]
            if not slot:
                continue
            s, e = starts[slot].min(), ends[slot].max()
            word = U["word"].iat[slot[0]]
            n_err = int(sum(U["is_error"].iat[k] for k in idx))
            wrows.append(np.concatenate([seg_stats(F, s, e), [e - s, len(slot), len(str(word))]]))
            wmeta.append((utt_id, spk, U["l1"].iat[slot[0]], fold_of[spk], gender_of[spk], int(wi), word, int(n_err > 0), n_err))
    pm = pd.DataFrame(pmeta, columns=["utt_id", "speaker", "l1", "fold", "gender", "phone_idx", "anchor_phone", "error_class",
                                      "ann_dur", "is_deviation", "is_err"])
    pnames = (["win_" + s for s in STAT_NAMES] + ["prev_" + d for d in _DIMS] + ["next_" + d for d in _DIMS]
              + ["has_prev", "has_next", "is_vowel"] + PHONE_NAMES)
    save_table("l2_phone", np.vstack(prows), pnames, pm)
    wm = pd.DataFrame(wmeta, columns=["utt_id", "speaker", "l1", "fold", "gender", "word_idx", "word", "y_has_error", "n_errors"])
    save_table("l2_word", np.vstack(wrows), STAT_NAMES + ["dur", "n_canon_phones", "n_chars"], wm)
    return {"phone": len(pm), "word": len(wm)}


def main(argv=None):
    import sys
    which = (argv or sys.argv[1:] or ["so762", "l2arctic"])
    if "so762" in which:
        print("so762", build_so762())
    if "l2arctic" in which:
        print("l2arctic", build_l2arctic())


if __name__ == "__main__":
    main()
