"""US 8.3: the generalization fix for the weakest subgroup (SO762 child/teen on MT-GN15).

Pure numpy / pandas (no torch), so the protocol can be unit-tested away from the GPU nodes. Three parts:

1. **Cross-validation folds** (`make_folds`, `write_folds`, `fold_split_csv`): speaker-grouped 5-fold CV over the 125 SO762 train + val speakers,
   stratified by age group. The 125 official test speakers are never read: they are marked `excluded` in every fold's split file.
   Inside fold k, the held-out fold is `test` (out-of-fold predictions), a small age-stratified set of the other speakers is `val` (early stopping), the rest `train`.
2. **Row weights for F1** (`row_weights`): inverse-frequency weights over (age group x label) on the TRAINING rows of one head.
3. **Selection** (`select_candidate`): the rule fixed in docs/sprint8_plan.md section 5, applied to pooled out-of-fold predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SPEAKERS = REPO / "data" / "splits" / "so762_speakers.csv"
CV_CSV = REPO / "data" / "splits" / "so762_cv_s8.csv"
N_FOLDS = 5
FOLD_SEED = 8
VAL_FRAC = 0.15                      # share of the non-held-out speakers used for early stopping inside a fold (per age group, rounded)


# --------------------------------------------------------------------------- #
# 1. folds
# --------------------------------------------------------------------------- #
def make_folds(speakers: pd.DataFrame | None = None, n_folds: int = N_FOLDS, seed: int = FOLD_SEED, val_frac: float = VAL_FRAC) -> pd.DataFrame:
    """One row per train+val speaker: speaker, age_group, official_split, cv_fold, and for each fold k a column `role_k` in {train, val, test}."""
    sp = pd.read_csv(speakers or SPEAKERS, dtype={"speaker": str}) if not isinstance(speakers, pd.DataFrame) else speakers.copy()
    sp = sp[sp["split"].isin(["train", "val"])].sort_values("speaker").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold = np.full(len(sp), -1, np.int64)
    for g in sorted(sp["age_group"].unique()):
        idx = np.flatnonzero((sp["age_group"] == g).to_numpy())
        fold[rng.permutation(idx)] = np.arange(len(idx)) % n_folds                 # round robin over a random order: sizes differ by at most 1 per group
    out = pd.DataFrame({"speaker": sp["speaker"], "age_group": sp["age_group"], "official_split": sp["split"], "cv_fold": fold})
    for k in range(n_folds):
        role = np.where(fold == k, "test", "train").astype(object)
        for g in sorted(sp["age_group"].unique()):
            pool = np.flatnonzero(((sp["age_group"] == g) & (fold != k)).to_numpy())
            n_val = max(1, int(round(val_frac * len(pool))))
            role[np.random.default_rng(seed + 1 + k).permutation(pool)[:n_val]] = "val"
        out[f"role_{k}"] = role
    return out


def write_folds(path: Path | None = None, **kw) -> pd.DataFrame:
    df = make_folds(**kw)
    path = path or CV_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fold_split_csv(k: int, out_dir: Path, folds: pd.DataFrame | None = None, speakers: pd.DataFrame | None = None) -> Path:
    """Split file for CV fold ``k`` in the format of data/splits/so762_speakers.csv (speaker, split); official test speakers -> `excluded`."""
    folds = folds if folds is not None else pd.read_csv(CV_CSV, dtype={"speaker": str})
    sp = pd.read_csv(SPEAKERS, dtype={"speaker": str}) if speakers is None else speakers
    role = folds.set_index("speaker")[f"role_{k}"]
    split = sp["speaker"].map(role).fillna("excluded")
    assert (split[sp["split"] == "test"] == "excluded").all(), "an official test speaker is part of the cross-validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"so762_cv_s8_fold{k}.csv"
    pd.DataFrame({"speaker": sp["speaker"], "split": split}).to_csv(path, index=False)
    return path


def check_folds(folds: pd.DataFrame, n_folds: int = N_FOLDS) -> None:
    """Raises if the fold file breaks a protocol rule (used by the tests and by the lab script before it submits anything)."""
    assert folds["speaker"].is_unique
    assert set(folds["cv_fold"]) == set(range(n_folds))
    for k in range(n_folds):
        r = folds[f"role_{k}"]
        assert set(r) == {"train", "val", "test"}
        assert ((r == "test") == (folds["cv_fold"] == k)).all()
        for g, part in folds.groupby("age_group"):
            assert (part[f"role_{k}"] == "test").any() and (part[f"role_{k}"] == "val").any() and (part[f"role_{k}"] == "train").any(), (k, g)


# --------------------------------------------------------------------------- #
# 2. F1 row weights
# --------------------------------------------------------------------------- #
def row_weights(group: np.ndarray, label: np.ndarray, train: np.ndarray, strength: float) -> np.ndarray:
    """F1 weight per row: (share of its age group among the TRAINING rows with the same label) ** -strength, rescaled so that every label keeps its
    total weight (mean weight 1 within each label). Together this is the inverse frequency of the (age group x label) cell, without moving mass between
    labels, so the class-balanced loss weights of the control (train.cw) keep their meaning and F1 only changes who is heard WITHIN a label.

    ``group``: age group per row; ``label``: class (classification) or a label bin (regression); ``train``: boolean mask of the training rows.
    strength 0 = no reweighting, 1 = groups contribute equally within each label, 0.5 = square root. Rows outside ``train`` get weight 1 (not used).
    """
    group, label, train = np.asarray(group).astype(str), np.asarray(label).astype(str), np.asarray(train, bool)
    df = pd.DataFrame({"g": group[train], "l": label[train]})
    share = df.groupby("l")["g"].transform(lambda s: s.map(s.value_counts(normalize=True))).to_numpy(float)
    w_tr = np.power(share, -float(strength))
    df["w"] = w_tr
    w_tr = w_tr / df.groupby("l")["w"].transform("mean").to_numpy(float)
    w = np.ones(len(group), float)
    w[train] = w_tr
    return w


def speaker_groups(speakers) -> np.ndarray:
    """Age group ('adult' | 'child/teen') per row, from the speaker id."""
    sp = pd.read_csv(SPEAKERS, dtype={"speaker": str}).set_index("speaker")["age_group"]
    out = pd.Series(np.asarray(speakers).astype(str)).map(sp)
    if out.isna().any():
        raise ValueError(f"speakers without an age group: {sorted(set(np.asarray(speakers).astype(str)[out.isna().to_numpy()]))[:5]}")
    return out.to_numpy()


# --------------------------------------------------------------------------- #
# F2 augmentation bank
# --------------------------------------------------------------------------- #
class AugBank:
    """Pre-computed child-like copies of adult training utterances (`<root>/<name>/<utt_id>.npy`, already normalised like the originals and of
    EXACTLY the same length, so every span, duration and label survives). In a training batch an utterance is swapped for its copy with probability ``prob``."""

    def __init__(self, root: Path, name: str, prob: float):
        self.dir, self.prob = Path(root) / name, float(prob)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"augmentation bank {self.dir} does not exist (run experiments/scripts/precompute_aug_s8.py first)")
        self.available = {p.stem for p in self.dir.glob("*.npy")}
        self.used = 0

    def maybe(self, utt_id, wav: np.ndarray, rng) -> np.ndarray:
        uid = str(utt_id)
        if uid not in self.available or rng.random() >= self.prob:
            return wav
        w = np.load(self.dir / f"{uid}.npy")
        if len(w) != len(wav):
            raise ValueError(f"{uid}: augmented copy has {len(w)} samples, original {len(wav)}")
        self.used += 1
        return w.astype(np.float32)


def label_bins(y: np.ndarray, train: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Regression heads: label bin = tercile of the TRAINING labels (edges from the training rows only)."""
    y = np.asarray(y, float)
    edges = np.quantile(y[np.asarray(train, bool)], np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, y, side="right")
