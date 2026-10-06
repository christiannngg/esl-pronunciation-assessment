"""
Speaker-independent splits for US 2.2.

SpeechOcean762
    * Keep the official 125/125 train/test speaker split (test never touched).
    * Carve a validation set out of the *train* speakers, stratified by
      age group x gender, ~15 % of speakers. Among ``n_candidates`` seeded draws we
      keep the one whose sentence-score distributions are closest to the remaining
      train speakers (max KS statistic). Only train-vs-val similarity is used; the
      test set is never consulted.

L2-ARCTIC (24 speakers = 6 L1 x 4 speakers, 2F + 2M per L1)
    * 4-fold grouped CV. Each fold holds exactly 1 speaker per L1 and 3F + 3M.
    * For outer fold k: test = fold k, val = fold (k+1) % 4, train = the other two.
    * Folds are deterministic given the seed; speaker lists are written to disk.

Everything is written to ``data/splits/`` so the exact speaker lists can be audited.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

SEED = 42
SO762_METRICS = ["accuracy", "fluency", "prosodic", "total"]


# --------------------------------------------------------------------------- #
# SpeechOcean762
# --------------------------------------------------------------------------- #
def _speaker_table(utts: pd.DataFrame) -> pd.DataFrame:
    return (utts.drop_duplicates("speaker")[["speaker", "split", "gender", "age", "age_group"]]
            .sort_values("speaker").reset_index(drop=True))


def _stratified_val_speakers(train_spk: pd.DataFrame, val_frac: float, rng: np.random.Generator) -> list:
    """Proportional allocation per (age_group, gender) stratum, at least 1 each."""
    n_val = int(round(val_frac * len(train_spk)))
    strata = train_spk.groupby(["age_group", "gender"])
    sizes = strata.size()
    raw = sizes / sizes.sum() * n_val
    alloc = np.maximum(1, np.floor(raw)).astype(int)
    # distribute the remainder by largest fractional part
    while alloc.sum() < n_val:
        frac = (raw - alloc).sort_values(ascending=False)
        alloc[frac.index[0]] += 1
    while alloc.sum() > n_val:
        alloc[alloc.idxmax()] -= 1
    val = []
    for key, grp in strata:
        k = int(alloc[key])
        val += list(rng.choice(sorted(grp["speaker"]), size=k, replace=False))
    return sorted(val)


def build_so762_splits(utts: pd.DataFrame, val_frac: float = 0.15, seed: int = SEED,
                       n_candidates: int = 200) -> pd.DataFrame:
    """Returns speaker table with final ``split`` in {train, val, test}."""
    spk = _speaker_table(utts)
    train_spk = spk[spk["split"] == "train"]
    best, best_score = None, np.inf
    for i in range(n_candidates):
        rng = np.random.default_rng(seed + i)
        val = _stratified_val_speakers(train_spk, val_frac, rng)
        is_val = utts["speaker"].isin(val)
        is_tr = (utts["split"] == "train") & ~is_val
        score = max(ks_2samp(utts.loc[is_val, m], utts.loc[is_tr, m]).statistic for m in SO762_METRICS)
        if score < best_score:
            best, best_score = val, score
    out = spk.copy()
    out.loc[out["speaker"].isin(best), "split"] = "val"
    out.attrs["val_max_ks"] = float(best_score)
    return out


# --------------------------------------------------------------------------- #
# L2-ARCTIC
# --------------------------------------------------------------------------- #
def build_l2arctic_folds(speakers: pd.DataFrame, n_folds: int = 4, seed: int = SEED) -> pd.DataFrame:
    """
    ``speakers``: columns speaker, l1, gender ('F'/'M'). Needs every L1 to have exactly
    n_folds speakers, half F and half M, and an even number of L1s (so each fold can be 3F/3M).
    """
    assert n_folds == 4, "gender-balanced rotation below is written for 4 folds (2F + 2M per L1)"
    rng = np.random.default_rng(seed)
    rows = []
    l1s = sorted(speakers["l1"].unique())
    assert len(l1s) % 2 == 0
    for i, l1 in enumerate(l1s):
        g = speakers[speakers["l1"] == l1]
        f = list(rng.permutation(sorted(g[g["gender"] == "F"]["speaker"])))
        m = list(rng.permutation(sorted(g[g["gender"] == "M"]["speaker"])))
        assert len(f) == 2 and len(m) == 2, f"{l1}: expected 2F+2M, got {len(f)}F+{len(m)}M"
        for k in range(n_folds):
            female = ((i + k) % 2 == 0) if k < 2 else ((i + k) % 2 == 1)
            spk = f.pop(0) if female else m.pop(0)
            rows.append({"speaker": spk, "l1": l1, "gender": "F" if female else "M", "fold": k})
    return pd.DataFrame(rows).sort_values(["fold", "l1"]).reset_index(drop=True)


def fold_roles(folds: pd.DataFrame, outer: int, n_folds: int = 4) -> dict:
    val = (outer + 1) % n_folds
    return {
        "test": sorted(folds[folds["fold"] == outer]["speaker"]),
        "val": sorted(folds[folds["fold"] == val]["speaker"]),
        "train": sorted(folds[~folds["fold"].isin([outer, val])]["speaker"]),
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_so762(spk: pd.DataFrame, utts: pd.DataFrame, official: pd.DataFrame) -> dict:
    sets = {s: set(spk[spk["split"] == s]["speaker"]) for s in ["train", "val", "test"]}
    off_test = set(official[official["split"] == "test"]["speaker"])
    res = {
        "disjoint": not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]),
        "all_speakers_covered": sum(len(v) for v in sets.values()) == official["speaker"].nunique(),
        "test_equals_official_test": sets["test"] == off_test,
        "n_speakers": {k: len(v) for k, v in sets.items()},
    }
    u = utts.merge(spk[["speaker", "split"]].rename(columns={"split": "final"}), on="speaker")
    res["n_utterances"] = u.groupby("final").size().to_dict()
    res["strata_counts"] = {s: spk[spk["split"] == s].groupby(["age_group", "gender"]).size().astype(int)
                            .rename(lambda x: x).to_dict() for s in sets}
    res["strata_counts"] = {s: {f"{a}|{g}": int(n) for (a, g), n in d.items()} for s, d in res["strata_counts"].items()}
    tr = u[u["final"] == "train"]
    res["mean_scores"] = {s: {m: round(float(u[u["final"] == s][m].mean()), 3) for m in SO762_METRICS}
                          for s in ["train", "val", "test"]}
    res["ks_val_vs_train"] = {m: round(float(ks_2samp(u[u["final"] == "val"][m], tr[m]).statistic), 3)
                              for m in SO762_METRICS}
    return res


def check_l2arctic(folds: pd.DataFrame, n_folds: int = 4) -> dict:
    res = {"every_speaker_once": bool(folds["speaker"].is_unique), "n_speakers": int(len(folds)), "folds": {}}
    ok = res["every_speaker_once"]
    for k in range(n_folds):
        f = folds[folds["fold"] == k]
        info = {"n": int(len(f)), "F": int((f["gender"] == "F").sum()), "M": int((f["gender"] == "M").sum()),
                "l1_each_once": bool(f["l1"].is_unique and f["l1"].nunique() == folds["l1"].nunique())}
        res["folds"][k] = info
        ok &= info["F"] == info["M"] and info["l1_each_once"]
        r = fold_roles(folds, k, n_folds)
        ok &= not (set(r["train"]) & set(r["val"]) or set(r["train"]) & set(r["test"]) or set(r["val"]) & set(r["test"]))
        ok &= len(r["train"]) + len(r["val"]) + len(r["test"]) == len(folds)
    res["all_checks_pass"] = bool(ok)
    return res


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]


def write_splits(so_spk: pd.DataFrame, so_checks: dict, folds: pd.DataFrame, l2_checks: dict,
                 out_dir: Path, seed: int = SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    so_spk.to_csv(out_dir / "so762_speakers.csv", index=False)
    for s in ["train", "val", "test"]:
        (out_dir / f"so762_{s}_speakers.txt").write_text(
            "\n".join(so_spk[so_spk["split"] == s]["speaker"]) + "\n")
    folds.to_csv(out_dir / "l2arctic_folds.csv", index=False)
    for k in sorted(folds["fold"].unique()):
        r = fold_roles(folds, int(k))
        for role, lst in r.items():
            (out_dir / f"l2arctic_fold{k}_{role}_speakers.txt").write_text("\n".join(lst) + "\n")
    meta = {"seed": seed, "so762": {**so_checks, "val_max_ks_selected": so_spk.attrs.get("val_max_ks"),
                                    "sha256_16": _sha(so_spk)},
            "l2arctic": {**l2_checks, "sha256_16": _sha(folds)}}
    (out_dir / "splits_metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def build_all(out_dir: Optional[Path] = None, seed: int = SEED) -> dict:
    from src.data.loaders import load_so762, load_l2arctic
    so = load_so762(with_audio_info=False)
    l2 = load_l2arctic(with_audio_info=False)
    so_spk = build_so762_splits(so.utts, seed=seed)
    so_checks = check_so762(so_spk, so.utts, _speaker_table(so.utts))
    spk = l2.utts.drop_duplicates("speaker")[["speaker", "l1", "gender"]].copy()
    spk["gender"] = spk["gender"].str.upper()
    folds = build_l2arctic_folds(spk, seed=seed)
    l2_checks = check_l2arctic(folds)
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "data" / "splits"
    write_splits(so_spk, so_checks, folds, l2_checks, out_dir, seed)
    return {"so762": so_checks, "l2arctic": l2_checks}
