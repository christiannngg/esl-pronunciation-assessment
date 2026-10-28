"""US 7.1 add-on: bundles, the paired bootstrap on them, and merging into the report (numpy / pandas only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import report_us71 as R
from src.evaluation import report_us71_pairs as P


def _speakers(n_spk=12, per=20):
    return np.repeat([f"s{i}" for i in range(n_spk)], per)


def test_regression_pair_detects_a_better_system(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BUNDLES", tmp_path)
    rng = np.random.default_rng(0)
    spk = _speakers()
    y = rng.normal(size=len(spk))
    P.save_bundle("B", "sent_total", y, y + rng.normal(0, 1.0, len(y)), spk)
    P.save_bundle("FT-base", "sent_total", y, y + rng.normal(0, 0.3, len(y)), spk)
    r = P.paired("sent_total", "B", "FT-base", P.load_bundle("B", "sent_total"), P.load_bundle("FT-base", "sent_total"), n_boot=200, seed=0)
    assert r["delta"] < 0 and r["verdict"] == "better" and r["lo"] < r["delta"] < r["hi"]


def test_classification_pair_uses_macro_f1_and_integer_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BUNDLES", tmp_path)
    rng = np.random.default_rng(1)
    spk = _speakers()
    y = rng.integers(0, 2, len(spk))
    good = np.where(rng.random(len(y)) < 0.9, y, 1 - y)
    bad = np.where(rng.random(len(y)) < 0.6, y, 1 - y)
    r = P.paired("phone_mispron_t1.0", "B", "FT-base", {"y": y, "pred": bad, "speaker": spk}, {"y": y, "pred": good, "speaker": spk}, n_boot=200, seed=0)
    assert r["delta"] > 0 and r["verdict"] == "better"


def test_different_rows_are_refused():
    spk = _speakers()
    y = np.zeros(len(spk))
    a = {"y": y, "pred": y, "speaker": spk}
    b = {"y": y.copy()[::-1] + np.arange(len(y)), "pred": y, "speaker": spk}
    assert P.paired("sent_total", "B", "F", a, b, n_boot=10, seed=0) is None


def test_bundle_roundtrip_and_score(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BUNDLES", tmp_path)
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    P.save_bundle("F", "sent_total", y, p, ["a", "a", "b"])
    b = P.load_bundle("F", "sent_total")
    assert abs(P.score_bundle("sent_total", b) - np.sqrt(4 / 3)) < 1e-9 and P.load_bundle("F", "sent_fluency") is None


def test_merge_replaces_only_matching_missing_rows():
    pairs = pd.DataFrame([{"task": "sent_total", "a": "B", "b": "FT-base", "delta": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "verdict": "MISSING (x)"},
                          {"task": "sent_total", "a": "B", "b": "MT-GN15", "delta": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "verdict": "MISSING (x)"},
                          {"task": "sent_total", "a": "B", "b": "F", "delta": -0.04, "lo": -0.09, "hi": -0.01, "p": 0.0, "verdict": "better"}])
    comp = pd.DataFrame([{"task": "sent_total", "a": "B", "b": "FT-base", "delta": -0.1, "lo": -0.15, "hi": -0.05, "p": 0.0, "verdict": "better"}])
    m = R.merge_computed(pairs, comp)
    assert len(m) == 3 and (m["verdict"].str.startswith("MISSING").sum() == 1)
    assert m[(m.b == "FT-base")].iloc[0]["delta"] == -0.1 and R.merge_computed(pairs, pd.DataFrame()).equals(pairs)
