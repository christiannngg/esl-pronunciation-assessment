"""Sprint 5 (US 5.2) tests: curve statistics, overfitting criteria, speaker-independent difference CI, sentence-text overlap. Numpy / pandas only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import diagnose_finetune as D
from src.evaluation import metrics as M
from src.models import finetune as ft
from src.models.baselines import build_tasks, load_config as load_baseline_config

THR = ft.load_config()["diagnose52"]
TASKS = build_tasks(load_baseline_config())


def _job(vp, tp, vl, tl, best, spe=10, freeze=10, seed=0, task="sent_accuracy"):
    return {"hist_val_primary": np.array(vp, float), "hist_trsub_primary": np.array(tp, float), "hist_val_loss": np.array(vl, float),
            "hist_train_loss": np.array(tl, float), "best_epoch": best, "steps_per_epoch": spe, "freeze_steps": freeze, "seed": seed, "fold": 0}


def test_rel_gap_signs_and_units():
    assert D.rel_gap("reg", 1.0, 0.5) == pytest.approx(0.5)               # RMSE: val 1.0, train 0.5 -> train better by half
    assert D.rel_gap("clf", 0.6, 0.9) == pytest.approx(1 / 3)             # macro-F1: train 0.9, val 0.6
    assert D.rel_gap("reg", 0.5, 0.5) == 0.0


def test_job_summary_regression_overfit_curve():
    d = _job(vp=[1.2, 1.0, 0.95, 0.97, 1.05, 1.1], tp=[1.1, 0.8, 0.5, 0.3, 0.2, 0.1], vl=[0.6, 0.5, 0.48, 0.5, 0.6, 0.7],
             tl=[0.7, 0.4, 0.2, 0.1, 0.05, 0.03], best=2)
    s = D.job_summary(TASKS["sent_accuracy"], d, THR)
    assert s["best_epoch"] == 2 and s["head_epochs"] == 1 and not s["early_best"] and not s["head_only_best"]
    assert s["gap_best"] == pytest.approx((0.95 - 0.5) / 0.95)
    assert s["gain_over_head_only"] == pytest.approx(1.2 - 0.95) and s["gain_rel"] == pytest.approx((1.2 - 0.95) / 1.2)   # reference = head-only epoch 0
    assert s["val_loss_rise"] == pytest.approx(0.7 / 0.48 - 1) and s["train_loss_drop"] == pytest.approx((0.2 - 0.03) / 0.2)


def test_job_summary_classification_uses_natural_f1_and_head_only_best():
    # history stores -macro-F1: val F1 0.60, 0.58, 0.55 (best = head-only epoch 0); train subset F1 0.9, 0.95, 0.99
    d = _job(vp=[-0.60, -0.58, -0.55], tp=[-0.9, -0.95, -0.99], vl=[0.9, 1.0, 1.2], tl=[0.9, 0.5, 0.3], best=0, task="phone_mispron_t1.0")
    s = D.job_summary(TASKS["phone_mispron_t1.0"], d, THR)
    assert s["head_only_best"] and s["early_best"] and s["val_at_best"] == pytest.approx(0.60) and s["trainsub_at_best"] == pytest.approx(0.9)
    assert s["gap_best"] == pytest.approx((0.9 - 0.6) / 0.9) and s["gain_over_head_only"] == 0.0 and s["train_loss_drop"] == pytest.approx((0.9 - 0.3) / 0.9)


def _frame(rows):
    return pd.DataFrame([D.job_summary(TASKS["sent_accuracy"], r, THR) for r in rows])


def test_flags_counts_criteria_and_risk_levels():
    overfit = _job([1.2, 1.0, 0.95, 0.97, 1.05, 1.1], [1.1, 0.8, 0.5, 0.3, 0.2, 0.1], [0.6, 0.5, 0.48, 0.5, 0.6, 0.7], [0.7, 0.4, 0.2, 0.1, 0.05, 0.03], 2)
    f = D.aggregate_flags("sent_accuracy", _frame([overfit, overfit]), THR, None)
    assert f["C1_gap"] and f["C2_diverge"] and not f["C3_early_best"] and f["C5_seen_gap"] is None
    assert f["n_fired"] >= 2 and f["overfit_risk"] in ("medium", "high") and "C1" in f["criteria_fired"]
    healthy = _job([1.2, 1.1, 1.0, 0.98, 0.97, 0.97], [1.1, 1.0, 0.95, 0.93, 0.9, 0.9], [0.6, 0.55, 0.5, 0.49, 0.49, 0.49], [0.7, 0.6, 0.5, 0.45, 0.42, 0.4], 5)
    g = D.aggregate_flags("sent_accuracy", _frame([healthy]), THR, {"rel_gap": 0.01, "ci_excludes_0": False})
    assert g["n_fired"] == 0 and g["overfit_risk"] == "low" and g["C5_seen_gap"] is False
    seen = D.aggregate_flags("sent_accuracy", _frame([healthy]), THR, {"rel_gap": 0.3, "ci_excludes_0": True})
    assert seen["C5_seen_gap"] is True
    # a CI that includes 0 does not fire C5 even with a large point estimate
    assert D.aggregate_flags("sent_accuracy", _frame([healthy]), THR, {"rel_gap": 0.3, "ci_excludes_0": False})["C5_seen_gap"] is False


def test_flags_early_best_and_low_gain():
    early = _job([1.0, 0.99, 1.05, 1.1, 1.2], [1.0, 0.7, 0.5, 0.3, 0.2], [0.5, 0.5, 0.6, 0.7, 0.8], [0.6, 0.4, 0.2, 0.1, 0.1], 1)
    f = D.aggregate_flags("sent_accuracy", _frame([early, early]), THR, None)
    assert f["C3_early_best"] and f["C4_low_gain"] and f["share_early_best"] == 1.0


def test_indep_diff_recovers_a_known_difference_and_its_ci_excludes_zero():
    rng = np.random.default_rng(0)
    fast = M.fast_reg
    spk_a, spk_b = np.repeat(np.arange(30), 20), np.repeat(np.arange(100, 130), 20)
    ya, yb = rng.normal(size=600), rng.normal(size=600)
    pa, pb = ya + rng.normal(0, 0.2, 600), yb + rng.normal(0, 0.6, 600)           # a is predicted much better than b
    r = D.indep_diff(spk_a, ya, pa, spk_b, yb, pb, fast, "rmse", n_boot=300)
    assert r["diff"] < -0.2 and r["hi"] < 0                                         # RMSE(a) - RMSE(b) clearly negative
    same = D.indep_diff(spk_a, ya, pa, spk_b, yb, yb + rng.normal(0, 0.2, 600), fast, "rmse", n_boot=300)
    assert same["lo"] < 0 < same["hi"]


def test_text_overlap_and_texts_from_word_meta(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "features" / "tables"
    root.mkdir(parents=True)
    words = pd.DataFrame({"utt_id": ["a", "a", "b", "b", "c", "c", "d", "d"], "word_idx": [1, 0, 0, 1, 0, 1, 0, 1],
                          "word": ["World", "Hello", "Hello", "World", "Other", "Text", "Hello", "World"]})
    words.to_csv(root / "so762_word_meta.csv.gz", index=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    t = D.sentence_texts(TASKS["sent_accuracy"])
    assert t == {"a": "hello world", "b": "hello world", "c": "other text", "d": "hello world"}
