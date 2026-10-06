"""Sprint 3 tests: features, aggregation helpers, metrics, split discipline, validation-only selection."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.evaluation import metrics as M
from src.features import aggregate as ag
from src.features.mfcc import DIM, HOP, SR, mfcc_from_signal, seg_frames

REPO = Path(__file__).resolve().parents[1]


# ---- features -------------------------------------------------------------- #
def test_mfcc_shape_cmvn_and_finite():
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(SR)).astype(np.float32)          # 1 s noise
    F = mfcc_from_signal(y)
    assert F.shape == (1 + SR // HOP, DIM) and F.dtype == np.float32
    assert np.isfinite(F).all()
    assert np.allclose(F.mean(0), 0, atol=1e-4) and np.allclose(F.std(0), 1, atol=1e-3)


def test_mfcc_very_short_signal_does_not_crash():
    F = mfcc_from_signal(np.zeros(400, np.float32) + 1e-3 * np.arange(400, dtype=np.float32) / 400)
    assert F.shape[1] == DIM and np.isfinite(F).all()


def test_seg_frames_clamps_and_keeps_one_frame():
    assert seg_frames(100, 0.0, 0.5) == (0, 50)
    i0, i1 = seg_frames(100, 0.2, 0.2)                                 # zero-length interval still yields a frame
    assert i1 - i0 == 1
    i0, i1 = seg_frames(100, 5.0, 6.0)                                 # past the end -> last frame
    assert (i0, i1) == (99, 100)


def test_seg_stats_dimension_and_values():
    F = np.vstack([np.full(DIM, 1.0), np.full(DIM, 3.0)]).astype(np.float32).repeat(50, 0)
    s = ag.seg_stats(F, 0.0, 1.0)
    assert s.shape == (4 * DIM,) and len(ag.STAT_NAMES) == 4 * DIM
    assert np.allclose(s[:DIM], 2.0) and np.allclose(s[DIM:2 * DIM], 1.0)   # mean, std
    assert np.allclose(s[2 * DIM:3 * DIM], 1.0) and np.allclose(s[3 * DIM:], 3.0)  # min, max


def test_fixed_window_is_independent_of_interval_length():
    a = ag.fixed_window(1.00, 1.04)       # a 40 ms "deleted phone" marker
    b = ag.fixed_window(0.90, 1.14)       # a long correct phone with the same midpoint
    assert np.allclose(a, b) and np.isclose(a[1] - a[0], 2 * ag.WIN_HALF_S)


def test_anchor_phone_for_additions_is_following_slot_unit_not_missing():
    canon = [None, "K", None, "AE", "T", None]     # silence, K, ADDITION, AE, T, trailing silence
    anchor = ag._anchor_phones(canon)
    assert anchor[2] == "AE"                        # addition attaches forward
    assert anchor[5] == "T"                         # nothing follows -> previous slot unit
    assert all(a is not None for a in anchor)       # the class cannot be read off a missing canonical phone


def test_pause_additions_attach_to_following_word():
    word_idx = np.array([np.nan, 1.0, 1.0, np.nan, 2.0, np.nan])
    ec = np.array(["silence", "correct", "correct", "addition", "correct", "silence"])
    w = ag._attach_words(word_idx, ec)
    assert w[3] == 2.0 and np.isnan(w[0]) and np.isnan(w[5])


def test_onehot_phone():
    v = ag.onehot("AA")
    assert v.sum() == 1 and v.shape == (ag.N_PHONES,)
    assert ag.onehot(None).sum() == 0


# ---- metrics --------------------------------------------------------------- #
def test_regression_metrics_values():
    y, p = np.array([1., 2., 3., 4.]), np.array([1., 2., 4., 6.])
    r = M.regression_metrics(y, p)
    assert np.isclose(r["mae"], 0.75) and np.isclose(r["rmse"], np.sqrt((0 + 0 + 1 + 4) / 4))
    assert np.isclose(r["pcc"], np.corrcoef(y, p)[0, 1])
    assert np.isclose(M.fast_reg(y, p)["pcc"], r["pcc"])


def test_fast_clf_matches_sklearn():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    f = M.fast_clf_factory(4)(y, p)
    assert np.isclose(f["macro_f1"], f1_score(y, p, average="macro"))
    assert np.isclose(f["balanced_acc"], balanced_accuracy_score(y, p))
    yb, pb = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    fb = M.fast_clf_factory(2, positive=1)(yb, pb)
    assert np.isclose(fb["f1"], f1_score(yb, pb))


def test_speaker_bootstrap_ci_brackets_point_estimate_and_resamples_speakers():
    rng = np.random.default_rng(2)
    spk = np.repeat(np.arange(30), 20)
    y = rng.normal(size=600)
    p = y + rng.normal(scale=0.5, size=600)
    ci = M.speaker_bootstrap(spk, y, p, M.fast_reg, n_boot=200)
    pt = M.fast_reg(y, p)
    for k in ("mae", "rmse", "pcc"):
        assert ci[k][0] <= pt[k] <= ci[k][1]


def test_paired_bootstrap_identical_predictions_gives_zero_delta():
    rng = np.random.default_rng(3)
    spk = np.repeat(np.arange(20), 10)
    y = rng.normal(size=200)
    p = y + rng.normal(size=200)
    r = M.paired_speaker_bootstrap(spk, y, p, p, M.fast_reg, "mae", n_boot=100)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0


# ---- split discipline on the real tables (skipped when the tables are absent) -------------- #
def _tables_ready():
    from src.utils.paths import get_data_root
    return (get_data_root() / "processed" / "features" / "tables" / "so762_phone_X.npy").exists()


@pytest.mark.skipif(not _tables_ready(), reason="feature tables not built")
def test_so762_speakers_are_disjoint_and_counts_match_sprint2():
    from src.models.baselines import build_tasks, load_config, load_task
    cfg = load_config()
    for name, n in (("sent_total", {"train": 2120, "val": 380, "test": 2500}),):
        _, _, meta, _ = load_task(build_tasks(cfg)[name])
        sets = {s: set(meta.loc[meta.split == s, "speaker"]) for s in n}
        assert sets["train"].isdisjoint(sets["val"]) and sets["train"].isdisjoint(sets["test"]) and sets["val"].isdisjoint(sets["test"])
        assert meta.split.value_counts().to_dict() == n
    # the table's official-split flag is replaced by the Sprint 2 val split: no val speaker is in train
    _, _, meta, _ = load_task(build_tasks(cfg)["phone_score"])
    assert (meta.split == "val").sum() > 0


@pytest.mark.skipif(not _tables_ready(), reason="feature tables not built")
def test_l2arctic_folds_are_speaker_disjoint_and_cover_everyone_once_as_test():
    from src.models.baselines import build_tasks, load_config, load_task, split_masks
    cfg = load_config()
    task = build_tasks(cfg)["l2_phone_error"]
    _, _, meta, _ = load_task(task)
    seen = []
    for k in range(4):
        m = split_masks(task, meta, k)
        spk = {s: set(meta.loc[m[s], "speaker"]) for s in m}
        assert spk["train"].isdisjoint(spk["val"]) and spk["train"].isdisjoint(spk["test"]) and spk["val"].isdisjoint(spk["test"])
        assert len(spk["test"]) == 6 and len(spk["val"]) == 6 and len(spk["train"]) == 12
        seen += sorted(spk["test"])
    assert sorted(seen) == sorted(meta["speaker"].unique()) and len(seen) == 24


# ---- validation-only selection ------------------------------------------------------- #
def test_select_config_never_reads_test_predictions(tmp_path, monkeypatch):
    from src.evaluation import report_baseline as rb
    from src.models import baselines as bl
    monkeypatch.setattr(bl, "results_root", lambda variant: tmp_path / variant)
    cfg = bl.load_config()
    task = bl.build_tasks(cfg)["sent_total"]
    n = 40
    meta = pd.DataFrame({"split": ["train"] * 20 + ["val"] * 10 + ["test"] * 10, "speaker": list("ab" * 20)})
    y = np.arange(n, dtype=float)
    for model, config, _ in bl.configs_for(task, cfg):
        out = bl.job_path("t", task.name, model, config, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        good = config == "alpha10"
        np.savez(out, val_pred=y[20:30] + (0 if good else 5.0))       # NO test_pred key: reading it would raise
    cfg_id, _ = rb.select_config("t", task, "ridge", cfg, y, meta)
    assert cfg_id == "alpha10"


# ---- US 3.3 helpers --------------------------------------------------------------------- #
def test_resample_indices_flattens_skewed_target():
    from src.models.baselines import resample_indices
    y = np.r_[np.full(900, 2.0), np.full(100, 0.2)]
    idx = resample_indices(y, [-0.01, 0.5, 2.01], 0.5, len(y), seed=0)
    assert len(idx) == len(y)
    assert (y[idx] < 0.5).mean() > 0.10 and (y[idx] < 0.5).mean() < 0.5      # low scores up-weighted, not fully balanced


def test_fit_weights_improves_macro_f1_on_imbalanced_probabilities():
    from src.evaluation.report_us33 import fit_weights
    from src.models.baselines import Task
    task = Task("t", "phoneme", "so762", "x", "clf", "y", "split", labels=(0, 1), positive=1)
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(int)
    p1 = np.clip(0.04 + 0.12 * y + 0.05 * rng.standard_normal(4000), 0.001, 0.99)   # informative but never above 0.5
    P = np.c_[1 - p1, p1]
    f = M.fast_clf_factory(2, 1)
    base = f(y, P.argmax(1))["macro_f1"]
    w = fit_weights(task, P, y)
    assert f(y, (P * w).argmax(1))["macro_f1"] > base + 0.05


def test_variant_configs_reuse_us31_selected_config():
    from src.models import baselines as bl
    if not (REPO / "results" / "tables" / "baseline" / "us31_selected.csv").exists():
        pytest.skip("us31 results not present")
    cfg = bl.load_config()
    t = bl.build_tasks(cfg)["l2_phone_error"]
    got = {m: (c, p) for m, c, p in bl.configs_for(t, cfg, "cw")}
    assert "svm" not in got and set(got) == {"logreg", "rf"}          # svm skipped on the 60k-row task
    assert isinstance(got["rf"][1]["min_samples_leaf"], int)
    assert bl.configs_for(bl.build_tasks(cfg)["sent_total"], cfg, "cw") == []   # class weights apply to classification only
