"""US 8.1: sufficient-statistic metrics, bootstrap helpers, test-manifest rebuild and an end-to-end run on synthetic tables (numpy / pandas only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import report_us81 as S


def _spk(n_spk=12, per=20):
    return np.repeat([f"s{i:02d}" for i in range(n_spk)], per)


def _codes(spk):
    return np.unique(spk, return_inverse=True)[1]


# ----------------------------------------------------------------------------- metrics
def test_regression_metrics_match_direct_formulas():
    rng = np.random.default_rng(0)
    spk = _spk()
    y = rng.normal(size=len(spk))
    p = 0.5 * y + rng.normal(0, 0.5, len(y))
    m = S.reg_metrics(S.reg_stats(y, p, _codes(spk), 12).sum(0))
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((p - y) ** 2)))
    assert m["bias"] == pytest.approx(np.mean(p - y))
    assert m["pcc"] == pytest.approx(np.corrcoef(y, p)[0, 1])
    assert m["slope"] == pytest.approx(np.polyfit(y, p, 1)[0])
    assert m["y_sd"] == pytest.approx(y.std())


def test_regression_metrics_broadcast_over_leading_axis_and_flag_constant_labels():
    st = S.reg_stats(np.arange(10.0), np.arange(10.0) + 1, np.zeros(10, int), 1)
    stack = np.stack([st, st * 2, st * 3])                               # three "bootstrap draws" of the same data
    m = S.reg_metrics(stack.sum(1))
    assert m["rmse"].shape == (3,) and np.allclose(m["rmse"], 1.0) and np.allclose(m["bias"], 1.0)
    flat = S.reg_metrics(S.reg_stats(np.ones(5), np.arange(5.0), np.zeros(5, int), 1).sum(0))
    assert np.isnan(flat["pcc"]) and np.isnan(flat["slope"])


def _macro_f1(y, p, k):
    f1 = []
    for c in range(k):
        if (y == c).sum() == 0:
            continue
        tp = ((y == c) & (p == c)).sum()
        prec = tp / max((p == c).sum(), 1)
        rec = tp / (y == c).sum()
        f1.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1))


def test_classification_metrics_match_direct_and_skip_absent_classes():
    rng = np.random.default_rng(1)
    spk = _spk()
    y = rng.integers(0, 2, len(spk))                                      # class 2 never occurs in y but is predicted sometimes
    p = np.where(rng.random(len(y)) < 0.2, 2, np.where(rng.random(len(y)) < 0.7, y, 1 - y))
    m = S.clf_metrics(S.clf_stats(y, p, _codes(spk), 12, 3).sum(0), 3)
    assert m["macro_f1"] == pytest.approx(_macro_f1(y, p, 3))
    yb, pb = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    b = S.clf_metrics(S.clf_stats(yb, pb, np.zeros(400, int), 1, 2).sum(0), 2)
    assert b["recall"] == pytest.approx(((yb == 1) & (pb == 1)).sum() / (yb == 1).sum())
    assert b["precision"] == pytest.approx(((yb == 1) & (pb == 1)).sum() / (pb == 1).sum())
    assert b["pos_rate"] == pytest.approx((yb == 1).mean())


# ----------------------------------------------------------------------------- bootstrap helpers
def test_weights_are_reproducible_and_each_draw_has_n_speakers():
    w = S.weights(15, 40, seed=3)
    assert w.shape == (40, 15) and np.all(w.sum(1) == 15)
    assert np.array_equal(w, S.weights(15, 40, seed=3)) and not np.array_equal(w, S.weights(15, 40, seed=4))


def test_identical_systems_have_a_zero_delta_with_a_degenerate_interval():
    rng = np.random.default_rng(2)
    spk = _spk()
    y, p = rng.normal(size=len(spk)), rng.normal(size=len(spk))
    c = S.Cell("sent_total", y, p, spk)
    st = c.stats(np.ones(len(y), bool))
    r = S.delta_ci("sent_total", "rmse", st, st, S.weights(len(st), 100, 0))
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["p"] == 1.0


def test_holm_matches_a_worked_example_and_keeps_input_order():
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert np.allclose(S.holm([0.5]), [0.5]) and S.holm([0.6, 0.9]).max() <= 1.0


def test_pvalue_and_interval():
    assert S.pvalue(np.arange(1, 101)) == 0.0
    assert S.pvalue(np.r_[-np.ones(50), np.ones(50)]) == 1.0
    lo, hi = S.interval(np.arange(101))
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)


def _two_group_cells(task, make_pred, n_spk=20, per=200, seed=0):
    rng = np.random.default_rng(seed)
    spk = np.repeat([f"s{i:02d}" for i in range(2 * n_spk)], per)
    child = np.repeat(np.arange(2 * n_spk) >= n_spk, per)
    y = rng.integers(0, 2, len(spk)) if task in S.N_CLASSES else rng.normal(5, 1.5, len(spk))
    return S.Cell(task, y, make_pred(rng, y, child), spk), ~child, child


def test_adverse_gap_is_positive_when_child_speakers_are_served_worse_for_both_metric_kinds():
    reg, adult, child = _two_group_cells("sent_accuracy", lambda r, y, c: y + r.normal(0, np.where(c, 1.2, 0.3)))
    clf, adult2, child2 = _two_group_cells("phone_mispron_t1.0", lambda r, y, c: np.where(r.random(len(y)) < np.where(c, 0.6, 0.9), y, 1 - y), seed=1)
    for cell, a, c in [(reg, adult, child), (clf, adult2, child2)]:
        sa, sc = cell.stats(a), cell.stats(c)
        adv, dr, vc, va = S.gap_draws(cell.task, sa, sc, S.weights(len(sa), 200, 0), S.weights(len(sc), 200, 1))
        lo, hi = S.interval(dr)
        assert adv > 0 and lo > 0 and S.pvalue(dr) < 0.05
    assert vc < va                                                         # macro-F1 is lower for the group that is served worse


def test_readings():
    assert S.reading_gap(0.3, 0.1, 0.5, 0.01) == "child/teen worse"
    assert S.reading_gap(0.3, 0.1, 0.5, 0.2) == "no significant gap"
    assert S.reading_gap(-0.3, -0.5, -0.1, np.nan) == "child/teen better (unadjusted)"
    assert S.reading_change(-0.1, -0.2, -0.05, 0.01) == "gap narrower" and S.reading_change(0.1, -0.1, 0.2, 0.5) == "no significant change"


# ----------------------------------------------------------------------------- manifest rebuild, on synthetic tables
@pytest.fixture
def so762_tables(tmp_path, monkeypatch):
    feats, splits, bundles = tmp_path / "tables", tmp_path / "splits", tmp_path / "bundles"
    for d in (feats, splits, bundles):
        d.mkdir()
    rng = np.random.default_rng(5)
    n_spk, per = 24, 10
    spk = [f"{i:04d}" for i in range(n_spk)]
    rows = [(f"{s}{u:03d}", s) for s in spk for u in range(per)]
    meta = pd.DataFrame({"utt_id": [r[0] for r in rows], "speaker": [r[1] for r in rows]})
    meta["age_group"] = np.where(meta["speaker"].astype(int) < 12, "adult", "child/teen")
    meta["gender"] = np.where(meta["speaker"].astype(int) % 2 == 0, "m", "f")
    meta["split"] = "train"
    for c in ("y_accuracy", "y_fluency", "y_prosodic", "y_total"):
        meta[c] = rng.integers(3, 11, len(meta)).astype(float)
    meta.to_csv(feats / "so762_sentence_meta.csv.gz", index=False)
    np.save(feats / "so762_sentence_side_X.npy", np.column_stack([rng.uniform(2, 10, len(meta)), rng.integers(3, 12, len(meta)), rng.integers(10, 40, len(meta))]))
    (feats / "so762_sentence_side_names.json").write_text(json.dumps(["dur", "n_words", "n_phones"]))
    pd.DataFrame({"speaker": spk, "split": ["test"] * n_spk, "gender": "m", "age": 20, "age_group": "adult"}).to_csv(splits / "so762_speakers.csv", index=False)
    for mod, v in [("FEATURES", feats), ("SPLITS", splits), ("BUNDLES", bundles)]:
        monkeypatch.setattr(S, mod, v)
    S._CACHE.clear()
    yield meta, bundles
    S._CACHE.clear()


def test_so762_frame_follows_the_test_rows_and_builds_balanced_slices(so762_tables):
    meta, _ = so762_tables
    f = S.test_frame("sent_accuracy")
    assert len(f) == len(meta) and np.array_equal(f["speaker"], meta["speaker"]) and np.array_equal(f["y_enc"], meta["y_accuracy"])
    assert set(f["age"]) == {"adult", "child/teen"} and set(f["score_bin"]) <= {"<=6", "7-8", "9-10"}
    assert set(f["length"]) == set(S.LENGTH_LABELS) and len(f.attrs["length_cuts"]) == 2
    per_spk = f.groupby("speaker")["prof"].nunique()
    assert (per_spk == 1).all() and f.groupby("prof")["speaker"].nunique().tolist() == [8, 8, 8]
    masks = S.group_masks(f)
    assert set(masks) == {"all", "age", "gender", "length", "proficiency", "score_bin"}
    assert masks["age"]["adult"].sum() + masks["age"]["child/teen"].sum() == len(f)


def test_check_bundle_rejects_rows_that_do_not_match_the_manifest(so762_tables):
    meta, _ = so762_tables
    f = S.test_frame("sent_accuracy")
    good = {"y": meta["y_accuracy"].to_numpy(), "pred": meta["y_accuracy"].to_numpy(), "speaker": meta["speaker"].to_numpy()}
    S.check_bundle("sent_accuracy", f, good)
    for bad in ({**good, "speaker": good["speaker"][::-1].copy()}, {**good, "y": good["y"] + 1}, {**good, "y": good["y"][:-1], "pred": good["pred"][:-1], "speaker": good["speaker"][:-1]}):
        with pytest.raises(ValueError):
            S.check_bundle("sent_accuracy", f, bad)


def test_build_end_to_end_on_synthetic_bundles(so762_tables):
    meta, bundles = so762_tables
    rng = np.random.default_rng(9)
    y = meta["y_accuracy"].to_numpy()
    child = (meta["age_group"] == "child/teen").to_numpy()
    for name, noise in [("B", 1.5), ("FT-base", 0.6), ("MT-GN15", 0.6)]:
        pred = y + rng.normal(0, np.where(child, noise * 1.5, noise))
        np.savez_compressed(bundles / f"{name}__sent_accuracy.npz", y=y, pred=pred, speaker=meta["speaker"].to_numpy().astype(str))
    res = S.build(systems=["B", "FT-base", "MT-GN15"], n_boot=100, seed=0, tasks=["sent_accuracy"])
    sl = res["slices"]
    allrows = sl[(sl["family"] == "all") & (sl["metric"] == "rmse")].set_index("system")["value"]
    pred_ft = np.load(bundles / "FT-base__sent_accuracy.npz")["pred"]
    assert allrows["FT-base"] == pytest.approx(np.sqrt(np.mean((pred_ft - y) ** 2)))
    assert set(sl["family"]) == {"all", "age", "gender", "length", "proficiency", "score_bin"}
    g = res["age_gap"].set_index("system")
    assert set(g.index) == {"B", "FT-base", "MT-GN15"} and g.loc["MT-GN15", "family"] == "A" and g.loc["B", "family"] == "exploratory"
    assert (g["adverse_gap"] > 0).all() and g.loc["MT-GN15", "p_holm"] >= g.loc["MT-GN15", "p_raw"]
    ch = res["gap_change"]
    assert set(zip(ch["a"], ch["b"])) == {("FT-base", "MT-GN15")} and ch["family"].tolist() == ["B"]
    assert {"verdict"} <= set(res["pairs"].columns) and set(res["pairs"]["a"]) <= {"B", "F", "FT-base"}
    assert len(res["speakers"]) == 3 * 24


def test_l2_rows_come_in_fold_order_with_utterance_durations_and_error_rate_proficiency(tmp_path, monkeypatch):
    feats = tmp_path / "tables"
    feats.mkdir()
    spk = [f"p{i}" for i in range(6)]
    fold = {s: i % 3 for i, s in enumerate(spk)}
    ew, ph = [], []
    for s in spk:
        for u in range(3):
            ew.append({"utt_id": f"{s}_{u}", "speaker": s, "l1": "Hindi" if s < "p3" else "Korean", "fold": fold[s], "gender": "F", "word_idx": 0, "word": "x", "y_has_error": u % 2, "n_errors": u % 2})
            for k in range(2):
                ph.append({"utt_id": f"{s}_{u}", "speaker": s, "l1": "Hindi", "fold": fold[s], "gender": "F", "phone_idx": k, "anchor_phone": "AA", "error_class": "correct" if (k + u) % 2 else "deletion",
                           "ann_dur": 0.1 * (u + 1), "is_deviation": 0, "is_err": 0})
    pd.DataFrame(ew).to_csv(feats / "l2_word_meta.csv.gz", index=False)
    pd.DataFrame(ph).to_csv(feats / "l2_phone_meta.csv.gz", index=False)
    monkeypatch.setattr(S, "FEATURES", feats)
    S._CACHE.clear()
    rows = S.test_rows("l2_word")
    folds = pd.read_csv(feats / "l2_word_meta.csv.gz")["fold"].to_numpy()[rows]
    assert list(folds) == sorted(folds) and len(rows) == len(ew)
    dur = S.utterance_durations("l2")
    assert dur["p0_0"] == pytest.approx(0.2) and dur["p0_2"] == pytest.approx(0.6)
    prof = S.speaker_proficiency("l2")
    assert prof.between(-1, 0).all()
    f = S.test_frame("l2_word_error")
    assert set(f["l1"]) == {"Hindi", "Korean"} and set(f["y_enc"]) == {0, 1} and "score_bin" not in f
    S._CACHE.clear()
