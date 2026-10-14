"""US 4.2 diagnosis tests (CPU, no torch): pure table helpers with hand-computed numbers, and an end-to-end run on the synthetic world of
test_frozen_heads (numpy stand-in trainer) that checks validation-only access, alignment with a Sprint 3 stand-in, and the output tables."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import diagnose_frozen as DF
from src.evaluation import report_frozen as RF
from src.models import frozen_heads as FH
from test_frozen_heads import ENC, _run, task, world  # noqa: F401  (world is a fixture)


# ---- pure helpers ----------------------------------------------------------- #
def test_error_mask_for_binary_positive_and_for_correct_label_tasks():
    assert list(DF.error_mask(task("phone_mispron_t1.0"), np.array([0, 1, 1, 0]))) == [False, True, True, False]
    assert list(DF.error_mask(task("l2_phone_error"), np.array(["correct", "deletion", "addition", "correct"]))) == [False, True, True, False]


def test_overfit_gap_sign_convention():
    assert DF.overfit_gap(task("phone_score"), 1.2, 0.8) == pytest.approx(0.4)             # RMSE: val worse than train -> positive
    assert DF.overfit_gap(task("l2_phone_error"), 0.45, 0.60) == pytest.approx(0.15)       # macro-F1: train higher -> positive
    assert DF.overfit_gap(task("phone_score"), 0.8, 1.2) < 0


def test_mlp_gain_and_best_head_follow_the_metric_direction():
    reg, clf = task("phone_score"), task("l2_phone_error")
    assert DF.mlp_gain(reg, 0.35, 0.33) == pytest.approx(0.02) and DF.mlp_gain(clf, 0.45, 0.50) == pytest.approx(0.05)
    assert DF.mlp_gain(reg, 0.33, 0.35) < 0 and DF.mlp_gain(clf, 0.50, 0.45) < 0
    assert DF.pick_best_head(reg, {"linear": {"val_primary": 0.35}, "mlp": {"val_primary": 0.33}}) == "mlp"          # RMSE: lower wins
    assert DF.pick_best_head(clf, {"linear": {"val_primary": 0.45}, "mlp": {"val_primary": 0.50}}) == "mlp"         # macro-F1: higher wins
    assert DF.pick_best_head(clf, {"linear": {"val_primary": 0.55}, "mlp": {"val_primary": 0.50}}) == "linear"
    assert DF.pick_best_head(reg, {"linear": {"val_primary": 0.3}, "mlp": {"val_primary": 0.3}}) == "linear"


def test_auc_metrics_perfect_inverted_and_per_class():
    t = task("phone_mispron_t1.0")
    y = np.array([0, 0, 1, 1])
    good = np.array([[.9, .1], [.8, .2], [.3, .7], [.2, .8]])
    assert DF.auc_metrics(t, y, good)["auc_error"] == 1.0 and DF.auc_metrics(t, y, good[:, ::-1])["auc_error"] == 0.0
    assert DF.auc_metrics(t, y, good)["error_share"] == 0.5
    t4 = task("l2_phone_error")                                                        # labels: correct, substitution, deletion, addition
    y4 = np.array(["correct", "correct", "substitution", "deletion", "addition", "correct"])
    P = np.array([[.9, .05, .03, .02], [.8, .1, .05, .05], [.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7], [.7, .1, .1, .1]])
    m = DF.auc_metrics(t4, y4, P)
    assert m["auc_error"] == 1.0 and m["auc_substitution"] == 1.0 and m["auc_deletion"] == 1.0 and m["auc_addition"] == 1.0 and m["auc_correct"] == 1.0
    P2 = P.copy(); P2[:, [1, 2]] = P2[:, [2, 1]]                                         # swap the substitution / deletion columns: per-class AUCs must drop
    assert DF.auc_metrics(t4, y4, P2)["auc_substitution"] < 1.0 and DF.auc_metrics(t4, y4, P2)["auc_error"] == 1.0


def test_make_frame_checks_lengths_and_does_not_clobber_meta():
    meta = pd.DataFrame({"speaker": ["a", "b"], "y_phone_score": [1.0, 2.0]})
    f = DF.make_frame([1, 2], [1.5, 1.5], meta)
    assert list(f.columns) == ["speaker", "y_phone_score", "y", "pred"] and list(f.pred) == [1.5, 1.5]
    with pytest.raises(AssertionError):
        DF.make_frame([1, 2, 3], [1, 2, 3], meta)


def test_check_same_rows_detects_length_and_label_mismatch_for_numbers_and_strings():
    a = pd.DataFrame({"y": [1.0, 2.0, 3.0]}); b = pd.DataFrame({"y": [1, 2, 3]})
    DF.check_same_rows(a, b, "t")                                                              # float vs int labels, same values: fine
    with pytest.raises(ValueError, match="rows"):
        DF.check_same_rows(a, b.iloc[:2], "t")
    with pytest.raises(ValueError, match="differ"):
        DF.check_same_rows(a, pd.DataFrame({"y": [1, 3, 2]}), "t")
    s = pd.DataFrame({"y": ["correct", "deletion"]})
    DF.check_same_rows(s, s.copy(), "t")
    with pytest.raises(ValueError, match="differ"):
        DF.check_same_rows(s, pd.DataFrame({"y": ["deletion", "correct"]}), "t")


def test_clf_summary_hand_computed():
    t = task("phone_mispron_t1.0")
    f = pd.DataFrame({"y": [1, 1, 1, 0, 0, 0, 0, 0], "pred": [1, 1, 0, 1, 0, 0, 0, 0]})
    s = DF.clf_summary(t, f)
    assert s["error_share_true"] == pytest.approx(3 / 8) and s["error_share_pred"] == pytest.approx(3 / 8)
    assert s["error_recall"] == pytest.approx(2 / 3) and s["error_precision"] == pytest.approx(2 / 3)
    # class 1: P=R=2/3 -> F1 2/3 ; class 0: P=R=4/5 -> macro 0.7333
    assert s["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_score_bins_show_regression_to_the_mean_and_the_slope():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 3, 5, 8, 10], 2000, p=[.03, .05, .07, .15, .70]).astype(float)
    pred = 9.0 + 0.1 * (y - 9.0)                                                             # shrunk towards the mean: slope 0.1
    f = pd.DataFrame({"y": y, "pred": pred, "speaker": "s"})
    bins = DF.score_bins(f, DF.BIN_SPECS["word_accuracy"]).set_index("true_bin")
    assert bins.loc["0-4", "bias"] > 5 and bins.loc["10", "bias"] < 0 and bins["share"].sum() == pytest.approx(1)
    assert DF.reg_summary(f)["slope_pred_on_true"] == pytest.approx(0.1, abs=1e-6)


def test_bins_are_assigned_by_true_value_not_by_prediction():
    f = pd.DataFrame({"y": [0.0, 10.0], "pred": [10.0, 0.0]})
    b = DF.score_bins(f, DF.BIN_SPECS["word_accuracy"]).set_index("true_bin")
    assert b.loc["0-4", "mean_pred"] == 10 and b.loc["10", "mean_pred"] == 0


def test_by_group_regression_and_classification():
    t = task("phone_score")
    f = pd.DataFrame({"y": [0., 1., 2., 0., 1., 2.], "pred": [0., 1., 2., 2., 1., 0.], "speaker": list("aabbcc"), "age_group": ["c", "c", "c", "a", "a", "a"]})
    g = DF.by_group(t, f, "age_group").set_index("group")
    assert g.loc["c", "rmse"] == pytest.approx(0) and g.loc["a", "rmse"] == pytest.approx(np.sqrt(8 / 3)) and g.loc["a", "n_speakers"] == 2
    tc = task("l2_word_error")
    fc = pd.DataFrame({"y": [1, 1, 0, 0], "pred": [1, 0, 0, 0], "speaker": list("aabb"), "l1": ["x", "x", "y", "y"]})
    gc = DF.by_group(tc, fc, "l1").set_index("group")
    assert gc.loc["x", "error_recall"] == 0.5 and gc.loc["y", "error_share_true"] == 0


def test_phoneme_error_recall_support_join_and_spearman():
    t = task("l2_phone_error")
    rows = []
    for ph, n_err, n_hit in (("A", 8, 7), ("B", 6, 4), ("C", 5, 2), ("D", 4, 0)):                 # recall falls with training support
        rows += [(ph, "deletion", "deletion")] * n_hit + [(ph, "deletion", "correct")] * (n_err - n_hit) + [(ph, "correct", "correct")] * 10
    f = pd.DataFrame(rows, columns=["anchor_phone", "y", "pred"])
    sup = pd.DataFrame({"train_n": [100] * 4, "train_err": [40, 30, 20, 10], "train_error_rate": [.4, .3, .2, .1]}, index=pd.Index(list("ABCD"), name="anchor_phone"))
    t_ = DF.phoneme_error_recall(t, f, "anchor_phone", sup).set_index("phone")
    assert t_.loc["A", "error_recall"] == pytest.approx(7 / 8) and t_.loc["D", "error_recall"] == 0 and t_.loc["B", "n_err"] == 6
    assert t_.loc["A", "pred_err_rate"] == pytest.approx(7 / 18) and t_.loc["C", "train_err"] == 20
    sp = DF.support_spearman(t_.reset_index(), min_err=3)
    assert sp["rho_support"] == pytest.approx(1.0) and sp["n_phonemes"] == 4
    assert np.isnan(DF.support_spearman(t_.reset_index(), min_err=7)["rho_support"])            # < 3 phonemes left -> NaN, not an exception


def test_train_error_support_uses_training_rows_only_and_averages_folds():
    t = task("l2_phone_error")                                                                    # folds 0-3: test=k, val=(k+1)%4, train = the other two
    meta = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3], 4), "anchor_phone": "AH"})
    y = np.array(["deletion"] * 4 + ["correct"] * 4 + ["deletion"] * 2 + ["correct"] * 2 + ["correct"] * 4 + ["correct"] * 4)[:16]
    out = DF.train_error_support(t, y, meta, "anchor_phone")
    # fold k trains on folds {k+2, k+3} mod 4 (test=k, val=k+1): errors per fold 4 / 0 / 2 / 0
    # f0 -> {2,3}: 2 ; f1 -> {3,0}: 4 ; f2 -> {0,1}: 4 ; f3 -> {1,2}: 2   (mean 3.0, 8 training rows each)
    assert out.loc["AH", "train_n"] == 8 and out.loc["AH", "train_err"] == pytest.approx(3.0) and out.loc["AH", "train_error_rate"] == pytest.approx(3 / 8)


# ---- end to end on the synthetic world -------------------------------------- #
def _baseline_frames(monkeypatch, tasks_):
    """US 3.1 stand-in: constant / majority prediction on the same validation rows (what a no-signal baseline looks like)."""
    def fake(t, bcfg=None):
        y_all, meta = FH.load_labels(t)
        m = FH.split_masks(t, meta, 0)["val"] if t.splitter == "split" else FH.split_masks(t, meta, 0)["val"]
        if t.splitter == "fold":                                                                  # pooled over folds, in fold order, like the real files
            parts = [meta[FH.split_masks(t, meta, f)["val"]] for f in t.folds]
            yy = np.concatenate([y_all[FH.split_masks(t, meta, f)["val"]] for f in t.folds])
            mm = pd.concat(parts, ignore_index=True)
        else:
            yy, mm = y_all[m], meta[m]
        pred = np.full(len(yy), yy.mean()) if t.kind == "reg" else np.full(len(yy), "correct" if isinstance(t.labels[0], str) else 0, dtype=object if isinstance(t.labels[0], str) else int)
        return "dummy", mm.assign(y=yy, pred=pred).reset_index(drop=True)
    monkeypatch.setattr(DF, "us31_val_frame", fake)


@pytest.fixture
def diagnosed(world, monkeypatch, tmp_path):
    names = ["phone_score", "l2_phone_error", "l2_word_error"]
    _run(world, "search", only_tasks=names)
    _run(world, "final", only_tasks=names)
    _baseline_frames(monkeypatch, names)
    seen = []
    orig = RF.pooled
    monkeypatch.setattr(RF, "pooled", lambda base, task_, head, config, stage, split, *a, **k: (seen.append(split), orig(base, task_, head, config, stage, split, *a, **k))[1])
    tabs = DF.run([ENC], 12, "emb+side", world["base"], None, min_n=5, min_err=1, out=tmp_path / "out")
    return {"tabs": tabs, "seen": seen, "out": tmp_path / "out"}


def test_diagnosis_reads_validation_predictions_only(diagnosed):
    assert diagnosed["seen"] and set(diagnosed["seen"]) == {"val"}


def test_tables_are_written_and_cover_every_task_head_and_system(diagnosed):
    tabs, out = diagnosed["tabs"], diagnosed["out"]
    for name in ("head_gaps_val", "linear_vs_mlp_val", "reg_summary_val", "clf_summary_val", "clf_per_class_val", "phoneme_error_recall_val", "phoneme_support_spearman_val"):
        assert (out / f"{name}.csv").exists(), name
    assert (out / "summary_val.md").read_text().startswith("# US 4.2 diagnosis (validation only)")
    g = tabs["head_gaps_val"]
    assert set(zip(g.task, g["head"])) == {(t, h) for t in ("phone_score", "l2_phone_error", "l2_word_error") for h in ("linear", "mlp")}
    auc = tabs["clf_auc_val"].query("system == @ENC")
    assert set(auc.task) == {"l2_phone_error", "l2_word_error"} and (auc.auc_error > 0.5).all() and {"auc_deletion", "auc_addition"} <= set(auc.columns)
    assert set(tabs["reg_summary_val"].system) == {"US 3.1", ENC} and set(tabs["clf_summary_val"].system) == {"US 3.1", ENC}


def test_frozen_head_beats_the_constant_baseline_where_the_embedding_carries_signal(diagnosed):
    r = diagnosed["tabs"]["reg_summary_val"].query("task == 'phone_score'").set_index("system")
    assert r.loc[ENC, "rmse"] < r.loc["US 3.1", "rmse"] and r.loc[ENC, "pcc"] > 0.3
    assert r.loc[ENC, "slope_pred_on_true"] > r.loc["US 3.1", "slope_pred_on_true"] - 1e-9          # constant baseline has slope 0


def test_per_class_and_confusion_tables_have_all_classes_and_rows_sum_to_one(diagnosed):
    pc = diagnosed["tabs"]["clf_per_class_val"].query("task == 'l2_phone_error' and system == @ENC")
    assert list(pc["class"]) == ["correct", "substitution", "deletion", "addition"]
    cf = diagnosed["tabs"]["clf_confusion_rownorm_val"].query("task == 'l2_phone_error' and system == @ENC")
    assert np.allclose(cf[[c for c in cf.columns if c.startswith("pred_")]].sum(1), 1)
    base = diagnosed["tabs"]["clf_per_class_val"].query("task == 'l2_phone_error' and system == 'US 3.1'").set_index("class")
    assert base.loc["deletion", "recall"] == 0 and base.loc["correct", "recall"] == 1            # the all-"correct" stand-in


def test_gap_table_flags_overfitting_with_the_documented_sign(diagnosed):
    g = diagnosed["tabs"]["head_gaps_val"]
    reg = g.query("task == 'phone_score'")
    assert np.allclose(reg.overfit_gap, reg.val_primary - reg.train_primary)
    clf = g.query("task == 'l2_word_error'")
    assert np.allclose(clf.overfit_gap, clf.train_primary - clf.val_primary, equal_nan=True) and clf.train_primary.notna().all()
    w = diagnosed["tabs"]["linear_vs_mlp_val"].query("task == 'phone_score'").iloc[0]
    assert w.mlp_gain == pytest.approx(w.val_linear - w.val_mlp)


def test_mismatched_baseline_rows_are_an_error(world, monkeypatch, tmp_path):
    _run(world, "search", only_tasks=["phone_score"])
    def bad(t, bcfg=None):
        return "dummy", pd.DataFrame({"y": [1.0, 2.0], "pred": [1.0, 1.0], "speaker": ["a", "b"]})
    monkeypatch.setattr(DF, "us31_val_frame", bad)
    with pytest.raises(ValueError, match="rows"):
        DF.run([ENC], 12, "emb+side", world["base"], ["phone_score"], out=tmp_path / "o")


def test_missing_results_give_a_clear_message(world, monkeypatch, tmp_path):
    _baseline_frames(monkeypatch, [])
    with pytest.raises(SystemExit, match="no head results"):
        DF.run([ENC], 12, "emb+side", world["base"], ["phone_score"], out=tmp_path / "o")


def test_truncated_result_file_gives_an_actionable_message(world, monkeypatch, tmp_path):
    _run(world, "search", only_tasks=["phone_score"])
    base = FH.results_root(ENC, 12, "emb+side", world["base"])
    victim = next((base / "phone_score").rglob("fold0.npz"))
    victim.write_bytes(b"")                                                                      # what an interrupted write / partial copy leaves
    assert DF.unreadable_results(base) == [victim]
    _baseline_frames(monkeypatch, [])
    with pytest.raises(SystemExit, match="unreadable result file.*--force"):
        DF.run([ENC], 12, "emb+side", world["base"], ["phone_score"], out=tmp_path / "o")


def test_cli_out_dir_keeps_ablation_tables_apart(world, monkeypatch, tmp_path):
    _run(world, "search", only_tasks=["phone_score"])
    _baseline_frames(monkeypatch, [])
    DF.main(["--encoders", ENC, "--tasks", "phone_score", "--results-root", str(world["base"]), "--out-dir", str(tmp_path / "side_out"), "--min-phoneme-n", "5"])
    assert (tmp_path / "side_out" / "reg_summary_val.csv").exists() and (tmp_path / "side_out" / "summary_val.md").exists()
