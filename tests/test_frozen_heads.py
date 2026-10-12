"""Sprint 4 (US 4.1) tests for the frozen-encoder heads and the benchmark report.

Runs on a laptop CPU without torch: a synthetic data root with the real file schemas is built, and ``train_head`` is replaced by a
numpy / scikit-learn stand-in with the same interface, so these tests cover everything around the optimiser (row alignment with the
Sprint 3 tables, splits, standardisation, layer selection, side features, seed averaging, resume, validation-only selection, the
US 3.3 decision weights, bootstrap tables). The torch trainer itself has its own tests at the bottom (skipped without torch).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import report_frozen as RF
from src.models import frozen_heads as FH

LAYER_SIGNAL, LAYER_NOISE, DIM, NL = 12, 3, 12, 13
ENC = "wavlm_base_plus"


# ---- synthetic data root ---------------------------------------------------- #
def _emb(rng, z, n):
    """(n, 13, DIM) float16: the label signal sits in layer 12 only, every other layer is noise."""
    E = rng.normal(size=(n, NL, DIM)).astype(np.float32)
    E[:, LAYER_SIGNAL, 0] = z + 0.3 * rng.normal(size=n)
    return E.astype(np.float16)


def _build(tmp_path, rng):
    tdir, cache = tmp_path / "tables", tmp_path / "cache" / ENC
    tdir.mkdir(); cache.mkdir(parents=True)
    spk = [f"{i:04d}" for i in range(24)]
    split = ["train"] * 12 + ["val"] * 6 + ["test"] * 6
    pd.DataFrame({"speaker": spk, "split": split}).to_csv(tmp_path / "so762_speakers.csv", index=False)

    def so_rows(per_spk):
        return pd.DataFrame({"speaker": np.repeat(spk, per_spk)}).assign(
            utt_id=lambda d: [f"{s}{k:03d}" for s, k in zip(d.speaker, np.tile(np.arange(per_spk), len(spk)))],
            gender="m", age_group="adult", split="train")                      # the official flag is deliberately wrong: must be overridden

    sent = so_rows(6)
    z = rng.normal(size=len(sent))
    for m in ("accuracy", "fluency", "prosodic", "total"):
        sent[f"y_{m}"] = 5 + 2 * z + 0.2 * rng.normal(size=len(sent))
    phone = so_rows(10).assign(word_idx=0, phone_idx=lambda d: d.groupby("utt_id").cumcount(), phone="AH")
    zp = rng.normal(size=len(phone))
    phone["y_phone_score"] = np.clip(1 + 0.7 * zp, 0, 2)
    l2 = pd.DataFrame({"speaker": np.repeat([f"L{i}" for i in range(8)], 40)})
    l2["utt_id"] = [f"{s}_{k}" for s, k in zip(l2.speaker, np.tile(np.arange(40), 8))]
    l2["fold"] = l2.speaker.map({f"L{i}": i % 4 for i in range(8)}); l2["l1"] = "Arabic"
    zl = rng.normal(size=len(l2))
    l2_phone = l2.assign(phone_idx=0, anchor_phone="AH", error_class=np.select([zl > 1.0, zl > 0.4, zl > -0.4], ["substitution", "deletion", "correct"], "addition"))
    l2_word = l2.assign(word_idx=0, word="the", y_has_error=(zl > 0).astype(int), n_errors=0)
    tables = {"so762_sentence": (sent, z), "so762_phone": (phone, zp), "l2_phone": (l2_phone, zl), "l2_word": (l2_word, zl)}
    for name, (df, zz) in tables.items():
        df.to_csv(tdir / f"{name}_meta.csv.gz", index=False)
        np.save(cache / f"{name}_X.npy", _emb(rng, zz, len(df)))
        np.save(tdir / f"{name}_side_X.npy", rng.normal(size=(len(df), 3)).astype(np.float32))
    return tdir, cache.parent, tables


def _fake_train_head(Xtr, ytr, Xva, yva, others, *, kind, n_classes, head, hp, seed, tcfg, device, cw=None):
    """Same contract as FH.train_head: ridge (regression) / logistic regression (classification), seed-independent."""
    from sklearn.linear_model import LogisticRegression, Ridge
    if kind == "reg":
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
        pred = m.predict
    else:
        cwd = None if cw is None else {i: float(w) for i, w in enumerate(cw)}
        m = LogisticRegression(max_iter=300, class_weight=cwd).fit(Xtr, ytr)

        def pred(X):
            P = np.zeros((len(X), n_classes))
            P[:, m.classes_] = m.predict_proba(X)
            return P
    return {"val": pred(Xva), **{k: pred(v) for k, v in others.items()}, "best_epoch": 3, "epochs_run": 8, "val_loss": 0.0}


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    tdir, cache, tables = _build(tmp_path, rng)
    cfg = {"heads": {"device": "cpu", "primary_layer": 12, "heads": ["linear", "mlp"], "seeds": [0, 1, 2], "mlp": {"hidden": 8},
                     "train": {"lr": 0.01, "batch_size": 64, "max_epochs": 3, "patience": 2, "train_eval_rows": 50},
                     "tasks": ["sent_accuracy", "phone_score", "phone_mispron_t1.0", "l2_phone_error", "l2_word_error"],
                     "grids": {"reg": {"linear": {"wd": [1e-4, 1e-2]}, "mlp": {"dropout": [0.1], "wd": [1e-2]}},
                               "clf": {"linear": {"wd": [1e-3], "cw": ["none", "balanced"]}, "mlp": {"dropout": [0.1], "wd": [1e-2], "cw": ["none", "balanced"]}}}}}
    monkeypatch.setattr(FH, "tables_dir", lambda: tdir)
    monkeypatch.setattr(FH, "SPLIT_CSV", tmp_path / "so762_speakers.csv")
    monkeypatch.setattr(FH, "load_config", lambda: cfg)
    monkeypatch.setattr(FH, "pick_device", lambda name="auto": "cpu")
    monkeypatch.setattr(FH, "train_head", _fake_train_head)
    return {"tdir": tdir, "cache": cache, "tables": tables, "base": tmp_path / "res", "cfg": cfg}


def task(name):
    return FH.build_tasks(FH.load_baseline_config())[name]


# ---- data / alignment ------------------------------------------------------- #
def test_labels_use_sprint2_speaker_split_not_the_official_flag(world):
    y, meta = FH.load_labels(task("phone_score"))
    assert set(meta["split"]) == {"train", "val", "test"} and len(y) == len(meta)
    assert meta.loc[meta.speaker == "0003", "split"].eq("train").all() and meta.loc[meta.speaker == "0020", "split"].eq("test").all()


def test_binarised_target_and_row_order_are_the_meta_order(world):
    y, meta = FH.load_labels(task("phone_mispron_t1.0"))
    raw = pd.read_csv(world["tdir"] / "so762_phone_meta.csv.gz")["y_phone_score"].to_numpy()
    assert (y == (raw < 1.0).astype(int)).all()


def test_layer_selection_side_features_and_modes(world):
    t = task("sent_accuracy")
    n = len(FH.load_labels(t)[1])
    E = np.load(world["cache"] / ENC / "so762_sentence_X.npy", mmap_mode="r")
    S = np.load(world["tdir"] / "so762_sentence_side_X.npy")
    X = FH.load_features(t, ENC, 12, "emb+side", n, world["cache"])
    assert X.shape == (n, DIM + 3) and X.dtype == np.float32
    assert np.array_equal(X[:, :DIM], E[:, 12].astype(np.float32)) and np.array_equal(X[:, DIM:], S)
    assert FH.load_features(t, ENC, 12, "emb", n, world["cache"]).shape == (n, DIM)
    assert np.array_equal(FH.load_features(t, ENC, 12, "side", n, world["cache"]), S)
    avg = FH.load_features(t, ENC, "10-12", "emb", n, world["cache"])
    assert np.allclose(avg, E[:, 10:13].astype(np.float32).mean(1), atol=1e-3)


def test_row_count_mismatch_with_sprint3_table_is_an_error(world):
    with pytest.raises(AssertionError, match="Sprint 3 table"):
        FH.load_features(task("sent_accuracy"), ENC, 12, "emb", 7, world["cache"])
    with pytest.raises(AssertionError):
        FH.load_features(task("sent_accuracy"), ENC, 13, "emb", len(FH.load_labels(task("sent_accuracy"))[1]), world["cache"])


def test_standardise_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(1)
    tr, other = rng.normal(5, 2, (500, 4)).astype(np.float32), rng.normal(50, 1, (50, 4)).astype(np.float32)
    tr[:, 3] = 7.0                                                            # constant column must not divide by zero
    a, b = FH.standardise(tr, other)
    assert np.allclose(a.mean(0), 0, atol=1e-4) and np.allclose(a[:, :3].std(0), 1, atol=1e-4) and np.isfinite(a).all()
    assert b[:, :3].mean() > 10                                              # the other split is shifted by TRAIN statistics, not its own


def test_head_grid_ids_are_unique_and_classification_has_class_weight_groups(world):
    h = world["cfg"]["heads"]
    ids = [c for c, _ in FH.head_grid(h, task("l2_phone_error"), "mlp")]
    assert len(set(ids)) == len(ids) == 2 and {hp["cw"] for _, hp in FH.head_grid(h, task("l2_phone_error"), "mlp")} == {"none", "balanced"}
    assert all("cw" not in hp for _, hp in FH.head_grid(h, task("phone_score"), "linear"))


# ---- run / resume / final stage -------------------------------------------- #
def _run(world, stage, **kw):
    FH.run(ENC, 12, "emb+side", stage, cache=world["cache"], base=world["base"], **kw)


def test_search_stage_writes_every_job_and_resumes_without_refitting(world, capsys):
    _run(world, "search", only_tasks=["phone_score", "l2_word_error"])
    base = FH.results_root(ENC, 12, "emb+side", world["base"])
    assert len(list((base / "phone_score").rglob("fold0.npz"))) == 3                       # linear x2 + mlp x1
    assert len(list((base / "l2_word_error").rglob("fold*.npz"))) == (2 + 2) * 4          # (linear cw x2 + mlp cw x2) x 4 folds
    stamp = {p: p.stat().st_mtime_ns for p in base.rglob("*.npz")}
    capsys.readouterr()
    _run(world, "search", only_tasks=["phone_score", "l2_word_error"])
    assert "0 fitted" in capsys.readouterr().out and stamp == {p: p.stat().st_mtime_ns for p in base.rglob("*.npz")}


def test_prediction_files_are_aligned_to_the_split_masks_and_seeds(world):
    _run(world, "search", only_tasks=["l2_phone_error", "phone_score"])
    base = FH.results_root(ENC, 12, "emb+side", world["base"])
    for tname, cfgname, head in (("phone_score", "wd0.0001", "linear"), ("l2_phone_error", "cwbalanced_wd0.001", "linear")):
        t = task(tname)
        y, meta = FH.load_labels(t)
        for fold in t.folds:
            m = FH.split_masks(t, meta, fold)
            d = np.load(FH.job_path(base, tname, head, cfgname, fold), allow_pickle=True)
            assert len(d["val_pred"]) == m["val"].sum() and len(d["test_pred"]) == m["test"].sum()
            assert list(d["seeds"]) == [0]
            if t.kind == "clf":                                                      # probabilities in task.labels order, preds decoded to label values
                assert list(d["classes"]) == [str(c) for c in t.labels] and d["test_proba"].shape == (m["test"].sum(), len(t.labels))
                assert set(d["test_pred"]) <= set(t.labels) and np.allclose(d["test_proba"].sum(1), 1, atol=1e-4)


def test_final_stage_refits_only_selected_configs_with_all_seeds(world):
    _run(world, "search", only_tasks=["l2_word_error"])
    _run(world, "final", only_tasks=["l2_word_error"])
    base = FH.results_root(ENC, 12, "emb+side", world["base"])
    finals = sorted(p.parent.name for p in (base / "l2_word_error").rglob("fold0.npz") if p.parent.name.endswith("__final"))
    assert len(finals) == 4                                                          # 2 heads x {none, balanced}, one selected config each
    any_final = next(p for p in (base / "l2_word_error").rglob("fold0.npz") if p.parent.name.endswith("__final"))
    assert list(np.load(any_final, allow_pickle=True)["seeds"]) == [0, 1, 2]


def test_selection_reads_only_validation_predictions(world):
    t = task("phone_score")
    y, meta = FH.load_labels(t)
    base = world["base"] / "x"
    m = FH.split_masks(t, meta, 0)
    for cfgname, noise in (("wd0.0001", 0.9), ("wd0.01", 0.1)):
        p = FH.job_path(base, t.name, "linear", cfgname, 0)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, val_pred=y[m["val"]] + noise)                                    # NO test keys at all: selection must not need them
    sel = FH.select_config(base, t, world["cfg"]["heads"], "linear", "none", y, meta)
    assert sel[1] == "wd0.01" and sel[0] == pytest.approx(0.1)



def test_trainer_sees_only_training_rows_standardised_on_train_and_the_right_class_weights(world, monkeypatch):
    calls = []

    def spy(Xtr, ytr, Xva, yva, others, **kw):
        calls.append({"ntr": len(Xtr), "nva": len(Xva), "nte": len(others["test"]), "mu": Xtr.mean(), "mu_va": Xva.mean(), "cw": kw["cw"], "ytr": ytr.copy()})
        return _fake_train_head(Xtr, ytr, Xva, yva, others, **kw)

    monkeypatch.setattr(FH, "train_head", spy)
    _run(world, "search", only_tasks=["l2_word_error"], only_heads=["linear"])
    t = task("l2_word_error")
    y, meta = FH.load_labels(t)
    assert len(calls) == 2 * 4
    for c, (config, hp), fold in zip(calls, [(c, h) for c, h in FH.head_grid(world["cfg"]["heads"], t, "linear") for _ in t.folds], t.folds * 2):
        m = FH.split_masks(t, meta, fold)
        assert (c["ntr"], c["nva"], c["nte"]) == (m["train"].sum(), m["val"].sum(), m["test"].sum())
        assert abs(c["mu"]) < 1e-3                                                  # standardised on the training rows
        assert (c["cw"] is None) == (hp["cw"] == "none")
        if c["cw"] is not None:
            assert np.allclose(c["cw"], FH.balanced_weights(c["ytr"], 2)) and c["cw"].argmin() == np.bincount(c["ytr"]).argmax()      # majority class gets the smallest weight


def test_decision_weights_change_predictions_only_in_the_cw_group(world):
    rng = np.random.default_rng(0)
    t = task("l2_word_error")
    n = 400
    y = (rng.random(n) < 0.3).astype(int)
    P1 = np.clip(0.12 + 0.2 * y + 0.05 * rng.normal(size=n), 0.01, 0.99)             # under-confident positive class: argmax predicts almost only 0
    P = np.c_[1 - P1, P1]
    meta = pd.DataFrame({"speaker": rng.choice(list("abcd"), n), "fold": rng.integers(0, 4, n)})
    d = {"y": y, "pred": P.argmax(1), "proba": P, "meta": meta}
    plain = RF.apply_variants(t, "none", d, d)
    cw = RF.apply_variants(t, "balanced", d, d)
    assert [v[0] for v in plain] == ["plain"] and [v[0] for v in cw] == ["cw", "cw+thr"]
    assert (cw[0][2] == d["pred"]).all() and d["pred"].sum() <= 2
    thr = cw[1][2]
    assert thr.sum() > 0.15 * n
    assert RF.primary(t, y, thr) > RF.primary(t, y, d["pred"])                       # tuned weights recover the minority class (macro-F1 up)


# ---- report ----------------------------------------------------------------- #
def _fit_all(world, layer=12):
    FH.run(ENC, layer, "emb+side", "search", cache=world["cache"], base=world["base"])
    FH.run(ENC, layer, "emb+side", "final", cache=world["cache"], base=world["base"])


def test_report_headline_variants_ci_and_learning(world):
    _fit_all(world)
    df, preds = RF.evaluate(ENC, 12, "emb+side", n_boot=20, root=world["base"])
    t = df[df.split == "test"]
    assert set(t.task) == {"sent_accuracy", "phone_score", "phone_mispron_t1.0", "l2_phone_error", "l2_word_error"}
    assert set(t[t.task == "l2_word_error"].variant) == {"plain", "cw", "cw+thr"} and set(t[t.task == "sent_accuracy"].variant) == {"plain"}
    assert (t.groupby(["task", "head"]).headline.sum() == 1).all()                   # exactly one headline variant per (task, head)
    assert t.rmse_lo.dropna().le(t.rmse.dropna()).all() and (t.dropna(subset=["rmse"]).rmse <= t.dropna(subset=["rmse"]).rmse_hi).all()
    y, p = preds[("sent_accuracy", "linear")]["y"], preds[("sent_accuracy", "linear")]["pred"]
    assert np.sqrt(np.mean((p - y) ** 2)) < 0.5 * y.std()                           # the layer-12 signal is found, so rows and embeddings are aligned
    f1 = t[(t.task == "l2_word_error") & t.headline].macro_f1
    assert (f1 > 0.7).all()
    assert (t[t.kind == "reg"].train_primary > 0).all()


def test_wrong_layer_does_not_learn_which_proves_the_layer_index_is_used(world):
    FH.run(ENC, LAYER_NOISE, "emb", "search", only_tasks=["sent_accuracy"], only_heads=["linear"], cache=world["cache"], base=world["base"])
    FH.run(ENC, 12, "emb", "search", only_tasks=["sent_accuracy"], only_heads=["linear"], cache=world["cache"], base=world["base"])
    t = task("sent_accuracy")
    y, meta = FH.load_labels(t)
    score = {}
    for layer in (LAYER_NOISE, 12):
        base = FH.results_root(ENC, layer, "emb", world["base"])
        score[layer] = FH.select_config(base, t, world["cfg"]["heads"], "linear", "none", y, meta)[0]
    assert score[12] < 0.5 * score[LAYER_NOISE]


# ---- side-feature exporter -------------------------------------------------- #
def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_side", Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "export_side_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exporter_picks_non_acoustic_columns_and_blocks_l2_duration(tmp_path):
    ex = _load_exporter()
    names = ["mean_c0", "own_mean_c0", "dur", "log_dur", "has_prev", "has_next", "word_initial", "word_final", "n_phones_in_word", "is_vowel", "prev_c0", "ph_AH", "ph_B"]
    idx = ex.pick(names, ex.SIDE_COLUMNS["so762_phone"])
    assert [names[i] for i in idx] == ["dur", "log_dur", "has_prev", "has_next", "word_initial", "word_final", "n_phones_in_word", "is_vowel", "ph_AH", "ph_B"]
    n2 = ["win_mean_c0", "has_prev", "has_next", "is_vowel", "ph_AH", "ph_B"]
    X = np.random.default_rng(0).normal(size=(5, len(n2))).astype(np.float32)
    np.save(tmp_path / "l2_phone_X.npy", X); (tmp_path / "l2_phone_names.json").write_text(json.dumps(n2))
    assert ex.export("l2_phone", tmp_path) == (5, 5)
    assert json.loads((tmp_path / "l2_phone_side_names.json").read_text()) == ["has_prev", "has_next", "is_vowel", "ph_AH", "ph_B"]
    bad = n2[:1] + ["dur"] + n2[1:]
    np.save(tmp_path / "l2_phone_X.npy", np.zeros((5, len(bad)), np.float32)); (tmp_path / "l2_phone_names.json").write_text(json.dumps(bad))
    ex.SIDE_COLUMNS = {**ex.SIDE_COLUMNS, "l2_phone": ["has_prev", "dur", "ph_"]}
    with pytest.raises(AssertionError, match="annotated duration"):
        ex.export("l2_phone", tmp_path)


# ---- the real torch trainer (runs on Colab / any machine with torch) ---------- #
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed; these run on Colab")
TCFG = {"lr": 0.01, "batch_size": 64, "max_epochs": 60, "patience": 8}


def _toy(kind, n=600, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    z = X[:, 0] - 0.5 * X[:, 1]
    y = z + 0.1 * rng.normal(size=n) if kind == "reg" else (z > 0.8).astype(int)       # ~20% positives
    return X, y


@needs_torch
@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_torch_regression_head_learns(head):
    X, y = _toy("reg")
    r = FH.train_head(X[:400], y[:400], X[400:500], y[400:500], {"test": X[500:]}, kind="reg", n_classes=1, head=head,
                      hp={"wd": 1e-4, "dropout": 0.1, "hidden": 32}, seed=0, tcfg=TCFG, device="cpu")
    assert np.corrcoef(r["test"], y[500:])[0, 1] > 0.95 and r["test"].shape == (100,)


@needs_torch
@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_torch_classification_head_learns_and_class_weights_raise_recall(head):
    X, y = _toy("clf")
    kw = dict(kind="clf", n_classes=2, head=head, hp={"wd": 1e-4, "dropout": 0.1, "hidden": 32}, seed=0, tcfg=TCFG, device="cpu")
    plain = FH.train_head(X[:400], y[:400], X[400:500], y[400:500], {"test": X[500:]}, **kw)
    cw = FH.train_head(X[:400], y[:400], X[400:500], y[400:500], {"test": X[500:]}, cw=FH.balanced_weights(y[:400], 2), **kw)
    assert plain["test"].shape == (100, 2) and np.allclose(plain["test"].sum(1), 1, atol=1e-5)
    assert (plain["test"].argmax(1) == y[500:]).mean() > 0.9
    assert cw["test"][:, 1].mean() > plain["test"][:, 1].mean()                    # weighted loss shifts mass to the minority class


@needs_torch
def test_torch_training_is_deterministic_for_a_seed_and_differs_across_seeds():
    X, y = _toy("reg")
    kw = dict(kind="reg", n_classes=1, head="mlp", hp={"wd": 1e-4, "dropout": 0.3, "hidden": 16}, tcfg={**TCFG, "max_epochs": 5}, device="cpu")
    a = FH.train_head(X[:400], y[:400], X[400:500], y[400:500], {}, seed=3, **kw)["val"]
    b = FH.train_head(X[:400], y[:400], X[400:500], y[400:500], {}, seed=3, **kw)["val"]
    c = FH.train_head(X[:400], y[:400], X[400:500], y[400:500], {}, seed=4, **kw)["val"]
    assert np.allclose(a, b, atol=1e-6) and not np.allclose(a, c, atol=1e-6)


@needs_torch
def test_torch_early_stopping_restores_best_epoch():
    X, y = _toy("reg", n=120)
    rng = np.random.default_rng(0)
    y_noisy = y + rng.normal(0, 3.0, len(y))                                       # pure over-fitting regime: val loss must stop improving early
    r = FH.train_head(X[:60], y_noisy[:60], X[60:], y_noisy[60:], {}, kind="reg", n_classes=1, head="mlp",
                      hp={"wd": 0.0, "dropout": 0.0, "hidden": 256}, seed=0, tcfg={**TCFG, "max_epochs": 200, "patience": 5, "lr": 0.01}, device="cpu")
    assert r["epochs_run"] < 200 and r["best_epoch"] <= r["epochs_run"] - 5
