"""Sprint 4 (US 4.3) tests: layer-weighted sum, sentence regularisation (PCA / pros), word pooling, layer shortlist, out-of-fold probabilities.

Runs on a laptop CPU with torch (the heads are trained for real on a tiny synthetic data root with the real file schemas); skipped without
torch. The label signal sits in ONE layer (and, for words, in the max-pooled block only), so each fix has something it must find.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation import report_frozen as RF
from src.models import frozen_fix as FF
from src.models import frozen_heads as FH

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed; these run on Colab")

REPO = Path(__file__).resolve().parents[1]
ENC, DIM, NL, SIG = "wavlm_base_plus", 12, 13, 7           # the signal sits in layer 7 only
N_SPK = 24


def _spk_table(per_spk, prefix="s"):
    spk = [f"{i:04d}" for i in range(N_SPK)]
    d = pd.DataFrame({"speaker": np.repeat(spk, per_spk)})
    d["utt_id"] = [f"{s}_{prefix}{k:03d}" for s, k in zip(d.speaker, np.tile(np.arange(per_spk), N_SPK))]
    d["gender"] = "m"
    d["age_group"] = d.speaker.map({s: ("adult" if i % 2 == 0 else "child/teen") for i, s in enumerate(spk)})
    d["split"] = "train"                                  # the official flag is deliberately wrong: the Sprint 2 speaker split must win
    return d


def _emb(rng, z, width, signal_cols=(0,), layers=(SIG,)):
    E = rng.normal(size=(len(z), NL, width)).astype(np.float32)
    for l in layers:
        for c in signal_cols:
            E[:, l, c] = z + 0.3 * rng.normal(size=len(z))
    return E.astype(np.float16)


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    tdir, cache = tmp_path / "tables", tmp_path / "cache" / ENC
    tdir.mkdir(); cache.mkdir(parents=True)
    spk = [f"{i:04d}" for i in range(N_SPK)]
    pd.DataFrame({"speaker": spk, "split": ["train"] * 12 + ["val"] * 6 + ["test"] * 6}).to_csv(tmp_path / "so762_speakers.csv", index=False)

    sent = _spk_table(40)
    z = rng.normal(size=len(sent))
    for m in ("accuracy", "fluency", "prosodic", "total"):
        sent[f"y_{m}"] = 5 + 2 * z + 0.2 * rng.normal(size=len(sent))
    word = _spk_table(60)
    word = word.assign(word_idx=0)
    zw = rng.normal(size=len(word))
    word["y_word_accuracy"] = 6 + 2 * zw + 0.2 * rng.normal(size=len(word))
    phone = _spk_table(60).assign(word_idx=0, phone_idx=0, phone="AH")
    zp = rng.normal(size=len(phone))
    phone["y_phone_score"] = np.clip(1 + 0.7 * zp, 0, 2)
    l2 = pd.DataFrame({"speaker": np.repeat([f"L{i}" for i in range(8)], 60)})
    l2["utt_id"] = [f"{s}_{k}" for s, k in zip(l2.speaker, np.tile(np.arange(60), 8))]
    l2["fold"] = l2.speaker.map({f"L{i}": i % 4 for i in range(8)})
    zl = rng.normal(size=len(l2))
    l2_word = l2.assign(word_idx=0, word="the", y_has_error=(zl > 0).astype(int), n_errors=0, l1="Arabic")

    l2_phone = l2.assign(phone_idx=0, anchor_phone="AH", error_class=np.select([zl > 1.0, zl > 0.4, zl > -0.4], ["substitution", "deletion", "correct"], "addition"))
    tables = {"so762_sentence": (sent, z, 2 * DIM), "so762_word": (word, zw, DIM), "so762_phone": (phone, zp, DIM), "l2_word": (l2_word, zl, DIM),
              "l2_phone": (l2_phone, zl, DIM)}
    for name, (df, zz, width) in tables.items():
        df.to_csv(tdir / f"{name}_meta.csv.gz", index=False)
        if name == "so762_word":                                 # mean block = pure noise, the signal is in the max block of layer 7
            np.save(cache / f"{name}_X.npy", _emb(rng, zz, width, signal_cols=(), layers=()))
            np.save(cache / f"{name}_ms_X.npy", _emb(rng, zz, 2 * width, signal_cols=(0,), layers=(SIG,)))
        else:
            np.save(cache / f"{name}_X.npy", _emb(rng, zz, width))
        np.save(tdir / f"{name}_side_X.npy", rng.normal(size=(len(df), 3)).astype(np.float32))
        np.save(tdir / f"{name}_pros_X.npy", rng.normal(size=(len(df), 4)).astype(np.float32))

    cfg = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())
    cfg["heads"].update({"device": "cpu", "seeds": [0, 1], "mlp": {"hidden": 8},
                         "train": {"lr": 0.02, "batch_size": 64, "max_epochs": 40, "patience": 8, "train_eval_rows": 200}})
    cfg["fix43"]["seeds_final"] = [0, 1]
    for r in cfg["fix43"]["recipes"].values():
        r["seeds"] = [0, 1]
    cfg["fix43"]["recipes"]["ws"]["grids"] = {"reg": {"linear": {"wd": [1e-4]}, "mlp": {"dropout": [0.1], "wd": [1e-2]}},
                                              "clf": {"linear": {"wd": [1e-3], "cw": ["none", "balanced"]}, "mlp": {"dropout": [0.1], "wd": [1e-2], "cw": ["balanced"]}}}
    cfg["heads"]["probe"]["tasks"] = ["sent_accuracy"]
    monkeypatch.setattr(FH, "tables_dir", lambda: tdir)
    monkeypatch.setattr(FF, "tables_dir", lambda: tdir)
    monkeypatch.setattr(FH, "SPLIT_CSV", tmp_path / "so762_speakers.csv")
    monkeypatch.setattr(FH, "load_config", lambda: cfg)
    monkeypatch.setattr(FH, "pick_device", lambda name="auto": "cpu")
    return {"tdir": tdir, "cache": cache.parent, "root": tmp_path / "res43", "frozen": tmp_path / "res41", "cfg": cfg, "tmp": tmp_path}


def task(name):
    return FH.build_tasks(FH.load_baseline_config())[name]


def _best_val(base, t, hcfg, head, y, meta, group="none"):
    return FH.select_config(base, t, hcfg, head, group, y, meta)


# ---- configuration ----------------------------------------------------------- #
def test_layer_spec_parsing():
    assert FF.parse_layer_spec("ws", (0, 12)) == (list(range(13)), "ws")
    assert FF.parse_layer_spec("avg6-12") == ([6, 7, 8, 9, 10, 11, 12], "avg")
    assert FF.parse_layer_spec(9) == ([9], "single") and FF.parse_layer_spec("9") == ([9], "single")


def test_yaml_recipes_are_complete_and_grids_expand(world):
    cfg = world["cfg"]
    for name in cfg["fix43"]["recipes"]:
        h, spec = FF.recipe_cfg(cfg, name)
        assert spec["inputs"] and spec["layer"] is not None
        for t in FH.main_tasks(None, h):
            for head in h["heads"]:
                assert FH.head_grid(h, t, head), (name, t.name, head)
    h, _ = FF.recipe_cfg(cfg, "sent_avgpca_pros")
    ids = [c for c, _ in FH.head_grid(h, task("sent_accuracy"), "linear")]
    assert any("pca32" in c for c in ids) and len(ids) == len(set(ids))


def test_unknown_recipe_and_pool_are_errors(world):
    with pytest.raises(KeyError, match="unknown recipe"):
        FF.recipe_cfg(world["cfg"], "nope")
    with pytest.raises(ValueError, match="only defined for the so762_word"):
        FF._pool_parts("max", "so762_sentence")
    with pytest.raises(ValueError, match="unknown pool"):
        FF._pool_parts("median", "so762_word")


# ---- data ------------------------------------------------------------------- #
@pytest.mark.parametrize("pool,width", [("mean", DIM), ("max", DIM), ("std", DIM), ("mean+max", 2 * DIM), ("mean+max+std", 3 * DIM)])
def test_word_pool_widths_and_blocks(world, pool, width):
    t = task("word_accuracy")
    n = len(FH.load_labels(t)[1])
    layers = [6, 7, 8]
    E = FF.load_embedding(t, ENC, layers, "ws", pool, n, "cpu", world["cache"]).numpy()
    assert E.shape == (n, 3, width) and E.dtype == np.float16
    M = np.load(world["cache"] / ENC / "so762_word_ms_X.npy", mmap_mode="r")
    Mean = np.load(world["cache"] / ENC / "so762_word_X.npy", mmap_mode="r")
    blocks = {"mean": Mean[:, layers, :], "max": M[:, layers, :DIM], "std": M[:, layers, DIM:]}
    ref = np.concatenate([blocks[p] for p in pool.split("+")], -1)
    assert np.array_equal(E, np.asarray(ref))


def test_avg_layer_input_is_the_mean_of_the_window_and_row_mismatch_is_an_error(world):
    t = task("sent_accuracy")
    n = len(FH.load_labels(t)[1])
    layers = list(range(6, 13))
    E = FF.load_embedding(t, ENC, layers, "avg", "base", n, "cpu", world["cache"]).float().numpy()
    raw = np.load(world["cache"] / ENC / "so762_sentence_X.npy")[:, 6:13].astype(np.float32).mean(1, keepdims=True)
    assert E.shape == (n, 1, 2 * DIM) and np.allclose(E, raw, atol=2e-3)
    with pytest.raises(AssertionError, match="Sprint 3 table"):
        FF.load_embedding(t, ENC, layers, "avg", "base", n + 1, "cpu", world["cache"])


def test_standardisation_uses_train_rows_only(world):
    t = task("sent_accuracy")
    spec = {"layer": "ws", "pool": "base", "inputs": ["emb", "side", "pros"]}
    data = FF.build_data(t, ENC, spec, world["cfg"], "cpu", world["cache"])
    m = FH.split_masks(t, data["meta"], 0)
    tr = np.flatnonzero(m["train"])
    st = FF._train_stats(data, tr, 0)
    E = data["E"].float().numpy()
    assert np.allclose(st["mu"].numpy(), E[tr].mean(0), atol=2e-3) and np.allclose(st["sd"].numpy(), E[tr].std(0), atol=2e-3)
    assert not np.allclose(st["mu"].numpy(), E.mean(0), atol=1e-6)
    X = st["X"].numpy()
    assert X.shape == (len(E), 3 + 4) and np.allclose(X[tr].mean(0), 0, atol=1e-4) and np.allclose(X[tr].std(0), 1, atol=1e-3)


def test_pca_is_fitted_on_train_rows_orthonormal_and_nested(world):
    t = task("sent_accuracy")
    data = FF.build_data(t, ENC, {"layer": "avg6-12", "pool": "base", "inputs": ["emb"]}, world["cfg"], "cpu", world["cache"])
    tr = np.flatnonzero(FH.split_masks(t, data["meta"], 0)["train"])
    st = FF._train_stats(data, tr, 0)
    V8, s8 = FF._pca_components(data, st, tr, 8, 0)
    V4, s4 = FF._pca_components(data, st, tr, 4, 0)
    assert V8.shape == (8, 2 * DIM) and np.allclose(V8 @ V8.T, np.eye(8), atol=1e-4)
    assert V4.shape == (4, 2 * DIM) and s4 > 0
    ws_data = FF.build_data(t, ENC, {"layer": "ws", "pool": "base", "inputs": ["emb"]}, world["cfg"], "cpu", world["cache"])
    st2 = FF._train_stats(ws_data, tr, 0)
    with pytest.raises(AssertionError, match="PCA needs a fixed layer"):
        FF._pca_components(ws_data, st2, tr, 4, 0)


# ---- fix 2: learned layer-weighted sum --------------------------------------- #
def test_layer_weighted_sum_finds_the_informative_layer_and_files_match_the_report_contract(world):
    FF.run_recipe(ENC, "ws", "search", only_tasks=["sent_accuracy"], only_heads=["linear"], cache=world["cache"], root=world["root"])
    hcfg, _ = FF.recipe_cfg(world["cfg"], "ws")
    t = task("sent_accuracy")
    y, meta = FH.load_labels(t)
    base = FF.recipe_base(ENC, "ws", world["root"])
    sel = _best_val(base, t, hcfg, "linear", y, meta)
    assert sel is not None
    d = np.load(FH.job_path(base, t.name, "linear", sel[1], 0), allow_pickle=True)
    w = d["ws_weights"]
    assert w.shape == (NL,) and np.isclose(w.sum(), 1.0, atol=1e-5)
    assert int(w.argmax()) == SIG and w[SIG] > 3.0 / NL, w.round(3)            # uniform start is 1/13
    pooled = RF.pooled(base, t, "linear", sel[1], "search", "val", y, meta)    # the existing report helper reads the new files unchanged
    assert pooled["pred"].shape == pooled["y"].shape and np.isfinite(pooled["pred"]).all()
    rmse = float(np.sqrt(np.mean((pooled["pred"] - pooled["y"]) ** 2)))
    assert rmse < 0.6 * float(np.std(pooled["y"]))                              # it learned the signal


def test_single_layer_input_cannot_see_signal_in_another_layer(world):
    spec = {"layer": 3, "pool": "base", "inputs": ["emb", "side"]}
    world["cfg"]["fix43"]["recipes"]["probe3"] = {**spec, "tasks": ["sent_accuracy"], "grids": {"reg": {"linear": {"wd": [1e-4]}, "mlp": {"dropout": [0.1], "wd": [1e-2]}}}}
    FF.run_recipe(ENC, "probe3", "search", only_tasks=["sent_accuracy"], only_heads=["linear"], cache=world["cache"], root=world["root"])
    hcfg, _ = FF.recipe_cfg(world["cfg"], "probe3")
    t = task("sent_accuracy")
    y, meta = FH.load_labels(t)
    base = FF.recipe_base(ENC, "probe3", world["root"])
    sel = _best_val(base, t, hcfg, "linear", y, meta)
    d = np.load(FH.job_path(base, t.name, "linear", sel[1], 0), allow_pickle=True)
    assert "ws_weights" not in d.files
    pooled = RF.pooled(base, t, "linear", sel[1], "search", "val", y, meta)
    assert float(np.sqrt(np.mean((pooled["pred"] - pooled["y"]) ** 2))) > 0.8 * float(np.std(pooled["y"]))


def test_search_resume_final_stage_and_seed_averaging(world):
    kw = dict(only_tasks=["phone_score"], cache=world["cache"], root=world["root"])
    FF.run_recipe(ENC, "ws", "search", **kw)
    base = FF.recipe_base(ENC, "ws", world["root"])
    files = sorted(base.rglob("fold0.npz"))
    assert len(files) == 2                                                    # linear x1 + mlp x1 configs
    mt = {p: p.stat().st_mtime_ns for p in files}
    FF.run_recipe(ENC, "ws", "search", **kw)                                  # resume: nothing is re-fitted
    assert {p: p.stat().st_mtime_ns for p in files} == mt
    FF.run_recipe(ENC, "ws", "final", **kw)
    fin = sorted(base.rglob("*__final/fold0.npz"))
    assert len(fin) == 2
    d = np.load(fin[0], allow_pickle=True)
    assert list(d["seeds"]) == [0, 1] and len(d["val_score_seeds"]) == 2 and len(d["best_epochs"]) == 2
    assert d["val_pred"].ndim == 1 and np.isfinite(d["train_score"])


def test_classification_files_store_probabilities_and_work_with_the_report_variants(world):
    kw = dict(only_tasks=["phone_mispron_t1.0"], cache=world["cache"], root=world["root"])
    FF.run_recipe(ENC, "ws", "search", **kw)
    FF.run_recipe(ENC, "ws", "final", **kw)
    hcfg, _ = FF.recipe_cfg(world["cfg"], "ws")
    t = task("phone_mispron_t1.0")
    y, meta = FH.load_labels(t)
    base = FF.recipe_base(ENC, "ws", world["root"])
    sel = FH.select_config(base, t, hcfg, "linear", "balanced", y, meta)
    cfg_id = sel[1]
    dv = RF.pooled(base, t, "linear", cfg_id, "final", "val", y, meta, proba=True)
    dt = RF.pooled(base, t, "linear", cfg_id, "final", "test", y, meta, proba=True)
    assert dv["proba"].shape == (len(dv["y"]), 2) and np.allclose(dv["proba"].sum(1), 1, atol=1e-4)
    assert "train_proba" in dt
    variants = RF.apply_variants(t, "balanced", dv, dt)
    assert [v[0] for v in variants] == ["cw", "cw+thr"]


def test_l2_four_fold_task_runs_all_folds(world):
    FF.run_recipe(ENC, "ws", "search", only_tasks=["l2_word_error"], only_heads=["linear"], cache=world["cache"], root=world["root"])
    base = FF.recipe_base(ENC, "ws", world["root"])
    folds = sorted(p.name for p in base.rglob("fold*.npz"))
    assert sorted(set(folds)) == ["fold0.npz", "fold1.npz", "fold2.npz", "fold3.npz"]


# ---- fix 3: sentence regularisation ------------------------------------------ #
@pytest.mark.parametrize("recipe,extra_cols,pca", [("sent_ws", 3, False), ("sent_ws_pros", 7, False), ("sent_avgpca", 3, True), ("sent_avgpca_pros", 7, True)])
def test_sentence_recipes_run_with_regularisation_pca_and_pros(world, recipe, extra_cols, pca):
    hcfg, spec = FF.recipe_cfg(world["cfg"], recipe)
    hcfg["grids"] = {"reg": {k: {kk: vv[:1] if kk not in ("pca",) else vv[:2] for kk, vv in g.items()} for k, g in hcfg["grids"]["reg"].items()}, "clf": hcfg["grids"]["clf"]}
    world["cfg"]["fix43"]["recipes"][recipe]["grids"] = hcfg["grids"]               # keep the test fast: first value of each hyper-parameter
    FF.run_recipe(ENC, recipe, "search", only_tasks=["sent_total"], only_heads=["linear", "mlp"], cache=world["cache"], root=world["root"])
    base = FF.recipe_base(ENC, recipe, world["root"])
    files = sorted(base.rglob("fold0.npz"))
    assert files
    for f in files:
        d = np.load(f, allow_pickle=True)
        k = min(int(next((p[3:] for p in f.parent.name.split("_") if p.startswith("pca")), 0)), 2 * DIM) if pca else 0     # PCA is capped by the width
        assert int(d["n_features"]) == (k if pca else NL * 2 * DIM) + extra_cols, (f.parent.name, int(d["n_features"]))
        assert ("ws_weights" in d.files) == (not pca)


def test_regularisation_hyper_parameters_reach_the_model(world):
    import torch
    Head = FF._model_cls()
    m = Head(13, 24, 3, 1, "mlp", {"dropout": 0.4, "in_drop": 0.25, "hidden": 16, "wd": 0.1}, True)
    assert m.in_drop.p == 0.25 and m.head[2].p == 0.4 and m.head[0].in_features == 27 and m.head[0].out_features == 16
    assert m.alpha.requires_grad and torch.allclose(m.layer_weights(), torch.full((13,), 1 / 13))
    lin = Head(13, 24, 3, 1, "linear", {"wd": 0.1}, False)
    assert not lin.alpha.requires_grad and isinstance(lin.head, torch.nn.Linear)
    m.eval()
    E, X = torch.randn(5, 13, 24), torch.randn(5, 3)
    assert torch.allclose(m(E, X), m(E, X))                          # eval mode: dropout off, deterministic
    m.train()
    assert not torch.allclose(m(E, X), m(E, X))                      # train mode: input dropout + dropout active


def test_strong_weight_decay_shrinks_the_prediction_spread(world):
    """The decay strength in the grid must matter: a heavily decayed linear head predicts closer to the mean."""
    recipes = world["cfg"]["fix43"]["recipes"]
    spread = {}
    for name, wd in (("wd_loose", 1e-6), ("wd_tight", 3.0)):
        recipes[name] = {"layer": "ws", "inputs": ["emb", "side"], "tasks": ["sent_accuracy"], "grids": {"reg": {"linear": {"wd": [wd]}}}}
        FF.run_recipe(ENC, name, "search", only_heads=["linear"], cache=world["cache"], root=world["root"])
        hcfg, _ = FF.recipe_cfg(world["cfg"], name)
        t = task("sent_accuracy")
        y, meta = FH.load_labels(t)
        base = FF.recipe_base(ENC, name, world["root"])
        sel = _best_val(base, t, hcfg, "linear", y, meta)
        spread[name] = float(np.std(RF.pooled(base, t, "linear", sel[1], "search", "val", y, meta)["pred"]))
    assert spread["wd_tight"] < spread["wd_loose"], spread


# ---- fix 4: word pooling ------------------------------------------------------ #
def test_word_max_pooling_finds_a_signal_that_mean_pooling_cannot_see(world):
    scores = {}
    for name in ("word_mean_pros", "word_max_pros"):
        world["cfg"]["fix43"]["recipes"][name]["grids"] = {"reg": {"linear": {"wd": [1e-4]}, "mlp": {"dropout": [0.1], "wd": [1e-2]}}}
        FF.run_recipe(ENC, name, "search", only_heads=["linear"], cache=world["cache"], root=world["root"])
        hcfg, _ = FF.recipe_cfg(world["cfg"], name)
        t = task("word_accuracy")
        y, meta = FH.load_labels(t)
        scores[name] = _best_val(FF.recipe_base(ENC, name, world["root"]), t, hcfg, "linear", y, meta)[0]      # validation RMSE
    assert scores["word_max_pros"] < 0.7 * scores["word_mean_pros"], scores


# ---- fix 1: layer shortlist ---------------------------------------------------- #
def test_shortlist_ranks_layers_by_probe_validation_score_within_the_window(world):
    for layer in range(6, 13):
        FH.run("%s" % ENC, layer, "emb+side", "search", only_tasks=["sent_accuracy"], cache=world["cache"], base=world["frozen"], probe=True)
    sl = FF.shortlist_layers(ENC, ["sent_accuracy"], (6, 12), 2, frozen_root=world["frozen"])
    assert sl["sent_accuracy"][0] == SIG and len(sl["sent_accuracy"]) == 2
    sl_narrow = FF.shortlist_layers(ENC, ["sent_accuracy"], (9, 12), 1, frozen_root=world["frozen"])
    assert sl_narrow["sent_accuracy"][0] in (9, 10, 11, 12)                     # layer 7 is outside the window


def test_shortlist_reports_missing_probe_results_instead_of_guessing(world, capsys):
    out = FF.shortlist_layers(ENC, ["sent_accuracy"], (6, 12), 2, frozen_root=world["frozen"])
    assert out == {} and "no probe results" in capsys.readouterr().out


def test_run_layers_dry_run_changes_nothing_and_the_real_run_fits_the_shortlisted_layers(world, capsys):
    for layer in range(6, 13):
        FH.run(ENC, layer, "emb+side", "search", only_tasks=["sent_accuracy"], cache=world["cache"], base=world["frozen"], probe=True)
    before = sorted(p for p in world["frozen"].rglob("*.npz"))
    FF.run_layers(ENC, 2, tasks=["sent_accuracy"], cache=world["cache"], frozen_root=world["frozen"], dry_run=True)
    out = capsys.readouterr().out
    assert '"sent_accuracy": [7,' in out and "jobs: " in out
    assert sorted(p for p in world["frozen"].rglob("*.npz")) == before
    FF.run_layers(ENC, 1, tasks=["sent_accuracy"], cache=world["cache"], frozen_root=world["frozen"])
    base7 = FH.results_root(ENC, 7, "emb+side", world["frozen"])
    assert list(base7.rglob("*__final/fold0.npz")), "the shortlisted layer got the final stage"
    assert list(base7.glob("sent_accuracy/mlp/*/fold0.npz")), "and the MLP head, not only the linear probe"


# ---- fix 5: out-of-fold probabilities ------------------------------------------- #
def test_mispron_oof_covers_each_train_row_once_with_speaker_disjoint_folds(world):
    kw = dict(only_tasks=["phone_mispron_t1.0"], cache=world["cache"], root=world["root"])
    FF.run_recipe(ENC, "ws", "search", **kw)
    FF.run_recipe(ENC, "ws", "final", **kw)
    out = FF.run_mispron_oof(ENC, n_folds=3, cache=world["cache"], frozen_root=world["frozen"], root=world["root"])
    d = np.load(out, allow_pickle=True)
    t = task("phone_mispron_t1.0")
    y, meta = FH.load_labels(t)
    train_rows = np.flatnonzero((meta["split"] == "train").to_numpy())
    assert np.array_equal(np.sort(d["rows"]), train_rows) and len(set(d["rows"])) == len(d["rows"])
    assert d["proba"].shape == (len(train_rows), 2) and np.isfinite(d["proba"]).all() and np.allclose(d["proba"].sum(1), 1, atol=1e-4)
    spk_fold = pd.DataFrame({"speaker": meta.loc[d["rows"], "speaker"].to_numpy(), "fold": d["fold"]}).drop_duplicates()
    assert spk_fold["speaker"].is_unique and set(spk_fold["fold"]) == {0, 1, 2}
    assert np.array_equal(d["y"], y[d["rows"]])
    src = json.loads(str(d["source"]))
    assert src["kind"] == "ws" and src["head"] in ("linear", "mlp")
    # informative: out-of-fold probabilities rank the (synthetic) positives well
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(d["y"], d["proba"][:, 1]) > 0.75
    mt = out.stat().st_mtime_ns
    assert FF.run_mispron_oof(ENC, n_folds=3, cache=world["cache"], frozen_root=world["frozen"], root=world["root"]) == out and out.stat().st_mtime_ns == mt


def test_mispron_source_errors_without_results_and_ignores_probe_only_and_search_only_systems(world):
    with pytest.raises(SystemExit, match="no class-weighted"):
        FF.find_mispron_source(ENC, world["frozen"], world["root"])
    for layer in (8, 12):                                                      # single-config probe files (US 4.2) are not candidates
        FH.run(ENC, layer, "emb+side", "search", only_tasks=["phone_mispron_t1.0"], cache=world["cache"], base=world["frozen"], probe=True)
    FF.run_recipe(ENC, "ws", "search", only_tasks=["phone_mispron_t1.0"], cache=world["cache"], root=world["root"])      # search only
    with pytest.raises(SystemExit, match="no class-weighted"):
        FF.find_mispron_source(ENC, world["frozen"], world["root"])
    FF.run_recipe(ENC, "ws", "final", only_tasks=["phone_mispron_t1.0"], cache=world["cache"], root=world["root"])
    src = FF.find_mispron_source(ENC, world["frozen"], world["root"])
    assert src["kind"] == "ws" and src["head"] in ("linear", "mlp")
