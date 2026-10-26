"""Sprint 6 (US 6.1) tests for src/models/multitask.py.

Numpy-only tests (helpers, loss weights, config, data alignment on the synthetic SO762 world of test_finetune.py) run anywhere.
The torch tests use the same tiny random WavLM (a few thousand parameters), so they also run on a laptop CPU. They check, in this order:
the multi-task model computes exactly what the single-task model computes; gradients reach the right parameters; the gradient diagnostics equal an
explicit computation; a full job writes single-task-compatible files; and a multi-task job with ONE head reproduces the single-task job.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import finetune as ft
from src.models import multitask as mt
from src.models.baselines import build_tasks, load_config as load_baseline_config

from test_finetune import SR, _task, _tiny_encoder, world  # noqa: F401  (fixtures / helpers shared with the Sprint 5 tests)


# ---- pure helpers ------------------------------------------------------------ #
def test_normalise_weights_sum_to_one_default_equal_and_reject_nonpositive():
    w = mt.normalise_weights(["a", "b", "c", "d"], None)
    assert all(abs(v - 0.25) < 1e-12 for v in w.values())
    w = mt.normalise_weights(["a", "b"], {"a": 3.0})
    assert w["a"] == pytest.approx(0.75) and w["b"] == pytest.approx(0.25) and sum(w.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mt.normalise_weights(["a"], {"a": 0.0})


def test_error_and_mean_rel():
    assert mt.error_from_primary("reg", 0.8) == 0.8
    assert mt.error_from_primary("clf", -0.7) == pytest.approx(0.3)                 # primary_score = -macro-F1
    assert mt.mean_rel({"a": 0.5, "b": 0.3}, {"a": 1.0, "b": 0.6}) == pytest.approx(0.5)
    assert mt.mean_rel({"a": 1.0}, {"a": 1.0}) == 1.0                                # epoch 0 is the reference


def test_cosine_from_gram_matches_vectors_and_handles_zero_gradient():
    V = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    n, c = mt.cosine_from_gram(V @ V.T)
    np.testing.assert_allclose(n, [1, 2, 1, 0])
    assert c[0, 1] == 0 and c[0, 2] == pytest.approx(-1.0) and c[0, 0] == pytest.approx(1.0)
    assert (c[3] == 0).all() and (c[:, 3] == 0).all()                                # zero gradient: cosine 0, not NaN
    s = mt.grad_shares(n)
    assert s.sum() == pytest.approx(1.0) and s[1] == pytest.approx(0.5)
    np.testing.assert_allclose(mt.grad_shares(np.zeros(3)), np.full(3, 1 / 3))


def test_fixed_weighter_is_the_weighted_sum_and_future_schemes_are_refused_for_now():
    torch = pytest.importorskip("torch")
    names = ["a", "b"]
    wt = mt.make_weighter({"scheme": "fixed", "weights": {"a": 1.0, "b": 3.0}}, names, {"a": "reg", "b": "clf"})
    losses = {"a": torch.tensor(2.0, requires_grad=True), "b": torch.tensor(4.0, requires_grad=True)}
    total, w = wt.combine(losses)
    assert float(total.detach()) == pytest.approx(0.25 * 2.0 + 0.75 * 4.0) and w == {"a": 0.25, "b": 0.75} and wt.parameters() == []
    eq, _ = mt.make_weighter({"scheme": "fixed"}, names, {}).combine(losses)
    assert float(eq.detach()) == pytest.approx(3.0)                                          # equal weights = mean of the losses
    with pytest.raises(NotImplementedError):
        mt.make_weighter({"scheme": "gradnorm"}, names, {})


# ---- configuration ----------------------------------------------------------- #
def test_config_inherits_the_single_task_recipe_and_lists_every_task_once():
    cfg, base = mt.load_config(), ft.load_config()
    for k in ("layers", "layer_mode", "head", "hidden", "head_dropout", "freeze_cnn", "freeze_layers"):
        assert cfg["model"][k] == base["model"][k], k
    for k in ("max_epochs", "patience", "lr_head", "lr_encoder", "lr_alpha", "wd_head", "wd_encoder", "grad_clip", "head_warmup_epochs", "warmup_frac", "schedule",
              "amp", "max_batch_seconds", "pad_tolerance", "cw"):
        assert cfg["train"][k] == base["train"][k], k
    assert cfg["encoder"] == base["encoder"] and cfg["train"]["stop_metric"] == "mean_rel" and cfg["train"]["loss"]["scheme"] == "fixed"
    reg = build_tasks(load_baseline_config())
    grouped = [t for g in cfg["groups"].values() for t in g["tasks"]]
    assert sorted(grouped) == sorted(cfg["tasks"]) and len(set(grouped)) == len(grouped) == 9
    assert all(t in reg for t in grouped)
    assert {reg[t].dataset for t in cfg["groups"]["so762"]["tasks"]} == {"so762"} and {reg[t].dataset for t in cfg["groups"]["l2"]["tasks"]} == {"l2arctic"}
    assert cfg["root"] != base["root"] and cfg["ckpt_root"] != base["ckpt_root"]                       # never overwrites Sprint 5 files
    assert mt.seeds_for_group(cfg, reg, "so762") == [0, 1, 2] and mt.seeds_for_group(cfg, reg, "l2") == [0, 1, 2] and mt.seeds_for_group(cfg, reg, "l2", [5]) == [5]


def test_variants_resolve_here_first_then_in_finetune_yaml():
    lr = mt.load_config("lr")                                                       # defined in finetune.yaml (US 5.3), applied to the multi-task recipe
    assert lr["train"]["lr_encoder"] == ft.load_config("lr")["train"]["lr_encoder"] != mt.load_config()["train"]["lr_encoder"]
    assert mt.load_config("sharedalpha")["model"]["alpha_mode"] == "shared"
    with pytest.raises(KeyError):
        mt.load_config("nope")


# ---- data alignment on the synthetic world ----------------------------------- #
def _group(world, names=("sent_accuracy", "sent_total", "phone_score", "phone_mispron_t1.0")):
    return mt.MTData([_task(n) for n in names], False, split_csv=world / "speakers.csv")


def test_mtdata_heads_see_the_same_utterances_rows_and_splits(world):
    md = _group(world)
    assert set(md.table_head) == {"so762_sentence", "so762_phone"} and md.table_kind == {"so762_sentence": "utt", "so762_phone": "segment"}
    us = md.utt_split(0)
    assert sorted(len(us[s]) for s in us) == [12, 12, 24]                           # 6 / 3 / 3 speakers x 4 utterances
    for n in md.names:                                                              # every head's rows for an utterance are its own table's rows
        td = md.td[n]
        for u in (0, 7, 30):
            np.testing.assert_array_equal(md.spans_all(u)[md.tasks[n].table][0], td.rows_of[md.map[n][u]])
    np.testing.assert_array_equal(md.rows("phone_score", us["val"]), md.rows("phone_mispron_t1.0", us["val"]))     # same table, same rows
    assert len(set(map(tuple, (md.rows(n, us["val"]) for n in ["sent_accuracy", "sent_total"])))) == 1


def test_mtdata_refuses_heads_that_disagree_on_the_splits(world):
    md = _group(world)
    real = md.td["phone_score"].utt_split
    md.td["phone_score"].utt_split = lambda fold: {**real(fold), "val": real(fold)["test"], "test": real(fold)["val"]}
    md._splits.clear()
    with pytest.raises(ValueError, match="disagree"):
        md.utt_split(0)


def test_collate_mt_rows_match_the_pooled_unit_order(world):
    torch = pytest.importorskip("torch")
    md = _group(world)
    utts = md.utt_split(0)["train"][:3]
    side_t = {n: torch.as_tensor(md.td[n].side) for n in md.names}
    yt = {n: torch.as_tensor(np.array(md.td[n].yi), dtype=torch.float32) for n in md.names}
    x, lens, spans, rows, side, ys = mt.collate_mt(md, utts, "cpu", side_t, yt)
    assert x.shape[0] == 3 and (lens <= x.shape[1]).all()
    assert len(rows["sent_accuracy"]) == 3 and len(rows["phone_score"]) == 12
    np.testing.assert_array_equal(rows["phone_score"], rows["phone_mispron_t1.0"])
    for n in md.names:
        assert side[n].shape[0] == len(rows[n]) == ys[n].shape[0]
        np.testing.assert_allclose(ys[n].numpy(), md.td[n].yi[rows[n]], rtol=1e-6)
    assert len(spans["so762_phone"]) == 3 and len(spans["so762_sentence"]) == 3


# ---- torch: model equivalence and gradients ---------------------------------- #
def _specs(md):
    return [{"name": n, "table": md.tasks[n].table, "pool_kind": md.td[n].pool_kind, "d_side": md.td[n].side.shape[1],
             "d_out": 1 if md.tasks[n].kind == "reg" else len(md.tasks[n].labels)} for n in md.names]


def test_pool_units_and_heads_equal_the_single_task_model(world):
    torch = pytest.importorskip("torch")
    md = _group(world)
    enc = _tiny_encoder()
    enc.eval()
    ST = ft.build_model_class()
    MT = mt.build_mt_model_class()
    model = MT(enc, [0, 1, 2, 3], _specs(md), "mlp", 8, 0.0).eval()
    utts = md.utt_split(0)["train"][:3]
    x, lens, spans, rows, side, _ = mt.collate_mt(md, utts, "cpu", {n: torch.as_tensor(md.td[n].side) for n in md.names}, None)
    with torch.no_grad():
        out = model(x, lens, spans, side)
        for n in md.names:
            td, task = md.td[n], md.tasks[n]
            d_out = 1 if task.kind == "reg" else len(task.labels)
            single = ST(enc, [0, 1, 2, 3], td.pool_kind, td.side.shape[1], d_out, "mlp", 8, 0.0).eval()
            single.head.load_state_dict(model.heads[mt.head_key(n)].state_dict())
            ref = single(x, lens, spans[task.table], side[n])
            assert out[n].shape == (len(rows[n]), d_out)
            torch.testing.assert_close(out[n], ref, rtol=1e-5, atol=1e-6)
    assert set(model.layer_weights()) == set(md.names) and all(abs(w.sum() - 1) < 1e-5 for w in model.layer_weights().values())


def test_each_head_trains_only_its_own_parameters_and_the_shared_encoder_gets_all_heads(world):
    torch = pytest.importorskip("torch")
    md = _group(world)
    model = mt.build_mt_model_class()(_tiny_encoder(), [0, 1, 2, 3], _specs(md), "mlp", 8, 0.0)
    utts = md.utt_split(0)["train"][:2]
    x, lens, spans, rows, side, _ = mt.collate_mt(md, utts, "cpu", {n: torch.as_tensor(md.td[n].side) for n in md.names}, None)
    out = model(x, lens, spans, side)
    out["phone_score"].sum().backward()
    g = {n: p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters()}
    assert g["heads.phone_score.0.weight"] and g["alpha.phone_score"]
    assert not g.get("heads.sent_accuracy.0.weight", False) and not g.get("alpha.sent_accuracy", False)              # a head's loss never touches another head
    assert g["enc.encoder.layers.2.attention.q_proj.weight"] and not any(v for n, v in g.items() if "feature_extractor" in n)
    assert "phone_mispron_t1.0" not in model.heads and mt.head_key("phone_mispron_t1.0") == "phone_mispron_t1_0" in model.heads   # '.' is illegal in ModuleDict keys


def test_shared_alpha_mode_has_one_layer_weight_vector(world):
    pytest.importorskip("torch")
    md = _group(world)
    model = mt.build_mt_model_class()(_tiny_encoder(), [0, 1, 2, 3], _specs(md), "mlp", 8, 0.0, "shared")
    assert list(model.alpha.keys()) == ["shared"]
    lw = model.layer_weights()
    assert all(lw[n] is lw[md.names[0]] or np.array_equal(lw[n], lw[md.names[0]]) for n in md.names)


def test_param_groups_cover_every_trainable_parameter_once_and_use_single_task_kinds(world):
    pytest.importorskip("torch")
    md = _group(world)
    model = mt.build_mt_model_class()(_tiny_encoder(), [0, 1, 2, 3], _specs(md), "mlp", 8, 0.0)
    tcfg = ft.load_config()["train"]
    groups = mt.param_groups_mt(model, tcfg)
    ids = [id(p) for g in groups for p in g["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(ids) == len(set(ids)) == len(trainable)
    assert {g["kind"] for g in groups} == {"head", "alpha", "enc"}
    assert any(g.get("layer") == 3 for g in groups) and all(g["lr"] > 0 for g in groups)
    top = mt.probe_parameters(model)
    assert top and all(ft.layer_index(n) == 3 for n, p in model.enc.named_parameters() if any(p is q for q in top))


def test_head_gram_equals_explicit_gradients(world):
    torch = pytest.importorskip("torch")
    md = _group(world)
    model = mt.build_mt_model_class()(_tiny_encoder(), [0, 1, 2, 3], _specs(md), "mlp", 8, 0.0)
    utts = md.utt_split(0)["train"][:3]
    x, lens, spans, rows, side, ys = mt.collate_mt(md, utts, "cpu", {n: torch.as_tensor(md.td[n].side) for n in md.names},
                                                   {n: torch.as_tensor(md.td[n].yi, dtype=torch.float32) for n in md.names})
    out = model(x, lens, spans, side)
    losses = {n: ((out[n].squeeze(-1) if md.tasks[n].kind == "reg" else out[n][:, 1]) - ys[n]).pow(2).mean() for n in md.names}
    probe = mt.probe_parameters(model)
    G = mt.head_gram(losses, md.names, probe)
    vecs = []
    for n in md.names:
        g = torch.autograd.grad(losses[n], probe, retain_graph=True)
        vecs.append(torch.cat([a.reshape(-1) for a in g]).double().numpy())
    V = np.stack(vecs)
    np.testing.assert_allclose(G, V @ V.T, rtol=1e-4, atol=1e-10)
    n_, c_ = mt.cosine_from_gram(G)
    assert np.allclose(np.diag(c_), 1.0) and (np.abs(c_) <= 1 + 1e-9).all()
    # the same graph must still be usable for the real backward pass (diagnostics must not consume it)
    sum(losses.values()).backward()


# ---- torch: full jobs --------------------------------------------------------- #
def _mt_cfg(world, **train):
    cfg = mt.load_config()
    return ft.deep_update(cfg, {
        "model": {"layers": [0, 3], "hidden": 16, "head_dropout": 0.0},
        "train": {"max_epochs": 3, "patience": 2, "head_warmup_epochs": 1.0, "max_batch_seconds": 4, "train_eval_utts": 4, "final_train_eval_rows": 1000,
                  "amp": "off", "save_checkpoint": "best", "lr_encoder": 1e-3, "lr_head": 5e-3, "diag": {"every_steps": 1, "probe": "top_layer"}, **train},
        "root": str(world / "mres"), "ckpt_root": str(world / "mck")})


def test_run_group_end_to_end_writes_single_task_compatible_files(world):
    pytest.importorskip("torch")
    cfg = _mt_cfg(world)
    md = _group(world)
    cache = {"load": lambda mcfg: (_tiny_encoder(0), False)}
    r = mt.run_group("so762", md, cache, cfg, "t", 0, 0, "cpu", log=lambda *a: None)
    us = md.utt_split(0)
    assert set(r["val"]) == set(md.names) and np.isfinite(list(r["val"].values())).all()
    for n in md.names:
        d = np.load(mt.job_file(cfg, "t", n, 0, 0), allow_pickle=True)
        task = md.tasks[n]
        np.testing.assert_array_equal(d["val_rows"], np.sort(md.rows(n, us["val"])))
        np.testing.assert_array_equal(d["test_rows"], np.sort(md.rows(n, us["test"])))
        assert len(d["val_pred"]) == len(d["val_rows"]) and len(d["hist_val_loss"]) == int(d["epochs_run"]) <= 3 and len(d["layer_weights"]) == 4
        assert abs(d["layer_weights"].sum() - 1) < 1e-5 and str(d["mt_group"]) == "so762" and int(d["mt_n_heads"]) == 4
        if task.kind == "clf":
            assert d["val_proba"].shape == (len(d["val_rows"]), 2) and np.allclose(d["val_proba"].sum(1), 1, atol=1e-5)
        for k in ("train_score", "val_score", "best_epoch", "epochs_run", "seed", "fit_s", "hist_train_loss", "hist_val_primary", "hist_trsub_primary", "train_rows"):
            assert k in d.files, k
        assert np.isfinite(d["hist_val_primary"]).all()
    g = np.load(mt.group_file(cfg, "t", "so762", 0, 0), allow_pickle=True)
    E = len(g["hist_mean_rel"])
    assert g["hist_mean_rel"][0] == pytest.approx(1.0)                              # epoch 0 is the reference
    assert g["hist_gnorm"].shape == (E, 4) and g["hist_gcos"].shape == (E, 4, 4) and g["hist_w"].shape == (E, 4)
    assert np.isnan(g["hist_gnorm"][0]).all() and E > 1 and np.isfinite(g["hist_gnorm"][1]).all() and (g["hist_gn"][1:] > 0).all()   # encoder frozen in epoch 0
    np.testing.assert_allclose(g["hist_w"].sum(1), 1.0)
    assert mt.ckpt_file(cfg, "t", "so762", 0, 0).exists()
    # same seed -> same predictions
    mt.run_group("so762", md, cache, cfg, "t2", 0, 0, "cpu", log=lambda *a: None)
    for n in md.names:
        a, b = np.load(mt.job_file(cfg, "t", n, 0, 0), allow_pickle=True), np.load(mt.job_file(cfg, "t2", n, 0, 0), allow_pickle=True)
        key = "val_pred" if md.tasks[n].kind == "reg" else "val_proba"
        np.testing.assert_allclose(a[key].astype(float), b[key].astype(float), atol=1e-4)


@pytest.mark.parametrize("name", ["sent_accuracy", "phone_mispron_t1.0"])
def test_a_one_head_multitask_job_reproduces_the_single_task_job(world, name):
    """With one head the multi-task code path must equal the Sprint 5 job (same seed, same recipe): the comparison of Sprint 6 is then only about sharing."""
    pytest.importorskip("torch")
    cfg = _mt_cfg(world)
    st_cfg = ft.load_config()
    st_cfg = ft.deep_update(st_cfg, {"model": cfg["model"], "train": {k: v for k, v in cfg["train"].items() if k in st_cfg["train"]}, "root": str(world / "sres"),
                                     "ckpt_root": str(world / "sck")})
    task = _task(name)
    td = ft.TaskData(task, False, split_csv=world / "speakers.csv")
    cache = {"load": lambda mcfg: (_tiny_encoder(0), False)}
    ft.run_job(task, td, cache, st_cfg, "st", 0, 0, "cpu", 0.0, log=lambda *a: None)
    md = mt.MTData([task], False, split_csv=world / "speakers.csv")
    mt.run_group("g", md, cache, cfg, "mt", 0, 0, "cpu", log=lambda *a: None)
    a, b = np.load(ft.job_file(st_cfg, "st", name, 0, 0), allow_pickle=True), np.load(mt.job_file(cfg, "mt", name, 0, 0), allow_pickle=True)
    assert int(a["best_epoch"]) == int(b["best_epoch"]) and int(a["epochs_run"]) == int(b["epochs_run"])
    np.testing.assert_array_equal(a["test_rows"], b["test_rows"])
    key = "test_pred" if task.kind == "reg" else "test_proba"
    np.testing.assert_allclose(a[key].astype(float), b[key].astype(float), atol=2e-4)
    np.testing.assert_allclose(a["hist_val_primary"], b["hist_val_primary"], atol=2e-4)


def test_status_and_dry_run_cli_on_the_synthetic_world(world, capsys, monkeypatch):
    """`status` counts finished groups; `dry-run` builds the real tables (here: only the SO762 tables of the synthetic world exist, so only check status)."""
    pytest.importorskip("torch")
    cfg = _mt_cfg(world)
    monkeypatch.setattr(mt, "load_config", lambda variant=None, path=None: cfg)
    class A:  # noqa: D401
        tag, variant, groups = "nothing", None, ["so762"]
    mt.cmd_status(A)
    assert "0/3" in capsys.readouterr().out
