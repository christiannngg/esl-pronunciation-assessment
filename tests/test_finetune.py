"""Sprint 5 (US 5.1) tests for src/models/finetune.py.

Numpy-only tests (batching, schedule, holdout, config, TaskData on a synthetic data root) run anywhere.
The torch tests need torch + transformers and use a tiny random WavLM (a few thousand parameters), so they also run on a laptop CPU.
The real-checkpoint equivalence check against the cached Sprint 4 embeddings is `python -m src.models.finetune verify` (needs the cache).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import ssl_embeddings as se
from src.models import finetune as ft
from src.models.baselines import build_tasks, load_config as load_baseline_config

SR = 16000


# ---- pure helpers ------------------------------------------------------------ #
def test_normalize_wave_matches_hf_rule_and_passthrough():
    w = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 3 + 1
    z = ft.normalize_wave(w, True)
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-3 and z.dtype == np.float32
    np.testing.assert_allclose(z, (w - w.mean()) / np.sqrt(w.var() + 1e-7), atol=1e-6)
    np.testing.assert_array_equal(ft.normalize_wave(w, False), w)


def test_make_batches_covers_everything_once_and_bounds_padding():
    rng = np.random.default_rng(1)
    lengths = rng.integers(2 * SR, 20 * SR, 300)
    for r in (None, np.random.default_rng(5)):
        batches = ft.make_batches(lengths, 60 * SR, 0.05, r)
        flat = sorted(i for b in batches for i in b)
        assert flat == list(range(300))
        for b in batches:
            L = lengths[b]
            assert L.max() <= L.min() * 1.05 + 1
            assert len(b) == 1 or len(b) * L.max() <= 60 * SR
    # deterministic without rng; different compositions with jitter
    assert ft.make_batches(lengths, 60 * SR, 0.05) == ft.make_batches(lengths, 60 * SR, 0.05)


def test_make_batches_single_long_utterance_is_its_own_batch():
    b = ft.make_batches(np.array([30 * SR, 31 * SR]), 20 * SR, 0.5)
    assert sorted(len(x) for x in b) == [1, 1]


def test_lr_scales_freeze_warmup_decay():
    total, freeze, warm = 100, 20, 10
    enc = [ft.lr_scales(s, total, freeze, warm)[0] for s in range(total)]
    head = [ft.lr_scales(s, total, freeze, warm)[1] for s in range(total)]
    assert all(e == 0 for e in enc[:freeze])                       # encoder frozen during the head warm-up
    assert enc[freeze] > 0 and enc[freeze + warm - 1] > enc[freeze]            # warming up
    assert max(enc) <= 1 and enc[-1] < 0.05 and head[-1] < 0.05                 # both decay to ~0
    assert head[0] > 0 and head[freeze - 1] == pytest.approx(1.0)               # head trains at full lr while the encoder waits
    post = enc[freeze + warm:]
    assert all(a >= b for a, b in zip(post, post[1:]))                          # monotone decay after warm-up
    cos = [ft.lr_scales(s, total, freeze, warm, "cosine")[0] for s in range(freeze + warm, total)]
    assert cos[-1] < 0.05 and all(a >= b for a, b in zip(cos, cos[1:]))


def test_holdout_utterances_per_speaker_and_deterministic():
    spk = np.repeat([f"s{i}" for i in range(10)], 20)
    utt = np.arange(200).astype(str)
    m = ft.holdout_utterances(spk, utt, 0.2, seed=0)
    assert m.sum() == 40 and all(m[spk == s].sum() == 4 for s in set(spk))
    np.testing.assert_array_equal(m, ft.holdout_utterances(spk, utt, 0.2, seed=0))
    assert not np.array_equal(m, ft.holdout_utterances(spk, utt, 0.2, seed=1))
    assert ft.holdout_utterances(spk, utt, 0.0).sum() == 0
    two = ft.holdout_utterances(np.array(["a", "a"]), np.array(["x", "y"]), 0.99)
    assert two.sum() == 1                                                      # always leaves one training utterance


def test_layer_index_maps_parameters_to_hidden_state_index():
    assert ft.layer_index("encoder.layers.0.attention.q_proj.weight") == 1
    assert ft.layer_index("encoder.layers.11.feed_forward.output_dense.bias") == 12
    assert ft.layer_index("feature_projection.projection.weight") == 0
    assert ft.layer_index("encoder.pos_conv_embed.conv.weight") == 0


def test_config_is_consistent_with_the_task_registry():
    cfg = ft.load_config()
    tasks = build_tasks(load_baseline_config())
    assert set(cfg["tasks"]) <= set(tasks)
    frozen = __import__("yaml").safe_load((ft.REPO / "configs" / "frozen.yaml").read_text())
    assert cfg["encoder"] in frozen["encoders"]
    assert cfg["model"]["hf"]["layerdrop"] == 0.0 and cfg["model"]["hf"]["apply_spec_augment"] is False      # 5.1 = plain recipe
    assert cfg["holdout_seen_frac"] == 0.0
    merged = ft.deep_update(cfg, {"model": {"freeze_layers": 6}, "train": {"lr_encoder": 1e-5}})
    assert merged["model"]["freeze_layers"] == 6 and merged["model"]["head"] == cfg["model"]["head"] and merged["train"]["lr_encoder"] == 1e-5
    assert cfg["model"]["freeze_layers"] == 0                                  # deep_update does not mutate
    with pytest.raises(KeyError):
        ft.load_config("no_such_variant")


# ---- synthetic world: SO762-shaped tables + audio ---------------------------- #
@pytest.fixture()
def world(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "data"
    tab, ali, aud = root / "processed" / "features" / "tables", root / "processed" / "alignments", root / "processed" / "audio" / "so762"
    for d in (tab, ali, aud):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    spk = [f"s{i:02d}" for i in range(12)]
    split = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
    pd.DataFrame({"speaker": spk, "split": split}).to_csv(tmp_path / "speakers.csv", index=False)
    s_rows, p_rows, a_rows = [], [], []
    for si, s in enumerate(spk):
        for i in range(4):
            u = f"{s}_u{i}"
            n = int((0.6 + 0.1 * i) * SR)
            sf.write(aud / f"{u}.wav", rng.standard_normal(n).astype(np.float32) * 0.1, SR, subtype="FLOAT")
            s_rows.append({"utt_id": u, "y_accuracy": float(rng.integers(3, 10)), "y_fluency": 5.0, "y_prosodic": 5.0, "y_total": 5.0,
                           "speaker": s, "split": "x", "gender": "f", "age_group": "adult"})
            for k in range(4):
                p_rows.append({"utt_id": u, "word_idx": k // 2, "phone_idx": k % 2, "phone": "AH", "y_phone_score": float((k + i + si) % 3 - 0.0) * 0.8,
                               "speaker": s, "split": "x", "gender": "f", "age_group": "adult"})
                a_rows.append({"utt_id": u, "word_idx": k // 2, "phone_idx": k % 2, "start": 0.05 + 0.1 * k, "end": 0.15 + 0.1 * k})
    sent, ph = pd.DataFrame(s_rows), pd.DataFrame(p_rows)
    sent.to_csv(tab / "so762_sentence_meta.csv.gz", index=False)
    ph.to_csv(tab / "so762_phone_meta.csv.gz", index=False)
    pd.DataFrame(a_rows).to_csv(ali / "so762_phones.csv.gz", index=False)
    np.save(tab / "so762_sentence_side_X.npy", rng.standard_normal((len(sent), 3)).astype(np.float32))
    np.save(tab / "so762_phone_side_X.npy", rng.standard_normal((len(ph), 5)).astype(np.float32))
    monkeypatch.setenv("DATA_ROOT", str(root))
    return tmp_path


def _task(name):
    return build_tasks(load_baseline_config())[name]


def test_taskdata_rows_utterances_and_splits(world):
    td = ft.TaskData(_task("phone_mispron_t1.0"), False, split_csv=world / "speakers.csv")
    assert len(td.utt_ids) == 48 and all(len(r) == 4 for r in td.rows_of) and td.pool_kind == "segment"
    us = td.utt_split(0)
    assert {k: len(v) for k, v in us.items()} == {"train": 24, "val": 12, "test": 12}
    assert not (set(td.utt_speaker[us["train"]]) & set(td.utt_speaker[us["test"]]))          # speaker-independent
    rows, i0, i1 = td.spans(5)
    T = se.n_frames(int(td.lengths[5]))
    e0, e1 = se.unit_frames(T, td.t0[rows], td.t1[rows])
    np.testing.assert_array_equal(i0, e0); np.testing.assert_array_equal(i1, e1)
    assert (i1 > i0).all() and int(td.lengths[5]) == len(td.wav(5))
    ts = ft.TaskData(_task("sent_accuracy"), True, split_csv=world / "speakers.csv")
    assert ts.pool_kind == "utt" and abs(ts.wav(0).mean()) < 1e-4                           # normalised on load
    assert (ts.spans(0)[1] == -1).all()


# ---- torch tests (tiny random WavLM) ---------------------------------------- #
def _tiny_encoder(seed=0):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    cfg = tr.WavLMConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, conv_dim=(16,) * 7,
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_buckets=32, max_bucket_distance=64,
                         hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
                         apply_spec_augment=False)
    enc = tr.WavLMModel(cfg)
    enc.freeze_feature_encoder()
    return enc


def test_pooling_matches_the_numpy_reference_used_for_the_frozen_cache():
    torch = pytest.importorskip("torch")
    Model = ft.build_model_class()
    enc = _tiny_encoder()
    w = torch.randn(1, int(0.9 * SR))
    for kind in ("utt", "segment"):
        m = Model(enc, [0, 1, 2, 3], kind, 0, 1, "linear", 8, 0.0).eval()
        with torch.no_grad():
            H = m.hidden_states(w, False)
            T = se.n_frames(w.shape[1])
            assert H.shape[2] == T                                              # frame count of the HF model == the geometry used for spans
            i0, i1 = se.unit_frames(T, np.array([0.0, 0.1, 0.5, 0.85]), np.array([0.9, 0.12, 0.7, 0.9]))
            P = m.pool(H, [w.shape[1]], [(i0, i1)]).numpy()
        Hn = H[0].numpy()                                                       # (K, T, D)
        if kind == "utt":
            np.testing.assert_allclose(P[0], se.pool_utterance(Hn), rtol=1e-4, atol=1e-5)
        else:
            np.testing.assert_allclose(P, se.pool_units(Hn, i0, i1), rtol=1e-4, atol=1e-5)


def test_gradients_flow_only_where_expected():
    torch = pytest.importorskip("torch")
    Model = ft.build_model_class()
    for freeze in (0, 2):
        enc = _tiny_encoder()
        mcfg = {"freeze_cnn": True, "freeze_layers": freeze}
        if freeze:
            for mod in (enc.feature_projection, enc.encoder.pos_conv_embed, getattr(enc.encoder, "layer_norm", None)):
                for p in mod.parameters():
                    p.requires_grad_(False)
            for layer in enc.encoder.layers[:freeze]:
                for p in layer.parameters():
                    p.requires_grad_(False)
        m = Model(enc, [0, 1, 2, 3], "utt", 2, 1, "mlp", 8, 0.0)
        w = torch.randn(2, SR // 2)
        out = m(w, [SR // 2] * 2, [(None, None)] * 2, torch.randn(2, 2))
        out.sum().backward()
        g = {n: p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in m.named_parameters()}
        assert not any(v for n, v in g.items() if "feature_extractor" in n)                  # CNN never trains
        assert g["head.0.weight"] and g["alpha"]
        assert g["enc.encoder.layers.2.attention.q_proj.weight"]
        assert g["enc.encoder.layers.0.attention.q_proj.weight"] == (freeze == 0)


def test_param_groups_partition_trainable_parameters_and_apply_layer_decay():
    pytest.importorskip("torch")
    Model = ft.build_model_class()
    enc = _tiny_encoder()
    m = Model(enc, [0, 1, 2, 3], "utt", 2, 1, "linear", 8, 0.0)
    tcfg = {"lr_head": 1e-3, "lr_alpha": 2e-2, "lr_encoder": 1e-4, "layer_lr_decay": 0.5, "wd_head": 0.01, "wd_encoder": 0.01}
    groups = ft.param_groups(m, tcfg)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == sum(p.requires_grad for p in m.parameters())
    lr = {g["layer"]: g["base_lr"] for g in groups if g["kind"] == "enc"}
    assert lr[3] == pytest.approx(1e-4) and lr[2] == pytest.approx(5e-5) and lr[0] == pytest.approx(1e-4 * 0.5 ** 3)
    nodecay = [g for g in groups if g["kind"] == "enc" and g["weight_decay"] == 0.0]
    assert nodecay and all(g["weight_decay"] in (0.0, 0.01) for g in groups if g["kind"] == "enc")


def _tiny_cfg(world):
    cfg = ft.load_config()
    cfg = ft.deep_update(cfg, {
        "model": {"layers": [0, 3], "hidden": 16, "head_dropout": 0.0},
        "train": {"max_epochs": 3, "patience": 2, "head_warmup_epochs": 1.0, "max_batch_seconds": 4, "train_eval_utts": 4,
                  "final_train_eval_rows": 1000, "amp": "off", "save_checkpoint": "best", "lr_encoder": 1e-3, "lr_head": 5e-3},
        "root": str(world / "res"), "ckpt_root": str(world / "ck")})
    return cfg


@pytest.mark.parametrize("name", ["sent_accuracy", "phone_mispron_t1.0"])
def test_run_job_end_to_end_on_a_tiny_model(world, name):
    pytest.importorskip("torch")
    cfg = _tiny_cfg(world)
    task = _task(name)
    td = ft.TaskData(task, False, split_csv=world / "speakers.csv")
    cache = {"load": lambda mcfg: (_tiny_encoder(0), False)}
    r = ft.run_job(task, td, cache, cfg, "t", 0, 0, "cpu", 0.0, log=lambda *a: None)
    d = np.load(ft.job_file(cfg, "t", name, 0, 0), allow_pickle=True)
    us = td.utt_split(0)
    np.testing.assert_array_equal(d["val_rows"], np.sort(td.rows(us["val"])))
    np.testing.assert_array_equal(d["test_rows"], np.sort(td.rows(us["test"])))
    assert len(d["val_pred"]) == len(d["val_rows"]) and len(d["hist_val_loss"]) == int(d["epochs_run"]) <= 3
    assert len(d["layer_weights"]) == 4 and abs(d["layer_weights"].sum() - 1) < 1e-5
    if task.kind == "clf":
        assert d["val_proba"].shape == (len(d["val_rows"]), 2) and np.allclose(d["val_proba"].sum(1), 1, atol=1e-5)
    assert np.isfinite(d["hist_val_primary"]).all() and np.isfinite(r["val"]) and ft.ckpt_file(cfg, "t", name, 0, 0).exists()
    # same seed -> same predictions
    ft.run_job(task, td, cache, cfg, "t2", 0, 0, "cpu", 0.0, log=lambda *a: None)
    d2 = np.load(ft.job_file(cfg, "t2", name, 0, 0), allow_pickle=True)
    np.testing.assert_allclose(d["val_pred"].astype(float) if task.kind == "reg" else d["val_proba"],
                               d2["val_pred"].astype(float) if task.kind == "reg" else d2["val_proba"], atol=1e-4)


def test_seen_speaker_holdout_is_never_trained_on_and_comes_from_training_speakers(world):
    pytest.importorskip("torch")
    cfg = _tiny_cfg(world)
    task = _task("sent_accuracy")
    td = ft.TaskData(task, False, split_csv=world / "speakers.csv")
    cache = {"load": lambda mcfg: (_tiny_encoder(0), False)}
    ft.run_job(task, td, cache, cfg, "h", 0, 0, "cpu", 0.5, log=lambda *a: None)
    d = np.load(ft.job_file(cfg, "h", task.name, 0, 0), allow_pickle=True)
    us = td.utt_split(0)
    seen_rows = d["seen_rows"]
    assert len(seen_rows) == 12                                                    # half of 6 speakers x 4 utterances
    assert set(td.meta["speaker"].to_numpy()[seen_rows]) <= set(td.utt_speaker[us["train"]])
    assert d["n_train"] == 12 and not (set(d["train_rows"]) & set(seen_rows))        # trained on the other half only
    assert np.isfinite(d["hist_seen_primary"]).all()


# ---- report: seed averaging, row alignment, decision-rule choice (no model needed) ---------------------------------------- #
def _fake_jobs(task, rng, n_rows, speakers, seeds=(0, 1)):
    """Per-fold lists of npz-like dicts with the keys run_job writes, rows partitioned val / test / train."""
    clf = task.kind == "clf"
    idx = rng.permutation(n_rows)
    parts = {"val": idx[:60], "test": idx[60:140], "train": idx[140:200]}
    jobs = {0: []}
    for s in seeds:
        d = {"seed": s, "val_score": 0.5, "train_score": 0.4, "best_epoch": 3, "epochs_run": 6, "fit_s": 1.0}
        for k, rows in parts.items():
            rows = np.sort(rows)
            d[f"{k}_rows"] = rows
            if clf:
                p = rng.dirichlet([1, 1], len(rows)).astype(np.float32)
                d[f"{k}_proba"] = p
                d[f"{k}_pred"] = p.argmax(1)
            else:
                d[f"{k}_pred"] = rng.normal(size=len(rows)).astype(np.float32)
        jobs[0].append(d)
    return jobs


def test_report_averages_seeds_and_keeps_sprint4_row_order():
    pytest.importorskip("sklearn")
    from src.evaluation import report_us51 as R
    rng = np.random.default_rng(0)
    n = 200
    meta = pd.DataFrame({"speaker": np.repeat([f"s{i}" for i in range(20)], 10), "age_group": "adult", "utt_id": [f"u{i}" for i in range(n)]})
    task = _task("sent_accuracy")
    y_all = rng.normal(size=n)
    jobs = _fake_jobs(task, rng, n, meta.speaker.to_numpy())
    P = R.pooled_ft(task, jobs, y_all, meta)
    exp = np.mean([j["test_pred"] for j in jobs[0]], 0)
    np.testing.assert_allclose(P["test"]["pred"], exp)
    np.testing.assert_array_equal(P["test"]["y"], y_all[jobs[0][0]["test_rows"]])
    assert (P["test"]["meta"]["utt_id"].to_numpy() == meta["utt_id"].to_numpy()[jobs[0][0]["test_rows"]]).all()
    jobs[0][1]["test_rows"] = jobs[0][1]["test_rows"][::-1].copy()
    with pytest.raises(AssertionError, match="disagree"):
        R.pooled_ft(task, jobs, y_all, meta)
    v, pv, pt, tr, val = R.headline_variant(task, P["val"], P["test"])
    assert v == "plain" and len(pv) == 60 and len(pt) == 80


def test_report_classification_picks_between_argmax_and_decision_weights_on_validation():
    pytest.importorskip("sklearn")
    from src.evaluation import report_us51 as R
    rng = np.random.default_rng(1)
    n = 200
    meta = pd.DataFrame({"speaker": np.repeat([f"s{i}" for i in range(20)], 10), "age_group": "adult", "utt_id": [f"u{i}" for i in range(n)]})
    task = _task("phone_mispron_t1.0")
    y_all = (rng.random(n) < 0.3).astype(int)
    jobs = _fake_jobs(task, rng, n, meta.speaker.to_numpy())
    for j in jobs[0]:                                        # make the probabilities informative so the choice is meaningful
        for k in ("val", "test", "train"):
            y = y_all[j[f"{k}_rows"]]
            p1 = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, len(y)), 0.01, 0.99)
            j[f"{k}_proba"] = np.stack([1 - p1, p1], 1).astype(np.float32)
        j["train_y"] = y_all[j["train_rows"]]
    P = R.pooled_ft(task, jobs, y_all, meta)
    v, pv, pt, tr, val = R.headline_variant(task, P["val"], P["test"])
    assert v in ("cw", "cw+thr") and set(np.unique(pt)) <= {0, 1} and 0 <= val <= 1 and 0 <= tr <= 1
