"""Sprint 4 (US 4.1) tests: frame geometry, pooling, Sprint 3 row alignment, resumable extraction, frozen-encoder wrapper.

Everything except the last two tests runs on a laptop CPU with no torch and no real data:
the unit tables are built from a tiny synthetic data root that has the real file schemas.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import ssl_embeddings as se

REPO = Path(__file__).resolve().parents[1]


# ---- frame geometry --------------------------------------------------------- #
def test_n_frames_known_values():
    assert se.n_frames(16000) == 49                       # 1 s -> 49 frames (20 ms stride, 25 ms window)
    assert se.n_frames(400) == 1
    assert se.n_frames(399) == 0 and se.n_frames(0) == 0
    assert se.n_frames(400 + 320) == 2                    # one more stride -> one more frame
    assert all(se.n_frames(n + 320) == se.n_frames(n) + 1 for n in (400, 8000, 16000, 100000))


def test_unit_frames_uses_frame_centres():
    # centres are 12.5 ms + 20 ms * i, so [0.1, 0.3) holds i = 5..14
    assert tuple(int(v[0]) for v in se.unit_frames(100, 0.1, 0.3)) == (5, 15)
    c = se.frame_centers(100)
    i0, i1 = (int(v[0]) for v in se.unit_frames(100, 0.1, 0.3))
    assert c[i0] >= 0.1 and c[i0 - 1] < 0.1 and c[i1 - 1] < 0.3 and c[i1] >= 0.3


def test_frame_geometry_is_pinned_to_the_conv_front_end():
    # frame i covers samples [320 i, 320 i + 400): centre = (320 i + 200) / 16000 s
    assert se.FIRST_CENTER_S == pytest.approx(0.0125) and se.STRIDE_S == pytest.approx(0.02)
    assert se.frame_centers(3) == pytest.approx([0.0125, 0.0325, 0.0525])
    # [0.11, 0.20) holds centres 0.1125 ... 0.1925 = frames 5..9 (a 0-based centre convention would give 6..9)
    assert tuple(int(v[0]) for v in se.unit_frames(100, 0.11, 0.20)) == (5, 10)


def test_unit_frames_short_empty_and_out_of_range_units_keep_one_frame():
    T = 100
    assert tuple(int(v[0]) for v in se.unit_frames(T, 0.0, 0.01)) == (0, 1)        # no centre inside -> nearest frame
    i0, i1 = (int(v[0]) for v in se.unit_frames(T, 0.5, 0.5))                       # zero-length marker (deleted phone)
    assert i1 - i0 == 1 and abs(se.frame_centers(T)[i0] - 0.5) <= se.STRIDE_S / 2 + 1e-9
    assert tuple(int(v[0]) for v in se.unit_frames(10, 5.0, 6.0)) == (9, 10)       # past the end -> last frame
    assert tuple(int(v[0]) for v in se.unit_frames(10, -2.0, -1.0)) == (0, 1)      # before the start -> first frame
    with pytest.raises(ValueError):
        se.unit_frames(0, 0.0, 1.0)


def test_unit_frames_vectorised_matches_scalar_and_is_valid():
    rng = np.random.default_rng(0)
    T = 60
    t0 = rng.uniform(-0.1, 1.3, 200)
    t1 = t0 + rng.uniform(0.0, 0.4, 200)
    i0, i1 = se.unit_frames(T, t0, t1)
    assert ((0 <= i0) & (i0 < i1) & (i1 <= T)).all()
    for k in range(0, 200, 7):
        a, b = se.unit_frames(T, t0[k], t1[k])
        assert (a[0], b[0]) == (i0[k], i1[k])


# ---- pooling ---------------------------------------------------------------- #
def test_pool_units_matches_naive_mean_per_layer():
    rng = np.random.default_rng(1)
    H = rng.standard_normal((3, 40, 8)).astype(np.float32)
    i0 = np.array([0, 5, 39, 10]); i1 = np.array([40, 6, 40, 30])
    out = se.pool_units(H, i0, i1)
    assert out.shape == (4, 3, 8)
    for u in range(4):
        np.testing.assert_allclose(out[u], H[:, i0[u]:i1[u], :].astype(np.float64).mean(1), rtol=0, atol=1e-12)


def test_pool_units_is_accurate_with_large_offsets():
    rng = np.random.default_rng(2)
    H = (1000.0 + rng.standard_normal((2, 900, 4))).astype(np.float32)
    out = se.pool_units(H, np.array([100]), np.array([107]))
    np.testing.assert_allclose(out[0], H[:, 100:107, :].astype(np.float64).mean(1), rtol=1e-12)


def test_pool_units_max_std_matches_naive_and_single_frame_has_zero_std():
    rng = np.random.default_rng(3)
    H = rng.standard_normal((3, 40, 8)).astype(np.float32)
    i0 = np.array([0, 5, 39, 10]); i1 = np.array([40, 6, 40, 30])
    out = se.pool_units_max_std(H, i0, i1)
    assert out.shape == (4, 3, 16)
    for u in range(4):
        seg = H[:, i0[u]:i1[u], :].astype(np.float64)
        np.testing.assert_allclose(out[u, :, :8], seg.max(1), atol=1e-12)
        np.testing.assert_allclose(out[u, :, 8:], seg.std(1), atol=1e-12)
    assert (out[1, :, 8:] == 0).all() and (out[2, :, 8:] == 0).all()          # one-frame spans: std 0, max = the frame


def test_word_ms_unit_has_the_rows_and_spans_of_the_word_unit(data_root):
    tabs = se.build_tables(["so762_word", "so762_word_ms"], HALF)
    w, ms = tabs["so762_word"], tabs["so762_word_ms"]
    pd.testing.assert_frame_equal(w.df, ms.df)
    assert w.keys_sha256 == ms.keys_sha256 and ms.kind == "segment_ms" and w.kind == "segment"
    assert ms.width(768) == 2 * 768 and w.width(768) == 768


def test_pool_utterance_is_mean_then_population_std():
    rng = np.random.default_rng(3)
    H = rng.standard_normal((3, 25, 6)).astype(np.float32)
    v = se.pool_utterance(H)
    assert v.shape == (3, 12)
    np.testing.assert_allclose(v[:, :6], H.astype(np.float64).mean(1))
    np.testing.assert_allclose(v[:, 6:], H.astype(np.float64).std(1))


def test_fixed_window_matches_sprint3_definition():
    pytest.importorskip("librosa")
    from src.features.aggregate import fixed_window
    a, b = fixed_window(0.4, 0.6, 0.075)
    assert (a, b) == pytest.approx((0.5 - 0.075, 0.5 + 0.075))


# ---- synthetic data root with the real schemas ------------------------------ #
@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    tab, ali = root / "processed" / "features" / "tables", root / "processed" / "alignments"
    tab.mkdir(parents=True); ali.mkdir(parents=True)

    words = pd.DataFrame([("u1", 0, "WE", 0.10, 0.50), ("u1", 1, "CALL", 0.50, 0.90), ("u2", 0, "HI", 0.00, 0.40)],
                         columns=["utt_id", "word_idx", "word", "start", "end"])
    phones = pd.DataFrame([("u1", 0, 0, 0.10, 0.30), ("u1", 0, 1, 0.30, 0.50), ("u1", 1, 0, 0.50, 0.70), ("u1", 1, 1, 0.70, 0.90),
                           ("u2", 0, 0, 0.00, 0.01), ("u2", 0, 1, 0.01, 0.40)],
                          columns=["utt_id", "word_idx", "phone_idx", "start", "end"])
    words.to_csv(ali / "so762_words.csv.gz", index=False)
    phones.to_csv(ali / "so762_phones.csv.gz", index=False)

    rows = [  # phone_idx, start, end, error_class, word_idx, corpus
        (0, 0.00, 0.20, "silence", np.nan, "arctic"), (1, 0.20, 0.30, "correct", 0, "arctic"), (2, 0.30, 0.40, "correct", 0, "arctic"),
        (3, 0.40, 0.45, "addition", np.nan, "arctic"), (4, 0.50, 0.50, "deletion", 1, "arctic"),
        (5, 0.50, 0.70, "substitution", 1, "arctic"), (6, 0.70, 0.90, "silence", np.nan, "arctic"),
        (0, 0.00, 1.00, "correct", 0, "suitcase")]
    clean = pd.DataFrame(rows, columns=["phone_idx", "start", "end", "error_class", "word_idx", "corpus"])
    clean.insert(0, "utt_id", ["A_arctic_a1"] * 7 + ["A_suitcase"])
    clean.insert(0, "speaker", "A")
    clean.to_csv(ali / "l2arctic_phones_clean.csv.gz", index=False)

    # Sprint 3 meta tables - deliberately NOT sorted like the alignment files, so row order must come from the meta
    pd.DataFrame({"utt_id": ["u2", "u1"], "speaker": ["s2", "s1"]}).to_csv(tab / "so762_sentence_meta.csv.gz", index=False)
    pd.DataFrame({"utt_id": ["u1", "u2", "u1"], "word_idx": [1, 0, 0], "speaker": ["s1", "s2", "s1"]}).to_csv(tab / "so762_word_meta.csv.gz", index=False)
    pd.DataFrame({"utt_id": ["u2", "u1", "u1", "u2", "u1", "u1"], "word_idx": [0, 1, 0, 0, 0, 1], "phone_idx": [1, 1, 0, 0, 1, 0],
                  "speaker": ["s2", "s1", "s1", "s2", "s1", "s1"]}).to_csv(tab / "so762_phone_meta.csv.gz", index=False)
    pd.DataFrame({"utt_id": ["A_arctic_a1"] * 5, "speaker": "A", "phone_idx": [1, 2, 4, 5, 3]}).to_csv(tab / "l2_phone_meta.csv.gz", index=False)
    pd.DataFrame({"utt_id": ["A_arctic_a1"] * 2, "speaker": "A", "word_idx": [0, 1]}).to_csv(tab / "l2_word_meta.csv.gz", index=False)
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


HALF = 0.075


def test_unit_tables_follow_meta_order_and_take_times_from_alignments(data_root):
    t = se.build_tables(se.UNITS, HALF)
    s = t["so762_sentence"].df
    assert list(s["utt_id"]) == ["u2", "u1"] and s[["t0", "t1"]].isna().all().all()
    w = t["so762_word"].df
    assert list(zip(w.utt_id, w.word_idx)) == [("u1", 1), ("u2", 0), ("u1", 0)]
    assert list(zip(w.t0, w.t1)) == [(0.5, 0.9), (0.0, 0.4), (0.1, 0.5)]
    p = t["so762_phone"].df
    assert list(zip(p.utt_id, p.word_idx, p.phone_idx)) == [("u2", 0, 1), ("u1", 1, 1), ("u1", 0, 0), ("u2", 0, 0), ("u1", 0, 1), ("u1", 1, 0)]
    assert list(zip(p.t0, p.t1)) == [(0.01, 0.4), (0.7, 0.9), (0.1, 0.3), (0.0, 0.01), (0.3, 0.5), (0.5, 0.7)]


def test_l2_units_use_fixed_window_and_non_addition_word_spans(data_root):
    t = se.build_tables(se.UNITS, HALF)
    p = t["l2_phone"].df
    assert list(p["phone_idx"]) == [1, 2, 4, 5, 3]                      # meta order; silence and suitcase rows are not in the table
    mid = {1: 0.25, 2: 0.35, 4: 0.50, 5: 0.60, 3: 0.425}
    for r in p.itertuples():
        assert r.t0 == pytest.approx(mid[r.phone_idx] - HALF) and r.t1 == pytest.approx(mid[r.phone_idx] + HALF)
    assert (p["t1"] - p["t0"]).round(9).eq(2 * HALF).all()              # constant width: the annotated duration is not visible
    w = t["l2_word"].df
    # word 0 = the two correct phones; word 1 = deleted marker + substitution; the addition never extends a span
    assert list(zip(w.word_idx, w.t0, w.t1)) == [(0, 0.2, 0.4), (1, 0.5, 0.7)]


def test_unit_tables_fail_loudly_when_an_alignment_row_is_missing(data_root):
    ali = data_root / "processed" / "alignments" / "so762_phones.csv.gz"
    df = pd.read_csv(ali, dtype={"utt_id": str})
    df[~((df.utt_id == "u1") & (df.word_idx == 1) & (df.phone_idx == 1))].to_csv(ali, index=False)
    with pytest.raises(ValueError, match="no alignment interval"):
        se.build_unit_table("so762_phone", HALF)


def test_keys_hash_changes_with_row_order(data_root):
    a = se.build_unit_table("so762_word", HALF).keys_sha256
    b = se.build_unit_table("so762_word", HALF).keys_sha256
    assert a == b
    m = data_root / "processed" / "features" / "tables" / "so762_word_meta.csv.gz"
    df = pd.read_csv(m, dtype={"utt_id": str, "speaker": str})
    df.iloc[::-1].to_csv(m, index=False)
    assert se.build_unit_table("so762_word", HALF).keys_sha256 != a


def test_utterance_list_is_deterministic_and_limit_is_per_dataset(data_root):
    t = se.build_tables(se.UNITS, HALF)
    u = se.utterance_list(t)
    assert list(zip(u.dataset, u.utt_id)) == [("so762", "u2"), ("so762", "u1"), ("l2arctic", "A_arctic_a1")]
    assert list(se.utterance_list(t, limit=1).utt_id) == ["u2", "A_arctic_a1"]


# ---- extraction core: fake encoder, naive reference, resume ----------------- #
L, D = 3, 5


def _wav(dataset, utt_id, speaker):
    return np.random.default_rng(sum(map(ord, utt_id))).standard_normal(16000).astype(np.float32)


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, wav):
        self.calls.append(float(wav[:4].sum()))
        T = se.n_frames(len(wav))
        return np.random.default_rng(int(abs(wav[:4].sum()) * 1e6) % 2**32).standard_normal((L, T, D)).astype(np.float32)


def _fp(tables):
    return {"code": "test", "keys": {n: t.keys_sha256 for n, t in tables.items()}}


def test_run_extraction_matches_naive_pooling(data_root, tmp_path):
    tables = se.build_tables(se.UNITS, HALF)
    utts = se.utterance_list(tables)
    enc = FakeEncoder()
    out = tmp_path / "cache"
    st = se.run_extraction(tables, utts, enc, _wav, out, n_layers=L, dim=D, shard_size=2, fingerprint=_fp(tables))
    assert st["complete"] and st["utterances"] == 3 and len(enc.calls) == 3
    # naive reference for every row of every unit
    for name, t in tables.items():
        X = np.load(out / f"{name}_X.npy")
        assert X.dtype == np.float16 and X.shape == (len(t.df), L, t.width(D))
        for k, r in enumerate(t.df.itertuples()):
            wav = _wav(None, r.utt_id, None)
            H = FakeEncoder()(wav).astype(np.float64)
            if t.kind == "utt":
                ref = np.concatenate([H.mean(1), H.std(1)], -1)
            elif t.kind == "segment_ms":
                a, b = (int(v[0]) for v in se.unit_frames(H.shape[1], r.t0, r.t1))
                ref = np.concatenate([H[:, a:b, :].max(1), H[:, a:b, :].std(1)], -1)
            else:
                a, b = (int(v[0]) for v in se.unit_frames(H.shape[1], r.t0, r.t1))
                ref = H[:, a:b, :].mean(1)
            np.testing.assert_allclose(X[k], ref, atol=2e-3, rtol=2e-3, err_msg=f"{name} row {k}")
    rep = se.verify_cache(out, tables, D, L)
    assert all(r["shape_ok"] and r["non_finite_rows"] == 0 and r["unfilled_rows"] == 0 for r in rep.values())


def test_run_extraction_resumes_after_a_disconnect_without_redoing_shards(data_root, tmp_path):
    tables = se.build_tables(se.UNITS, HALF)
    utts = se.utterance_list(tables)
    full = tmp_path / "full"
    se.run_extraction(tables, utts, FakeEncoder(), _wav, full, n_layers=L, dim=D, shard_size=1, fingerprint=_fp(tables))

    part = tmp_path / "part"
    e1 = FakeEncoder()
    s1 = se.run_extraction(tables, utts, e1, _wav, part, n_layers=L, dim=D, shard_size=1, fingerprint=_fp(tables), stop_after_shards=1)
    assert not s1["complete"] and len(e1.calls) == 1
    assert json.loads((part / "progress.json").read_text())["done_shards"] == [0]
    assert not all(r["unfilled_rows"] == 0 for r in se.verify_cache(part, tables, D, L).values())      # the cache really is partial

    e2 = FakeEncoder()
    s2 = se.run_extraction(tables, utts, e2, _wav, part, n_layers=L, dim=D, shard_size=1, fingerprint=_fp(tables))
    assert s2["complete"] and len(e2.calls) == 2 and not set(e1.calls) & set(e2.calls)            # finished shard is not recomputed
    for name in tables:
        np.testing.assert_array_equal(np.load(full / f"{name}_X.npy"), np.load(part / f"{name}_X.npy"))


def test_run_extraction_refuses_to_mix_settings(data_root, tmp_path):
    tables = se.build_tables(se.UNITS, HALF)
    utts = se.utterance_list(tables)
    out = tmp_path / "c"
    se.run_extraction(tables, utts, FakeEncoder(), _wav, out, n_layers=L, dim=D, shard_size=5, fingerprint=_fp(tables))
    with pytest.raises(RuntimeError, match="different settings"):
        se.run_extraction(tables, utts, FakeEncoder(), _wav, out, n_layers=L, dim=D, shard_size=5, fingerprint={**_fp(tables), "code": "other"})


def test_run_extraction_rejects_a_frame_count_that_disagrees_with_the_geometry(data_root, tmp_path):
    tables = se.build_tables(se.UNITS, HALF)
    utts = se.utterance_list(tables)
    bad = lambda wav: np.zeros((L, se.n_frames(len(wav)) + 1, D), np.float32)
    with pytest.raises(ValueError, match="frames, expected"):
        se.run_extraction(tables, utts, bad, _wav, tmp_path / "x", n_layers=L, dim=D, fingerprint=_fp(tables))


def test_smoke_verify_only_expects_the_limited_utterances(data_root, tmp_path):
    tables = se.build_tables(se.UNITS, HALF)
    utts = se.utterance_list(tables, limit=1)
    out = tmp_path / "smoke"
    se.run_extraction(tables, utts, FakeEncoder(), _wav, out, n_layers=L, dim=D, shard_size=5, fingerprint=_fp(tables))
    full_view = se.verify_cache(out, tables, D, L)
    assert any(r["unfilled_rows"] > 0 for r in full_view.values())                                   # rows of skipped utterances
    smoke_view = se.verify_cache(out, tables, D, L, utts)
    assert all(r["unfilled_rows"] == 0 and r["shape_ok"] for r in smoke_view.values())


# ---- real data (runs where the Sprint 2/3 tables exist: your laptop, or Colab after unpacking the pack) ----------- #
def _real_data_present():
    from src.utils.paths import get_data_root
    r = get_data_root() / "processed"
    return (r / "alignments" / "so762_words.csv.gz").exists() and (r / "features" / "tables" / "l2_phone_meta.csv.gz").exists()


@pytest.mark.skipif(not _real_data_present(), reason="processed Sprint 2/3 tables not found under the data root")
def test_real_unit_tables_resolve_every_sprint3_row():
    cfg = se.load_config()
    tables = se.build_tables(cfg["units"], cfg["l2_window_half_s"])
    for name, t in tables.items():
        meta = se._read_meta(name)
        assert len(t.df) == len(meta)
        assert (t.df["utt_id"].to_numpy() == meta["utt_id"].to_numpy()).all()
        if t.kind == "segment":
            assert t.df[["t0", "t1"]].notna().all().all() and (t.df["t1"] > t.df["t0"]).all()
    assert len(se.utterance_list(tables)) >= 5000


# ---- the real wrapper around transformers (tiny random model saved locally, so no download) ---------------------- #
def test_frozen_encoder_wrapper_shapes_frame_counts_and_freezing(tmp_path):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    cfg = tf.Wav2Vec2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                            conv_dim=(8,) * 7, conv_kernel=se.CONV_KERNEL, conv_stride=se.CONV_STRIDE,
                            num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2, feat_extract_norm="group")
    torch.manual_seed(0)
    tf.Wav2Vec2Model(cfg).save_pretrained(tmp_path)
    tf.Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True).save_pretrained(tmp_path)

    enc = se.FrozenEncoder(str(tmp_path), None, "cpu")
    assert (enc.n_layers, enc.dim) == (3, 16) and enc.kernel == se.CONV_KERNEL and enc.stride == se.CONV_STRIDE
    assert not any(p.requires_grad for p in enc.model.parameters()) and not enc.model.training
    for n in (400, 1234, 16000, 31999):
        wav = np.random.default_rng(n).standard_normal(n).astype(np.float32)
        H = enc(wav)
        assert H.shape == (3, se.n_frames(n), 16) and H.dtype == np.float32 and np.isfinite(H).all()
    np.testing.assert_array_equal(enc(wav), enc(wav))                     # deterministic in eval mode


# ---- Colab data packer -------------------------------------------------------------------------------------------- #
def _load_packer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pack_colab_data", REPO / "experiments" / "scripts" / "pack_colab_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pack_collects_exactly_the_files_the_extraction_needs(data_root):
    aud = data_root / "processed" / "audio"
    (aud / "so762").mkdir(parents=True); (aud / "l2arctic" / "A").mkdir(parents=True)
    for u in ("u1", "u2"):
        (aud / "so762" / f"{u}.wav").write_bytes(b"x")
    (aud / "l2arctic" / "A" / "A_arctic_a1.wav").write_bytes(b"x")
    (aud / "l2arctic" / "A" / "A_arctic_unused.wav").write_bytes(b"x")                 # not in any Sprint 3 table -> not packed
    names = sorted(n for _, n in _load_packer().collect(data_root))
    assert "processed/audio/l2arctic/A/A_arctic_a1.wav" in names and "processed/audio/so762/u1.wav" in names
    assert not any("unused" in n for n in names)
    assert sum(n.endswith("_meta.csv.gz") for n in names) == 5 and "processed/alignments/so762_words.csv.gz" in names
    (aud / "so762" / "u2.wav").unlink()
    with pytest.raises(FileNotFoundError):
        _load_packer().collect(data_root)


def test_pros_pack_checks_row_alignment_before_anything_is_uploaded(data_root):
    tab = data_root / "processed" / "features" / "tables"
    packer = _load_packer()
    for n in packer.UNITS:
        meta = pd.read_csv(tab / f"{n}_meta.csv.gz", dtype=str)
        meta.to_csv(tab / f"{n}_pros_meta.csv.gz", index=False)
        np.save(tab / f"{n}_pros_X.npy", np.zeros((len(meta), 2), np.float32))
        (tab / f"{n}_pros_names.json").write_text("[]")
    names = sorted(n for _, n in packer.collect_pros(data_root))
    assert len(names) == 10 and "processed/features/tables/so762_word_pros_X.npy" in names
    pd.read_csv(tab / "so762_word_meta.csv.gz", dtype=str).iloc[::-1].to_csv(tab / "so762_word_pros_meta.csv.gz", index=False)     # same rows, other order
    with pytest.raises(ValueError, match="not aligned"):
        packer.collect_pros(data_root)
