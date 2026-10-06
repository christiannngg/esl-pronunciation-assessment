"""Unit tests for the US 2.2 preprocessing pipeline (synthetic data, no datasets needed)."""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.preprocessing import alignment as al
from src.preprocessing import splits as sp
from src.preprocessing.audio import TARGET_SR, process_array, process_file
from src.preprocessing.labels import clean_l2arctic_phones, clean_phone


# ------------------------------- audio ------------------------------------ #
def _tone(sr, secs=1.0, amp=0.1, dc=0.0):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * 440 * t) + dc).astype(np.float32)


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100])
def test_resample_length_and_rate(sr):
    y, qc = process_array(_tone(sr), sr)
    assert len(y) == TARGET_SR and qc.duration_s == pytest.approx(1.0, abs=1e-3)


def test_peak_and_dc():
    y, qc = process_array(_tone(16000, amp=0.1, dc=0.05), 16000)
    assert np.abs(y).max() == pytest.approx(0.95, abs=1e-3)
    assert abs(float(y.mean())) < 1e-3


def test_gain_cap_and_low_level_flag():
    y, qc = process_array(_tone(16000, amp=0.001), 16000)
    assert qc.gain_db <= 20.0 + 1e-6 and qc.flag_low_level
    assert np.abs(y).max() < 0.95  # not boosted to full scale


def test_clipping_flag_and_stereo_downmix():
    x = np.stack([_tone(16000, amp=2.0), _tone(16000, amp=2.0)], axis=1)
    x = np.clip(x, -1, 1)
    y, qc = process_array(x, 16000)
    assert y.ndim == 1 and qc.orig_channels == 2 and qc.flag_clipped


def test_no_trimming_of_silence(tmp_path):
    x = np.concatenate([np.zeros(8000), _tone(16000, 0.5), np.zeros(8000)]).astype(np.float32)
    sf.write(tmp_path / "a.wav", x, 16000)
    qc = process_file(tmp_path / "a.wav", tmp_path / "b.wav")
    assert sf.info(tmp_path / "b.wav").frames == len(x) and sf.info(tmp_path / "b.wav").subtype == "PCM_16"
    assert qc.duration_s == pytest.approx(1.5)


def test_silent_file_is_not_amplified():
    y, qc = process_array(np.zeros(16000, np.float32), 16000)
    assert np.abs(y).max() == 0 and qc.flag_low_level


# ------------------------------- labels ----------------------------------- #
@pytest.mark.parametrize("raw,base,dev", [("D_", "D", False), ("ER)", "ER", False), ("AX", "AH", False),
                                          ("R*", "R", True), ("IY1", "IY", False)])
def test_clean_phone(raw, base, dev):
    c = clean_phone(raw)
    assert c.base == base and c.is_deviation == dev


@pytest.mark.parametrize("raw", ["sil", "sp", "spn", "", "err"])
def test_clean_phone_non_phones(raw):
    assert clean_phone(raw).base is None


def test_error_flag_policy():
    df = pd.DataFrame({"cpl": ["AH", "R", "T"], "ppl": ["AH", "R*", "D"], "error_type": ["correct", "substitution", "substitution"],
                       "ppl_is_err": [False, False, False], "ppl_is_deviation": [False, True, False]})
    assert clean_l2arctic_phones(df)["is_error"].tolist() == [False, True, True]
    assert clean_l2arctic_phones(df, count_deviation_as_error=False)["is_error"].tolist() == [False, False, True]


# ------------------------------- splits ----------------------------------- #
def _so_utts():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(40):
        split = "train" if i < 24 else "test"
        for j in range(5):
            rows.append({"speaker": f"{i:04d}", "split": split, "gender": "m" if i % 2 else "f", "age": 10 + i,
                         "age_group": "adult" if i % 3 else "child/teen",
                         **{m: int(rng.integers(3, 10)) for m in sp.SO762_METRICS}})
    return pd.DataFrame(rows)


def test_so762_split_properties():
    u = _so_utts()
    a = sp.build_so762_splits(u, n_candidates=10)
    b = sp.build_so762_splits(u, n_candidates=10)
    assert a.equals(b)  # deterministic
    chk = sp.check_so762(a, u, sp._speaker_table(u))
    assert chk["disjoint"] and chk["test_equals_official_test"] and chk["all_speakers_covered"]
    assert set(a[a["split"] == "test"]["speaker"]) == set(u[u["split"] == "test"]["speaker"])
    assert (a["split"] == "val").sum() >= 1


def _l2_speakers():
    rows = []
    for l1 in ["A", "B", "C", "D", "E", "F"]:
        for g in ["F", "F", "M", "M"]:
            rows.append({"speaker": f"{l1}{len(rows)}", "l1": l1, "gender": g})
    return pd.DataFrame(rows)


def test_l2arctic_folds():
    f = sp.build_l2arctic_folds(_l2_speakers())
    chk = sp.check_l2arctic(f)
    assert chk["all_checks_pass"]
    assert f.groupby("fold").size().tolist() == [6, 6, 6, 6]
    assert sp.build_l2arctic_folds(_l2_speakers()).equals(f)
    for k in range(4):
        r = sp.fold_roles(f, k)
        assert not set(r["train"]) & set(r["test"]) and not set(r["val"]) & set(r["test"])


# ------------------------------ alignment --------------------------------- #
def _write_tg(path, words, phones):
    def tier(name, ivs, k):
        s = f'    item [{k}]:\n        class = "IntervalTier"\n        name = "{name}"\n        xmin = 0\n        xmax = {ivs[-1][1]}\n        intervals: size = {len(ivs)}\n'
        for i, (a, b, t) in enumerate(ivs, 1):
            s += f'        intervals [{i}]:\n            xmin = {a}\n            xmax = {b}\n            text = "{t}"\n'
        return s
    path.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n'
                    + tier("words", words, 1) + tier("phones", phones, 2))


CANON = [{"text": "WE", "phones": ["W", "IY0"], "phone_scores": [2, 1.6], "accuracy": 10, "stress": 10, "total": 10},
         {"text": "GO", "phones": ["G", "OW1"], "phone_scores": [2, 2], "accuracy": 9, "stress": 10, "total": 9}]
WORDS = [(0, 0.1, ""), (0.1, 0.4, "we"), (0.4, 0.8, "go"), (0.8, 1.0, "")]
PHONES = [(0, 0.1, ""), (0.1, 0.2, "W"), (0.2, 0.4, "IY0"), (0.4, 0.55, "G"), (0.55, 0.8, "OW1"), (0.8, 1.0, "")]


def test_ingest_ok(tmp_path):
    _write_tg(tmp_path / "u.TextGrid", WORDS, PHONES)
    w, p, problem = al.ingest_so762_textgrid(tmp_path / "u.TextGrid", "u", CANON)
    assert problem is None and len(w) == 2 and len(p) == 4
    assert [r["phone"] for r in p] == ["W", "IY", "G", "OW"] and p[1]["phone_score"] == 1.6
    assert all(p[i]["end"] <= p[i + 1]["start"] + 1e-9 for i in range(3))


def test_ingest_rejects_wrong_phones_and_word_count(tmp_path):
    bad = PHONES.copy()
    bad[3] = (0.4, 0.55, "K")
    _write_tg(tmp_path / "a.TextGrid", WORDS, bad)
    assert al.ingest_so762_textgrid(tmp_path / "a.TextGrid", "a", CANON)[2].startswith("phone_mismatch")
    _write_tg(tmp_path / "b.TextGrid", WORDS[:2], PHONES)
    assert al.ingest_so762_textgrid(tmp_path / "b.TextGrid", "b", CANON)[2].startswith("word_count")


def test_dictionary_roundtrip(tmp_path):
    canon = {"u1": CANON, "u2": [{**CANON[0], "phones": ["W", "IY2"]}]}
    dic = al.build_so762_dictionary(canon)
    assert dic["we"] == [("W", "IY0"), ("W", "IY2")]
    assert al.word_token("WE", ["W", "IY2"], dic) == "we__1"
    assert al.utt_transcript(CANON, dic) .startswith("we__0")
    al.write_mfa_dictionary(dic, tmp_path / "d.dict")
    assert al.dictionary_phone_inventory(tmp_path / "d.dict") == {"W", "IY0", "IY2", "G", "OW1"}
    assert "we__1\tW IY2" in (tmp_path / "d.dict").read_text()


def test_bare_vowels_get_stress_digit(tmp_path):
    al.write_mfa_dictionary({"hi": [("HH", "IH")]}, tmp_path / "d.dict")
    assert "HH IH0" in (tmp_path / "d.dict").read_text()


def test_ingest_accepts_token_labels(tmp_path):
    words = [(0, 0.1, ""), (0.1, 0.4, "we__0"), (0.4, 0.8, "go__0"), (0.8, 1.0, "")]
    _write_tg(tmp_path / "u.TextGrid", words, PHONES)
    assert al.ingest_so762_textgrid(tmp_path / "u.TextGrid", "u", CANON)[2] is None


def test_boundary_consistency(tmp_path):
    _write_tg(tmp_path / "a.TextGrid", WORDS, PHONES)
    shifted = [(a + (0.01 if 0 < a < 1 else 0), b, t) for a, b, t in PHONES]
    _write_tg(tmp_path / "b.TextGrid", WORDS, shifted)
    r = al.boundary_consistency(tmp_path / "a.TextGrid", tmp_path / "b.TextGrid")
    assert r["n"] == 4 and np.allclose(r["abs_diff_s"], 0.01)
