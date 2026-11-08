"""Serving side features == the saved training side features (Sprint 3 tables), for the same alignment. Skips when the data tables are absent."""
import json

import numpy as np
import pandas as pd
import pytest

from src.serving import side_features as SF
from src.utils.paths import get_data_root

DATA = get_data_root() / "processed"
TABLES, ALI = DATA / "features" / "tables", DATA / "alignments"
pytestmark = pytest.mark.skipif(not (TABLES / "so762_phone_side_X.npy").exists() or not (ALI / "so762_phones.csv.gz").exists(), reason="processed SO762 tables not available")


def _words_of(utt, W, P):
    out = []
    for w in W[W.utt_id == utt].sort_values("word_idx").itertuples():
        ph = P[(P.utt_id == utt) & (P.word_idx == w.word_idx)].sort_values("phone_idx")
        out.append({"word": w.word, "start": w.start, "end": w.end,
                    "phones": [{"phone": r.phone, "start": r.start, "end": r.end} for r in ph.itertuples()]})
    return out


@pytest.fixture(scope="module")
def tables():
    W = pd.read_csv(ALI / "so762_words.csv.gz", dtype={"utt_id": str})
    P = pd.read_csv(ALI / "so762_phones.csv.gz", dtype={"utt_id": str})
    wm = pd.read_csv(TABLES / "so762_word_meta.csv.gz", dtype={"utt_id": str})
    pm = pd.read_csv(TABLES / "so762_phone_meta.csv.gz", dtype={"utt_id": str})
    return W, P, wm, pm, np.load(TABLES / "so762_word_side_X.npy"), np.load(TABLES / "so762_phone_side_X.npy")


def test_column_names_match_saved_tables():
    assert json.loads((TABLES / "so762_sentence_side_names.json").read_text()) == SF.SENTENCE_COLUMNS
    assert json.loads((TABLES / "so762_word_side_names.json").read_text()) == SF.WORD_COLUMNS
    assert json.loads((TABLES / "so762_phone_side_names.json").read_text()) == SF.PHONE_COLUMNS


def test_word_and_phone_side_match_saved_rows(tables):
    W, P, wm, pm, WX, PX = tables
    utts = sorted(wm.utt_id.unique())[::25]                              # every 25th utterance (~100), all splits
    wi, pi = wm.groupby("utt_id").indices, pm.groupby("utt_id").indices
    for u in utts:
        words = _words_of(u, W, P)
        np.testing.assert_allclose(SF.word_side(words), WX[wi[u]], rtol=1e-5, atol=1e-5, err_msg=f"word side, utt {u}")
        np.testing.assert_allclose(SF.phone_side(words), PX[pi[u]], rtol=1e-5, atol=1e-5, err_msg=f"phone side, utt {u}")


def test_sentence_duration_matches_saved_rows():
    pytest.importorskip("soundfile")
    import soundfile as sf
    meta = pd.read_csv(TABLES / "so762_sentence_meta.csv.gz", dtype={"utt_id": str})
    X = np.load(TABLES / "so762_sentence_side_X.npy")
    for i in range(0, len(meta), 100):
        u = meta.utt_id[i]
        n = sf.info(DATA / "audio" / "so762" / f"{u}.wav").frames
        np.testing.assert_allclose(SF.sentence_side(n, X[i, 1], X[i, 2])[0], X[i], rtol=1e-5, atol=1e-5, err_msg=f"utt {u}")
