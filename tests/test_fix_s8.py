"""US 8.3 building blocks: CV folds, F1 weights, F2 bank (numpy only)."""
import numpy as np
import pandas as pd
import pytest

from src.models import fix_s8 as F


def speakers(n_adult=30, n_child=25, n_test=20):
    rows = [(f"{i:04d}", "train" if i % 5 else "val", "adult") for i in range(n_adult)]
    rows += [(f"{1000 + i:04d}", "train" if i % 5 else "val", "child/teen") for i in range(n_child)]
    rows += [(f"{2000 + i:04d}", "test", "adult" if i % 2 else "child/teen") for i in range(n_test)]
    return pd.DataFrame(rows, columns=["speaker", "split", "age_group"])


def test_folds_are_speaker_grouped_stratified_and_exclude_test():
    sp = speakers()
    df = F.make_folds(sp)
    F.check_folds(df)
    assert len(df) == 55 and not df["speaker"].str.startswith("2").any()          # no official test speaker
    assert df.groupby(["cv_fold", "age_group"]).size().unstack().max().max() - df.groupby(["cv_fold", "age_group"]).size().unstack().min().min() <= 1
    for k in range(F.N_FOLDS):
        role = df[f"role_{k}"]
        assert (df.loc[role == "test", "cv_fold"] == k).all()                       # held-out fold = test role
        assert (df.loc[role != "test", "cv_fold"] != k).all()                       # every other speaker trains or validates
    # deterministic
    assert F.make_folds(sp).equals(df)
    assert not F.make_folds(sp, seed=9).equals(df)


def test_fold_split_file_marks_official_test_as_excluded(tmp_path):
    sp = speakers()
    df = F.make_folds(sp)
    p = F.fold_split_csv(2, tmp_path, folds=df, speakers=sp)
    out = pd.read_csv(p, dtype={"speaker": str})
    assert set(out["split"]) == {"train", "val", "test", "excluded"}
    assert (out.loc[out["speaker"].str.startswith("2"), "split"] == "excluded").all()
    assert (out["speaker"] == sp["speaker"]).all()                                  # same speaker order / set as the official file


def test_row_weights_equalise_groups_within_each_label_and_keep_label_mass():
    group = np.array(["adult"] * 8 + ["child/teen"] * 2 + ["adult"] * 3 + ["child/teen"] * 3)
    label = np.array([0] * 10 + [1] * 6)
    train = np.ones(len(group), bool)
    w1 = F.row_weights(group, label, train, 1.0)
    for l in (0, 1):
        m = label == l
        assert w1[m].mean() == pytest.approx(1.0)                                   # label mass unchanged
        a, c = w1[m & (group == "adult")].sum(), w1[m & (group == "child/teen")].sum()
        assert a == pytest.approx(c)                                                # strength 1: groups equal inside the label
    assert (F.row_weights(group, label, train, 0.0) == 1.0).all()                   # strength 0: nothing changes
    w5 = F.row_weights(group, label, train, 0.5)
    assert w5[(label == 0) & (group == "child/teen")][0] < w1[(label == 0) & (group == "child/teen")][0]    # half strength is milder
    assert w5[(label == 0) & (group == "child/teen")][0] > 1.0


def test_row_weights_use_training_rows_only():
    group = np.array(["adult", "adult", "child/teen", "child/teen", "adult"])
    label = np.zeros(5, int)
    train = np.array([True, True, True, False, False])
    w = F.row_weights(group, label, train, 1.0)
    assert w[3] == 1.0 and w[4] == 1.0                                              # rows outside the training mask are neutral
    assert w[2] > w[0]                                                              # the lone child/teen training row outweighs an adult row
    # a held-out row must not change the training weights
    w2 = F.row_weights(group[:3], label[:3], np.ones(3, bool), 1.0)
    assert np.allclose(w[:3], w2)


def test_label_bins_edges_from_training_rows_only():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    train = np.array([True] * 6 + [False])
    b = F.label_bins(y, train, 3)
    assert b[-1] == 2 and set(b[:6]) == {0, 1, 2}


def test_aug_bank_swaps_with_probability_and_requires_equal_length(tmp_path):
    d = tmp_path / "bank"
    d.mkdir()
    np.save(d / "u1.npy", np.ones(10, np.float32))
    np.save(d / "bad.npy", np.ones(7, np.float32))
    rng = np.random.default_rng(0)
    bank = F.AugBank(tmp_path, "bank", 1.0)
    orig = np.zeros(10, np.float32)
    assert (bank.maybe("u1", orig, rng) == 1).all() and bank.used == 1
    assert (bank.maybe("not_there", orig, rng) == 0).all()                          # child/teen or unknown utterances are untouched
    with pytest.raises(ValueError):
        bank.maybe("bad", orig, rng)
    never = F.AugBank(tmp_path, "bank", 0.0)
    assert (never.maybe("u1", orig, rng) == 0).all()
    with pytest.raises(FileNotFoundError):
        F.AugBank(tmp_path, "missing", 0.5)


def _load_precompute():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("precompute_aug_s8", Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "precompute_aug_s8.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_precompute_helpers_keep_length_and_are_deterministic():
    m = _load_precompute()
    assert len(m.fit_length(np.ones(90, np.float32), 100)) == 100 and (m.fit_length(np.ones(90, np.float32), 100)[90:] == 0).all()
    assert len(m.fit_length(np.ones(120, np.float32), 100)) == 100
    a, b = m.semitones_for("000010012", "shift_2_4", 2.0, 4.0), m.semitones_for("000010012", "shift_2_4", 2.0, 4.0)
    assert a == b and 2.0 <= a <= 4.0
    assert m.semitones_for("000010012", "shift_4_6", 4.0, 6.0) >= 4.0                  # shifts are always upward (child-like)
    assert len({m.semitones_for(str(i), "shift_4_6", 4.0, 6.0) for i in range(50)}) > 40  # drawn per utterance, not constant
