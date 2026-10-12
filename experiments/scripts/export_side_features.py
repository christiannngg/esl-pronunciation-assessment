"""
Export the NON-ACOUSTIC side features of the Sprint 3 tables, so frozen-encoder heads can use exactly the same side information
as the MFCC baselines (canonical phone identity, durations, position in word, ...) without shipping the MFCC matrices to Colab.

    python experiments/scripts/export_side_features.py            # -> <data_root>/processed/features/tables/<unit>_side_X.npy (+ _side_names.json)
    python experiments/scripts/export_side_features.py --check    # re-reads and checks shapes / finiteness only

Only columns that carry no MFCC statistics are exported (listed in SIDE_COLUMNS). The L2-ARCTIC tables never contain the
annotated duration (leakage control from Sprint 3), and this script asserts it. Row order is the Sprint 3 order.
Standalone: numpy / pandas / json only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_data_root  # noqa: E402

PHONES = "ph_"                      # canonical-phone one-hot (so762_phone) / anchor-phone one-hot (l2_phone)
SIDE_COLUMNS = {
    "so762_sentence": ["dur", "n_words", "n_phones"],
    "so762_word": ["dur", "n_phones", "phone_dur_mean", "phone_dur_std", "n_chars", "word_pos"],
    "so762_phone": ["dur", "log_dur", "has_prev", "has_next", "word_initial", "word_final", "n_phones_in_word", "is_vowel", PHONES],
    "l2_word": ["dur", "n_canon_phones", "n_chars"],
    "l2_phone": ["has_prev", "has_next", "is_vowel", PHONES],
}
FORBIDDEN_L2 = ("ann_dur", "dur", "log_dur")          # annotated-duration leakage (l2_phone only; l2_word's dur is the word span from non-addition units)


def pick(names: list[str], spec: list[str]) -> list[int]:
    idx = []
    for s in spec:
        idx += [i for i, n in enumerate(names) if n.startswith(PHONES)] if s == PHONES else [names.index(s)]
    return idx


def export(unit: str, tdir: Path) -> tuple[int, int]:
    names = json.loads((tdir / f"{unit}_names.json").read_text())
    X = np.load(tdir / f"{unit}_X.npy", mmap_mode="r")
    cols = pick(names, SIDE_COLUMNS[unit])
    sel = [names[i] for i in cols]
    if unit == "l2_phone":
        assert not any(n in FORBIDDEN_L2 for n in sel), "annotated duration must not reach the L2 phone heads"
    out = np.ascontiguousarray(X[:, cols], dtype=np.float32)
    assert np.isfinite(out).all(), f"{unit}: non-finite side features"
    np.save(tdir / f"{unit}_side_X.npy", out)
    (tdir / f"{unit}_side_names.json").write_text(json.dumps(sel))
    return out.shape


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    tdir = get_data_root() / "processed" / "features" / "tables"
    for unit in SIDE_COLUMNS:
        if a.check:
            X = np.load(tdir / f"{unit}_side_X.npy", mmap_mode="r")
            names = json.loads((tdir / f"{unit}_side_names.json").read_text())
            assert X.shape[1] == len(names) and np.isfinite(np.asarray(X)).all()
            shape = X.shape
        else:
            shape = export(unit, tdir)
        print(f"[side] {unit:15s} {shape}")


if __name__ == "__main__":
    main()
