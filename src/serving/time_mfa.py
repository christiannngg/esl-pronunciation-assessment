"""
Sprint 9 alignment spike: how long does ONE utterance take through MFA `align_one`, and does it reproduce the training alignment?

    python -m src.serving.time_mfa                 # 5 utterances
    python -m src.serving.time_mfa --n 8 --show-cmd

For each chosen SO762 utterance (from data/processed/mfa/so762_corpus: wav + token transcript, as used in training) this runs
    mfa align_one <wav> <lab> data/processed/mfa/so762_lexicon.dict english_us_arpa <out.TextGrid> --beam 100 --retry_beam 400
in a temporary folder, prints the wall time of every call (the first one is cold), and compares the phone boundaries with the TextGrid made during
training (data/processed/mfa/so762_aligned). The ingest step is ``alignment.ingest_so762_textgrid`` (same checks as training).
Needs the `mfa` command (your esl-pronunciation env). Nothing is written outside the temp folder.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from src.preprocessing import alignment as AL
from src.utils.paths import get_data_root

MFA = get_data_root() / "processed" / "mfa"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--show-cmd", action="store_true")
    ap.add_argument("--utts", nargs="+", default=None, help="explicit utterance ids instead of an evenly spaced sample")
    ap.add_argument("--keep", default=None, help="folder to keep the new TextGrids in (for inspection)")
    a = ap.parse_args(argv)
    if shutil.which("mfa") is None:
        raise SystemExit("`mfa` not found: activate the esl-pronunciation conda env")
    labs = sorted((MFA / "so762_corpus").rglob("*.lab"))
    pick = [l for l in labs if l.stem in set(a.utts)] if a.utts else [labs[i] for i in np.linspace(0, len(labs) - 1, a.n).astype(int)]
    times, diffs, meds = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, lab in enumerate(pick):
            utt = lab.stem
            wav = lab.with_suffix(".wav")
            out = tmp / f"{utt}.TextGrid"
            cmd = ["mfa", "align_one", str(wav.resolve()), str(lab), str(MFA / "so762_lexicon.dict"), "english_us_arpa", str(out),
                   "--beam", "100", "--retry_beam", "400"]
            if a.show_cmd and k == 0:
                print(" ".join(cmd))
            t = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.perf_counter() - t
            if r.returncode != 0 or not out.exists():
                print(f"[{utt}] FAILED in {dt:.1f}s\n{(r.stderr or r.stdout)[-1500:]}")
                continue
            times.append(dt)
            ref = next((MFA / "so762_aligned").rglob(f"{utt}.TextGrid"), None)
            msg = ""
            if ref is not None:
                try:
                    ours, theirs = AL._phone_intervals(out), AL._phone_intervals(ref)
                    if [p[2] for p in ours] == [p[2] for p in theirs]:
                        d = np.abs(np.array([[o[0] - t_[0], o[1] - t_[1]] for o, t_ in zip(ours, theirs)])) * 1000
                        diffs.append(d.max())
                        meds.append(float(np.median(d)))
                        msg = f"phones {len(ours)}, max boundary diff vs training TextGrid {d.max():.1f} ms (median {np.median(d):.1f})"
                    else:
                        msg = "phone sequence differs from the training TextGrid"
                except Exception as e:
                    msg = f"comparison failed: {e}"
            print(f"[{utt}] {dt:5.1f}s {'(cold)' if k == 0 else ''} {msg}")
            if a.keep:
                Path(a.keep).mkdir(parents=True, exist_ok=True)
                shutil.copy(out, Path(a.keep) / out.name)
    if times:
        warm = times[1:] or times
        print(f"\nwall time per call: first {times[0]:.1f}s, warm median {np.median(warm):.1f}s, max {max(times):.1f}s ({len(times)} ok of {len(pick)})")
    if diffs:
        d = np.array(diffs)
        print(f"compared {len(d)} utterances: max boundary diff  median {np.median(d):.0f} ms, p90 {np.percentile(d, 90):.0f} ms, worst {d.max():.0f} ms")
        print(f"  utterances with max diff <= 30 ms: {(d <= 30).sum()}, 30-100 ms: {((d > 30) & (d <= 100)).sum()}, > 100 ms: {(d > 100).sum()}")


if __name__ == "__main__":
    main()
