"""
Bundle the processed files Colab needs for Sprint 4 into ONE tar (no raw data, no MFCC/prosody caches).

    python experiments/scripts/pack_colab_data.py --dry-run
    python experiments/scripts/pack_colab_data.py                     # -> dist/esl_colab_pack.tar  (dist/ is gitignored)

Upload the tar to Drive (MyDrive/esl-pronunciation-data/), then the Colab notebook unpacks it onto Colab's local
disk and sets DATA_ROOT to the unpacked folder. Archive paths are relative to the data root:

    processed/audio/so762/<utt>.wav                       every SO762 utterance
    processed/audio/l2arctic/<speaker>/<utt>.wav           only the annotated utterances that have Sprint 3 rows
    processed/alignments/*                                 MFA / cleaned L2-ARCTIC alignment tables
    processed/features/tables/<unit>_meta.csv.gz           Sprint 3 row order, keys and labels (not the MFCC feature matrices)

Splits and configs come from the git repo itself, not from this pack.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # run as a plain script from anywhere
from src.utils.paths import get_data_root  # noqa: E402

UNITS = ("so762_sentence", "so762_word", "so762_phone", "l2_word", "l2_phone")


def collect(root: Path) -> list[tuple[Path, str]]:
    tab, ali, aud = root / "processed" / "features" / "tables", root / "processed" / "alignments", root / "processed" / "audio"
    files: list[tuple[Path, str]] = []

    so = pd.read_csv(tab / "so762_sentence_meta.csv.gz", usecols=["utt_id"], dtype=str)
    files += [(aud / "so762" / f"{u}.wav", f"processed/audio/so762/{u}.wav") for u in so["utt_id"]]

    l2 = pd.concat([pd.read_csv(tab / f"{n}_meta.csv.gz", usecols=["utt_id", "speaker"], dtype=str) for n in ("l2_phone", "l2_word")])
    l2 = l2.drop_duplicates()
    files += [(aud / "l2arctic" / s / f"{u}.wav", f"processed/audio/l2arctic/{s}/{u}.wav") for u, s in zip(l2["utt_id"], l2["speaker"])]

    files += [(p, f"processed/alignments/{p.name}") for p in sorted(ali.iterdir()) if p.is_file()]
    files += [(tab / f"{n}_meta.csv.gz", f"processed/features/tables/{n}_meta.csv.gz") for n in UNITS]

    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[:3]}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist/esl_colab_pack.tar")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    root = get_data_root()
    files = collect(root)
    total = sum(p.stat().st_size for p, _ in files)
    n_wav = sum(1 for _, n in files if n.endswith(".wav"))
    print(f"[pack] data root: {root}\n[pack] {len(files)} files ({n_wav} wav), {total / 1e9:.2f} GB")
    if a.dry_run:
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with tarfile.open(out, "w") as tar:                 # plain tar: wav files do not compress, and it unpacks fastest
        manifest = json.dumps({"files": len(files), "wav": n_wav, "bytes": total, "created": time.strftime("%Y-%m-%d %H:%M:%S")}).encode()
        info = tarfile.TarInfo("PACK_MANIFEST.json")
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
        for p, name in files:
            tar.add(p, arcname=name, recursive=False)
    print(f"[pack] wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
