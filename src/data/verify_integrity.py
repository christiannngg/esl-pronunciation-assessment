"""
Integrity verification for SpeechOcean762 and L2-ARCTIC after download.

Checks file counts against the known-good totals published by each
dataset's maintainers, so a partial/corrupt download is caught before
Sprint 2 preprocessing builds on top of it.

Usage:
    python src/data/verify_integrity.py
    python src/data/verify_integrity.py --dataset speechocean762
    python src/data/verify_integrity.py --dataset l2arctic
"""

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import speechocean762_root, l2arctic_root  # noqa: E402

# NOTE: The paper/README describe "5000 English sentences" as a rounded
# headline figure. The actual repo (as of this writing) contains 5245 WAV
# files, confirmed against a real clone — don't fail on utterance count
# mismatches, just report the actual total for visibility.
SPEECHOCEAN_EXPECTED_SPEAKERS = 250

# Ground truth from https://psi.engr.tamu.edu/l2-arctic-corpus-docs (v5.0 "File summary" table)
L2ARCTIC_EXPECTED_WAV_COUNTS = {
    "ABA": 1129, "SKA": 974, "YBAA": 1130, "ZHAA": 1132,
    "BWC": 1130, "LXC": 1131, "NCC": 1131, "TXHC": 1132,
    "ASI": 1131, "RRBI": 1130, "SVBI": 1132, "TNI": 1131,
    "HJK": 1131, "HKK": 1131, "YDCK": 1131, "YKWK": 1131,
    "EBVS": 1007, "ERMS": 1132, "MBMPS": 1132, "NJS": 1131,
    "HQTV": 1132, "PNV": 1132, "THV": 1132, "TLV": 1132,
}
L2ARCTIC_EXPECTED_ANNOTATION_COUNTS = {
    "ABA": 150, "SKA": 150, "YBAA": 149, "ZHAA": 150,
    "BWC": 150, "LXC": 150, "NCC": 150, "TXHC": 150,
    "ASI": 150, "RRBI": 150, "SVBI": 150, "TNI": 150,
    "HJK": 150, "HKK": 150, "YDCK": 150, "YKWK": 150,
    "EBVS": 150, "ERMS": 150, "MBMPS": 150, "NJS": 150,
    "HQTV": 150, "PNV": 150, "THV": 150, "TLV": 150,
}
L2ARCTIC_TOTAL_WAV = 26867
L2ARCTIC_TOTAL_ANNOTATIONS = 3599


def _is_readable_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() > 0
    except Exception:
        return False


def verify_speechocean762(sample_wav_check: int = 25) -> bool:
    root = speechocean762_root()
    print(f"\n=== SpeechOcean762 ({root}) ===")
    if not root.exists():
        print(f"  FAIL: directory does not exist. Did you clone it here?")
        return False

    ok = True

    # scores.json / scores-detail.json live in resource/, not the repo root
    # (the README's example tree is misleading on this point).
    scores_path = root / "resource" / "scores.json"
    if not scores_path.exists():
        print(f"  FAIL: scores.json missing (looked in {scores_path})")
        ok = False
    else:
        with open(scores_path) as f:
            scores = json.load(f)
        print(f"  OK: scores.json has {len(scores)} utterances")
        n_utterances = len(scores)

    wave_dir = root / "WAVE"
    if not wave_dir.exists():
        print("  FAIL: WAVE/ directory missing")
        ok = False
    else:
        speaker_dirs = [d for d in wave_dir.iterdir() if d.is_dir()]
        if len(speaker_dirs) != SPEECHOCEAN_EXPECTED_SPEAKERS:
            print(f"  FAIL: found {len(speaker_dirs)} speaker dirs in WAVE/, expected {SPEECHOCEAN_EXPECTED_SPEAKERS}")
            ok = False
        else:
            print(f"  OK: {len(speaker_dirs)} speaker directories in WAVE/")

        # Glob case-insensitively (repo uses ".WAV") in a single pass rather
        # than two separate patterns, and report the total rather than fail
        # against the paper's rounded "5000" figure — the actual repo has
        # 5245 WAV files as of this writing, confirmed against a real clone.
        all_wavs = [p for p in wave_dir.rglob("*") if p.suffix.lower() == ".wav"]
        print(f"  INFO: {len(all_wavs)} .WAV files found (paper cites ~5000 as a rounded figure)")
        if scores_path.exists() and len(all_wavs) != n_utterances:
            print(f"  WARNING: WAV count ({len(all_wavs)}) doesn't match scores.json utterance count ({n_utterances})")

        sample = all_wavs[:sample_wav_check]
        bad = [str(p) for p in sample if not _is_readable_wav(p)]
        if bad:
            print(f"  FAIL: {len(bad)}/{len(sample)} sampled WAV files unreadable: {bad[:5]}")
            ok = False
        else:
            print(f"  OK: sampled {len(sample)} WAV files, all readable")

    for split in ("train", "test"):
        split_dir = root / split
        required = {"spk2age", "spk2gender", "spk2utt", "text", "utt2spk", "wav.scp"}
        missing = required - {p.name for p in split_dir.glob("*")} if split_dir.exists() else required
        if missing:
            print(f"  FAIL: {split}/ missing files: {sorted(missing)}")
            ok = False
        else:
            print(f"  OK: {split}/ has all expected Kaldi-style files")

    return ok


def verify_l2arctic(sample_wav_check: int = 25) -> bool:
    root = l2arctic_root()
    print(f"\n=== L2-ARCTIC ({root}) ===")
    if not root.exists():
        print(f"  FAIL: directory does not exist. Did you download/extract it here?")
        return False

    ok = True
    total_wav = 0
    total_annotations = 0

    for speaker, expected_wav in L2ARCTIC_EXPECTED_WAV_COUNTS.items():
        speaker_dir = root / speaker
        if not speaker_dir.exists():
            print(f"  FAIL: speaker {speaker} directory missing")
            ok = False
            continue

        wav_dir = speaker_dir / "wav"
        transcript_dir = speaker_dir / "transcript"
        textgrid_dir = speaker_dir / "textgrid"
        annotation_dir = speaker_dir / "annotation"

        n_wav = len(list(wav_dir.glob("*.wav"))) if wav_dir.exists() else 0
        n_annot = len(list(annotation_dir.glob("*.TextGrid"))) if annotation_dir.exists() else 0
        total_wav += n_wav
        total_annotations += n_annot

        expected_annot = L2ARCTIC_EXPECTED_ANNOTATION_COUNTS[speaker]
        speaker_ok = True
        if n_wav != expected_wav:
            print(f"  FAIL: {speaker}/wav has {n_wav} files, expected {expected_wav}")
            speaker_ok = False
        if n_annot != expected_annot:
            print(f"  FAIL: {speaker}/annotation has {n_annot} files, expected {expected_annot}")
            speaker_ok = False
        if not transcript_dir.exists() or not textgrid_dir.exists():
            print(f"  FAIL: {speaker} missing transcript/ or textgrid/ subfolder")
            speaker_ok = False
        if speaker_ok:
            print(f"  OK: {speaker} ({n_wav} wav, {n_annot} annotations)")
        else:
            ok = False

    if total_wav != L2ARCTIC_TOTAL_WAV:
        print(f"  FAIL: total wav files = {total_wav}, expected {L2ARCTIC_TOTAL_WAV}")
        ok = False
    if total_annotations != L2ARCTIC_TOTAL_ANNOTATIONS:
        print(f"  FAIL: total annotations = {total_annotations}, expected {L2ARCTIC_TOTAL_ANNOTATIONS}")
        ok = False

    all_wavs = list(root.glob("*/wav/*.wav"))
    sample = all_wavs[:sample_wav_check]
    bad = [str(p) for p in sample if not _is_readable_wav(p)]
    if bad:
        print(f"  FAIL: {len(bad)}/{len(sample)} sampled WAV files unreadable: {bad[:5]}")
        ok = False
    else:
        print(f"  OK: sampled {len(sample)} WAV files, all readable")

    license_file = root / "LICENSE"
    readme_file = root / "README.md"
    if not license_file.exists():
        print("  WARNING: LICENSE file missing from root — expected per L2-ARCTIC docs")
    if not readme_file.exists():
        print("  WARNING: README.md missing from root — expected per L2-ARCTIC docs")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["speechocean762", "l2arctic", "both"], default="both")
    args = parser.parse_args()

    results = {}
    if args.dataset in ("speechocean762", "both"):
        results["speechocean762"] = verify_speechocean762()
    if args.dataset in ("l2arctic", "both"):
        results["l2arctic"] = verify_l2arctic()

    print("\n=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if all(results.values()) else 1)