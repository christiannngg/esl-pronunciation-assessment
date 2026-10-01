"""
Build a data manifest across both datasets: speaker counts, utterance
counts, split sizes, and demographic/L1-background breakdowns. This is
the US 1.2 deliverable and directly feeds Sprint 2's EDA (US 2.1).

Usage:
    python src/data/build_manifest.py
Writes: data/manifest.json and data/manifest.md (human-readable summary)
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import speechocean762_root, l2arctic_root, get_data_root  # noqa: E402

# Speaker -> (L1 background, gender), from https://psi.engr.tamu.edu/l2-arctic-corpus/
L2ARCTIC_SPEAKER_INFO = {
    "ABA": ("Arabic", "M"), "SKA": ("Arabic", "F"), "YBAA": ("Arabic", "M"), "ZHAA": ("Arabic", "F"),
    "BWC": ("Mandarin", "M"), "LXC": ("Mandarin", "F"), "NCC": ("Mandarin", "F"), "TXHC": ("Mandarin", "M"),
    "ASI": ("Hindi", "M"), "RRBI": ("Hindi", "M"), "SVBI": ("Hindi", "F"), "TNI": ("Hindi", "F"),
    "HJK": ("Korean", "F"), "HKK": ("Korean", "M"), "YDCK": ("Korean", "F"), "YKWK": ("Korean", "M"),
    "EBVS": ("Spanish", "M"), "ERMS": ("Spanish", "M"), "MBMPS": ("Spanish", "F"), "NJS": ("Spanish", "F"),
    "HQTV": ("Vietnamese", "M"), "PNV": ("Vietnamese", "F"), "THV": ("Vietnamese", "F"), "TLV": ("Vietnamese", "M"),
}


def build_speechocean762_manifest() -> dict:
    root = speechocean762_root()
    if not root.exists():
        return {"available": False, "reason": f"{root} does not exist"}

    scores_path = root / "resource" / "scores.json"
    with open(scores_path) as f:
        scores = json.load(f)

    wave_dir = root / "WAVE"
    speaker_dirs = sorted(d.name for d in wave_dir.iterdir() if d.is_dir()) if wave_dir.exists() else []

    def _read_kaldi_list(split: str, fname: str) -> list[str]:
        p = root / split / fname
        return p.read_text().splitlines() if p.exists() else []

    train_utts = _read_kaldi_list("train", "text")
    test_utts = _read_kaldi_list("test", "text")

    ages, genders = [], []
    for split in ("train", "test"):
        for line in _read_kaldi_list(split, "spk2age"):
            parts = line.split()
            if len(parts) == 2:
                ages.append(parts[1])
        for line in _read_kaldi_list(split, "spk2gender"):
            parts = line.split()
            if len(parts) == 2:
                genders.append(parts[1])

    return {
        "available": True,
        "license": "CC BY 4.0",
        "total_utterances": len(scores),
        "total_speakers": len(speaker_dirs),
        "train_utterances": len(train_utts),
        "test_utterances": len(test_utts),
        "gender_counts": dict(Counter(genders)),
        "age_distribution_sample": dict(Counter(ages)),
        "annotation_levels": ["sentence", "word", "phoneme"],
        "score_fields": ["accuracy", "completeness", "fluency", "prosodic", "total"],
    }


def build_l2arctic_manifest() -> dict:
    root = l2arctic_root()
    if not root.exists():
        return {"available": False, "reason": f"{root} does not exist"}

    per_speaker = {}
    l1_counts = Counter()
    gender_counts = Counter()
    total_wav = 0
    total_annotations = 0

    for speaker, (l1, gender) in L2ARCTIC_SPEAKER_INFO.items():
        speaker_dir = root / speaker
        if not speaker_dir.exists():
            continue
        n_wav = len(list((speaker_dir / "wav").glob("*.wav"))) if (speaker_dir / "wav").exists() else 0
        n_annot = len(list((speaker_dir / "annotation").glob("*.TextGrid"))) if (speaker_dir / "annotation").exists() else 0
        per_speaker[speaker] = {"l1": l1, "gender": gender, "n_wav": n_wav, "n_annotations": n_annot}
        l1_counts[l1] += n_wav
        gender_counts[gender] += n_wav
        total_wav += n_wav
        total_annotations += n_annot

    suitcase_dir = root / "suitcase_corpus"
    has_suitcase = suitcase_dir.exists()

    return {
        "available": True,
        "license": "CC BY-NC 4.0 (non-commercial use only)",
        "total_speakers": len(per_speaker),
        "total_wav_files": total_wav,
        "total_manual_annotations": total_annotations,
        "utterances_by_l1_background": dict(l1_counts),
        "utterances_by_gender": dict(gender_counts),
        "per_speaker": per_speaker,
        "error_types_annotated": ["substitution", "deletion", "addition"],
        "has_suitcase_corpus": has_suitcase,
    }


def main():
    manifest = {
        "speechocean762": build_speechocean762_manifest(),
        "l2arctic": build_l2arctic_manifest(),
    }

    out_dir = Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "manifest.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {json_path}")

    md_lines = ["# Data Manifest", ""]
    for name, info in manifest.items():
        md_lines.append(f"## {name}")
        if not info.get("available"):
            md_lines.append(f"- **Not found.** {info.get('reason', '')}")
            md_lines.append("")
            continue
        for key, val in info.items():
            if key == "per_speaker":
                continue
            md_lines.append(f"- **{key}**: {val}")
        if "per_speaker" in info:
            md_lines.append("")
            md_lines.append("| Speaker | L1 | Gender | # Wav | # Annotations |")
            md_lines.append("|---|---|---|---|---|")
            for spk, d in info["per_speaker"].items():
                md_lines.append(f"| {spk} | {d['l1']} | {d['gender']} | {d['n_wav']} | {d['n_annotations']} |")
        md_lines.append("")

    md_path = out_dir / "manifest.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()