"""
Tidy-DataFrame loaders for SpeechOcean762 and L2-ARCTIC.

Built for US 2.1 (EDA) and reused by US 2.2 (preprocessing / splits), so every
downstream consumer parses the raw releases the same way.

Public API
----------
SpeechOcean762
    load_so762()            -> SO762(utts, words, phones)      consensus scores
    load_so762_experts()    -> SO762Experts(sent, words, phones) per-expert scores
L2-ARCTIC
    load_l2arctic()         -> L2Arctic(utts, phones, words)   manual-annotation labels
    parse_textgrid()        -> {tier_name: [(xmin, xmax, text), ...]}
    parse_phone_label()     -> dict describing one annotated phone label

Only the standard library + pandas are required (audio headers are read with the
`wave` module, so no soundfile/librosa dependency for EDA).
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd

from src.data.build_manifest import L2ARCTIC_SPEAKER_INFO
from src.utils.paths import get_data_root, l2arctic_root, speechocean762_root

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

# SpeechOcean762 ages have a clean gap between 15 and 19, so "< 18" separates
# the child/teen half of the corpus from the adult half without ambiguity.
ADULT_AGE_MIN = 18

SO762_SENTENCE_METRICS = ["accuracy", "completeness", "fluency", "prosodic", "total"]
SILENCE_LABELS = {"sil", "sp", "spn", ""}


def _strip_stress(phone: str) -> str:
    """'AH0' -> 'AH' (ARPAbet stress digit removed)."""
    return re.sub(r"\d", "", phone)


def wav_info(path: Path) -> tuple:
    """(duration_s, sample_rate, channels, sample_width_bytes) from the WAV header only."""
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            return w.getnframes() / sr, sr, w.getnchannels(), w.getsampwidth()
    except Exception:  # unreadable / non-PCM header; reported as NaN, never silently dropped
        return float("nan"), None, None, None


def _audio_info_table(paths: dict, cache_csv: Optional[Path]) -> pd.DataFrame:
    """key -> header info for many wavs, cached because it touches ~32k files."""
    if cache_csv is not None and cache_csv.exists():
        cached = pd.read_csv(cache_csv, dtype={"key": str})
        if set(cached["key"]) == set(paths):
            return cached
    rows = []
    for key, p in paths.items():
        dur, sr, ch, sw = wav_info(p)
        rows.append({"key": key, "duration_s": dur, "sample_rate": sr, "channels": ch, "sample_width": sw})
    table = pd.DataFrame(rows)
    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_csv, index=False)
    return table


def _cache_dir() -> Path:
    return get_data_root() / "processed" / "eda_cache"


def _read_pairs(path: Path) -> dict:
    """Kaldi-style 'key<ws>value' file -> dict (value may contain spaces)."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                parts = re.split(r"\s+", line.strip(), maxsplit=1)
                out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


# --------------------------------------------------------------------------- #
# SpeechOcean762
# --------------------------------------------------------------------------- #

class SO762(NamedTuple):
    utts: pd.DataFrame    # one row per utterance: sentence-level consensus scores + metadata
    words: pd.DataFrame   # one row per word: accuracy / stress / total
    phones: pd.DataFrame  # one row per canonical phone: mean-of-experts score in [0, 2]


class SO762Experts(NamedTuple):
    sent: pd.DataFrame    # long: utt_id, expert, accuracy, completeness, fluency, prosodic, total
    words: pd.DataFrame   # long: utt_id, word_idx, expert, accuracy, stress, total
    phones: pd.DataFrame  # long: utt_id, word_idx, phone_idx, expert, score in {0, 1, 2}
    n_phone_mismatch: int  # words whose expert phone string did not match ref-phones length


def load_so762(root: Optional[Path] = None, with_audio_info: bool = True) -> SO762:
    """Consensus (scores.json) labels plus speaker metadata and official split."""
    root = Path(root) if root else speechocean762_root()
    scores = json.loads((root / "resource" / "scores.json").read_text())

    utt_split, utt_spk, utt_wav = {}, {}, {}
    for split in ("train", "test"):
        for utt, spk in _read_pairs(root / split / "utt2spk").items():
            utt_split[utt], utt_spk[utt] = split, spk
        for utt, rel in _read_pairs(root / split / "wav.scp").items():
            utt_wav[utt] = root / rel
    spk_age, spk_gender = {}, {}
    for split in ("train", "test"):
        spk_age.update(_read_pairs(root / split / "spk2age"))
        spk_gender.update(_read_pairs(root / split / "spk2gender"))

    utt_rows, word_rows, phone_rows = [], [], []
    for utt_id, rec in scores.items():
        spk = utt_spk.get(utt_id)
        split = utt_split.get(utt_id)
        words = rec["words"]
        n_phones = sum(len(w["phones"]) for w in words)
        utt_rows.append({
            "utt_id": utt_id, "speaker": spk, "split": split,
            "gender": spk_gender.get(spk), "age": int(spk_age[spk]) if spk in spk_age else None,
            "text": rec["text"], "n_words": len(words), "n_phones": n_phones,
            **{m: rec[m] for m in SO762_SENTENCE_METRICS},
            "wav_path": str(utt_wav[utt_id]) if utt_id in utt_wav else None,
        })
        for w_idx, w in enumerate(words):
            phones = w["phones"].split() if isinstance(w["phones"], str) else list(w["phones"])
            pron = {m["index"]: m["pronounced-phone"] for m in w.get("mispronunciations", [])}
            word_rows.append({
                "utt_id": utt_id, "speaker": spk, "split": split, "word_idx": w_idx, "word": w["text"],
                "accuracy": w["accuracy"], "stress": w["stress"], "total": w["total"], "n_phones": len(phones),
            })
            for p_idx, (ph, sc) in enumerate(zip(phones, w["phones-accuracy"])):
                phone_rows.append({
                    "utt_id": utt_id, "speaker": spk, "split": split, "word_idx": w_idx, "phone_idx": p_idx,
                    "phone": ph, "phone_base": _strip_stress(ph), "score": sc, "pronounced": pron.get(p_idx),
                })

    utts = pd.DataFrame(utt_rows)
    utts["age_group"] = utts["age"].map(lambda a: "adult" if a is not None and a >= ADULT_AGE_MIN else "child/teen")
    if with_audio_info:
        info = _audio_info_table({u: Path(p) for u, p in zip(utts.utt_id, utts.wav_path) if p},
                                 _cache_dir() / "so762_audio_info.csv")
        utts = utts.merge(info.rename(columns={"key": "utt_id"}), on="utt_id", how="left")
    return SO762(utts, pd.DataFrame(word_rows), pd.DataFrame(phone_rows))


_PH_TOKEN = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|\S+")


def parse_expert_phone_string(s: str) -> tuple:
    """
    scores-detail.json notation: plain = 2, {x} = 1, (x) = 0, [x] = inserted phone.
    Returns (scores for the canonical phones, n_inserted).
    """
    scores, inserted = [], 0
    for tok in _PH_TOKEN.findall(s):
        if tok.startswith("["):
            inserted += 1
        elif tok.startswith("("):
            scores.append(0)
        elif tok.startswith("{"):
            scores.append(1)
        else:
            scores.append(2)
    return scores, inserted


def load_so762_experts(root: Optional[Path] = None) -> SO762Experts:
    """Per-expert raw scores (5 experts) from scores-detail.json."""
    root = Path(root) if root else speechocean762_root()
    detail = json.loads((root / "resource" / "scores-detail.json").read_text())
    sent_rows, word_rows, phone_rows, mismatch = [], [], [], 0
    for utt_id, rec in detail.items():
        n_exp = len(rec["accuracy"])
        for e in range(n_exp):
            sent_rows.append({"utt_id": utt_id, "expert": e, **{m: rec[m][e] for m in SO762_SENTENCE_METRICS}})
        for w_idx, w in enumerate(rec["words"]):
            ref_n = len(w["ref-phones"].split())
            for e in range(n_exp):
                word_rows.append({"utt_id": utt_id, "word_idx": w_idx, "expert": e,
                                  "accuracy": w["accuracy"][e], "stress": w["stress"][e], "total": w["total"][e]})
                scores, _ = parse_expert_phone_string(w["phones"][e])
                if len(scores) != ref_n:
                    mismatch += 1
                    continue
                for p_idx, sc in enumerate(scores):
                    phone_rows.append({"utt_id": utt_id, "word_idx": w_idx, "phone_idx": p_idx, "expert": e, "score": sc})
    return SO762Experts(pd.DataFrame(sent_rows), pd.DataFrame(word_rows),
                        pd.DataFrame(phone_rows).astype({"score": "int8"}), mismatch)


# --------------------------------------------------------------------------- #
# L2-ARCTIC
# --------------------------------------------------------------------------- #

_TIER_SPLIT = re.compile(r"^[ \t]*item \[\d+\]:[ \t]*$", re.M)
_TIER_NAME = re.compile(r'name = "(.*?)"')
_INTERVAL = re.compile(
    r'intervals \[\d+\]:\s*xmin = (\S+)\s*xmax = (\S+)\s*text = "(.*?)"[ \t]*'
    r"(?=\r?\n\s*intervals \[|\r?\n\s*item \[|\s*\Z)", re.S)


def parse_textgrid(path: Path, tiers: Optional[set] = None) -> dict:
    """
    Minimal long-format TextGrid reader (no external dependency). Handles both the
    tab-indented forced-alignment files and the space-indented annotation files.
    Returns {tier_name: [(xmin, xmax, text), ...]} for IntervalTiers.
    """
    text = Path(path).read_text(encoding="utf8", errors="replace")
    out = {}
    for chunk in _TIER_SPLIT.split(text)[1:]:
        m = _TIER_NAME.search(chunk)
        if not m or (tiers is not None and m.group(1) not in tiers):
            continue
        out[m.group(1)] = [(float(a), float(b), t) for a, b, t in _INTERVAL.findall(chunk)]
    return out


def parse_phone_label(raw: str) -> dict:
    """
    Decode one 'phones'-tier label from a manual annotation file.

      plain label ('AO1')   -> correct          (forced-alignment label unchanged)
      'CPL,PPL,s'           -> substitution     (CPL = canonical, PPL = perceived)
      'CPL,sil,d'           -> deletion
      'sil,PPL,a'           -> addition
      sil / sp / spn / ''   -> silence marker

    Tags may contain stray whitespace and mixed case ('B, F, S', 'D,SIL,D'); both are normalised.
    PPL may be 'err' (annotator could not name the sound) or carry '*' (accented variant of a phone).
    """
    lab = re.sub(r"\s+", "", raw)
    base = {"label_raw": raw, "cpl": None, "ppl": None, "error_type": None,
            "ppl_is_err": False, "ppl_is_deviation": False}
    if "," not in lab:
        if lab.lower() in SILENCE_LABELS:
            return {**base, "cpl": lab.lower() or None, "error_type": "silence"}
        return {**base, "cpl": lab.upper(), "error_type": "correct"}

    parts = lab.split(",")
    if len(parts) != 3 or parts[2].lower() not in {"s", "d", "a"}:
        return {**base, "error_type": "malformed"}

    def norm(tok: str) -> str:
        return tok.lower() if tok.lower() in {"sil", "err"} else tok.upper()

    cpl, ppl = norm(parts[0]), norm(parts[1])
    etype = {"s": "substitution", "d": "deletion", "a": "addition"}[parts[2].lower()]
    return {**base, "cpl": cpl, "ppl": ppl, "error_type": etype,
            "ppl_is_err": ppl == "err", "ppl_is_deviation": ppl.endswith("*")}


class L2Arctic(NamedTuple):
    utts: pd.DataFrame    # every wav (scripted sentences + suitcase narratives) with coverage flags
    phones: pd.DataFrame  # one row per phones-tier interval of every manually annotated file
    words: pd.DataFrame   # annotated words with derived per-word error counts


def _assign_word(words: list, mid: float) -> Optional[int]:
    for i, (a, b, t) in enumerate(words):
        if t.strip() and a <= mid < b:
            return i
    return None


def load_l2arctic(root: Optional[Path] = None, with_audio_info: bool = True) -> L2Arctic:
    root = Path(root) if root else l2arctic_root()
    utt_rows, phone_rows, word_rows = [], [], []

    def _scan(speaker_dir: Path, speaker: str, corpus: str, name_of):
        l1, gender = L2ARCTIC_SPEAKER_INFO[speaker]
        wavs = sorted((speaker_dir / "wav").glob("*.wav"))
        ann = {p.stem.lower(): p for p in (speaker_dir / "annotation").glob("*.TextGrid")}
        fa = {p.stem.lower() for p in (speaker_dir / "textgrid").glob("*.TextGrid")} \
            if (speaker_dir / "textgrid").exists() else set()
        for wav in wavs:
            stem = wav.stem
            tr = speaker_dir / "transcript" / f"{stem}.txt"
            utt_id = name_of(stem)
            utt_rows.append({
                "speaker": speaker, "l1": l1, "gender": gender, "corpus": corpus, "utt_id": utt_id,
                "prompt_id": stem if corpus == "arctic" else None,
                "text": tr.read_text(encoding="utf8", errors="replace").strip() if tr.exists() else None,
                "wav_path": str(wav), "has_annotation": stem.lower() in ann, "has_forced_alignment": stem.lower() in fa,
            })
        # Annotation files without a matching wav would be a data problem: surface them
        wav_stems = {w.stem.lower() for w in wavs}
        for stem, path in ann.items():
            if stem not in wav_stems:
                utt_rows.append({"speaker": speaker, "l1": l1, "gender": gender, "corpus": corpus,
                                 "utt_id": name_of(stem), "prompt_id": None, "text": None, "wav_path": None,
                                 "has_annotation": True, "has_forced_alignment": stem in fa})
        for stem, path in ann.items():
            utt_id = name_of(path.stem)
            tg = parse_textgrid(path, tiers={"words", "phones"})
            word_iv, phone_iv = tg.get("words", []), tg.get("phones", [])
            counts = {i: {"n_phones": 0, "n_sub": 0, "n_del": 0, "n_add": 0} for i in range(len(word_iv))}
            for p_idx, (a, b, raw) in enumerate(phone_iv):
                d = parse_phone_label(raw)
                w_i = _assign_word(word_iv, (a + b) / 2)
                phone_rows.append({
                    "speaker": speaker, "l1": l1, "corpus": corpus, "utt_id": utt_id, "phone_idx": p_idx,
                    "start": a, "end": b, "dur": b - a, "word_idx": w_i,
                    "word": word_iv[w_i][2].strip().lower() if w_i is not None else None, **d,
                })
                if w_i is not None:
                    c = counts[w_i]
                    if d["error_type"] in {"correct", "substitution", "deletion"}:
                        c["n_phones"] += 1
                    c["n_sub"] += d["error_type"] == "substitution"
                    c["n_del"] += d["error_type"] == "deletion"
                    c["n_add"] += d["error_type"] == "addition"
            for w_i, (a, b, t) in enumerate(word_iv):
                if t.strip():
                    word_rows.append({"speaker": speaker, "l1": l1, "corpus": corpus, "utt_id": utt_id,
                                      "word_idx": w_i, "word": t.strip().lower(), "start": a, "end": b, **counts[w_i]})

    for speaker in L2ARCTIC_SPEAKER_INFO:
        if (root / speaker).exists():
            _scan(root / speaker, speaker, "arctic", lambda s, spk=speaker: f"{spk}_{s}")
    suitcase = root / "suitcase_corpus"
    if suitcase.exists():
        # suitcase files are named by speaker code (aba.wav, aba.TextGrid) in flat folders
        for wav in sorted((suitcase / "wav").glob("*.wav")):
            spk = wav.stem.upper()
            if spk not in L2ARCTIC_SPEAKER_INFO:
                continue
            l1, gender = L2ARCTIC_SPEAKER_INFO[spk]
            tr = suitcase / "transcript" / f"{wav.stem}.txt"
            ann = suitcase / "annotation" / f"{wav.stem}.TextGrid"
            utt_rows.append({
                "speaker": spk, "l1": l1, "gender": gender, "corpus": "suitcase", "utt_id": f"{spk}_suitcase",
                "prompt_id": None, "text": tr.read_text(encoding="utf8", errors="replace").strip() if tr.exists() else None,
                "wav_path": str(wav), "has_annotation": ann.exists(), "has_forced_alignment": False})
            if ann.exists():
                utt_id = f"{spk}_suitcase"
                tg = parse_textgrid(ann, tiers={"words", "phones"})
                word_iv, phone_iv = tg.get("words", []), tg.get("phones", [])
                counts = {i: {"n_phones": 0, "n_sub": 0, "n_del": 0, "n_add": 0} for i in range(len(word_iv))}
                for p_idx, (a, b, raw) in enumerate(phone_iv):
                    d = parse_phone_label(raw)
                    w_i = _assign_word(word_iv, (a + b) / 2)
                    phone_rows.append({
                        "speaker": spk, "l1": l1, "corpus": "suitcase", "utt_id": utt_id, "phone_idx": p_idx,
                        "start": a, "end": b, "dur": b - a, "word_idx": w_i,
                        "word": word_iv[w_i][2].strip().lower() if w_i is not None else None, **d})
                    if w_i is not None:
                        c = counts[w_i]
                        if d["error_type"] in {"correct", "substitution", "deletion"}:
                            c["n_phones"] += 1
                        c["n_sub"] += d["error_type"] == "substitution"
                        c["n_del"] += d["error_type"] == "deletion"
                        c["n_add"] += d["error_type"] == "addition"
                for w_i, (a, b, t) in enumerate(word_iv):
                    if t.strip():
                        word_rows.append({"speaker": spk, "l1": l1, "corpus": "suitcase", "utt_id": utt_id,
                                          "word_idx": w_i, "word": t.strip().lower(), "start": a, "end": b, **counts[w_i]})

    utts = pd.DataFrame(utt_rows)
    if with_audio_info:
        info = _audio_info_table({u: Path(p) for u, p in zip(utts.utt_id, utts.wav_path) if p},
                                 _cache_dir() / "l2arctic_audio_info.csv")
        utts = utts.merge(info.rename(columns={"key": "utt_id"}), on="utt_id", how="left")
    return L2Arctic(utts, pd.DataFrame(phone_rows), pd.DataFrame(word_rows))
