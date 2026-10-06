"""
Forced-alignment support for US 2.2.

SpeechOcean762 ships NO timestamps, so phone/word boundaries come from the
Montreal Forced Aligner (MFA), run on the user's machine via ``run_mfa.py``.
L2-ARCTIC's manual annotation TextGrids already carry boundaries, so for that corpus
MFA is run only as a *consistency check* against the provided forced alignments
(the manual annotations are mostly edited forced alignments, hence not independent ground truth).

This module is pure Python (no MFA import) so it can be unit-tested anywhere:
  * building corpus directories and the SO762 pronunciation dictionary
  * ingesting MFA TextGrids into tidy word / phone tables with structural checks
  * L2-ARCTIC consistency statistics
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loaders import parse_textgrid
from src.preprocessing.labels import SILENCE

_DIGITS = re.compile(r"\d")


def strip_stress(p: str) -> str:
    return _DIGITS.sub("", p)


def _is_sil(label: str) -> bool:
    t = label.strip().lower()
    return t in SILENCE or t.startswith("<")


# --------------------------------------------------------------------------- #
# SpeechOcean762: canonical (intended) pronunciations from scores.json
# --------------------------------------------------------------------------- #
def so762_canonical(scores_json: Path) -> dict:
    """utt_id -> list of words: {text, phones(list, stress digits kept), phone_scores, word_total, ...}"""
    scores = json.loads(Path(scores_json).read_text())
    out = {}
    for utt, rec in scores.items():
        out[utt] = [{"text": w["text"], "phones": w["phones"], "phone_scores": w["phones-accuracy"],
                     "accuracy": w["accuracy"], "stress": w["stress"], "total": w["total"]}
                    for w in rec["words"]]
    return out


def build_so762_dictionary(canonical: dict) -> dict:
    """lowercase word -> sorted set of observed pronunciations (tuples, stress digits kept)."""
    d = defaultdict(set)
    for words in canonical.values():
        for w in words:
            d[w["text"].lower()].add(tuple(w["phones"]))
    return {k: sorted(v) for k, v in sorted(d.items())}


_BARE_VOWELS = {"IH", "UH"}  # appear without a stress digit in scores.json; the MFA model only knows IH0/UH0 etc.


def _mfa_phone(p: str) -> str:
    return p + "0" if p in _BARE_VOWELS else p


def word_token(word: str, pron, dic: dict) -> str:
    """Unique dictionary token for one (word, pronunciation) pair, e.g. 'to__1'.
    Using one token per canonical pronunciation FORCES MFA to align the pronunciation the
    corpus intended, instead of picking whichever variant fits the audio best."""
    return f"{word.lower()}__{dic[word.lower()].index(tuple(pron))}"


def utt_transcript(canon_words: list, dic: dict) -> str:
    return " ".join(word_token(w["text"], w["phones"], dic) for w in canon_words)


def base_word(label: str) -> str:
    return re.sub(r"__\d+$", "", label.strip().lower())


def write_mfa_dictionary(dic: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for w, prons in dic.items():
            for i, p in enumerate(prons):
                f.write(f"{w}__{i}\t{' '.join(_mfa_phone(x) for x in p)}\n")


def dictionary_phone_inventory(dict_path: Path) -> set:
    """Phone symbols used by an MFA .dict/.txt dictionary (handles optional probability columns)."""
    phones = set()
    for line in Path(dict_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        toks = parts[1:]
        while toks and re.fullmatch(r"[\d.eE+-]+", toks[0]):  # pron-prob / silence-prob columns
            toks.pop(0)
        phones.update(toks)
    return phones


def prepare_corpus(items: list, corpus_dir: Path, text_of: dict, raw: bool = False) -> int:
    """
    items: [(speaker, utt_id, wav_path)]. Creates corpus_dir/<speaker>/<utt>.wav (symlink) and .lab.
    Transcripts are lowercase, punctuation stripped except apostrophes.
    """
    n = 0
    for spk, utt, wav in items:
        d = corpus_dir / spk
        d.mkdir(parents=True, exist_ok=True)
        link = d / f"{utt}.wav"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(Path(wav).resolve(), d.resolve()), link)  # relative: valid on any machine
        txt = text_of[utt] if raw else re.sub(r"[^a-z' ]", " ", text_of[utt].lower())
        (d / f"{utt}.lab").write_text(" ".join(txt.split()) + "\n")
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Ingest MFA output
# --------------------------------------------------------------------------- #
def _tier(tg: dict, *names):
    for n in names:
        for k in tg:
            if k.lower() == n:
                return tg[k]
    raise KeyError(f"none of tiers {names} in {list(tg)}")


def ingest_so762_textgrid(tg_path: Path, utt_id: str, canonical_words: list) -> tuple:
    """
    Align one MFA TextGrid to canonical words. Returns (word_rows, phone_rows, problem or None).
    Structural checks: #word intervals == #canonical words, text matches, the phones inside
    each word equal the canonical pronunciation (ignoring stress digits), boundaries ordered.
    """
    tg = parse_textgrid(tg_path)
    words = [(a, b, t) for a, b, t in _tier(tg, "words") if not _is_sil(t)]
    phones = [(a, b, t) for a, b, t in _tier(tg, "phones") if not _is_sil(t)]
    if len(words) != len(canonical_words):
        return [], [], f"word_count {len(words)} != {len(canonical_words)}"
    wrows, prows = [], []
    for wi, ((ws, we, wt), cw) in enumerate(zip(words, canonical_words)):
        if base_word(wt) != cw["text"].lower():
            return [], [], f"word_text '{wt}' != '{cw['text']}' at {wi}"
        inside = [(a, b, t) for a, b, t in phones if a >= ws - 1e-3 and b <= we + 1e-3]
        if [strip_stress(t) for _, _, t in inside] != [strip_stress(p) for p in cw["phones"]]:
            return [], [], f"phone_mismatch in word {wi} '{cw['text']}'"
        wrows.append({"utt_id": utt_id, "word_idx": wi, "word": cw["text"], "start": ws, "end": we,
                      "word_accuracy": cw["accuracy"], "word_stress": cw["stress"], "word_total": cw["total"]})
        for pi, ((a, b, t), cp, sc) in enumerate(zip(inside, cw["phones"], cw["phone_scores"])):
            if b < a:
                return [], [], f"negative_duration word {wi}"
            prows.append({"utt_id": utt_id, "word_idx": wi, "phone_idx": pi, "start": a, "end": b,
                          "mfa_phone": t, "phone": strip_stress(cp), "phone_stress": cp[len(strip_stress(cp)):] or None,
                          "phone_score": sc})
    prows_sorted = sorted(prows, key=lambda r: r["start"])
    if any(prows_sorted[i]["end"] > prows_sorted[i + 1]["start"] + 1e-3 for i in range(len(prows_sorted) - 1)):
        return [], [], "overlap"
    return wrows, prows, None


def ingest_so762(tg_dir: Path, canonical: dict, utt_ids: Optional[list] = None) -> tuple:
    """Walk an MFA output dir. Returns (words_df, phones_df, report_df of per-utt status)."""
    index = {p.stem: p for p in Path(tg_dir).rglob("*.TextGrid")}
    W, P, R = [], [], []
    for utt in (utt_ids or sorted(canonical)):
        if utt not in index:
            R.append({"utt_id": utt, "status": "no_textgrid"})
            continue
        w, p, problem = ingest_so762_textgrid(index[utt], utt, canonical[utt])
        if problem:
            R.append({"utt_id": utt, "status": "rejected", "reason": problem})
        else:
            W += w
            P += p
            R.append({"utt_id": utt, "status": "ok"})
    return pd.DataFrame(W), pd.DataFrame(P), pd.DataFrame(R)


# --------------------------------------------------------------------------- #
# L2-ARCTIC: MFA (ours) vs provided forced alignment (MFA v1.0)
# --------------------------------------------------------------------------- #
def _phone_intervals(path: Path) -> list:
    tg = parse_textgrid(path)
    return [(a, b, strip_stress(t.strip().upper())) for a, b, t in _tier(tg, "phones", "phone")
            if not _is_sil(t)]


def boundary_consistency(ours: Path, provided: Path) -> Optional[dict]:
    """Per-utterance start-boundary differences if phone sequences match, else None."""
    a, b = _phone_intervals(ours), _phone_intervals(provided)
    if [x[2] for x in a] != [x[2] for x in b] or not a:
        return None
    ds = np.array([abs(x[0] - y[0]) for x, y in zip(a, b)])
    return {"n": len(a), "abs_diff_s": ds}


def l2arctic_consistency(ours_dir: Path, provided_of: dict) -> dict:
    """provided_of: utt_id -> path of provided FA TextGrid. Returns summary dict."""
    index = {p.stem: p for p in Path(ours_dir).rglob("*.TextGrid")}
    diffs, n_match, n_mismatch, n_missing = [], 0, 0, 0
    for utt, prov in provided_of.items():
        if utt not in index or not Path(prov).exists():
            n_missing += 1
            continue
        r = boundary_consistency(index[utt], prov)
        if r is None:
            n_mismatch += 1
        else:
            n_match += 1
            diffs.append(r["abs_diff_s"])
    d = np.concatenate(diffs) if diffs else np.array([])
    return {"n_utts_compared": n_match, "n_utts_phone_sequence_mismatch": n_mismatch, "n_utts_missing": n_missing,
            "n_phones": int(d.size),
            "median_abs_start_diff_ms": float(np.median(d) * 1000) if d.size else None,
            "p90_abs_start_diff_ms": float(np.percentile(d, 90) * 1000) if d.size else None,
            "share_within_20ms": float((d <= 0.02).mean()) if d.size else None,
            "share_within_50ms": float((d <= 0.05).mean()) if d.size else None}


def _word_intervals(path: Path) -> list:
    tg = parse_textgrid(path)
    return [(a, b, base_word(t)) for a, b, t in _tier(tg, "words", "word") if not _is_sil(t)]


def word_boundary_consistency(ours_dir: Path, provided_of: dict) -> dict:
    """
    Word-level start/end differences, robust to the two aligners picking different phone variants
    (our english_us_arpa dictionary vs the CMU dictionary used for the provided alignments).
    """
    index = {p.stem: p for p in Path(ours_dir).rglob("*.TextGrid")}
    d, n_ok, n_bad = [], 0, 0
    for utt, prov in provided_of.items():
        if utt not in index or not Path(prov).exists():
            continue
        a, b = _word_intervals(index[utt]), _word_intervals(prov)
        if [x[2] for x in a] != [x[2] for x in b] or not a:
            n_bad += 1
            continue
        n_ok += 1
        d.append(np.abs(np.array([[x[0] - y[0], x[1] - y[1]] for x, y in zip(a, b)])).ravel())
    d = np.concatenate(d) if d else np.array([])
    return {"n_utts_compared": n_ok, "n_utts_word_sequence_mismatch": n_bad, "n_boundaries": int(d.size),
            "median_abs_diff_ms": float(np.median(d) * 1000) if d.size else None,
            "p90_abs_diff_ms": float(np.percentile(d, 90) * 1000) if d.size else None,
            "share_within_20ms": float((d <= 0.02).mean()) if d.size else None,
            "share_within_50ms": float((d <= 0.05).mean()) if d.size else None}


def annotation_shift(manual_phones: pd.DataFrame, provided_of: dict) -> dict:
    """
    How far did annotators move boundaries relative to the provided forced alignment?
    Silence and addition rows exist only in the annotation, so they are dropped; the remaining rows
    (canonical phone slots) are compared 1:1 when the counts match.
    """
    keep = manual_phones[~manual_phones["error_class"].isin(["silence", "addition"])]
    diffs, n_same, n_diff = [], 0, 0
    for utt, g in keep.groupby("utt_id"):
        prov = provided_of.get(utt)
        if prov is None or not Path(prov).exists():
            continue
        p = _phone_intervals(prov)
        g = g.sort_values("start")
        if len(p) != len(g):
            n_diff += 1
            continue
        n_same += 1
        diffs.append(np.abs(g["start"].to_numpy() - np.array([x[0] for x in p])))
    d = np.concatenate(diffs) if diffs else np.array([])
    return {"n_utts_same_phone_count": n_same, "n_utts_different_phone_count": n_diff,
            "median_abs_start_shift_ms": float(np.median(d) * 1000) if d.size else None,
            "share_unchanged_within_1ms": float((d <= 0.001).mean()) if d.size else None,
            "p90_abs_start_shift_ms": float(np.percentile(d, 90) * 1000) if d.size else None}
