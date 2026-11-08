"""
Sprint 9: MFA TextGrid -> the ``words`` structure the serving model takes (see side_features.py), using the SAME ingest and structural checks as training
(``alignment.ingest_so762_textgrid``: word count, word text, phones inside each word equal to the canonical pronunciation, ordered boundaries).
"""
from __future__ import annotations

from pathlib import Path

from src.preprocessing import alignment as AL


class AlignmentError(ValueError):
    """The alignment failed the structural checks (the app should show 'could not align your recording', not scores)."""


def words_from_textgrid(tg_path: Path | str, canonical: list[dict]) -> list[dict]:
    """canonical: [{"text": "call", "phones": ["K", "AO1", "L"]}, ...] (stress digits optional)."""
    cw = [{"text": w["text"], "phones": list(w["phones"]), "phone_scores": [0] * len(w["phones"]), "accuracy": 0, "stress": 0, "total": 0} for w in canonical]
    wrows, prows, problem = AL.ingest_so762_textgrid(Path(tg_path), "live", cw)
    if problem:
        raise AlignmentError(problem)
    return [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
             "phones": [{"phone": p["phone"], "start": float(p["start"]), "end": float(p["end"])} for p in prows if p["word_idx"] == w["word_idx"]]}
            for w in wrows]
