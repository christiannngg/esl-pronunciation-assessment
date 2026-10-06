"""Unit tests for the pure parsing helpers in src/data/loaders.py (no dataset files needed)."""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.data.loaders import parse_expert_phone_string, parse_phone_label, parse_textgrid


@pytest.mark.parametrize("raw, etype, cpl, ppl", [
    ("AO1", "correct", "AO1", None),
    ("sp", "silence", "sp", None),
    ("", "silence", None, None),
    ("D,T,s", "substitution", "D", "T"),
    ("P, B, S ", "substitution", "P", "B"),      # whitespace + upper-case tag
    ("R,sil,d", "deletion", "R", "sil"),
    ("D, SIL, D", "deletion", "D", "sil"),
    ("sil,K,a", "addition", "sil", "K"),
])
def test_parse_phone_label_types(raw, etype, cpl, ppl):
    d = parse_phone_label(raw)
    assert (d["error_type"], d["cpl"], d["ppl"]) == (etype, cpl, ppl)


def test_parse_phone_label_flags_and_malformed():
    assert parse_phone_label("R,R*,s")["ppl_is_deviation"] is True
    assert parse_phone_label("R,err,s")["ppl_is_err"] is True
    assert parse_phone_label("DH, ERR, S")["ppl_is_err"] is True
    assert parse_phone_label("A,B")["error_type"] == "malformed"
    assert parse_phone_label("A,B,x")["error_type"] == "malformed"


def test_parse_expert_phone_string():
    assert parse_expert_phone_string("W IY0") == ([2, 2], 0)
    assert parse_expert_phone_string("K {AO0} L") == ([2, 1, 2], 0)
    assert parse_expert_phone_string("B (EH0) (R)") == ([2, 0, 0], 0)
    assert parse_expert_phone_string("B EH0 [L] R") == ([2, 2, 2], 1)   # inserted phone is not scored


def test_parse_textgrid_both_indentation_styles(tmp_path):
    body = textwrap.dedent('''\
        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 1
        tiers? <exists>
        size = 2
        item []:
            item [1]:
                class = "IntervalTier"
                name = "words"
                xmin = 0
                xmax = 1
                intervals: size = 2
                intervals [1]:
                    xmin = 0
                    xmax = 0.4
                    text = ""
                intervals [2]:
                    xmin = 0.4
                    xmax = 1
                    text = "hello"
            item [2]:
                class = "IntervalTier"
                name = "phones"
                xmin = 0
                xmax = 1
                intervals: size = 1
                intervals [1]:
                    xmin = 0
                    xmax = 1
                    text = "HH,err,s"
        ''')
    for name, text in {"spaces.TextGrid": body, "tabs.TextGrid": body.replace("    ", "\t")}.items():
        p = tmp_path / name
        p.write_text(text)
        tg = parse_textgrid(p)
        assert tg["words"] == [(0.0, 0.4, ""), (0.4, 1.0, "hello")]
        assert tg["phones"] == [(0.0, 1.0, "HH,err,s")]
        assert set(parse_textgrid(p, tiers={"words"})) == {"words"}
