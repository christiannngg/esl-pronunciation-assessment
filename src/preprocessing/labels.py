"""
Label cleaning shared by both datasets (US 2.2).

* One 39-phone ARPAbet inventory, with lexical stress kept as a separate field.
* Repairs the typo / variant symbols found in the L2-ARCTIC annotations (see EDA register LA-7, X-2).
* Explicit, configurable policy for the '*' (accented-variant) and 'err' tags (LA-8).

Decisions (agreed after the EDA):
  - '*' tags are kept as errors, with ``is_deviation`` recorded so results can be reported both ways.
  - 'err' (sound not identified) stays a substitution, flagged with ``is_err``.
  - AX (schwa variant) maps to AH.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

ARPABET39 = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y",
    "Z", "ZH"})
VOWELS = frozenset({"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"})
PHONE_TO_ID = {p: i for i, p in enumerate(sorted(ARPABET39))}

# Variant symbols mapped onto the 39-phone set (applied after junk characters are stripped).
VARIANT_MAP = {"AX": "AH"}
SILENCE = frozenset({"sil", "sp", "spn", ""})
ERROR_CLASSES = ("correct", "substitution", "deletion", "addition")


@dataclass(frozen=True)
class CleanPhone:
    base: Optional[str]        # one of ARPABET39, or None when the symbol is not usable
    stress: Optional[int]      # 0/1/2 for vowels that carry a digit, else None
    is_deviation: bool         # trailing '*' (accented variant)
    repaired: bool             # the raw symbol needed fixing (typo characters or variant mapping)


def clean_phone(symbol: Optional[str]) -> CleanPhone:
    """'AH0' -> (AH, 0); "R*" -> (R, deviation); 'D_' -> (D, repaired); 'AX' -> (AH, repaired); 'err'/'sil' -> base None."""
    if symbol is None:
        return CleanPhone(None, None, False, False)
    raw = symbol.strip()
    if raw.lower() in SILENCE or raw.lower() == "err":
        return CleanPhone(None, None, False, False)
    deviation = "*" in raw
    letters = re.sub(r"[^A-Za-z]", "", re.sub(r"\d", "", raw)).upper()
    m = re.search(r"(\d)", raw)
    stress = int(m.group(1)) if m else None
    repaired = bool(re.search(r"[^A-Za-z0-9*]", raw))
    if letters in VARIANT_MAP:
        letters, repaired = VARIANT_MAP[letters], True
    if letters not in ARPABET39:
        return CleanPhone(None, stress, deviation, repaired)
    return CleanPhone(letters, stress, deviation, repaired)


def clean_l2arctic_phones(phones: pd.DataFrame, count_deviation_as_error: bool = True) -> pd.DataFrame:
    """
    Add cleaned columns to the phone table from ``loaders.load_l2arctic().phones``:
      cpl_base / cpl_stress   canonical phone (None for additions)
      ppl_base                perceived phone (None for deletions / unnamed 'err')
      is_deviation, is_err    tag flags
      label_repaired          True when a typo or variant had to be fixed
      error_class             correct / substitution / deletion / addition / silence
      is_error                binary error flag under the chosen '*' policy
    Rows whose canonical phone cannot be mapped to the 39-phone inventory are flagged ``unmapped`` (never silently dropped).
    """
    out = phones.copy()
    canon = out["cpl"].map(clean_phone)
    perc = out["ppl"].map(clean_phone)
    out["cpl_base"] = canon.map(lambda c: c.base)
    out["cpl_stress"] = canon.map(lambda c: c.stress)
    out["ppl_base"] = perc.map(lambda c: c.base)
    out["is_deviation"] = out["ppl_is_deviation"].astype(bool)
    out["is_err"] = out["ppl_is_err"].astype(bool)
    out["label_repaired"] = canon.map(lambda c: c.repaired) | perc.map(lambda c: c.repaired)
    out["error_class"] = out["error_type"]
    is_slot = out["error_class"].isin(["correct", "substitution", "deletion"])
    out["unmapped"] = is_slot & out["cpl_base"].isna()
    err = out["error_class"].isin(["substitution", "deletion", "addition"])
    if not count_deviation_as_error:
        err = err & ~((out["error_class"] == "substitution") & out["is_deviation"])
    out["is_error"] = err
    return out
