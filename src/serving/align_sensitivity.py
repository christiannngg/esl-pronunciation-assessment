"""
Sprint 9: how much do the model's outputs change when the alignment comes from a single live `mfa align_one` call instead of the stored training alignment?

    python -m src.serving.time_mfa --n 30 --keep results/tables/serving/mfa_live        # makes the live TextGrids (about 2 minutes)
    python -m src.serving.align_sensitivity --align-dir results/tables/serving/mfa_live  # needs the model; seed 0

For every kept TextGrid: predict with (a) the stored alignment (training) and (b) the live one, same audio, same model. Reports per utterance the largest
boundary difference and the output changes, then a summary split by utterances with small (<= 100 ms) and large (> 100 ms) boundary differences.
Context for reading the numbers: seed-to-seed validation RMSE differs by about 0.01-0.02 (US 8.3), and the test RMSE of sentence heads is about 0.9-1.1.
"""
from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import multitask as MT
from src.models.baselines import REPO, build_tasks
from src.models.baselines import load_config as load_baseline_config
from src.serving.aligner import AlignmentError, words_from_textgrid
from src.serving.model import DEFAULT_CKPT, DEFAULT_STATS, PronunciationModel
from src.serving.parity import ALI, words_of


def main(argv=None) -> None:
    faulthandler.enable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align-dir", default=str(REPO / "results/tables/serving/mfa_live"))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    cfg = MT.load_config("gradnorm_a15")
    md = MT.MTData(MT.group_tasks(cfg, "so762", build_tasks(load_baseline_config())), False)
    idx = {str(u): i for i, u in enumerate(md.utt_ids)}
    W = pd.read_csv(ALI / "so762_words.csv.gz", dtype={"utt_id": str})
    P = pd.read_csv(ALI / "so762_phones.csv.gz", dtype={"utt_id": str})
    model = PronunciationModel(a.ckpt, DEFAULT_STATS, a.device)
    rows, rejected = [], 0
    for tg in sorted(Path(a.align_dir).glob("*.TextGrid")):
        utt = tg.stem
        stored = words_of(utt, W, P)
        canon = [{"text": w["word"], "phones": [p["phone"] for p in w["phones"]]} for w in stored]
        try:
            live = words_from_textgrid(tg, canon)
        except AlignmentError as e:
            rejected += 1
            print(f"[{utt}] rejected by the structural checks: {e}")
            continue
        wav = md.wav(idx[utt])
        r0, r1 = model.predict(wav, stored)["raw"], model.predict(wav, live)["raw"]
        bd = max(abs(p0[k] - p1[k]) for w0, w1 in zip(stored, live) for p0, p1 in zip(w0["phones"], w1["phones"]) for k in ("start", "end")) * 1000
        rows.append({"utt": utt, "max_boundary_ms": bd,
                     "d_sent_total": abs(float(r0["sent_total"][0] - r1["sent_total"][0])),
                     "d_word_mean": float(np.abs(r0["word_accuracy"] - r1["word_accuracy"]).mean()),
                     "d_phone_mean": float(np.abs(r0["phone_score"] - r1["phone_score"]).mean()),
                     "flags_changed": int(((r0["phone_mispron_t1.0"] > 0.5) != (r1["phone_mispron_t1.0"] > 0.5)).sum()),
                     "n_phones": len(r0["phone_score"])})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no alignments compared")
        return
    out = REPO / "results" / "tables" / "serving"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "align_sensitivity.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for name, sub in (("all", df), ("boundary diff <= 100 ms", df[df.max_boundary_ms <= 100]), ("boundary diff > 100 ms", df[df.max_boundary_ms > 100])):
        if len(sub):
            print(f"{name:26s} n={len(sub):2d}  mean |d sent_total| {sub.d_sent_total.mean():.3f}  mean |d word_acc| {sub.d_word_mean.mean():.3f}  "
                  f"mean |d phone_score| {sub.d_phone_mean.mean():.3f}  flags changed {sub.flags_changed.sum()}/{sub.n_phones.sum()} phones")
    print(f"rejected by structural checks: {rejected}")


if __name__ == "__main__":
    main()
