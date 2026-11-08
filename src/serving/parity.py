"""
Sprint 9 gate G1 (parity): the served model, fed the stored forced alignment, must reproduce what each seed predicted during training.

    python -m src.serving.parity                       # seed 0, 40 test utterances, CPU
    python -m src.serving.parity --seeds 0 1 2 --n 60 --device mps

Reference = the per-seed job files ``experiments/results/multitask/mt63_gradnorm_a15/<head>/fold0/seed<s>.npz`` (``test_pred`` for regression heads,
``test_proba`` for ``phone_mispron_t1.0``; ``test_rows`` gives the table rows). The training numbers came from bf16 autocast on a GPU; here fp32 on
CPU / MPS, so small differences are expected. Tolerances, fixed BEFORE the first run:
    regression heads : mean |diff| <= 0.02 and max |diff| <= 0.10   (score units)
    mispronunciation : flag agreement >= 99 %  and max |p diff| <= 0.05
Writes results/tables/serving/parity_seed<s>.csv.  Also reports the encoder forward time per utterance.
"""
from __future__ import annotations

import faulthandler
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import finetune as FT
from src.models import multitask as MT
from src.models.baselines import REPO, build_tasks
from src.models.baselines import load_config as load_baseline_config
from src.serving.model import DEFAULT_STATS, PronunciationModel
from src.utils.paths import get_data_root

JOBS = REPO / "experiments/results/multitask/mt63_gradnorm_a15"
ALI = get_data_root() / "processed" / "alignments"
TOL = {"reg_mean": 0.02, "reg_max": 0.10, "clf_agree": 0.99, "clf_pmax": 0.05}
HEADS = ["sent_accuracy", "sent_fluency", "sent_prosodic", "sent_total", "word_accuracy", "phone_score", "phone_mispron_t1.0"]


def words_of(utt: str, W: pd.DataFrame, P: pd.DataFrame) -> list[dict]:
    out = []
    for w in W[W.utt_id == utt].sort_values("word_idx").itertuples():
        ph = P[(P.utt_id == utt) & (P.word_idx == w.word_idx)].sort_values("phone_idx")
        out.append({"word": w.word, "start": w.start, "end": w.end,
                    "phones": [{"phone": r.phone, "start": r.start, "end": r.end} for r in ph.itertuples()]})
    return out


def main(argv=None) -> None:
    faulthandler.enable()                 # a native crash (segfault) prints the Python stack
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    cfg = MT.load_config("gradnorm_a15")
    tasks = MT.group_tasks(cfg, "so762", build_tasks(load_baseline_config()))
    md = MT.MTData(tasks, False)
    test_u = md.utt_split(0)["test"]
    pick = test_u[np.linspace(0, len(test_u) - 1, a.n).astype(int)]
    W = pd.read_csv(ALI / "so762_words.csv.gz", dtype={"utt_id": str})
    P = pd.read_csv(ALI / "so762_phones.csv.gz", dtype={"utt_id": str})
    out_dir = REPO / "results" / "tables" / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for seed in a.seeds:
        ckpt = REPO / f"results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed{seed}.pt"
        t = time.time()
        model = PronunciationModel(ckpt, DEFAULT_STATS, a.device)
        print(f"seed {seed}: model loaded in {time.time() - t:.1f}s on {a.device}")
        ref = {}
        for h in HEADS:
            d = np.load(JOBS / h / "fold0" / f"seed{seed}.npz", allow_pickle=True)
            vals = d["test_proba"][:, 1] if h.startswith("phone_mispron") else d["test_pred"]
            ref[h] = (dict(zip(d["test_rows"].tolist(), range(len(vals)))), vals)
        mine = {h: [] for h in HEADS}
        theirs = {h: [] for h in HEADS}
        fwd = []
        for u in pick:
            words = words_of(str(md.utt_ids[u]), W, P)
            res = model.predict(md.wav(int(u)), words)
            fwd.append(model.last_forward_s)
            for h in HEADS:
                rows = md.rows(h, np.array([u]))
                pos, vals = ref[h]
                theirs[h].append(vals[[pos[int(r)] for r in rows]])
                mine[h].append(res["raw"][h])
        recs = []
        for h in HEADS:
            m, r = np.concatenate(mine[h]), np.concatenate(theirs[h])
            if h.startswith("phone_mispron"):
                agree = float(((m > 0.5) == (r > 0.5)).mean())
                pmax = float(np.abs(m - r).max())
                ok = agree >= TOL["clf_agree"] and pmax <= TOL["clf_pmax"]
                recs.append({"head": h, "n": len(m), "agree": agree, "max_abs_diff": pmax, "mean_abs_diff": float(np.abs(m - r).mean()), "pass": ok})
            else:
                dif = np.abs(m - r)
                ok = dif.mean() <= TOL["reg_mean"] and dif.max() <= TOL["reg_max"]
                recs.append({"head": h, "n": len(m), "agree": np.nan, "max_abs_diff": float(dif.max()), "mean_abs_diff": float(dif.mean()), "pass": bool(ok)})
            ok_all &= bool(recs[-1]["pass"])
        df = pd.DataFrame(recs)
        df.to_csv(out_dir / f"parity_seed{seed}.csv", index=False)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"encoder+heads forward, batch 1: median {np.median(fwd):.2f}s, p90 {np.percentile(fwd, 90):.2f}s over {len(fwd)} utterances (device {a.device})")
    print("PARITY PASS" if ok_all else "PARITY FAIL: see the table above")


if __name__ == "__main__":
    main()
