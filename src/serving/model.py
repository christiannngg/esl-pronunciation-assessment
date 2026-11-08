"""
Sprint 9: the served MT-GN15 model. Rebuilds the EXACT training module (``multitask.build_mt_model_class``) from serving/stats.json, loads one checkpoint,
and maps (waveform, aligned words) -> scores, with the same pooling spans and side features as training.

    m = PronunciationModel()                       # seed 0 checkpoint, CPU
    out = m.predict(wav_16k_float32, words)        # words: see src/serving/side_features.py
    out["sentence"]  -> {"sent_accuracy": float, ..., "sent_total": float}
    out["words"]     -> [{"word_accuracy": float}, ...]
    out["phones"]    -> [{"phone_score": float, "mispron_prob": float, "mispron": bool}, ...]   (rows ordered by word, then phone)

Scores are in the label units of SpeechOcean762 (sentence 0-10, word accuracy 0-10, phone 0-2). No clipping is applied here.
The classification flag is ``P(class 1) > 0.5`` (``stats.json`` ``decision``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.features import ssl_embeddings as se
from src.models import finetune as FT
from src.models import multitask as MT
from src.models.baselines import REPO
from src.serving import side_features as SF

DEFAULT_CKPT = REPO / "results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed0.pt"
DEFAULT_STATS = REPO / "serving" / "stats.json"
SENT, WORD, PHONE = "so762_sentence", "so762_word", "so762_phone"


class PronunciationModel:
    def __init__(self, ckpt: Path | str = DEFAULT_CKPT, stats: Path | str = DEFAULT_STATS, device: str = "cpu"):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.stats = json.loads(Path(stats).read_text())
        enc_s, mod_s = self.stats["encoder"], self.stats["model"]
        cfg = MT.load_config(self.stats["source"]["variant"])
        enc, _ = FT.load_encoder(enc_s["hf_id"], enc_s["revision"], cfg["model"])
        specs = [{"name": n, "table": h["table"], "pool_kind": h["pool_kind"], "d_side": h["d_side"], "d_out": h["d_out"]}
                 for n, h in self.stats["heads"].items()]
        self.model = MT.build_mt_model_class()(enc, mod_s["layer_ids"], specs, mod_s["head"], mod_s["hidden"], mod_s["head_dropout"], mod_s["alpha_mode"])
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)       # fp16 on disk -> fp32 compute
        self.model.eval().to(self.device)
        self.do_normalize = bool(enc_s["do_normalize"])
        self._side_stats = {n: (torch.tensor(h["side_mean"], dtype=torch.float32), torch.tensor(h["side_std"], dtype=torch.float32))
                            for n, h in self.stats["heads"].items()}

    # ------------------------------------------------------------------ inputs
    def _side(self, head: str, raw: np.ndarray):
        mu, sd = self._side_stats[head]
        return ((self.torch.as_tensor(raw, dtype=self.torch.float32) - mu) / sd).to(self.device)

    def predict(self, wav: np.ndarray, words: list[dict]) -> dict:
        torch = self.torch
        w = FT.normalize_wave(wav, self.do_normalize)
        T = se.n_frames(len(w))
        ws = np.array([x["start"] for x in words]), np.array([x["end"] for x in words])
        ps = [p for x in words for p in x["phones"]]
        pst, pen = np.array([p["start"] for p in ps]), np.array([p["end"] for p in ps])
        spans = {SENT: [(np.array([-1]), np.array([-1]))], WORD: [se.unit_frames(T, *ws)], PHONE: [se.unit_frames(T, pst, pen)]}
        raw = {SENT: SF.sentence_side(len(w), len(words), len(ps)), WORD: SF.word_side(words), PHONE: SF.phone_side(words)}
        side = {n: self._side(n, raw[h["table"]]) for n, h in self.stats["heads"].items()}
        x = torch.as_tensor(w[None], dtype=torch.float32, device=self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(x, np.array([len(w)]), spans, side, amp=False)
        self.last_forward_s = time.perf_counter() - t0
        res = {}
        for n, h in self.stats["heads"].items():
            o = out[n].float().cpu()
            if h["kind"] == "reg":
                res[n] = (o.squeeze(-1) * h["target_std"] + h["target_mean"]).numpy()
            else:
                res[n] = torch.softmax(o, 1)[:, 1].numpy()                                   # P(class 1)
        thr = self.stats["heads"]["phone_mispron_t1.0"]["decision"]["positive_threshold"]
        return {
            "sentence": {n: float(res[n][0]) for n in ("sent_accuracy", "sent_fluency", "sent_prosodic", "sent_total")},
            "words": [{"word_accuracy": float(v)} for v in res["word_accuracy"]],
            "phones": [{"phone_score": float(s), "mispron_prob": float(p), "mispron": bool(p > thr)}
                       for s, p in zip(res["phone_score"], res["phone_mispron_t1.0"])],
            "raw": res,
        }
