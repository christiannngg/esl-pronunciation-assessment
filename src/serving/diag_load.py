"""
Sprint 9: find where model loading crashes (segfault) on this machine. Prints a flushed line before every step.

    python -m src.serving.diag_load
    HF_HUB_OFFLINE=1 python -m src.serving.diag_load           # weights are cached after the first run
    python -m src.serving.diag_load --threads 1
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
import time

faulthandler.enable(all_threads=True)


def step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-forward", action="store_true")
    a = ap.parse_args()
    step("import torch")
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    step(f"torch {torch.__version__}, threads {torch.get_num_threads()}, mps {torch.backends.mps.is_available()}, python {sys.version.split()[0]}")
    import json

    from src.models import finetune as FT
    from src.models import multitask as MT
    from src.serving.model import DEFAULT_CKPT, DEFAULT_STATS
    stats = json.loads(DEFAULT_STATS.read_text())
    cfg = MT.load_config(stats["source"]["variant"])
    step("FT.load_encoder (Hugging Face)")
    enc, _ = FT.load_encoder(stats["encoder"]["hf_id"], stats["encoder"]["revision"], cfg["model"])
    step(f"encoder loaded: {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M parameters")
    if not a.no_forward:
        step("encoder forward on 1 s of noise (eval, no grad)")
        enc.eval()
        with torch.no_grad():
            h = enc(torch.randn(1, 16000), output_hidden_states=True).hidden_states
        step(f"encoder forward OK: {len(h)} hidden states, last {tuple(h[-1].shape)}")
    step("build multi-task module")
    specs = [{"name": n, "table": x["table"], "pool_kind": x["pool_kind"], "d_side": x["d_side"], "d_out": x["d_out"]} for n, x in stats["heads"].items()]
    m = stats["model"]
    model = MT.build_mt_model_class()(enc, m["layer_ids"], specs, m["head"], m["hidden"], m["head_dropout"], m["alpha_mode"])
    step("torch.load checkpoint")
    sd = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=True)
    step(f"checkpoint tensors: {len(sd)}")
    step("load_state_dict (strict)")
    res = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    step(f"load_state_dict OK: {res}")
    model.eval()
    step("DONE: all steps passed")


if __name__ == "__main__":
    main()
