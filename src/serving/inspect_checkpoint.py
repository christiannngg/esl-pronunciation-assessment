"""
Sprint 9 / iteration 1: open a MT-GN15 checkpoint and check it against serving/stats.json (layout only; no data, no GPU, no audio).

    python -m src.serving.inspect_checkpoint                                  # seed 0
    python -m src.serving.inspect_checkpoint --ckpt results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed1.pt

Prints the state-dict structure and asserts, for every head, that the first Linear layer expects
(pooled dim + d_side) inputs: 1536 + d_side for sentence heads (mean||std), 768 + d_side for word / phone heads.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.models.baselines import REPO


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO / "results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed0.pt"))
    ap.add_argument("--stats", default=str(REPO / "serving" / "stats.json"))
    a = ap.parse_args(argv)
    import torch

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    stats = json.loads(Path(a.stats).read_text())
    print(f"{a.ckpt}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M values, dtypes {dict(Counter(str(v.dtype) for v in sd.values()))}")
    prefixes = Counter(k.split(".")[0] for k in sd)
    print("top-level prefixes:", dict(prefixes))
    for k in [k for k in sd if k.startswith("alpha")]:
        print(f"  {k}: {tuple(sd[k].shape)}")
    D = 768
    ok = True
    for name, h in stats["heads"].items():
        key = "heads." + name.replace(".", "_") + ".0.weight"
        cands = [k for k in sd if k.startswith("heads.") and k.endswith("0.weight") and name.replace(".", "_") in k] or [k for k in sd if name in k and k.endswith("0.weight")]
        if not cands:
            print(f"  {name}: first-layer weight not found (looked for {key}); keys with 'heads.': {[k for k in sd if k.startswith('heads.')][:6]}")
            ok = False
            continue
        w = sd[cands[0]]
        exp = (2 * D if h["pool_kind"] == "utt" else D) + h["d_side"]
        status = "OK" if w.shape[1] == exp else f"MISMATCH (expected {exp})"
        ok &= w.shape[1] == exp
        print(f"  {name:20s} {cands[0]}: {tuple(w.shape)}  {status}")
    print("LAYOUT OK" if ok else "LAYOUT PROBLEM: read the lines above")


if __name__ == "__main__":
    main()
