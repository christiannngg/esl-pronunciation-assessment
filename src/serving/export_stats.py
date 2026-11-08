"""
Sprint 9 / iteration 1: export the constants the MT-GN15 checkpoint does NOT store, so the app can rebuild the model and the inputs.

    python -m src.serving.export_stats                      # -> serving/stats.json   (fold 0, SO762 group, variant gradnorm_a15)
    python -m src.serving.export_stats --out serving/stats.json --variant gradnorm_a15

Everything is computed with the SAME code path as ``multitask.run_group`` (same TaskData objects, same training rows of fold 0, same formulas), so the values
are the ones the checkpoint was trained with:
    * regression heads : target mean / std on the training rows           (run_group: ``d["mu"], d["sd"]``)
    * every head       : side-feature mean / std on the training rows, std < 1e-6 -> 1     (run_group: ``mu_s, sd_s``)
    * model layout     : encoder id + revision, layer ids, head type, hidden size, dropout, per-head pool kind / d_side / d_out
    * decision rule    : phone_mispron_t1.0 = P(class 1) > 0.5 on the (seed-averaged) probabilities  (recovered in iteration 1, see docs/us91_iter0_inventory.md)
No GPU and no audio are needed. Needs the Sprint 3 feature tables under data/processed/features/tables and the speaker splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.models import finetune as FT
from src.models import multitask as MT
from src.models.baselines import REPO, build_tasks
from src.models.baselines import load_config as load_baseline_config

GROUP, FOLD = "so762", 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="gradnorm_a15")
    ap.add_argument("--out", default=str(REPO / "serving" / "stats.json"))
    ap.add_argument("--no-hf", action="store_true", help="skip the Hugging Face feature-extractor lookup (do_normalize then recorded as null)")
    a = ap.parse_args(argv)

    cfg = MT.load_config(a.variant)
    registry = build_tasks(load_baseline_config())
    tasks = MT.group_tasks(cfg, GROUP, registry)
    do_norm = None
    if not a.no_hf:
        try:
            do_norm = bool(FT.peek_normalize(cfg))
        except Exception as e:                                   # offline / not cached: the caller can rerun without --no-hf later
            print(f"warning: could not read do_normalize from the feature extractor ({e}); recorded as null")
    md = MT.MTData(tasks, bool(do_norm))
    us = md.utt_split(FOLD)
    tr_u = us["train"]
    print(f"{GROUP} fold {FOLD}: {len(tr_u)} training utterances, {len(us['val'])} val, {len(us['test'])} test")

    import yaml
    enc_spec = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())["encoders"][cfg["encoder"]]
    mcfg = cfg["model"]
    layers = list(range(int(mcfg["layers"][0]), int(mcfg["layers"][1]) + 1)) if mcfg["layer_mode"] == "ws" else [int(mcfg["layers"][1])]

    heads = {}
    for t in tasks:
        n = t.name
        td = md.td[n]
        rows = md.rows(n, tr_u)
        reg = t.kind == "reg"
        side = td.side
        mu_s, sd_s = side[rows].mean(0), side[rows].std(0)
        sd_s[sd_s < 1e-6] = 1.0
        h = {
            "kind": t.kind, "table": t.table, "pool_kind": td.pool_kind,
            "d_side": int(side.shape[1]), "d_out": 1 if reg else len(t.labels),
            "n_train_rows": int(len(rows)),
            "side_mean": mu_s.astype(float).tolist(), "side_std": sd_s.astype(float).tolist(),
        }
        if reg:
            yi = td.yi
            h["target_mean"], h["target_std"] = float(yi[rows].mean()), float(yi[rows].std()) or 1.0
        else:
            h["classes"] = [int(c) for c in t.labels]
            h["decision"] = {"rule": "argmax_of_probabilities", "positive_threshold": 0.5,
                             "note": "saved MT-GN15 flags equal (mean of 3 seeds' P(class 1)) > 0.5 on all 47,369 test phones"}
        heads[n] = h
        print(f"  {n:20s} {t.kind} {td.pool_kind:7s} rows {len(rows):7d} d_side {side.shape[1]:3d}" + (f" mu {h['target_mean']:.4f} sd {h['target_std']:.4f}" if reg else ""))

    side_names = {}
    for table in sorted({h["table"] for h in heads.values()}):
        side_names[table] = json.loads((FT.se.tables_dir() / f"{table}_side_names.json").read_text())

    out = {
        "source": {"tag": "mt63_gradnorm_a15", "group": GROUP, "fold": FOLD, "variant": a.variant,
                   "checkpoints": "results/models/multitask/mt63_gradnorm_a15/so762/fold0_seed{0,1,2}.pt (served: seed 0)"},
        "encoder": {"key": cfg["encoder"], "hf_id": enc_spec["hf_id"], "revision": enc_spec.get("revision"), "do_normalize": do_norm,
                    "sample_rate": FT.se.SR},
        "model": {"layer_ids": layers, "layer_mode": mcfg["layer_mode"], "head": mcfg["head"], "hidden": int(mcfg["hidden"]),
                  "head_dropout": float(mcfg["head_dropout"]), "alpha_mode": mcfg.get("alpha_mode", "per_head")},
        "heads": heads,
        "side_names": side_names,
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
