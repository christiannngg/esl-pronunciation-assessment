"""
US 4.1 - frozen-encoder results table and comparison with the Sprint 3 baselines.

    python -m src.evaluation.report_frozen --encoders wavlm_base_plus hubert_base wav2vec2_base          # layer = heads.primary_layer
    python -m src.evaluation.report_frozen --encoders wavlm_base_plus --layer 8 --mode emb               # ablation / other layer

Needs the head predictions written by ``python -m src.models.frozen_heads`` (copy experiments/results/frozen/ back from Drive) and the
Sprint 3 outputs (results/tables/baseline/, experiments/results/<variant>/). Per (task, head):
  regression      plain
  classification  plain (no class weights)  |  cw (balanced class weights in the loss)  |  cw+thr (cw + validation-tuned decision weights, US 3.3)
The variant shown as the head's result is chosen on VALIDATION only (RMSE / macro-F1); test is scored once. CIs: speaker bootstrap.
References: US 3.1 = best Sprint 3 model per task by validation score; US 3.3 = best model after the validated fix. Deltas are paired
speaker-level bootstraps of the primary metric (RMSE for regression, macro-F1 for classification) on the identical test rows.

Outputs (results/tables/frozen/):
  <encoder>_L<layer>_<mode>_results.csv | .md     every task x head x variant, val and test, test CIs, train score, seed spread
  frozen_vs_baseline_L<layer>_<mode>.csv | .md    headline: best head per encoder vs US 3.1 / US 3.3
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.report_baseline import OUT as BASELINE_OUT
from src.evaluation.report_baseline import CLF_KEYS, REG_KEYS, _score
from src.evaluation.report_us33 import _groups, _test_preds, decode, fit_weights, cross_fit_predict
from src.models import frozen_heads as FH
from src.models.baselines import REPO, build_tasks, load_config as load_baseline_config, split_masks

warnings.filterwarnings("ignore")
OUT = REPO / "results" / "tables" / "frozen"


def _fast(task):
    return M.fast_reg if task.kind == "reg" else M.fast_clf_factory(len(task.labels), None if task.positive is None else list(task.labels).index(task.positive))


def _enc(task, arr):
    if task.kind == "reg":
        return np.asarray(arr, float)
    idx = {l: i for i, l in enumerate(task.labels)}
    return np.array([idx[v] for v in arr])


def primary(task, y, p) -> float:
    r = _score(task, y, p)
    return r["rmse"] if task.kind == "reg" else r["macro_f1"]


def better(task, a, b) -> bool:
    return a < b if task.kind == "reg" else a > b


def pooled(base, task, head, config, stage, split, y_all, meta, proba=False):
    ys, ps, ms, pr, tr = [], [], [], [], []
    for fold in task.folds:
        d = np.load(FH.job_path(base, task.name, head, config, fold, stage), allow_pickle=True)
        mask = split_masks(task, meta, fold)[split]
        ys.append(y_all[mask]); ps.append(d[f"{split}_pred"]); ms.append(meta[mask])
        if proba:
            pr.append(d[f"{split}_proba"])
        tr.append(float(d["train_score"]))
    out = {"y": np.concatenate(ys), "pred": np.concatenate(ps), "meta": pd.concat(ms, ignore_index=True), "train_score": float(np.mean(tr))}
    if proba:
        out["proba"] = np.vstack(pr)
    if "val_score_seeds" in d.files:
        out["seed_sd"] = float(np.std(d["val_score_seeds"]))
    return out


def apply_variants(task, group, dv, dt):
    """[(variant, val_pred, test_pred)] for one selected config: plain / cw argmax, and for the cw group also cw+thr (US 3.3 decision
    weights: tuned on the whole validation set for test; cross-fitted on validation for the validation estimate)."""
    out = [("plain" if group == "none" else "cw", dv["pred"], dt["pred"])]
    if group == "balanced":
        yi = _enc(task, dv["y"])
        pv = decode(task, cross_fit_predict(task, dv["proba"], yi, _groups(task, dv["meta"])))
        w = fit_weights(task, dv["proba"], yi)
        out.append(("cw+thr", pv, decode(task, (dt["proba"] * w).argmax(1))))
    return out


def evaluate(encoder: str, layer, mode: str, n_boot: int | None = None, root=None):
    """Long table + {(task, head): test dict of the headline variant} for one encoder."""
    bcfg = load_baseline_config()
    hcfg = FH.load_config()["heads"]
    n_boot = n_boot or bcfg["bootstrap"]["n_boot"]
    base = FH.results_root(encoder, layer, mode, root)
    rows, head_preds = [], {}
    for task in FH.main_tasks(None, hcfg):
        y_all, meta = FH.load_labels(task)
        for head in hcfg["heads"]:
            cands = []
            for group in FH.groups_for(task):
                sel = FH.select_config(base, task, hcfg, head, group, y_all, meta)
                if sel is None:
                    continue
                config = sel[1]
                stage = "final" if all(FH.job_path(base, task.name, head, config, f, "final").exists() for f in task.folds) else "search"
                dv = pooled(base, task, head, config, stage, "val", y_all, meta, proba=task.kind == "clf")
                dt = pooled(base, task, head, config, stage, "test", y_all, meta, proba=task.kind == "clf")
                for vname, pv, pt in apply_variants(task, group, dv, dt):
                    cands.append({"variant": vname, "config": config, "stage": stage, "pv": pv, "pt": pt, "dv": dv, "dt": dt,
                                  "val_primary": primary(task, dv["y"], pv)})
            if not cands:
                continue
            best = cands[0]
            for c in cands[1:]:
                best = c if better(task, c["val_primary"], best["val_primary"]) else best
            for c in cands:
                for split, y, p, mm in (("val", c["dv"]["y"], c["pv"], c["dv"]["meta"]), ("test", c["dt"]["y"], c["pt"], c["dt"]["meta"])):
                    row = {"encoder": encoder, "layer": layer, "mode": mode, "task": task.name, "level": task.level, "dataset": task.dataset,
                           "kind": task.kind, "head": head, "variant": c["variant"], "config": c["config"], "stage": c["stage"], "split": split,
                           "headline": c is best, **_score(task, y, p)}
                    if split == "test":
                        ci = M.speaker_bootstrap(mm["speaker"].to_numpy(), _enc(task, y), _enc(task, p), _fast(task), n_boot=n_boot, seed=bcfg["bootstrap"]["seed"])
                        row.update({f"{k}_{s}": v for k, (lo, hi) in ci.items() for s, v in (("lo", lo), ("hi", hi))})
                        row["train_primary"] = c["dt"]["train_score"] if task.kind == "reg" else -c["dt"]["train_score"]
                        row["val_seed_sd"] = c["dt"].get("seed_sd", np.nan)
                    rows.append(row)
            head_preds[(task.name, head)] = {"y": best["dt"]["y"], "pred": best["pt"], "meta": best["dt"]["meta"], "variant": best["variant"],
                                              "val_primary": best["val_primary"]}
    return pd.DataFrame(rows), head_preds


def baseline_refs(task, bcfg):
    ba = pd.read_csv(BASELINE_OUT / "us33_before_after.csv")
    g = ba[ba.task == task.name]
    asc = task.kind == "reg"
    r31 = g.sort_values("val_primary_before", ascending=asc).iloc[0]
    r33 = g.sort_values("val_primary_after", ascending=asc).iloc[0]
    variant, thr = r33.fix.replace("+thr", ""), r33.fix.endswith("+thr")
    return ({"model": r31.model, "fix": "none", **_test_preds(task, r31.model, "us31", False, bcfg)},
            {"model": r33.model, "fix": r33.fix, **_test_preds(task, r33.model, variant, thr, bcfg)})


def compare(preds_by_encoder: dict, layer, mode: str, bcfg) -> pd.DataFrame:
    rows = []
    for task in build_tasks(bcfg).values():
        if not any((task.name, h) in p for p in preds_by_encoder.values() for h in FH.HEADS):
            continue
        r31, r33 = baseline_refs(task, bcfg)
        fast = _fast(task)
        key = "rmse" if task.kind == "reg" else "macro_f1"
        y = _enc(task, r31["y"])
        spk = r31["meta"]["speaker"].to_numpy()
        for tag, ref in (("US 3.1", r31), ("US 3.3", r33)):
            rows.append({"task": task.name, "system": tag, "detail": f"{ref['model']} ({ref['fix']})", **fast(y, _enc(task, ref["pred"]))})
        for enc, preds in preds_by_encoder.items():
            cand = [(h, preds[(task.name, h)]) for h in FH.HEADS if (task.name, h) in preds]
            if not cand:
                continue
            head, p = sorted(cand, key=lambda hp: hp[1]["val_primary"], reverse=(task.kind == "clf"))[0]
            assert (_enc(task, p["y"]) == y).all(), f"{task.name}: frozen test rows differ from Sprint 3 test rows"
            pe = _enc(task, p["pred"])
            row = {"task": task.name, "system": enc, "detail": f"{head} ({p['variant']})", **fast(y, pe)}
            for tag, ref in (("us31", r31), ("us33", r33)):
                t = M.paired_speaker_bootstrap(spk, y, _enc(task, ref["pred"]), pe, fast, key, n_boot=500)
                row.update({f"d_{tag}": t["delta"], f"d_{tag}_lo": t["lo"], f"d_{tag}_hi": t["hi"], f"d_{tag}_p": t["p"]})
            rows.append(row)
    return pd.DataFrame(rows)


def _cell(r, kind):
    main = f"RMSE {r.rmse:.3f} / PCC {r.pcc:.3f}" if kind == "reg" else f"macro-F1 {r.macro_f1:.3f} / bal.acc {r.balanced_acc:.3f}"
    if pd.notna(r.get("d_us33")):
        main += f"<br>vs US 3.3 {r.d_us33:+.3f} [{r.d_us33_lo:+.3f}, {r.d_us33_hi:+.3f}] p={r.d_us33_p:.3f}"
    return main


def write_compare_md(df: pd.DataFrame, path, layer, mode) -> None:
    lines = [f"# Frozen encoders vs Sprint 3 baselines (layer {layer}, inputs: {mode}) - test set", "",
             "US 3.1 = best Sprint 3 model per task by validation score; US 3.3 = best model after the validated fix. Frozen rows: best head chosen on "
             "validation (config and class-weight / threshold variant too). Delta = frozen minus US 3.3 on RMSE (regression, lower is better) or macro-F1 "
             "(classification, higher is better), paired speaker bootstrap on identical test rows.", ""]
    for task, g in df.groupby("task", sort=False):
        kind = "reg" if g["rmse"].notna().any() else "clf"
        lines += [f"## {task}", "", "| system | detail | test |", "|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.system} | {r.detail} | {_cell(pd.Series(r._asdict()), kind)} |")
        lines.append("")
    path.write_text("\n".join(lines))


def write_results_md(df: pd.DataFrame, path, encoder, layer, mode) -> None:
    t = df[(df.split == "test") & df.headline]
    lines = [f"# {encoder}, layer {layer}, inputs {mode} - headline variant per task x head (test, 95% speaker-bootstrap CIs)", "",
             "train_primary / val_seed_sd expose over-fitting and seed noise: RMSE for regression, macro-F1 for classification.", "",
             "| task | head | variant | config | test | train_primary | val seed sd |", "|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        txt = (f"RMSE {r.rmse:.3f} [{r.rmse_lo:.3f}, {r.rmse_hi:.3f}] · PCC {r.pcc:.3f}" if r.kind == "reg"
               else f"macro-F1 {r.macro_f1:.3f} [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}] · bal.acc {r.balanced_acc:.3f}")
        lines.append(f"| {r.task} | {r.head} | {r.variant} | {r.config} | {txt} | {r.train_primary:.3f} | {r.val_seed_sd:.4f} |")
    path.write_text("\n".join(lines))


def log_wandb(df: pd.DataFrame, encoder, layer, mode) -> None:
    import os
    os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    from src.utils.tracking import init_run
    for (task, head), g in df[df.headline].groupby(["task", "head"], sort=False):
        run = init_run(job_type="frozen-encoder", config={"task": task, "head": head, "encoder": encoder, "layer": str(layer), "mode": mode,
                                                           "variant": g.iloc[0]["variant"], "config": g.iloc[0]["config"]},
                       tags=["sprint4", "frozen", encoder])
        for r in g.to_dict("records"):
            run.summary.update({f"{r['split']}/{k}": r[k] for k in REG_KEYS + CLF_KEYS + ["n"] if k in r and pd.notna(r[k])})
            if r["split"] == "test":
                run.summary.update({f"test/{k}": r[k] for k in r if k.endswith(("_lo", "_hi")) and pd.notna(r[k])})
        run.finish()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--mode", default="emb+side", choices=FH.MODES)
    ap.add_argument("--results-root", default=None, help="folder holding <encoder>/L<layer>/<mode>/... (default experiments/results/frozen)")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    layer = a.layer if a.layer is not None else FH.load_config()["heads"]["primary_layer"]
    OUT.mkdir(parents=True, exist_ok=True)
    bcfg = load_baseline_config()
    preds = {}
    from pathlib import Path
    for enc in a.encoders:
        df, preds[enc] = evaluate(enc, layer, a.mode, root=Path(a.results_root) if a.results_root else None)
        stem = OUT / f"{enc}_L{layer}_{a.mode}"
        df.to_csv(f"{stem}_results.csv", index=False)
        write_results_md(df, Path(f"{stem}_results.md"), enc, layer, a.mode)
        if a.wandb:
            log_wandb(df, enc, layer, a.mode)
        print(f"[report] {enc}: {len(df)} rows -> {stem}_results.csv")
    cmp = compare(preds, layer, a.mode, bcfg)
    cmp.to_csv(OUT / f"frozen_vs_baseline_L{layer}_{a.mode}.csv", index=False)
    write_compare_md(cmp, OUT / f"frozen_vs_baseline_L{layer}_{a.mode}.md", layer, a.mode)
    print(f"[report] comparison -> {OUT}/frozen_vs_baseline_L{layer}_{a.mode}.md")


if __name__ == "__main__":
    main()
