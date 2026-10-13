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
    ys, ps, ms, pr, tr, trp, try_ = [], [], [], [], [], [], []
    for fold in task.folds:
        d = np.load(FH.job_path(base, task.name, head, config, fold, stage), allow_pickle=True)
        mask = split_masks(task, meta, fold)[split]
        ys.append(y_all[mask]); ps.append(d[f"{split}_pred"]); ms.append(meta[mask])
        if proba:
            pr.append(d[f"{split}_proba"])
        tr.append(float(d["train_score"]))
        if "train_proba" in d.files:
            trp.append(d["train_proba"]); try_.append(d["train_y"])
    out = {"y": np.concatenate(ys), "pred": np.concatenate(ps), "meta": pd.concat(ms, ignore_index=True), "train_score": float(np.mean(tr))}
    if proba:
        out["proba"] = np.vstack(pr)
    if trp:
        out["train_proba"], out["train_y"] = np.vstack(trp), np.concatenate(try_)
    if "val_score_seeds" in d.files:
        out["seed_sd"] = float(np.std(d["val_score_seeds"]))
    return out


def apply_variants(task, group, dv, dt):
    """[(variant, val_pred, test_pred, train_primary)] for one selected config: plain / cw argmax, and for the cw group also cw+thr
    (US 3.3 decision weights: tuned on the whole validation set for test; cross-fitted on validation for the validation estimate).
    train_primary (RMSE / macro-F1 on a training subsample) is scored with the SAME decision rule as the variant; NaN when the
    result file predates the saved train probabilities (re-run the final stage with --force)."""
    base_train = dt["train_score"] if task.kind == "reg" else -dt["train_score"]
    out = [("plain" if group == "none" else "cw", dv["pred"], dt["pred"], base_train)]
    if group == "balanced":
        yi = _enc(task, dv["y"])
        pv = decode(task, cross_fit_predict(task, dv["proba"], yi, _groups(task, dv["meta"])))
        w = fit_weights(task, dv["proba"], yi)
        tr = np.nan
        if "train_proba" in dt:
            tr = primary(task, decode(task, dt["train_y"]), decode(task, (dt["train_proba"] * w).argmax(1)))
        out.append(("cw+thr", pv, decode(task, (dt["proba"] * w).argmax(1)), tr))
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
                for vname, pv, pt, trp in apply_variants(task, group, dv, dt):
                    cands.append({"variant": vname, "config": config, "stage": stage, "pv": pv, "pt": pt, "dv": dv, "dt": dt,
                                  "train_primary": trp, "val_primary": primary(task, dv["y"], pv)})
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
                        row["train_primary"] = c["train_primary"]
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


# --------------------------------------------------------------------------- #
# US 4.2: per-layer probe profile and input ablation
# --------------------------------------------------------------------------- #
ENC_STYLE = {"wav2vec2_base": ("#2a78d6", "o"), "hubert_base": ("#eb6834", "s"), "wavlm_base_plus": ("#1baf7a", "^")}   # fixed colour per entity
ENC_LABEL = {"wav2vec2_base": "wav2vec 2.0", "hubert_base": "HuBERT", "wavlm_base_plus": "WavLM"}


def layer_profile(encoders, layers, mode: str = "emb+side", root=None) -> pd.DataFrame:
    """One linear probe per (encoder, layer, task): the single probe config of configs/frozen.yaml heads.probe, scored with the plain
    argmax rule (regression: RMSE; classification: macro-F1 with balanced class weights in the loss, no decision weights).
    Layers are compared on VALIDATION; test is shown next to it, never used to pick a layer."""
    hcfg = FH.probe_cfg(FH.load_config()["heads"])
    rows = []
    for task in FH.main_tasks(None, hcfg):
        y_all, meta = FH.load_labels(task)
        group = "none" if task.kind == "reg" else "balanced"
        for enc in encoders:
            for layer in layers:
                base = FH.results_root(enc, layer, mode, root)
                sel = FH.select_config(base, task, hcfg, "linear", group, y_all, meta)
                if sel is None:
                    continue
                dv = pooled(base, task, "linear", sel[1], "search", "val", y_all, meta)
                dt = pooled(base, task, "linear", sel[1], "search", "test", y_all, meta)
                rv, rt = _score(task, dv["y"], dv["pred"]), _score(task, dt["y"], dt["pred"])
                k = "rmse" if task.kind == "reg" else "macro_f1"
                rows.append({"task": task.name, "kind": task.kind, "encoder": enc, "layer": int(layer), "mode": mode, "config": sel[1],
                             "val": rv[k], "test": rt[k], "val_aux": rv["pcc"] if task.kind == "reg" else rv["balanced_acc"],
                             "test_aux": rt["pcc"] if task.kind == "reg" else rt["balanced_acc"]})
    return pd.DataFrame(rows)


def write_layer_profile(df: pd.DataFrame, stem, mode: str) -> None:
    """CSV + markdown (best layer per encoder by validation) + a small-multiples figure of the validation curves."""
    df.to_csv(f"{stem}.csv", index=False)
    lines = [f"# Per-layer linear probe ({mode}) - layer chosen on VALIDATION, test shown at that layer", "",
             "Regression: RMSE (lower is better). Classification: macro-F1 (higher is better). One linear head, one seed per layer; layer 0 = conv front end.", ""]
    for task, g in df.groupby("task", sort=False):
        kind = g["kind"].iloc[0]
        lines += [f"## {task}", "", "| encoder | best layer (val) | val | test | layer 12 val | layer 12 test |", "|---|---|---|---|---|---|"]
        for enc, ge in g.groupby("encoder", sort=False):
            b = ge.sort_values("val", ascending=(kind == "reg")).iloc[0]
            l12 = ge[ge.layer == 12]
            last = f"{l12.val.iloc[0]:.3f} | {l12.test.iloc[0]:.3f}" if len(l12) else "n/a | n/a"
            lines.append(f"| {enc} | {int(b.layer)} | {b.val:.3f} | {b.test:.3f} | {last} |")
        lines.append("")
    open(f"{stem}.md", "w").write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tasks = list(dict.fromkeys(df.task))
    ncol = 3
    fig, axes = plt.subplots((len(tasks) + ncol - 1) // ncol, ncol, figsize=(12, 3.4 * ((len(tasks) + ncol - 1) // ncol)), squeeze=False, sharex=True)
    for ax, task in zip(axes.ravel(), tasks):
        g = df[df.task == task]
        kind = g["kind"].iloc[0]
        for enc, ge in g.groupby("encoder", sort=False):
            col, mk = ENC_STYLE.get(enc, ("#52514e", "o"))
            ge = ge.sort_values("layer")
            ax.plot(ge.layer, ge.val, color=col, marker=mk, ms=4.5, lw=1.8, label=ENC_LABEL.get(enc, enc))
            b = ge.sort_values("val", ascending=(kind == "reg")).iloc[0]
            ax.plot([b.layer], [b.val], marker=mk, ms=9, mfc="white", mec=col, mew=1.8, zorder=5)       # best layer by validation
        ax.set_title(f"{task}  ({'RMSE, lower is better' if kind == 'reg' else 'macro-F1, higher is better'})", fontsize=9, loc="left", color="#0b0b0b")
        ax.set_xticks(range(0, 13, 2))
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#9a9992")
        ax.tick_params(colors="#52514e", labelsize=8)
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("transformer layer (0 = conv front end)", fontsize=8, color="#52514e")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9)
    fig.text(0.5, 0.005, "validation score of a linear probe; open marker = best layer on validation", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(f"{stem}.png", dpi=160)
    plt.close(fig)


def best_head(preds, task_name, task_kind):
    cand = [(h, preds[(task_name, h)]) for h in FH.HEADS if (task_name, h) in preds]
    if not cand:
        return None
    return sorted(cand, key=lambda hp: hp[1]["val_primary"], reverse=(task_kind == "clf"))[0]


def ablation(encoder: str, layer, modes=("emb+side", "emb", "side"), root=None, n_boot: int = 500) -> pd.DataFrame:
    """Same protocol per input setting (best head and variant chosen on validation inside each setting); paired speaker bootstrap of
    the full setting (emb+side) minus each ablated setting on the identical test rows."""
    bcfg = load_baseline_config()
    preds = {}
    for m in modes:
        df, preds[m] = evaluate(encoder, layer, m, n_boot=20, root=root)
        if df.empty:
            raise SystemExit(f"[report] no results for {encoder} layer {layer} mode {m}; run frozen_heads with --mode {m}")
    rows = []
    for task in build_tasks(bcfg).values():
        if not any((task.name, h) in preds["emb+side"] for h in FH.HEADS):
            continue
        key = "rmse" if task.kind == "reg" else "macro_f1"
        fast = _fast(task)
        full_head, full = best_head(preds["emb+side"], task.name, task.kind)
        y = _enc(task, full["y"])
        spk = full["meta"]["speaker"].to_numpy()
        for m in modes:
            head, p = best_head(preds[m], task.name, task.kind)
            assert (_enc(task, p["y"]) == y).all()
            row = {"task": task.name, "encoder": encoder, "layer": layer, "mode": m, "head": head, "variant": p["variant"], **fast(y, _enc(task, p["pred"]))}
            if m != "emb+side":
                t = M.paired_speaker_bootstrap(spk, y, _enc(task, p["pred"]), _enc(task, full["pred"]), fast, key, n_boot=n_boot)
                row.update({"full_minus_this": t["delta"], "lo": t["lo"], "hi": t["hi"], "p": t["p"]})
            rows.append(row)
    return pd.DataFrame(rows)


def write_ablation_md(df: pd.DataFrame, path, encoder, layer) -> None:
    lines = [f"# Input ablation: {encoder}, layer {layer} - test set", "",
             "`emb+side` = embedding + Sprint 3 non-acoustic columns (main setting); `emb` = embedding only; `side` = non-acoustic columns only (no audio). "
             "Each setting uses its own validation-selected head / variant. Last column: (emb+side) minus this setting on the primary metric "
             "(RMSE: negative means the full setting is better; macro-F1: positive means it is better), paired speaker bootstrap.", ""]
    for task, g in df.groupby("task", sort=False):
        reg = g["rmse"].notna().any()
        lines += [f"## {task}", "", "| setting | head (variant) | test | full minus this (95% CI, p) |", "|---|---|---|---|"]
        for r in g.itertuples():
            val = f"RMSE {r.rmse:.3f} / PCC {r.pcc:.3f}" if reg else f"macro-F1 {r.macro_f1:.3f} / bal.acc {r.balanced_acc:.3f}"
            d = "" if pd.isna(getattr(r, "p", np.nan)) else f"{r.full_minus_this:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] p={r.p:.3f}"
            lines.append(f"| {r.mode} | {r.head} ({r.variant}) | {val} | {d} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--mode", default="emb+side", choices=FH.MODES)
    ap.add_argument("--results-root", default=None, help="folder holding <encoder>/L<layer>/<mode>/... (default experiments/results/frozen)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--layer-profile", default=None, metavar="LAYERS", help="per-layer probe table + figure, e.g. 0-12 (needs run --probe results)")
    ap.add_argument("--ablation", action="store_true", help="emb+side vs emb vs side for each encoder (needs run --mode emb / side results)")
    a = ap.parse_args(argv)
    layer = a.layer if a.layer is not None else FH.load_config()["heads"]["primary_layer"]
    OUT.mkdir(parents=True, exist_ok=True)
    bcfg = load_baseline_config()
    preds = {}
    from pathlib import Path
    root = Path(a.results_root) if a.results_root else None
    if a.layer_profile:
        lo, hi = (a.layer_profile.split("-") + [a.layer_profile])[:2] if "-" in a.layer_profile else (a.layer_profile, a.layer_profile)
        df = layer_profile(a.encoders, range(int(lo), int(hi) + 1), a.mode, root)
        if df.empty:
            raise SystemExit("[report] no probe results found; run `python -m src.models.frozen_heads run --probe --layers 0-12` first")
        write_layer_profile(df, OUT / f"layer_profile_{a.mode}", a.mode)
        print(f"[report] layer profile -> {OUT}/layer_profile_{a.mode}.(csv|md|png)")
        return
    if a.ablation:
        for enc in a.encoders:
            df = ablation(enc, layer, root=root)
            df.to_csv(OUT / f"ablation_{enc}_L{layer}.csv", index=False)
            write_ablation_md(df, OUT / f"ablation_{enc}_L{layer}.md", enc, layer)
            print(f"[report] ablation -> {OUT}/ablation_{enc}_L{layer}.md")
        return
    for enc in a.encoders:
        df, preds[enc] = evaluate(enc, layer, a.mode, root=Path(a.results_root) if a.results_root else None)
        if df.empty:
            where = FH.results_root(enc, layer, a.mode, Path(a.results_root) if a.results_root else None)
            raise SystemExit(f"[report] no head results found for {enc} under {where}\n"
                             f"        expected {where}/<task>/<head>/<config>/fold*.npz - copy the Drive frozen_results folder there or pass --results-root")
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
