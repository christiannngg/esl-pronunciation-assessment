"""
Aggregate fitted baseline jobs into results tables (US 3.1 / US 3.3 share this code).

Selection rule (validation only): per (task, model) pick the config with the best mean validation score over folds
(lowest RMSE for regression, highest balanced accuracy for classification; RMSE rather than MAE because on these
skewed targets MAE rewards a median-collapsed predictor, see the dummy_median row). Test predictions of ONLY that config are
loaded. For L2-ARCTIC the 4 test folds partition the 24 speakers, so pooled test predictions cover each speaker once.

Outputs (results/tables/baseline/):
  <variant>_results.csv        long table: task x model x {val, test} metrics, test 95% speaker-bootstrap CIs
  <variant>_results.md         readable table (test), best model per task in bold, dummy reference
  <variant>_selected.csv       selected config per task/model and val score
Predictions of the selected configs are written to experiments/results/<variant>/predictions/ (val and test files separate).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.models.baselines import (REPO, build_tasks, configs_for, job_path, load_config, load_task, results_root,
                                  split_masks)

warnings.filterwarnings("ignore", message="All-NaN slice")   # PCC of a constant (dummy) predictor is undefined
OUT = REPO / "results" / "tables" / "baseline"
CLF_KEYS = ["f1", "precision", "recall", "macro_f1", "balanced_acc", "accuracy"]
REG_KEYS = ["mae", "rmse", "pcc", "spearman"]


def _encode(task, arr):
    """Integer-code labels for fast metrics."""
    if task.labels is None:
        return np.asarray(arr)
    idx = {l: i for i, l in enumerate(task.labels)}
    return np.array([idx[v] for v in arr])


def _score(task, y, p) -> dict:
    if task.kind == "reg":
        return M.regression_metrics(y, p)
    out = M.classification_metrics(y, p, labels=list(task.labels), positive=task.positive)
    return out


def _load_split(variant, task, model, config, split, y, meta):
    ys, ps, ms = [], [], []
    for fold in task.folds:
        d = np.load(job_path(variant, task.name, model, config, fold), allow_pickle=True)
        mask = split_masks(task, meta, fold)[split]
        ys.append(y[mask]); ps.append(d[f"{split}_pred"]); ms.append(meta[mask])
    return np.concatenate(ys), np.concatenate(ps), pd.concat(ms, ignore_index=True)


def select_config(variant, task, model, cfg, y, meta):
    """Best config by mean per-fold VALIDATION score. Never touches test predictions."""
    best = None
    for m, config, _ in configs_for(task, cfg):
        if m != model:
            continue
        scores = []
        for fold in task.folds:
            d = np.load(job_path(variant, task.name, m, config, fold), allow_pickle=True)
            yv = y[split_masks(task, meta, fold)["val"]]
            r = _score(task, yv, d["val_pred"])
            scores.append(r["rmse"] if task.kind == "reg" else -r["balanced_acc"])
        s = float(np.mean(scores))
        if best is None or s < best[0]:
            best = (s, config)
    return best[1], (best[0] if task.kind == "reg" else -best[0])


def report(variant: str, wandb_log: bool = True, n_boot: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    n_boot = n_boot or cfg["bootstrap"]["n_boot"]
    OUT.mkdir(parents=True, exist_ok=True)
    pred_dir = results_root(variant) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, sel_rows = [], []
    for task in build_tasks(cfg).values():
        _X, y_all, meta, _names = load_task(task)
        del _X
        models = sorted({m for m, _, _ in configs_for(task, cfg)}, key=lambda m: (not m.startswith("dummy"), m))
        fast = M.fast_reg if task.kind == "reg" else M.fast_clf_factory(len(task.labels), None if task.positive is None else list(task.labels).index(task.positive))
        for model in models:
            config, val_score = select_config(variant, task, model, cfg, y_all, meta)
            sel_rows.append({"task": task.name, "model": model, "config": config, "val_selection_score": val_score})
            for split in ("val", "test"):
                y, p, mm = _load_split(variant, task, model, config, split, y_all, meta)
                r = _score(task, y, p)
                row = {"task": task.name, "level": task.level, "dataset": task.dataset, "kind": task.kind, "model": model,
                       "config": config, "split": split, **r}
                if split == "test":
                    ci = M.speaker_bootstrap(mm["speaker"].to_numpy(), _encode(task, y), _encode(task, p), fast, n_boot=n_boot, seed=cfg["bootstrap"]["seed"])
                    for k, (lo, hi) in ci.items():
                        row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
                rows.append(row)
                keep = [c for c in ["utt_id", "speaker", "l1", "fold", "gender", "age_group", "word_idx", "phone_idx", "phone", "anchor_phone", "word", "error_class"] if c in mm]
                pd.DataFrame({**{c: mm[c] for c in keep}, "y": y, "pred": p}).to_csv(
                    pred_dir / f"{task.name}__{model}__{split}.csv.gz", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{variant}_results.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / f"{variant}_selected.csv", index=False)
    write_markdown(df, OUT / f"{variant}_results.md", variant)
    if wandb_log:
        log_wandb(df, variant)
    return df


def _fmt(row, kind, positive):
    def ci(k, d=3):
        lo, hi = row.get(f"{k}_lo"), row.get(f"{k}_hi")
        return f"{row[k]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if pd.notna(lo) else f"{row[k]:.{d}f}"
    if kind == "reg":
        pcc = ci('pcc') if pd.notna(row['pcc']) else "n/a"
        return f"MAE {ci('mae')} · RMSE {ci('rmse')} · PCC {pcc}"
    if pd.notna(row.get("f1")):
        return f"F1 {ci('f1')} · bal.acc {ci('balanced_acc')}"
    return f"macro-F1 {ci('macro_f1')} · bal.acc {ci('balanced_acc')}"


def write_markdown(df: pd.DataFrame, path: Path, variant: str) -> None:
    t = df[df["split"] == "test"]
    lines = [f"# Baseline results ({variant}) - test set, 95% speaker-bootstrap CIs in brackets", "",
             "Hyper-parameters chosen on validation only (regression: RMSE; classification: balanced accuracy). Bold = best non-dummy model by "
             "PCC (regression) or F1 / macro-F1 (classification). dummy = mean / majority predictor, dummy_median = median predictor.",
             "Human-agreement reference (Sprint 2 EDA): sentence-level leave-one-out r about 0.77-0.80; phone-level Fleiss kappa 0.46.", ""]
    for task, g in t.groupby("task", sort=False):
        kind = g["kind"].iloc[0]
        key = "pcc" if kind == "reg" else ("f1" if g["f1"].notna().any() else "macro_f1")
        best = g[~g["model"].str.startswith("dummy")].sort_values(key, ascending=False).iloc[0]["model"]
        lines += [f"## {task}  ({g['level'].iloc[0]}, {g['dataset'].iloc[0]}, n={int(g['n'].iloc[0]):,})", "",
                  "| model | config | test |", "|---|---|---|"]
        for r in g.sort_values("model", key=lambda s: ~s.str.startswith("dummy")).to_dict("records"):
            txt = _fmt(r, kind, None)
            lines.append(f"| {'**' + r['model'] + '**' if r['model'] == best else r['model']} | {r['config']} | {txt} |")
        lines.append("")
    path.write_text("\n".join(lines))


def log_wandb(df: pd.DataFrame, variant: str) -> None:
    """One W&B run per (task, model). Offline unless WANDB_API_KEY is set; sync later with `wandb sync wandb/offline-run-*`."""
    os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    from src.utils.tracking import init_run
    for (task, model), g in df.groupby(["task", "model"], sort=False):
        cfgrow = g.iloc[0]
        run = init_run(job_type="baseline-eval",
                       config={"task": task, "model": model, "config": cfgrow["config"], "features": "mfcc+delta", "variant": variant},
                       tags=["sprint3", "traditional-ml", variant])
        for r in g.to_dict("records"):
            run.summary.update({f"{r['split']}/{k}": r[k] for k in REG_KEYS + CLF_KEYS + ["n"] if k in r and pd.notna(r[k])})
            if r["split"] == "test":
                run.summary.update({f"test/{k}": r[k] for k in r if k.endswith(("_lo", "_hi")) and pd.notna(r[k])})
        run.finish()
