"""
US 3.3 - choose the fix per (task, model) on VALIDATION, then score test once and write before/after tables.

    python -m src.evaluation.report_us33 val      # candidate table on validation only (no test predictions are read)
    python -m src.evaluation.report_us33 final    # freeze choices, evaluate test, paired speaker bootstrap vs US 3.1

Candidates (same hyper-parameters as US 3.1; ``+thr`` = decision weights tuned on validation):
  classification: us31 | cw (balanced class weights) | cw+thr | pros (pitch/energy/pause features) | cw_pros | cw_pros+thr
  regression:     us31 | rs (resample low scores, power 0.5) | pros | rs_pros
Primary selection metric (validation): macro-F1 for every classification task (F1 of the positive class is degenerate
when prevalence is ~40%: predicting all-positive scores 0.59), RMSE for regression. F1, balanced accuracy, PCC are reported too.
``+thr`` candidates are scored with cross-fitted weights (tuned on the other validation folds / speaker half), so the
validation estimate is not optimistic; the weights applied to test are tuned on the whole validation set.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.report_baseline import OUT, _encode, _score
from src.models.baselines import REPO, build_tasks, configs_for, job_path, load_config, load_task, split_masks

warnings.filterwarnings("ignore")
CLF_CANDS = [("us31", False), ("cw", False), ("cw", True), ("pros", False), ("cw_pros", False), ("cw_pros", True)]
REG_CANDS = [("us31", False), ("rs", False), ("pros", False), ("rs_pros", False)]
_cache: dict = {}


def _tables(variant, task, cfg):
    key = (task.name,)
    if key not in _cache:
        _, y, meta, _ = load_task(task)
        _cache[key] = (y, meta)
    return _cache[key]


def get(variant, task, model, split, cfg, need_proba=False):
    """Pooled predictions of the single config of (variant, model). None if the job set does not exist."""
    config = next((c for m, c, _ in configs_for(task, cfg, variant) if m == model), None)
    if config is None:
        return None
    y_all, meta = _tables(variant, task, cfg)
    ys, ps, ms, pr = [], [], [], []
    for fold in task.folds:
        path = job_path(variant, task.name, model, config, fold)
        if not path.exists():
            return None
        d = np.load(path, allow_pickle=True)
        mask = split_masks(task, meta, fold)[split]
        ys.append(y_all[mask]); ps.append(d[f"{split}_pred"]); ms.append(meta[mask])
        if need_proba:
            if f"{split}_proba" not in d:
                return None
            pr.append(d[f"{split}_proba"]); classes = list(d["classes"])
    out = {"y": np.concatenate(ys), "pred": np.concatenate(ps), "meta": pd.concat(ms, ignore_index=True), "config": config}
    if need_proba:
        out["proba"], out["classes"] = np.vstack(pr), classes
    return out


def primary(task, y, p) -> float:
    r = _score(task, y, p)
    if task.kind == "reg":
        return r["rmse"]
    return r["macro_f1"]


def better(task, a, b) -> bool:
    return a < b if task.kind == "reg" else a > b


# ---- decision weights ------------------------------------------------------ #
def _fast(task):
    return M.fast_clf_factory(len(task.labels), None if task.positive is None else list(task.labels).index(task.positive))


def _target(task, m):
    return m["macro_f1"]


def fit_weights(task, proba, y_idx) -> np.ndarray:
    K = proba.shape[1]
    f = _fast(task)
    grid = np.exp(np.linspace(np.log(0.1), np.log(30), 40))
    w = np.ones(K)
    for _ in range(3):
        for c in range(1, K) if K > 2 else [1]:
            best = (-1.0, w[c])
            for g in grid:
                w2 = w.copy(); w2[c] = g
                s = _target(task, f(y_idx, (proba * w2).argmax(1)))
                if s > best[0]:
                    best = (s, g)
            w[c] = best[1]
    return w


def cross_fit_predict(task, proba, y_idx, groups) -> np.ndarray:
    pred = np.empty(len(y_idx), int)
    for g in np.unique(groups):
        te = groups == g
        w = fit_weights(task, proba[~te], y_idx[~te])
        pred[te] = (proba[te] * w).argmax(1)
    return pred


def _groups(task, meta):
    if task.splitter == "fold":
        return meta["fold"].to_numpy()
    spk = sorted(meta["speaker"].unique())
    return meta["speaker"].map({s: i % 2 for i, s in enumerate(spk)}).to_numpy()


def proba_in_label_order(task, d):
    """Probability matrix with columns in ``task.labels`` order (saved class names are strings)."""
    cast = type(task.labels[0])
    order = [list(task.labels).index(cast(c)) for c in d["classes"]]
    P = np.zeros_like(d["proba"])
    P[:, order] = d["proba"]
    return P


def decode(task, idx):
    return np.array(task.labels, dtype=object)[idx] if isinstance(task.labels[0], str) else np.array(task.labels)[idx]


# ---- validation table ------------------------------------------------------ #
def candidate_rows(cfg, with_test_for=None):
    rows = []
    for task in build_tasks(cfg).values():
        cands = CLF_CANDS if task.kind == "clf" else REG_CANDS
        models = list(dict.fromkeys(m for m, _, _ in configs_for(task, cfg, "us31") if not m.startswith("dummy")))
        for model in models:
            for variant, thr in cands:
                if variant != "us31" and model not in [m for m, _, _ in configs_for(task, cfg, variant)]:
                    continue
                if variant == "us31" and thr:
                    continue
                if variant == "us31":
                    sel = pd.read_csv(OUT / "us31_selected.csv").query("task == @task.name and model == @model")["config"].iloc[0]
                    d = _from_us31(task, model, sel, "val")
                else:
                    d = get(variant, task, model, "val", cfg, need_proba=thr)
                if d is None:
                    continue
                if thr:
                    yi = _encode(task, d["y"])
                    pred_idx = cross_fit_predict(task, proba_in_label_order(task, d), yi, _groups(task, d["meta"]))
                    pred = decode(task, pred_idx)
                else:
                    pred = d["pred"]
                    if task.kind == "clf":
                        pred = pred.astype(type(task.labels[0])) if not isinstance(task.labels[0], str) else pred
                r = _score(task, d["y"], pred)
                rows.append({"task": task.name, "model": model, "variant": variant + ("+thr" if thr else ""), "primary": primary(task, d["y"], pred),
                             **{k: r[k] for k in ("mae", "rmse", "pcc") if k in r}, **{k: r[k] for k in ("f1", "precision", "recall", "macro_f1", "balanced_acc") if k in r}})
    return pd.DataFrame(rows)


def _from_us31(task, model, config, split):
    d = get("us31", task, model, split, {**load_config()}, need_proba=False) if False else None
    cfg = load_config()
    y_all, meta = _tables("us31", task, cfg)
    ys, ps, ms = [], [], []
    for fold in task.folds:
        z = np.load(job_path("us31", task.name, model, config, fold), allow_pickle=True)
        mask = split_masks(task, meta, fold)[split]
        ys.append(y_all[mask]); ps.append(z[f"{split}_pred"]); ms.append(meta[mask])
    return {"y": np.concatenate(ys), "pred": np.concatenate(ps), "meta": pd.concat(ms, ignore_index=True), "config": config}


MARGIN = {"reg": 0.01, "clf": 0.01}        # a fix must beat US 3.1 on validation by >= 1% relative RMSE / +0.01 macro-F1


def _test_preds(task, model, variant, thr, cfg):
    """Test predictions of a chosen candidate. Decision weights (if any) are tuned on the whole validation set."""
    if variant == "us31":
        sel = pd.read_csv(OUT / "us31_selected.csv").query("task == @task.name and model == @model")["config"].iloc[0]
        return _from_us31(task, model, sel, "test")
    d = get(variant, task, model, "test", cfg, need_proba=thr)
    if thr:
        dv = get(variant, task, model, "val", cfg, need_proba=True)
        w = fit_weights(task, proba_in_label_order(task, dv), _encode(task, dv["y"]))
        d["pred"] = decode(task, (proba_in_label_order(task, d) * w).argmax(1))
        d["weights"] = w
    return d


def final():
    cfg = load_config()
    cand = pd.read_csv(OUT / "us33_val_candidates.csv")
    rows = []
    for (tname, model), g in cand.groupby(["task", "model"]):
        task = build_tasks(cfg)[tname]
        base = g[g.variant == "us31"].iloc[0]
        g2 = g.sort_values("primary", ascending=(task.kind == "reg"))
        best = g2.iloc[0]
        gain = (base.primary - best.primary) / base.primary if task.kind == "reg" else best.primary - base.primary
        chosen = best.variant if gain >= MARGIN[task.kind] else "us31"
        variant, thr = (chosen.replace("+thr", ""), chosen.endswith("+thr"))
        before = _test_preds(task, model, "us31", False, cfg)
        after = before if chosen == "us31" else _test_preds(task, model, variant, thr, cfg)
        assert (before["y"] == after["y"]).all()
        fast = M.fast_reg if task.kind == "reg" else _fast(task)
        y, pb, pa = _encode(task, before["y"]), _encode(task, before["pred"]), _encode(task, after["pred"])
        spk = before["meta"]["speaker"].to_numpy()
        keys = ["mae", "rmse", "pcc"] if task.kind == "reg" else ["macro_f1", "balanced_acc"] + (["f1", "precision", "recall"] if task.positive is not None else [])
        mb, ma = fast(y, pb), fast(y, pa)
        row = {"task": tname, "level": task.level, "dataset": task.dataset, "model": model, "fix": chosen,
               "val_primary_before": base.primary, "val_primary_after": g[g.variant == chosen].primary.iloc[0], "n_test": len(y)}
        for k in keys:
            row[f"before_{k}"], row[f"after_{k}"], row[f"delta_{k}"] = mb[k], ma[k], ma[k] - mb[k]
        pkey = "rmse" if task.kind == "reg" else "macro_f1"
        if chosen != "us31":
            t = M.paired_speaker_bootstrap(spk, y, pb, pa, fast, pkey, n_boot=500)
            row.update({f"delta_{pkey}_lo": t["lo"], f"delta_{pkey}_hi": t["hi"], f"delta_{pkey}_p": t["p"]})
        else:
            row.update({f"delta_{pkey}_lo": 0.0, f"delta_{pkey}_hi": 0.0, f"delta_{pkey}_p": 1.0})
        rows.append(row)
        print(tname, model, chosen, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "us33_before_after.csv", index=False)
    write_md(df)


def write_md(df):
    lines = ["# US 3.3 before / after (test set, same model, hyper-parameters unchanged)", "",
             "Fix chosen per (task, model) on validation (must beat US 3.1 by >= 1% RMSE or +0.01 macro-F1, else US 3.1 is kept). "
             "Delta CI / p: paired speaker-level bootstrap on the primary metric (RMSE for regression, macro-F1 for classification). "
             "`cw` = balanced class weights, `+thr` = validation-tuned decision weights, `pros` = pitch / energy / pause features, `rs` = low-score resampling.", ""]
    lines += ["## Headline: best model per task (chosen on validation after the fix) - test set", "",
              "| task | model | fix | before | after | change (95% CI, p) |", "|---|---|---|---|---|---|"]
    for tname, g in df.groupby("task", sort=False):
        reg = g["before_rmse"].notna().any()
        r = (g.sort_values("val_primary_after") if reg else g.sort_values("val_primary_after", ascending=False)).iloc[0]
        if reg:
            lines.append(f"| {tname} | {r.model} | {r.fix} | RMSE {r.before_rmse:.3f} / PCC {r.before_pcc:.3f} | RMSE {r.after_rmse:.3f} / PCC {r.after_pcc:.3f} | "
                         f"{r.delta_rmse:+.3f} ([{r.delta_rmse_lo:+.3f}, {r.delta_rmse_hi:+.3f}], p={r.delta_rmse_p:.3f}) |")
        else:
            lines.append(f"| {tname} | {r.model} | {r.fix} | macro-F1 {r.before_macro_f1:.3f} / bal.acc {r.before_balanced_acc:.3f} | macro-F1 {r.after_macro_f1:.3f} / "
                         f"bal.acc {r.after_balanced_acc:.3f} | {r.delta_macro_f1:+.3f} ([{r.delta_macro_f1_lo:+.3f}, {r.delta_macro_f1_hi:+.3f}], p={r.delta_macro_f1_p:.3f}) |")
    lines.append("")
    for tname, g in df.groupby("task", sort=False):
        kind = "reg" if g["before_rmse"].notna().any() else "clf"
        lines += [f"## {tname}", ""]
        if kind == "reg":
            lines += ["| model | fix | RMSE before -> after (95% CI of delta, p) | PCC before -> after | MAE before -> after |", "|---|---|---|---|---|"]
            for r in g.itertuples():
                lines.append(f"| {r.model} | {r.fix} | {r.before_rmse:.3f} -> {r.after_rmse:.3f} ([{r.delta_rmse_lo:+.3f}, {r.delta_rmse_hi:+.3f}], p={r.delta_rmse_p:.3f}) | "
                             f"{r.before_pcc:.3f} -> {r.after_pcc:.3f} | {r.before_mae:.3f} -> {r.after_mae:.3f} |")
        else:
            lines += ["| model | fix | macro-F1 before -> after (95% CI of delta, p) | balanced acc | F1 (positive) |", "|---|---|---|---|---|"]
            for r in g.itertuples():
                f1 = f"{r.before_f1:.3f} -> {r.after_f1:.3f}" if hasattr(r, "before_f1") and pd.notna(r.before_f1) else "n/a"
                lines.append(f"| {r.model} | {r.fix} | {r.before_macro_f1:.3f} -> {r.after_macro_f1:.3f} ([{r.delta_macro_f1_lo:+.3f}, {r.delta_macro_f1_hi:+.3f}], p={r.delta_macro_f1_p:.3f}) | "
                             f"{r.before_balanced_acc:.3f} -> {r.after_balanced_acc:.3f} | {f1} |")
        lines.append("")
    (OUT / "us33_before_after.md").write_text("\n".join(lines))


def val():
    cfg = load_config()
    df = candidate_rows(cfg)
    df.to_csv(OUT / "us33_val_candidates.csv", index=False)
    base = df[df.variant == "us31"].set_index(["task", "model"])["primary"].to_dict()
    df["delta_primary_vs_us31"] = [r.primary - base.get((r.task, r.model), np.nan) for r in df.itertuples()]
    df.to_csv(OUT / "us33_val_candidates.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["task", "model"], columns="variant", values="primary").round(3).to_string())


if __name__ == "__main__":
    {"val": val, "final": final}[sys.argv[1]]()
