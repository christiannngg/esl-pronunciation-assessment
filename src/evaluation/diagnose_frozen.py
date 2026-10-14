"""
US 4.2 - failure-mode diagnosis of the frozen-encoder heads against the Sprint 3 baseline. VALIDATION rows only (test is never opened).

    python -m src.evaluation.diagnose_frozen --encoders wavlm_base_plus hubert_base wav2vec2_base
    python experiments/scripts/diagnose_us42.py --encoders wavlm_base_plus --tasks l2_phone_error phone_score

For every task the head result shown is the validation-selected headline (best head, config and class-weight / decision-weight variant, exactly
as in report_frozen) compared with the US 3.1 baseline's stored validation predictions on the SAME rows (asserted). Outputs in
results/analysis/us42/ (csv + summary_val.md):

  head_gaps_val            train vs validation score per task x head (over-fitting) and seed sd
  linear_vs_mlp_val        validation gain of the MLP over the linear probe (head capacity)
  reg_summary_val          RMSE / PCC / slope of prediction on truth / share of the human ceiling
  reg_by_score_bin_val     bias by true-score bin (regression to the mean)
  reg_by_group_val         by age group / gender
  phone_score_per_phoneme_val
  clf_summary_val | clf_per_class_val | clf_confusion_rownorm_val | clf_by_group_val (by L1 / age group / gender)
  phoneme_error_recall_val | phoneme_support_spearman_val   error recall per phoneme vs its training error support
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import metrics as M
from src.evaluation import report_frozen as RF
from src.models import frozen_heads as FH
from src.models.baselines import REPO, build_tasks, load_config as load_baseline_config, split_masks

warnings.filterwarnings("ignore")
OUT = REPO / "results" / "analysis" / "us42"
S3_TABLES = REPO / "results" / "tables" / "baseline"
S3_PRED = REPO / "experiments" / "results" / "us31" / "predictions"
S3_ANALYSIS = REPO / "results" / "analysis" / "us32"
PHONE_COL = {"phone_score": "phone", "phone_mispron_t1.0": "phone", "l2_phone_error": "anchor_phone"}
GROUP_COLS = ("age_group", "gender", "l1")
BIN_SPECS = {
    "phone_score": ([-0.01, 0.5, 1.0, 1.5, 1.99, 2.01], ["<0.5", "0.5-1", "1-1.5", "1.5-<2", "=2"]),
    "word_accuracy": ([-1, 4, 6, 8, 9, 10], ["0-4", "5-6", "7-8", "9", "10"]),
    "sentence": ([-1, 4, 6, 7, 8, 9, 10], ["0-4", "5-6", "7", "8", "9", "10"]),
}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def make_frame(y, pred, meta: pd.DataFrame) -> pd.DataFrame:
    """Validation rows: Sprint 3 meta columns + y + pred."""
    f = meta[[c for c in meta.columns if c not in ("y", "pred")]].reset_index(drop=True).copy()
    assert len(f) == len(y) == len(pred), (len(f), len(y), len(pred))
    f["y"], f["pred"] = np.asarray(y), np.asarray(pred)
    return f


def error_mask(task, s) -> np.ndarray:
    """True where the unit carries the error / positive class."""
    s = np.asarray(s)
    if task.positive is not None:
        return s.astype(float) == float(task.positive)
    return s.astype(str) != "correct"


def overfit_gap(task, val_primary: float, train_primary: float) -> float:
    """Positive = better on train than on validation (RMSE: val - train; macro-F1: train - val)."""
    return val_primary - train_primary if task.kind == "reg" else train_primary - val_primary


def mlp_gain(task, val_linear: float, val_mlp: float) -> float:
    """Validation gain of the MLP over the linear head (positive = MLP better)."""
    return val_linear - val_mlp if task.kind == "reg" else val_mlp - val_linear


def pick_best_head(task, per_head: dict) -> str:
    """Head with the best validation primary score (RMSE lower / macro-F1 higher); ties keep the first head."""
    best = None
    for h, e in per_head.items():
        if best is None or RF.better(task, e["val_primary"], per_head[best]["val_primary"]):
            best = h
    return best


def auc_metrics(task, y, proba) -> dict:
    """Threshold-free ranking quality of the head's probabilities (independent of class weights / decision weights):
    error-vs-rest ROC-AUC and average precision, plus one-vs-rest AUC per class for multi-class tasks."""
    labels = list(task.labels)
    e = error_mask(task, y)
    score = proba[:, labels.index(task.positive)] if task.positive is not None else 1.0 - proba[:, labels.index("correct")]
    out = {"auc_error": float(roc_auc_score(e, score)), "ap_error": float(average_precision_score(e, score)), "error_share": float(e.mean())}
    if len(labels) > 2:
        ys = np.asarray(y).astype(str)
        for k, l in enumerate(labels):
            out[f"auc_{l}"] = float(roc_auc_score(ys == str(l), proba[:, k]))
    return out


def sprint3_auc(task_name: str) -> dict:
    """Sprint 3 (US 3.2) unweighted logistic-regression validation AUCs, where that memo stored them (no probabilities of the best US 3.1 model are kept)."""
    names = {"l2_phone_error": ("l2_phone_imbalance_experiment_val.csv", {"auc_substitution": "auc_substitution", "auc_deletion": "auc_deletion",
                                                                        "auc_addition": "auc_addition", "auc_correct": "auc_correct"}),
             "phone_mispron_t1.0": ("phone_mispron_imbalance_experiment_val.csv", {"auc_error": "roc_auc", "ap_error": "pr_auc"})}
    if task_name not in names or not (S3_ANALYSIS / names[task_name][0]).exists():
        return {}
    d = pd.read_csv(S3_ANALYSIS / names[task_name][0])
    r = d[d.variant == "logreg unweighted"]
    if r.empty:
        return {}
    return {ours: float(r.iloc[0][theirs]) for ours, theirs in names[task_name][1].items() if theirs in r.columns and pd.notna(r.iloc[0][theirs])}


def unreadable_results(base) -> list:
    """Result files under ``base`` that cannot be opened (e.g. 0-byte files from an interrupted write or an incomplete copy from Drive)."""
    bad = []
    for p in sorted(Path(base).rglob("*.npz")):
        try:
            np.load(p, allow_pickle=True)["val_pred"]
        except Exception:
            bad.append(p)
    return bad


def check_same_rows(a: pd.DataFrame, b: pd.DataFrame, task_name: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{task_name}: frozen validation has {len(a)} rows, Sprint 3 stored predictions {len(b)}")
    ya, yb = a["y"].to_numpy(), b["y"].to_numpy()
    same = np.allclose(ya.astype(float), yb.astype(float)) if ya.dtype.kind in "fiu" and yb.dtype.kind in "fiu" else (ya.astype(str) == yb.astype(str)).all()
    if not same:
        raise ValueError(f"{task_name}: frozen and Sprint 3 validation rows differ (labels not equal row by row)")


def reg_summary(f: pd.DataFrame) -> dict:
    r = M.regression_metrics(f.y, f.pred)
    return {"n": r["n"], "n_speakers": f["speaker"].nunique() if "speaker" in f else np.nan, "rmse": r["rmse"], "mae": r["mae"], "pcc": r["pcc"],
            "slope_pred_on_true": float(np.polyfit(np.asarray(f.y, float), np.asarray(f.pred, float), 1)[0])}


def clf_summary(task, f: pd.DataFrame) -> dict:
    r = M.classification_metrics(f.y, f.pred, labels=list(task.labels))
    out = {"n": r["n"], "macro_f1": r["macro_f1"], "balanced_acc": r["balanced_acc"]}
    e, pe = error_mask(task, f.y), error_mask(task, f.pred)
    out.update({"error_share_true": float(e.mean()), "error_share_pred": float(pe.mean()),
                "error_recall": float(pe[e].mean()) if e.any() else np.nan, "error_precision": float(e[pe].mean()) if pe.any() else np.nan})
    return out


def score_bins(f: pd.DataFrame, spec) -> pd.DataFrame:
    edges, labels = spec
    b = pd.cut(f.y.astype(float), edges, labels=labels)
    rows = []
    for k, g in f.groupby(b, observed=True):
        rows.append({"true_bin": k, "n": len(g), "share": len(g) / len(f), "mean_true": g.y.astype(float).mean(), "mean_pred": g.pred.astype(float).mean(),
                     "bias": float((g.pred.astype(float) - g.y.astype(float)).mean()), "mae": float((g.pred.astype(float) - g.y.astype(float)).abs().mean())})
    return pd.DataFrame(rows)


def by_group(task, f: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for v, g in f.groupby(col):
        row = {"group": v, "n": len(g), "n_speakers": g["speaker"].nunique() if "speaker" in g else np.nan}
        if task.kind == "reg":
            r = M.regression_metrics(g.y, g.pred)
            row.update({"rmse": r["rmse"], "pcc": r["pcc"]})
        else:
            s = clf_summary(task, g)
            row.update({k: s[k] for k in ("macro_f1", "balanced_acc", "error_share_true", "error_recall", "error_precision")})
        rows.append(row)
    return pd.DataFrame(rows)


def phoneme_reg(f: pd.DataFrame, col: str, min_n: int = 40) -> pd.DataFrame:
    rows = []
    for v, g in f.groupby(col):
        if len(g) < min_n:
            continue
        r = M.regression_metrics(g.y, g.pred)
        rows.append({"phone": v, "n": r["n"], "mean_true": float(np.mean(g.y.astype(float))), "share_below_1": float((g.y.astype(float) < 1).mean()),
                     "rmse": r["rmse"], "pcc": r["pcc"]})
    return pd.DataFrame(rows)


def train_error_support(task, y_all, meta: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per phoneme: training rows and training error events (mean over the task's folds)."""
    parts = []
    for fold in task.folds:
        m = split_masks(task, meta, fold)["train"]
        d = meta.loc[m, [col]].assign(err=error_mask(task, np.asarray(y_all)[m]))
        parts.append(d.groupby(col).agg(train_n=("err", "size"), train_err=("err", "sum")))
    out = pd.concat(parts).groupby(level=0).mean()
    out["train_error_rate"] = out.train_err / out.train_n
    return out


def phoneme_error_recall(task, f: pd.DataFrame, col: str, support: pd.DataFrame | None = None) -> pd.DataFrame:
    e, pe = error_mask(task, f.y), error_mask(task, f.pred)
    d = f[[col]].assign(err=e, pred_err=pe)
    g = d.groupby(col).agg(n=("err", "size"), n_err=("err", "sum"), pred_err_rate=("pred_err", "mean"))
    g = g.join(d[d.err].groupby(col)["pred_err"].mean().rename("error_recall"))
    if support is not None:
        g = g.join(support)
    g.index.name = "phone"
    return g.reset_index()


def support_spearman(t: pd.DataFrame, min_err: int = 3) -> dict:
    s = t[t.n_err >= min_err].dropna(subset=["error_recall", "train_err"])
    if len(s) < 3:
        return {"rho_support": np.nan, "p_support": np.nan, "rho_rate": np.nan, "p_rate": np.nan, "n_phonemes": len(s)}
    a, b = spearmanr(s.error_recall, s.train_err), spearmanr(s.error_recall, s.train_error_rate)
    return {"rho_support": float(a[0]), "p_support": float(a[1]), "rho_rate": float(b[0]), "p_rate": float(b[1]), "n_phonemes": len(s)}


# --------------------------------------------------------------------------- #
# Frozen headline on validation, Sprint 3 reference
# --------------------------------------------------------------------------- #
def headline_val(base, task, hcfg: dict, head: str, y_all, meta):
    """Validation-selected headline of one (task, head): dict(y, pred, meta, variant, config, stage, train_primary, val_primary, seed_sd).
    Reads validation predictions and the stored train score only; test arrays are never accessed. None when no results exist."""
    cands = []
    for group in FH.groups_for(task):
        sel = FH.select_config(base, task, hcfg, head, group, y_all, meta)
        if sel is None:
            continue
        config = sel[1]
        stage = "final" if all(FH.job_path(base, task.name, head, config, f, "final").exists() for f in task.folds) else "search"
        dv = RF.pooled(base, task, head, config, stage, "val", y_all, meta, proba=task.kind == "clf")
        for vname, pv, _, trp in RF.apply_variants(task, group, dv, dv):     # the "test" slot is unused: val arrays are passed twice
            cands.append({"variant": vname, "config": config, "stage": stage, "y": dv["y"], "pred": pv, "meta": dv["meta"], "train_primary": trp,
                          "val_primary": RF.primary(task, dv["y"], pv), "seed_sd": dv.get("seed_sd", np.nan), "proba": dv.get("proba")})
    if not cands:
        return None
    best = cands[0]
    for c in cands[1:]:
        best = c if RF.better(task, c["val_primary"], best["val_primary"]) else best
    return best


def us31_val_frame(task, bcfg=None) -> tuple[str, pd.DataFrame]:
    """(model, validation frame) of the US 3.1 model that Sprint 3 selected on validation (same rule as report_frozen.baseline_refs)."""
    ba = pd.read_csv(S3_TABLES / "us33_before_after.csv")
    g = ba[ba.task == task.name]
    model = g.sort_values("val_primary_before", ascending=task.kind == "reg").iloc[0].model
    df = pd.read_csv(S3_PRED / f"{task.name}__{model}__val.csv.gz", dtype={"utt_id": str, "speaker": str})
    return model, df


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "" if pd.isna(v) else f"{v:.{nd}f}"
    return "" if v is None else str(v)


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(_fmt(v, nd) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def run(encoders, layer=None, mode="emb+side", root=None, only_tasks=None, min_n=40, min_err=3, out=None):
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    bcfg, hcfg = load_baseline_config(), FH.load_config()["heads"]
    layer = layer if layer is not None else hcfg["primary_layer"]
    ceil = {}
    if (S3_ANALYSIS / "regression_summary_val.csv").exists():
        c = pd.read_csv(S3_ANALYSIS / "regression_summary_val.csv")
        ceil = dict(zip(c.task, c.human_loo_r))
    checked = set()
    T = {k: [] for k in ("gaps", "wide", "reg", "bins", "reg_group", "phone_reg", "clf", "per_class", "conf", "clf_group", "ph_recall", "ph_rho", "auc")}
    for task in FH.main_tasks(None, hcfg):
        if only_tasks and task.name not in only_tasks:
            continue
        y_all, meta = FH.load_labels(task)
        systems, details, probas = {}, {}, {}
        for enc in encoders:
            base = FH.results_root(enc, layer, mode, root)
            per_head = {}
            if enc not in checked:
                checked.add(enc)
                bad = unreadable_results(base)
                if bad:
                    raise SystemExit(f"[diagnose] {len(bad)} unreadable result file(s) for {enc} (0 bytes or truncated, usually an interrupted write or an incomplete "
                                     "copy from Drive). Re-copy them from Drive or re-fit them with `python -m src.models.frozen_heads run --encoder "
                                     f"{enc} --tasks <task> --heads <head> --stage search --force`:\n  " + "\n  ".join(str(b.relative_to(base)) for b in bad[:20]))
            for head in hcfg["heads"]:
                e = headline_val(base, task, hcfg, head, y_all, meta)
                if e is None:
                    continue
                per_head[head] = e
                T["gaps"].append({"encoder": enc, "task": task.name, "head": head, "variant": e["variant"], "config": e["config"], "stage": e["stage"],
                                  "val_primary": e["val_primary"], "train_primary": e["train_primary"],
                                  "overfit_gap": overfit_gap(task, e["val_primary"], e["train_primary"]), "val_seed_sd": e["seed_sd"]})
            if not per_head:
                continue
            if {"linear", "mlp"} <= set(per_head):
                lin, mlp = per_head["linear"]["val_primary"], per_head["mlp"]["val_primary"]
                T["wide"].append({"encoder": enc, "task": task.name, "val_linear": lin, "val_mlp": mlp, "mlp_gain": mlp_gain(task, lin, mlp)})
            best_name = pick_best_head(task, per_head)
            e = per_head[best_name]
            systems[enc] = make_frame(e["y"], e["pred"], e["meta"])
            details[enc] = f"{best_name} ({e['variant']})"
            if e["proba"] is not None:
                probas[enc] = e["proba"]
        if not systems:
            continue
        model31, d31 = us31_val_frame(task, bcfg)
        ref = next(iter(systems.values()))
        f31 = make_frame(d31["y"], d31["pred"], d31.drop(columns=["y", "pred"]))
        check_same_rows(ref, f31, task.name)
        systems = {"US 3.1": f31, **systems}
        details = {"US 3.1": model31, **details}
        pc = PHONE_COL.get(task.name)
        for sysname, f in systems.items():
            base_row = {"task": task.name, "system": sysname, "detail": details[sysname]}
            gcols = [c for c in GROUP_COLS if c in f.columns]
            if task.kind == "reg":
                s = reg_summary(f)
                ce = ceil.get(task.name, np.nan)
                T["reg"].append({**base_row, **s, "human_loo_r": ce, "pcc_over_ceiling": s["pcc"] / ce if ce == ce else np.nan})
                spec = BIN_SPECS.get(task.name, BIN_SPECS["sentence"] if task.level == "sentence" else None)
                if spec:
                    T["bins"].append(score_bins(f, spec).assign(task=task.name, system=sysname))
                for col in gcols:
                    T["reg_group"].append(by_group(task, f, col).assign(task=task.name, system=sysname, by=col))
                if task.name == "phone_score":
                    T["phone_reg"].append(phoneme_reg(f, "phone", min_n).assign(system=sysname))
            else:
                T["clf"].append({**base_row, **clf_summary(task, f)})
                if sysname in probas:
                    T["auc"].append({**base_row, **auc_metrics(task, f.y, probas[sysname])})
                elif sysname == "US 3.1" and sprint3_auc(task.name):
                    T["auc"].append({**base_row, "detail": "logreg, unweighted (US 3.2 csv)", **sprint3_auc(task.name)})
                labels = list(task.labels)
                T["per_class"].append(M.per_class_table(f.y, f.pred, labels).assign(task=task.name, system=sysname))
                T["conf"].append(M.confusion(f.y, f.pred, labels, normalize=True).reset_index(names="true").assign(task=task.name, system=sysname))
                for col in gcols:
                    T["clf_group"].append(by_group(task, f, col).assign(task=task.name, system=sysname, by=col))
                if pc and pc in f.columns:
                    t = phoneme_error_recall(task, f, pc, train_error_support(task, y_all, meta, pc))
                    T["ph_recall"].append(t.assign(task=task.name, system=sysname))
                    T["ph_rho"].append({"task": task.name, "system": sysname, **support_spearman(t, min_err)})
        print(f"[diagnose] {task.name}: {', '.join(f'{k}: {v}' for k, v in details.items())}", flush=True)

    def cat(k):
        return pd.concat(T[k], ignore_index=True) if T[k] and isinstance(T[k][0], pd.DataFrame) else pd.DataFrame(T[k])

    tabs = {"head_gaps_val": cat("gaps"), "linear_vs_mlp_val": cat("wide"), "reg_summary_val": cat("reg"), "reg_by_score_bin_val": cat("bins"),
            "reg_by_group_val": cat("reg_group"), "phone_score_per_phoneme_val": cat("phone_reg"), "clf_summary_val": cat("clf"),
            "clf_per_class_val": cat("per_class"), "clf_confusion_rownorm_val": cat("conf"), "clf_by_group_val": cat("clf_group"),
            "phoneme_error_recall_val": cat("ph_recall"), "phoneme_support_spearman_val": cat("ph_rho"), "clf_auc_val": cat("auc")}
    if all(t.empty for t in tabs.values()):
        raise SystemExit(f"[diagnose] no head results found for {encoders} under {FH.results_root(encoders[0], layer, mode, root)} "
                         "(copy the Drive frozen_results folder there or pass --results-root)")
    for name, t in tabs.items():
        if not t.empty:
            t.to_csv(out / f"{name}.csv", index=False)
    write_summary(tabs, out / "summary_val.md", layer, mode)
    return tabs


def write_summary(tabs: dict, path, layer, mode) -> None:
    L = [f"# US 4.2 diagnosis (validation only) - layer {layer}, inputs {mode}", "",
         "Frozen result = validation-selected headline per task (best head and variant). US 3.1 = Sprint 3 best baseline on validation. Full tables: csv files in this folder.", ""]
    def sec(title, name, query=None, cols=None, nd=3):
        t = tabs.get(name)
        if t is None or t.empty:
            return
        if query:
            t = t.query(query)
        if cols:
            t = t[[c for c in cols if c in t.columns]]
        L.extend([f"## {title}", "", md_table(t, nd), ""])
    sec("Over-fitting: train vs validation (positive gap = better on train)", "head_gaps_val",
        cols=["encoder", "task", "head", "variant", "val_primary", "train_primary", "overfit_gap", "val_seed_sd"])
    sec("Head capacity: validation gain of MLP over linear (positive = MLP better)", "linear_vs_mlp_val")
    sec("Regression summary", "reg_summary_val", cols=["task", "system", "detail", "rmse", "pcc", "slope_pred_on_true", "human_loo_r", "pcc_over_ceiling"])
    sec("Regression bias by true-score bin (word / phone)", "reg_by_score_bin_val", query="task in ['word_accuracy', 'phone_score']",
        cols=["task", "system", "true_bin", "n", "share", "mean_true", "mean_pred", "bias"])
    sec("Classification summary", "clf_summary_val")
    sec("Threshold-free ranking quality (AUC of P(error) / per class)", "clf_auc_val")
    sec("Per class", "clf_per_class_val", query="task == 'l2_phone_error'")
    sec("L2-ARCTIC phone error by L1", "clf_by_group_val", query="task == 'l2_phone_error' and by == 'l1'",
        cols=["system", "group", "n", "macro_f1", "error_share_true", "error_recall", "error_precision"])
    sec("Error recall per phoneme vs training error support (Spearman)", "phoneme_support_spearman_val")
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["wavlm_base_plus", "hubert_base", "wav2vec2_base"])
    ap.add_argument("--layer", default=None)
    ap.add_argument("--mode", default="emb+side")
    ap.add_argument("--results-root", default=None, help="folder holding <encoder>/L<layer>/<mode>/ (default experiments/results/frozen)")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None, help="default results/analysis/us42 (use another folder for ablation modes so tables are not overwritten)")
    ap.add_argument("--min-phoneme-n", type=int, default=40)
    ap.add_argument("--min-error-n", type=int, default=3)
    a = ap.parse_args(argv)
    tabs = run(a.encoders, a.layer, a.mode, Path(a.results_root) if a.results_root else None, a.tasks, a.min_phoneme_n, a.min_error_n,
               Path(a.out_dir) if a.out_dir else None)
    print(f"[diagnose] wrote {sum(not t.empty for t in tabs.values())} tables to {a.out_dir or OUT}")


if __name__ == "__main__":
    main()
