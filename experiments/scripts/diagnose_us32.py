"""
US 3.2 - weakness diagnosis of the US 3.1 baselines. VALIDATION predictions / validation data only (no test).

    python experiments/scripts/diagnose_us32.py regression     # score-bin, age/gender, per-phoneme, human ceilings
    python experiments/scripts/diagnose_us32.py l2             # per-class P/R/F1, confusion, per-L1, per-phoneme, imbalance tests
    python experiments/scripts/diagnose_us32.py phone          # SO762 mispronunciation detection + imbalance tests

Outputs: results/analysis/us32/*.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import metrics as M
from src.models.baselines import build_model, build_tasks, job_path, load_config, load_task, split_masks

warnings.filterwarnings("ignore")
OUT = REPO / "results" / "analysis" / "us32"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
TASKS = build_tasks(cfg)
RES = pd.read_csv(REPO / "results" / "tables" / "baseline" / "us31_results.csv")
SEL = pd.read_csv(REPO / "results" / "tables" / "baseline" / "us31_selected.csv")
PRED = REPO / "experiments" / "results" / "us31" / "predictions"


def best_model(task_name: str) -> str:
    """Best non-dummy model for the task by VALIDATION score (PCC / F1 / macro-F1)."""
    g = RES[(RES.task == task_name) & (RES.split == "val") & ~RES.model.str.startswith("dummy")]
    key = "pcc" if TASKS[task_name].kind == "reg" else ("f1" if g["f1"].notna().any() else "macro_f1")
    return g.sort_values(key, ascending=False).iloc[0]["model"]


def val_preds(task_name: str, model: str | None = None) -> pd.DataFrame:
    return pd.read_csv(PRED / f"{task_name}__{model or best_model(task_name)}__val.csv.gz", dtype={"utt_id": str, "speaker": str})


def human_loo_r(df: pd.DataFrame, key: list, value: str) -> float:
    """Mean over raters of corr(rater, mean of the other raters): the human-agreement ceiling for a score."""
    wide = df.pivot_table(index=key, columns="expert", values=value)
    wide = wide.dropna()
    r = []
    for e in wide.columns:
        others = wide.drop(columns=e).mean(1)
        r.append(np.corrcoef(wide[e], others)[0, 1])
    return float(np.mean(r))


# --------------------------------------------------------------------------- #
def regression():
    rows, bins, groups, ceil = [], [], [], []
    from src.data.loaders import load_so762_experts
    ex = load_so762_experts()
    ceil_map = {f"sent_{m}": human_loo_r(ex.sent, ["utt_id"], m) for m in ("accuracy", "fluency", "prosodic", "total")}
    ceil_map["word_accuracy"] = human_loo_r(ex.words, ["utt_id", "word_idx"], "accuracy")
    ceil_map["phone_score"] = human_loo_r(ex.phones, ["utt_id", "word_idx", "phone_idx"], "score")
    for task_name in ["sent_accuracy", "sent_fluency", "sent_prosodic", "sent_total", "word_accuracy", "phone_score"]:
        model = best_model(task_name)
        p = val_preds(task_name, model)
        r = M.regression_metrics(p.y, p.pred)
        d = RES[(RES.task == task_name) & (RES.split == "val") & (RES.model == "dummy")].iloc[0]
        slope = np.polyfit(p.y, p.pred, 1)[0]
        rows.append({"task": task_name, "model": model, **{k: r[k] for k in ("mae", "rmse", "pcc")}, "dummy_rmse": d["rmse"],
                     "rmse_skill": 1 - r["rmse"] / d["rmse"], "human_loo_r": ceil_map[task_name], "pcc_over_ceiling": r["pcc"] / ceil_map[task_name],
                     "slope_pred_on_true": slope, "n": r["n"], "n_speakers": p.speaker.nunique()})
        # error by true-score bin (regression to the mean)
        if task_name == "phone_score":
            edges, labels = [-0.01, 0.5, 1.0, 1.5, 1.99, 2.01], ["<0.5", "0.5-1", "1-1.5", "1.5-<2", "=2"]
        elif task_name == "word_accuracy":
            edges, labels = [-1, 4, 6, 8, 9, 10], ["0-4", "5-6", "7-8", "9", "10"]
        else:
            edges, labels = [-1, 4, 6, 7, 8, 9, 10], ["0-4", "5-6", "7", "8", "9", "10"]
        p["bin"] = pd.cut(p.y, edges, labels=labels)
        for b, g in p.groupby("bin", observed=True):
            bins.append({"task": task_name, "model": model, "true_bin": b, "n": len(g), "share": len(g) / len(p), "mean_true": g.y.mean(),
                         "mean_pred": g.pred.mean(), "bias": (g.pred - g.y).mean(), "mae": (g.pred - g.y).abs().mean()})
        # age group / gender (SO762 val = 19 speakers: indicative only)
        from src.models.baselines import load_task as _lt
        _, _, meta, _ = _lt(TASKS[task_name])
        keys = [k for k in ("utt_id", "word_idx", "phone_idx") if k in p.columns and k in meta.columns]
        p2 = p.drop(columns=[c for c in ("age_group", "gender") if c in p.columns]).merge(meta[keys + ["age_group", "gender"]].drop_duplicates(keys), on=keys, how="left")
        for col in ("age_group", "gender"):
            for v, g in p2.groupby(col):
                rr = M.regression_metrics(g.y, g.pred)
                groups.append({"task": task_name, "model": model, "by": col, "group": v, "n": rr["n"], "n_speakers": g.speaker.nunique(),
                               "mae": rr["mae"], "rmse": rr["rmse"], "pcc": rr["pcc"]})
        if task_name == "phone_score":
            ph = []
            for v, g in p.groupby("phone"):
                if len(g) >= 40:
                    rr = M.regression_metrics(g.y, g.pred)
                    ph.append({"phone": v, "n": rr["n"], "mean_true": g.y.mean(), "share_below_1": (g.y < 1).mean(), "rmse": rr["rmse"], "pcc": rr["pcc"]})
            pd.DataFrame(ph).sort_values("rmse", ascending=False).to_csv(OUT / "phone_score_per_phoneme_val.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "regression_summary_val.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "regression_by_score_bin_val.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "regression_by_group_val.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    print(pd.DataFrame(bins).round(3).to_string())
    print(pd.DataFrame(groups).round(3).to_string())


# --------------------------------------------------------------------------- #
def _fit_scores(task, cfg, X, y, meta, class_weight=None, C=1.0, only_cols=None):
    """logreg per fold -> pooled validation probabilities (+ labels, hard predictions)."""
    P, Y, H, idx = [], [], [], []
    for fold in task.folds:
        m = split_masks(task, meta, fold)
        XX = X if only_cols is None else X[:, only_cols]
        est = build_model(task, "logreg", {"C": C}, cfg, class_weight=class_weight)
        est.fit(XX[m["train"]], y[m["train"]])
        P.append(est.predict_proba(XX[m["val"]])); Y.append(y[m["val"]]); H.append(est.predict(XX[m["val"]]))
        idx.append(np.flatnonzero(m["val"]))
        classes = est.classes_
    return np.vstack(P), np.concatenate(Y), np.concatenate(H), np.concatenate(idx), classes


def l2():
    task = TASKS["l2_phone_error"]
    labels = list(task.labels)
    model = best_model("l2_phone_error")
    p = val_preds("l2_phone_error", model)
    # per-class P/R/F1 + confusion
    pc = M.per_class_table(p.y, p.pred, labels)
    pc["train_share"] = np.nan
    X, y, meta, names = load_task(task)
    tr_counts = pd.Series(y[meta.fold.notna()]).value_counts()  # overall, for reference only (shares are near-identical across folds)
    pc["share_of_units"] = pc["class"].map(tr_counts / tr_counts.sum())
    pc.to_csv(OUT / "l2_phone_per_class_val.csv", index=False)
    M.confusion(p.y, p.pred, labels).to_csv(OUT / "l2_phone_confusion_counts_val.csv")
    M.confusion(p.y, p.pred, labels, normalize=True).to_csv(OUT / "l2_phone_confusion_rownorm_val.csv")
    print(model); print(pc.round(3).to_string()); print(M.confusion(p.y, p.pred, labels, True).round(3).to_string())
    # per-L1 and per-class recall by L1
    rows = []
    for l1, g in p.groupby("l1"):
        r = M.classification_metrics(g.y, g.pred, labels=labels)
        rec = M.per_class_table(g.y, g.pred, labels).set_index("class")["recall"]
        rows.append({"l1": l1, "n": len(g), "macro_f1": r["macro_f1"], "balanced_acc": r["balanced_acc"], **{f"recall_{c}": rec[c] for c in labels[1:]},
                     "error_rate": (g.y != "correct").mean()})
    pd.DataFrame(rows).to_csv(OUT / "l2_phone_per_l1_val.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    # per-phoneme error recall vs training support (imbalance at phoneme level)
    ph_rows = []
    for fold in task.folds:
        m = split_masks(task, meta, fold)
        trn = meta[m["train"]].assign(y=y[m["train"]])
        sup = trn.groupby("anchor_phone").agg(train_n=("y", "size"), train_err=("y", lambda s: (s != "correct").sum()))
        val = p[p.fold == fold].copy()
        val["err"] = val.y != "correct"
        val["pred_err"] = val.pred != "correct"
        g = val[val.err].groupby("anchor_phone").agg(val_err_n=("err", "size"), recall=("pred_err", "mean"))
        ph_rows.append(sup.join(g, how="inner").assign(fold=fold).reset_index())
    ph = pd.concat(ph_rows).groupby("anchor_phone").agg(train_err=("train_err", "mean"), train_n=("train_n", "mean"),
                                                         val_err_n=("val_err_n", "sum"), recall=("recall", "mean")).reset_index()
    ph["train_error_rate"] = ph.train_err / ph.train_n
    ph.to_csv(OUT / "l2_phone_per_phoneme_error_recall_val.csv", index=False)
    s1 = spearmanr(ph.recall, ph.train_err); s2 = spearmanr(ph.recall, ph.train_error_rate)
    print(f"per-phoneme error recall vs train error support: rho={s1[0]:.2f} (p={s1[1]:.3g}); vs train error rate: rho={s2[0]:.2f} (p={s2[1]:.3g}); n_phones={len(ph)}")
    pd.DataFrame([{"test": "spearman(recall, train_error_support)", "rho": s1[0], "p": s1[1], "n_phonemes": len(ph)},
                  {"test": "spearman(recall, train_error_rate)", "rho": s2[0], "p": s2[1], "n_phonemes": len(ph)}]).to_csv(OUT / "l2_phone_imbalance_spearman.csv", index=False)
    # imbalance experiment: logreg unweighted vs balanced vs phone-prior-only
    ph_cols = [i for i, n in enumerate(names) if n.startswith("ph_")]
    out = []
    for tag, kw in {"logreg unweighted": {}, "logreg balanced": {"class_weight": "balanced"},
                    "phone-identity only (prior)": {"only_cols": ph_cols}}.items():
        P, Y, H, idx, classes = _fit_scores(task, cfg, X, y, meta, **kw)
        r = M.classification_metrics(Y, H, labels=labels)
        pcs = M.per_class_table(Y, H, labels).set_index("class")
        row = {"variant": tag, "macro_f1": r["macro_f1"], "balanced_acc": r["balanced_acc"]}
        for c in labels:
            k = list(classes).index(c)
            row[f"recall_{c}"], row[f"precision_{c}"] = pcs.loc[c, "recall"], pcs.loc[c, "precision"]
            row[f"auc_{c}"] = roc_auc_score((Y == c).astype(int), P[:, k]); row[f"ap_{c}"] = average_precision_score((Y == c).astype(int), P[:, k])
        out.append(row)
        print(tag, {k: round(v, 3) for k, v in row.items() if k != "variant"}, flush=True)
    pd.DataFrame(out).to_csv(OUT / "l2_phone_imbalance_experiment_val.csv", index=False)


def phone():
    task = TASKS["phone_mispron_t1.0"]
    X, y, meta, names = load_task(task)
    m = split_masks(task, meta, 0)
    base_rate = {s: float(y[m[s]].mean()) for s in ("train", "val")}
    print("mispronounced base rate", base_rate)
    out = []
    for tag, kw in {"logreg unweighted": {}, "logreg balanced": {"class_weight": "balanced"}}.items():
        P, Y, H, idx, classes = _fit_scores(task, cfg, X, y, meta, **kw)
        r = M.classification_metrics(Y, H, labels=[0, 1], positive=1)
        out.append({"variant": tag, "val_base_rate": base_rate["val"], "pred_pos_rate": float(H.mean()), "f1": r["f1"], "precision": r["precision"],
                    "recall": r["recall"], "balanced_acc": r["balanced_acc"], "roc_auc": roc_auc_score(Y, P[:, 1]),
                    "pr_auc": average_precision_score(Y, P[:, 1])})
        print(out[-1], flush=True)
    for model in ("rf", "svm", "logreg"):
        pv = val_preds("phone_mispron_t1.0", model)
        out.append({"variant": f"stored val preds: {model}", "val_base_rate": base_rate["val"], "pred_pos_rate": float(pv.pred.mean())})
    pd.DataFrame(out).to_csv(OUT / "phone_mispron_imbalance_experiment_val.csv", index=False)
    # by phoneme: how are mispronunciations distributed, and where is recall lost
    pv = val_preds("phone_mispron_t1.0", best_model("phone_mispron_t1.0"))
    g = pv.groupby("phone").agg(n=("y", "size"), positives=("y", "sum"), recall=("pred", lambda s: np.nan)).reset_index()
    rec = pv[pv.y == 1].groupby("phone")["pred"].mean().rename("recall")
    g = g.drop(columns="recall").merge(rec, on="phone", how="left")
    g["pos_rate"] = g.positives / g.n
    g.sort_values("positives", ascending=False).to_csv(OUT / "phone_mispron_per_phoneme_val.csv", index=False)
    print(g.sort_values("positives", ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    {"regression": regression, "l2": l2, "phone": phone}[sys.argv[1]]()
