"""
Metrics and speaker-clustered bootstrap used by all Sprint 3+ baselines.

Regression:      MAE, RMSE, Pearson r (PCC), Spearman rho
Classification:  F1 (binary, positive class), macro-F1, balanced accuracy, per-class precision / recall / F1 / support

Confidence intervals resample SPEAKERS (not rows) with replacement, because rows from one speaker are not independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support)


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    const = np.ptp(y) == 0 or np.ptp(p) == 0
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "pcc": float("nan") if const else float(pearsonr(y, p)[0]),
            "spearman": float("nan") if const else float(spearmanr(y, p)[0]),
            "n": int(len(y))}


def classification_metrics(y, p, labels=None, positive=None) -> dict:
    """Binary tasks: pass ``positive`` to get F1 of that class. Always returns macro-F1 and balanced accuracy."""
    y, p = np.asarray(y), np.asarray(p)
    labels = sorted(set(y) | set(p)) if labels is None else list(labels)
    out = {"macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
           "balanced_acc": float(balanced_accuracy_score(y, p)),
           "accuracy": float((y == p).mean()), "n": int(len(y))}
    if positive is not None:
        out["f1"] = float(f1_score(y, p, pos_label=positive, zero_division=0))
        pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=[positive], zero_division=0)
        out["precision"], out["recall"] = float(pr[0]), float(rc[0])
    return out


def per_class_table(y, p, labels) -> pd.DataFrame:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return pd.DataFrame({"class": labels, "precision": pr, "recall": rc, "f1": f1, "support": sup})


def confusion(y, p, labels, normalize=False) -> pd.DataFrame:
    cm = confusion_matrix(y, p, labels=labels, normalize="true" if normalize else None)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def _speaker_index(speakers):
    uniq, inv = np.unique(np.asarray(speakers), return_inverse=True)
    return len(uniq), [np.flatnonzero(inv == k) for k in range(len(uniq))]


def speaker_bootstrap(speakers, y, p, metric_fn, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI of ``metric_fn(y, p) -> dict`` under resampling of speakers. Returns {metric: (lo, hi)}."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n_spk, idx = _speaker_index(speakers)
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[k] for k in rng.integers(0, n_spk, n_spk)])
        try:
            draws.append(metric_fn(y[rows], p[rows]))
        except ValueError:
            continue
    keys = [k for k in draws[0] if k != "n"]
    arr = {k: np.array([d[k] for d in draws], float) for k in keys}
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for k, v in arr.items()}


def paired_speaker_bootstrap(speakers, y, p_a, p_b, metric_fn, key, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired speaker-level test for ``metric(b) - metric(a)`` (before/after in US 3.3): delta, 95% CI, two-sided p."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    n_spk, idx = _speaker_index(speakers)
    deltas = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[k] for k in rng.integers(0, n_spk, n_spk)])
        try:
            deltas.append(metric_fn(y[rows], p_b[rows])[key] - metric_fn(y[rows], p_a[rows])[key])
        except ValueError:
            continue
    d = np.array(deltas)
    obs = metric_fn(y, p_b)[key] - metric_fn(y, p_a)[key]
    pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(obs), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p": float(min(pval, 1.0))}


# --------------------------------------------------------------------------- #
# Fast numpy versions for bootstrap loops (integer-coded labels)
# --------------------------------------------------------------------------- #
def fast_reg(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    sy, sp = y.std(), p.std()
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "pcc": float("nan") if sy == 0 or sp == 0 else float(((y - y.mean()) * (p - p.mean())).mean() / (sy * sp))}


def fast_clf_factory(n_classes: int, positive: int | None = None):
    """Returns f(y, p) -> {macro_f1, balanced_acc, accuracy[, f1, precision, recall]} for integer labels 0..K-1."""
    def f(y, p):
        y, p = np.asarray(y, int), np.asarray(p, int)
        cm = np.bincount(y * n_classes + p, minlength=n_classes ** 2).reshape(n_classes, n_classes).astype(float)
        tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
        rec = np.divide(tp, sup, out=np.zeros(n_classes), where=sup > 0)
        prec = np.divide(tp, pred, out=np.zeros(n_classes), where=pred > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_classes), where=(prec + rec) > 0)
        present = sup > 0
        out = {"macro_f1": float(f1[present].mean()), "balanced_acc": float(rec[present].mean()), "accuracy": float(tp.sum() / cm.sum())}
        if positive is not None:
            out.update({"f1": float(f1[positive]), "precision": float(prec[positive]), "recall": float(rec[positive])})
        return out
    return f
