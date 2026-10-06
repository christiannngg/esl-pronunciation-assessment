"""
US 3.1 diagnostic ablations (VALIDATION data only; nothing here touches test).

  l2_phone_error  logreg(C=1):  base features | + annotated interval duration (leaky) | - anchor-phone identity
  phone_score     ridge/RF:      base | - canonical-phone identity
Shows how much of each baseline comes from a cue that would not exist at inference time (annotated duration) or from phone priors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation import metrics as M
from src.models.baselines import build_model, build_tasks, load_config, load_task, split_masks

cfg = load_config()
tasks = build_tasks(cfg)
rows = []

# ---- L2-ARCTIC phone error type --------------------------------------------------------------- #
t = tasks["l2_phone_error"]
X, y, meta, names = load_task(t)
ph_cols = [i for i, n in enumerate(names) if n.startswith("ph_")]
keep_noph = [i for i in range(X.shape[1]) if i not in set(ph_cols)]
dur = np.c_[meta["ann_dur"].to_numpy(), np.log(np.maximum(meta["ann_dur"].to_numpy(), 1e-3))].astype(np.float32)
variants = {"base": X, "+annotated_duration (leaky)": np.hstack([X, dur]), "-anchor_phone_identity": X[:, keep_noph]}
for vname, XX in variants.items():
    ys, ps = [], []
    for fold in t.folds:
        m = split_masks(t, meta, fold)
        est = build_model(t, "logreg", {"C": 1}, cfg)
        est.fit(XX[m["train"]], y[m["train"]])
        ys.append(y[m["val"]]); ps.append(est.predict(XX[m["val"]]))
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    r = M.classification_metrics(yy, pp, labels=list(t.labels))
    pc = M.per_class_table(yy, pp, list(t.labels)).set_index("class")["recall"]
    rows.append({"task": t.name, "model": "logreg", "variant": vname, "macro_f1": r["macro_f1"], "balanced_acc": r["balanced_acc"],
                 **{f"recall_{c}": pc[c] for c in t.labels}})
    print(rows[-1], flush=True)

# ---- SO762 phone score ------------------------------------------------------------------------ #
t = tasks["phone_score"]
X, y, meta, names = load_task(t)
keep_noph = [i for i, n in enumerate(names) if not n.startswith("ph_")]
for vname, XX in {"base": X, "-canonical_phone_identity": X[:, keep_noph]}.items():
    m = split_masks(t, meta, 0)
    est = build_model(t, "ridge", {"alpha": 1000}, cfg)
    est.fit(XX[m["train"]], y[m["train"]])
    r = M.regression_metrics(y[m["val"]], est.predict(XX[m["val"]]))
    rows.append({"task": t.name, "model": "ridge", "variant": vname, "rmse": r["rmse"], "pcc": r["pcc"], "mae": r["mae"]})
    print(rows[-1], flush=True)

out = Path(__file__).resolve().parents[2] / "results" / "tables" / "baseline" / "us31_ablations_val.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
