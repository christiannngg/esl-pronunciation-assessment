"""
Sprint 3 traditional-ML baselines: task registry, model zoo, resumable fit jobs.

    python -m src.models.baselines run    --variant us31 --budget 140     # fits every missing job, stops before the time budget
    python -m src.models.baselines status --variant us31
    python -m src.models.baselines report --variant us31                  # val-based selection -> tables (+ W&B)

Protocol (no test leakage):
  * hyper-parameters are chosen on the validation split only (regression: val RMSE, classification: val balanced accuracy) (SO762 val speakers; L2-ARCTIC fold (k+1) % 4);
  * test predictions are written at fit time into a sealed file and only the validation-selected config is ever read back;
  * standardisation lives inside each pipeline, so it is fitted on training rows only;
  * US 3.1 uses NO class weights or resampling (that is the US 3.3 intervention).

One job = (variant, task, model, config, fold); outputs go to experiments/results/<variant>/<task>/<model>/<config>/fold<k>.npz
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.features.aggregate import load_table

REPO = Path(__file__).resolve().parents[2]
ERROR_CLASSES = ["correct", "substitution", "deletion", "addition"]


def load_config() -> dict:
    return yaml.safe_load((REPO / "configs" / "baseline.yaml").read_text())


def results_root(variant: str) -> Path:
    return REPO / "experiments" / "results" / variant


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    name: str
    level: str            # sentence | word | phoneme
    dataset: str          # so762 | l2arctic
    table: str
    kind: str             # reg | clf
    target: str
    splitter: str         # split (SO762 train/val/test) | fold (L2-ARCTIC grouped folds)
    labels: tuple | None = None
    positive: int | None = None
    thr: float | None = None
    models: tuple | None = None   # restrict to these model families (None = all)

    @property
    def folds(self) -> list[int]:
        return [0] if self.splitter == "split" else [0, 1, 2, 3]


def build_tasks(cfg: dict) -> dict[str, Task]:
    t = {}
    for m in ("accuracy", "fluency", "prosodic", "total"):
        t[f"sent_{m}"] = Task(f"sent_{m}", "sentence", "so762", "so762_sentence", "reg", f"y_{m}", "split")
    t["word_accuracy"] = Task("word_accuracy", "word", "so762", "so762_word", "reg", "y_word_accuracy", "split")
    t["phone_score"] = Task("phone_score", "phoneme", "so762", "so762_phone", "reg", "y_phone_score", "split")
    main = cfg["phone_binarize_threshold"]
    for thr in [main] + list(cfg["sensitivity_thresholds"]):
        models = None if thr == main else tuple(cfg["sensitivity_models"])
        t[f"phone_mispron_t{thr}"] = Task(f"phone_mispron_t{thr}", "phoneme", "so762", "so762_phone", "clf",
                                          "y_phone_score", "split", labels=(0, 1), positive=1, thr=thr, models=models)
    t["l2_phone_error"] = Task("l2_phone_error", "phoneme", "l2arctic", "l2_phone", "clf", "error_class", "fold",
                               labels=tuple(ERROR_CLASSES))
    t["l2_word_error"] = Task("l2_word_error", "word", "l2arctic", "l2_word", "clf", "y_has_error", "fold",
                              labels=(0, 1), positive=1)
    return t


def load_task(task: Task, spec: dict | None = None):
    """X, y, meta for the full table (all splits). ``spec['extra'] == 'pros'`` appends the US 3.3 pitch/energy/pause features."""
    X, meta, names = load_table(task.table)
    if spec and spec.get("extra") == "pros":
        E, meta_e, names_e = load_table(task.table + "_pros")
        assert len(E) == len(X) and (meta_e["utt_id"].to_numpy() == meta["utt_id"].to_numpy()).all()
        X, names = np.hstack([X, E]), names + ["x_" + n for n in names_e]
    if task.dataset == "so762":      # the feature meta carries the OFFICIAL train/test flag; apply the Sprint 2 train/val/test speakers
        sp = pd.read_csv(REPO / "data" / "splits" / "so762_speakers.csv", dtype={"speaker": str}).set_index("speaker")["split"]
        meta["split"] = meta["speaker"].map(sp)
        assert meta["split"].notna().all() and set(meta["split"]) == {"train", "val", "test"}
    y = meta[task.target].to_numpy()
    if task.thr is not None:
        y = (y < task.thr).astype(int)
    return X, y, meta.reset_index(drop=True), names


def split_masks(task: Task, meta: pd.DataFrame, fold: int) -> dict[str, np.ndarray]:
    if task.splitter == "split":
        return {s: (meta["split"] == s).to_numpy() for s in ("train", "val", "test")}
    f = meta["fold"].to_numpy()
    val_fold = (fold + 1) % 4
    return {"test": f == fold, "val": f == val_fold, "train": (f != fold) & (f != val_fold)}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def configs_for(task: Task, cfg: dict, variant: str = "us31") -> list[tuple[str, str, dict]]:
    """[(model, config_id, params)] for the task. us31: full grid + dummies. Other variants: only the config US 3.1 selected."""
    if variant != "us31":
        spec = cfg["variants"][variant]
        if task.kind not in spec.get("kinds", ["reg", "clf"]) or task.name not in spec.get("tasks", [task.name]):
            return []
        sel = pd.read_csv(REPO / "results" / "tables" / "baseline" / "us31_selected.csv")
        sel = sel[sel.task == task.name].set_index("model")["config"]
        out = []
        for model in cfg["variant_models"][task.kind]:
            if model in cfg.get("variant_skip", {}).get(task.name, []) or (task.models and model not in task.models):
                continue
            config = sel[model]
            (key,) = cfg["grids"][task.kind][model]            # every grid has exactly one hyper-parameter
            assert config.startswith(key), (config, key)
            v = config[len(key):]
            params = {key: float(v) if "." in v else int(v)}
            out.append((model, config, params))
        return out
    out = [("dummy", "default", {})]
    if task.kind == "reg":
        out.append(("dummy_median", "default", {}))   # median predictor: shows how much "MAE" is just skew on these targets
    for model, grid in cfg["grids"][task.kind].items():
        if task.models is not None and model not in task.models:
            continue
        keys = sorted(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, vals))
            out.append((model, "_".join(f"{k}{v}" for k, v in params.items()), params))
    if task.models is not None:
        out = [o for o in out if o[0].startswith("dummy") or o[0] in task.models]
    return out


def build_model(task: Task, model: str, params: dict, cfg: dict, class_weight=None):
    seed = cfg["seed"]
    rf = cfg["rf"]
    if task.kind == "reg":
        if model == "dummy":
            return DummyRegressor(strategy="mean")
        if model == "dummy_median":
            return DummyRegressor(strategy="median")
        if model == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
        if model == "svr":
            return TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=params["C"], gamma="scale", epsilon=0.1, cache_size=1000)),
                transformer=StandardScaler())
        if model == "rf":
            return RandomForestRegressor(n_estimators=rf["n_estimators"], max_features=rf["max_features"],
                                         min_samples_leaf=params["min_samples_leaf"], n_jobs=4, random_state=seed)
    else:
        if model == "dummy":
            return DummyClassifier(strategy="most_frequent")
        if model == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=300, class_weight=class_weight))
        if model == "svm":
            return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=params["C"], gamma="scale", cache_size=1000, class_weight=class_weight))
        if model == "rf":
            return RandomForestClassifier(n_estimators=rf["n_estimators"], max_features=rf["max_features"],
                                          min_samples_leaf=params["min_samples_leaf"], n_jobs=4, random_state=seed,
                                          class_weight=class_weight)
    raise ValueError(f"unknown model {model} for {task.kind}")


def job_path(variant: str, task: str, model: str, config: str, fold: int) -> Path:
    return results_root(variant) / task / model / config / f"fold{fold}.npz"


def resample_indices(y, bins, power, n, seed):
    """Draw n training rows with probability ~ (share of the row's score bin) ** -power: flattens a skewed target."""
    b = np.digitize(y, bins[1:-1])
    share = np.bincount(b, minlength=len(bins) - 1) / len(b)
    w = share[b] ** (-power)
    return np.random.default_rng(seed).choice(len(y), size=n, replace=True, p=w / w.sum())


def fit_job(task: Task, model: str, config: str, params: dict, fold: int, variant: str, cfg: dict,
            data=None, class_weight=None) -> dict:
    spec = cfg["variants"].get(variant, {})
    class_weight = class_weight or spec.get("class_weight")
    X, y, meta, _ = data if data is not None else load_task(task, spec)
    m = split_masks(task, meta, fold)
    tr = np.flatnonzero(m["train"])
    if spec.get("resample_power"):
        tr = tr[resample_indices(y[tr], cfg["resample_bins"][task.name], spec["resample_power"], len(tr), cfg["seed"])]
    if model in ("svr", "svm") and len(tr) > cfg["kernel_max_train"]:
        tr = np.sort(np.random.default_rng(cfg["seed"]).choice(tr, cfg["kernel_max_train"], replace=False))
    est = build_model(task, model, params, cfg, class_weight)
    t0 = time.time()
    est.fit(X[tr], y[tr])
    fit_s = time.time() - t0
    out = job_path(variant, task.name, model, config, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    def _pred(rows):
        p = est.predict(rows)
        return p.astype(str) if p.dtype == object else p      # string class labels -> fixed-width unicode (no pickling)
    extra = {}
    if task.kind == "clf" and hasattr(est, "predict_proba"):
        extra = {"val_proba": est.predict_proba(X[m["val"]]).astype(np.float32),
                 "test_proba": est.predict_proba(X[m["test"]]).astype(np.float32), "classes": np.array(est.classes_).astype(str)}
    np.savez_compressed(out, val_pred=_pred(X[m["val"]]), test_pred=_pred(X[m["test"]]),
                        n_train=len(tr), fit_s=fit_s, **extra)
    return {"fit_s": fit_s, "n_train": len(tr)}


def all_jobs(cfg: dict, variant: str, only_tasks=None, only_models=None):
    for task in build_tasks(cfg).values():
        if only_tasks and task.name not in only_tasks:
            continue
        for model, config, params in configs_for(task, cfg, variant):
            if only_models and model not in only_models:
                continue
            for fold in task.folds:
                yield task, model, config, params, fold


def run(variant: str, budget: float, only_tasks=None, only_models=None) -> None:
    cfg = load_config()
    t_start = time.time()
    cache_key, data = None, None
    todo = [j for j in all_jobs(cfg, variant, only_tasks, only_models) if not job_path(variant, j[0].name, j[1], j[2], j[4]).exists()]
    total = len(list(all_jobs(cfg, variant, only_tasks, only_models)))
    print(f"[run] {len(todo)} of {total} jobs missing for variant '{variant}'")
    default_s = {"rf": 60.0, "svm": 30.0, "svr": 15.0, "logreg": 15.0, "ridge": 5.0, "dummy": 2.0}
    seen: dict = {}                      # (task, model) -> slowest fit so far (this run or earlier runs)
    done = 0
    for task, model, config, params, fold in todo:
        key = (task.name, model)
        if key not in seen:
            prior = [float(np.load(f)["fit_s"]) for f in (results_root(variant) / task.name / model).glob("*/fold*.npz")]
            seen[key] = max(prior) if prior else default_s.get(model, 60.0) / 1.3
        if time.time() - t_start + 1.3 * seen[key] + 10 > budget:
            break
        if cache_key != task.name:
            data, cache_key = load_task(task, cfg["variants"].get(variant, {})), task.name
        info = fit_job(task, model, config, params, fold, variant, cfg, data=data)
        seen[key] = max(seen[key], info["fit_s"])
        done += 1
        print(f"[run] {task.name:20s} {model:7s} {config:22s} fold{fold}  fit {info['fit_s']:.1f}s  n_train={info['n_train']}", flush=True)
    print(f"[run] finished {done} jobs; {len(todo) - done} remaining")


def status(variant: str) -> None:
    cfg = load_config()
    rows = [(t.name, m, job_path(variant, t.name, m, c, f).exists()) for t, m, c, _, f in all_jobs(cfg, variant)]
    df = pd.DataFrame(rows, columns=["task", "model", "done"])
    print(df.groupby(["task"])["done"].agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "report"])
    ap.add_argument("--variant", default="us31")
    ap.add_argument("--budget", type=float, default=140.0)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.variant, a.budget, a.tasks, a.models)
    elif a.cmd == "status":
        status(a.variant)
    else:
        from src.evaluation.report_baseline import report
        report(a.variant)
