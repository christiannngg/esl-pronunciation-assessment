"""
US 5.1 - fine-tuned encoder results vs Sprints 3-4.

    python -m src.evaluation.report_us51 --tag base
    python -m src.evaluation.report_us51 --tag base --frozen-root experiments/results/frozen --fix-root experiments/results/frozen43

Reads the jobs written by ``python -m src.models.finetune run --tag <tag>`` (experiments/results/finetune/<tag>/<task>/fold<k>/seed<s>.npz).
Per task: seeds are averaged (regression: mean prediction, classification: mean probability, as in US 4.1), then, for classification, the
plain-argmax and the US 3.3 `cw+thr` (decision weights tuned on pooled VALIDATION rows) variants are scored and the one with the better
validation score is the headline. Everything is chosen on validation; test is scored once. Deltas are paired speaker-level bootstraps on the
identical test rows against the frozen systems of the SAME encoder (US 4.1 layer 12 and the US 4.3 validation-selected system).

Outputs (results/tables/finetune/):
    us51_headline.csv / us51_report.md    fine-tuned vs US 3.1 / US 3.3 / 4.1 / 4.3, deltas with CIs, train / val / test, best frozen encoder (reference only)
    us51_jobs.csv                         one row per (task, fold, seed): best epoch, epochs run, val / train / test primary metric (seed spread)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import metrics as M
from src.evaluation import report_frozen as RF
from src.evaluation import report_us43 as RU
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models.baselines import build_tasks, load_config as load_baseline_config

warnings.filterwarnings("ignore")
OUT = FT.REPO / "results" / "tables" / "finetune"


# --------------------------------------------------------------------------- #
# pooling the fine-tuned jobs into the dicts the Sprint 4 report code uses
# --------------------------------------------------------------------------- #
def load_jobs(cfg: dict, tag: str, task, seeds=None):
    """{fold: [npz, ...]} for every fold; None when any fold has no finished job."""
    seeds = seeds or FT.seeds_for(task, cfg, None)
    out = {}
    for fold in task.folds:
        fs = [FT.job_file(cfg, tag, task.name, fold, s) for s in seeds]
        fs = [f for f in fs if f.exists()]
        if not fs:
            return None
        out[fold] = [np.load(f, allow_pickle=True) for f in fs]
    return out


def _avg(task, ds: list, key: str):
    """Seed average of predictions (regression) or probabilities (classification) of one split; rows must agree across seeds."""
    rows = ds[0][f"{key}_rows"]
    for d in ds[1:]:
        assert (d[f"{key}_rows"] == rows).all(), f"{task.name}: seeds disagree on the {key} rows"
    if task.kind == "reg":
        return rows, np.mean([d[f"{key}_pred"].astype(float) for d in ds], 0)
    return rows, np.mean([d[f"{key}_proba"] for d in ds], 0)


def pooled_ft(task, jobs: dict, y_all, meta) -> dict:
    """Dict in the layout of report_frozen.pooled(), seed-averaged and pooled over folds (validation and test separately)."""
    out = {}
    for split in ("val", "test"):
        rows_l, pred_l, tr_p, tr_y, tr_rows = [], [], [], [], []
        for fold in task.folds:
            r, p = _avg(task, jobs[fold], split)
            rows_l.append(r); pred_l.append(p)
        rows = np.concatenate(rows_l)
        P = np.concatenate(pred_l) if task.kind == "reg" else np.vstack(pred_l)
        d = {"y": y_all[rows], "meta": meta.iloc[rows].reset_index(drop=True), "rows": rows}
        if task.kind == "reg":
            d["pred"] = P
        else:
            d["proba"] = P
            d["pred"] = FH.decode_labels(task, P.argmax(1))
        out[split] = d
    # train subsample, scored with the same seed-averaged rule
    tr_rows, tr_P = [], []
    for fold in task.folds:
        r, p = _avg(task, jobs[fold], "train")
        tr_rows.append(r); tr_P.append(p)
    rows = np.concatenate(tr_rows)
    P = np.concatenate(tr_P) if task.kind == "reg" else np.vstack(tr_P)
    if task.kind == "reg":
        score = FH.primary_score(task, y_all[rows], P)
        extra = {}
    else:
        score = FH.primary_score(task, y_all[rows], FH.decode_labels(task, P.argmax(1)))
        extra = {"train_proba": P, "train_y": FH.encode_y(task, y_all[rows])}
    sd = float(np.std([np.mean([d["val_score"] for d in jobs[f]]) for f in task.folds]))
    for split in ("val", "test"):
        out[split].update({"train_score": score, "seed_sd": sd, **(extra if split == "test" else {})})
    return out


def headline_variant(task, dv, dt):
    """(variant, val_pred, test_pred, train_primary, val_primary) with the better VALIDATION score (plain/cw argmax vs cw+thr)."""
    group = "balanced" if task.kind == "clf" else "none"
    cands = [(v, pv, pt, tr, RF.primary(task, dv["y"], pv)) for v, pv, pt, tr in RF.apply_variants(task, group, dv, dt)]
    best = cands[0]
    for c in cands[1:]:
        best = c if RF.better(task, c[4], best[4]) else best
    return best


# --------------------------------------------------------------------------- #
# frozen references (same encoder) for paired comparisons
# --------------------------------------------------------------------------- #
def frozen_refs(encoder: str, headline43: pd.DataFrame, frozen_root, fix_root):
    """{task: {"4.1": preds dict, "4.3": preds dict}} for the encoder, from the Sprint 4 result files; {} when they are not on this machine."""
    try:
        systems, _ = RU.collect_systems(encoder, frozen_root, fix_root, n_boot=20)
        gtab, gsys = RU.mispron_groups(encoder, frozen_root, fix_root)
        if gsys:
            systems.update(gsys)
    except Exception as e:                                           # pragma: no cover - depends on the machine
        print(f"[us51] could not load the frozen {encoder} results ({e}); paired deltas vs 4.1 / 4.3 are skipped")
        return {}
    tasks = build_tasks(load_baseline_config())
    refs = {}
    for r in headline43[headline43.encoder == encoder].itertuples():
        task = tasks[r.task]
        b41 = RU.best_of(systems, ["4.1 L12"], r.task, task.kind)
        b43 = RU.best_of(systems, [r.frozen_43_system], r.task, task.kind)
        refs[r.task] = {"4.1": None if b41 is None else b41[2], "4.3": None if b43 is None else b43[2]}
    return refs


def _pair(task, a: dict, b: dict, n_boot: int) -> dict | None:
    if a is None:
        return None
    try:
        return RU.pair(task, a, b, n_boot)
    except AssertionError as e:
        print(f"[us51] {task.name}: {e}")
        return None


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(tag: str, variant: str | None = None, frozen_root=None, fix_root=None, n_boot: int | None = None):
    cfg = FT.load_config(variant)
    bcfg = load_baseline_config()
    n_boot = n_boot or bcfg["bootstrap"]["n_boot"]
    tasks = build_tasks(bcfg)
    h43 = pd.read_csv(RF.OUT / "us43_headline.csv")
    refs = frozen_refs(cfg["encoder"], h43, frozen_root, fix_root)
    rows, jobs_rows = [], []
    for name in cfg["tasks"]:
        task = tasks[name]
        jobs = load_jobs(cfg, tag, task)
        if jobs is None:
            print(f"[us51] {name}: no finished jobs under tag {tag!r}; skipped")
            continue
        n_missing = sum(len(FT.seeds_for(task, cfg, None)) - len(jobs[f]) for f in task.folds)
        if n_missing:
            print(f"[us51] {name}: {n_missing} job(s) missing; using the finished ones")
        y_all, meta = FH.load_labels(task)
        P = pooled_ft(task, jobs, y_all, meta)
        dv, dt = P["val"], P["test"]
        variant_name, pv, pt, tr_prim, val_prim = headline_variant(task, dv, dt)
        key = "rmse" if task.kind == "reg" else "macro_f1"
        test_sys = {"y": dt["y"], "pred": pt, "meta": dt["meta"]}
        score = RF._score(task, dt["y"], pt)
        ci = M.speaker_bootstrap(dt["meta"]["speaker"].to_numpy(), RF._enc(task, dt["y"]), RF._enc(task, pt), RF._fast(task), n_boot=n_boot,
                                 seed=bcfg["bootstrap"]["seed"])
        ref = h43[(h43.encoder == cfg["encoder"]) & (h43.task == name)]
        ref = ref.iloc[0] if len(ref) else None
        best_other = None
        if ref is not None:
            g = h43[h43.task == name]
            best_other = g.sort_values("frozen_43", ascending=task.kind == "reg").iloc[0]
        r = {"tag": tag, "encoder": cfg["encoder"], "task": name, "level": task.level, "dataset": task.dataset, "metric": key,
             "us31": None if ref is None else ref.us31, "us33": None if ref is None else ref.us33,
             "frozen_41": None if ref is None else ref.frozen_41, "frozen_43": None if ref is None else ref.frozen_43,
             "frozen_43_system": None if ref is None else ref.frozen_43_system,
             "best_frozen_43_any_encoder": None if best_other is None else best_other.frozen_43,
             "best_frozen_43_encoder": None if best_other is None else best_other.encoder,
             "finetuned": score[key], "ft_lo": ci[key][0], "ft_hi": ci[key][1], "variant": variant_name, "val": val_prim, "train": tr_prim,
             "aux": score["pcc"] if task.kind == "reg" else score["balanced_acc"], "n_seeds": min(len(v) for v in jobs.values()),
             "seed_val_sd": dv["seed_sd"]}
        if task.kind == "clf" and task.positive is not None and len(task.labels) == 2 and "proba" in dt:
            yb = RF._enc(task, dt["y"])
            r["auc"], r["ap"] = float(roc_auc_score(yb, dt["proba"][:, 1])), float(average_precision_score(yb, dt["proba"][:, 1]))
        for label in ("4.1", "4.3"):
            d = _pair(task, (refs.get(name) or {}).get(label), test_sys, n_boot)
            if d is not None:
                r.update({f"d_vs_{label.replace('.', '')}": d["delta"], f"d_vs_{label.replace('.', '')}_lo": d["lo"],
                          f"d_vs_{label.replace('.', '')}_hi": d["hi"], f"d_vs_{label.replace('.', '')}_p": d["p"],
                          f"verdict_vs_{label.replace('.', '')}": RU.verdict(task.kind, d)})
        rows.append(r)
        for fold, ds in jobs.items():
            for d in ds:
                jobs_rows.append({"task": name, "fold": fold, "seed": int(d["seed"]), "best_epoch": int(d["best_epoch"]), "epochs_run": int(d["epochs_run"]),
                                  "val_primary": float(d["val_score"]) * (1 if task.kind == "reg" else -1),
                                  "train_primary": float(d["train_score"]) * (1 if task.kind == "reg" else -1), "fit_s": float(d["fit_s"])})
    return pd.DataFrame(rows), pd.DataFrame(jobs_rows)


def _f(v, nd=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def write_md(head: pd.DataFrame, jobs: pd.DataFrame, path: Path, tag: str) -> None:
    L = [f"# US 5.1 - fine-tuned encoder vs Sprints 3-4 (tag `{tag}`, test set; every choice made on validation)", "",
         "Metric: RMSE (regression, lower is better) or macro-F1 (classification, higher is better). Fine-tuned = seed-averaged predictions; for "
         "classification the plain class-weighted argmax or the validation-tuned `cw+thr` decision weights, whichever scored better on validation. "
         "Delta = fine-tuned minus the frozen system of the same encoder on the identical test rows, paired speaker bootstrap (95% CI, two-sided p).", "",
         "| task | metric | US 3.1 | US 3.3 | frozen 4.1 | frozen 4.3 (system) | **fine-tuned** [95% CI] | rule | delta vs 4.3 [95% CI] | delta vs 4.1 | verdict vs 4.3 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in head.itertuples():
        d43 = f"{r.d_vs_43:+.3f} [{r.d_vs_43_lo:+.3f}, {r.d_vs_43_hi:+.3f}] p={r.d_vs_43_p:.3f}" if hasattr(r, "d_vs_43") and not pd.isna(getattr(r, "d_vs_43")) else ""
        d41 = f"{r.d_vs_41:+.3f} (p={r.d_vs_41_p:.3f})" if hasattr(r, "d_vs_41") and not pd.isna(getattr(r, "d_vs_41")) else ""
        verd = getattr(r, "verdict_vs_43", "") if hasattr(r, "verdict_vs_43") else ""
        L.append(f"| {r.task} | {r.metric} | {_f(r.us31)} | {_f(r.us33)} | {_f(r.frozen_41)} | {_f(r.frozen_43)} ({r.frozen_43_system}) "
                 f"| **{_f(r.finetuned)}** [{_f(r.ft_lo)}, {_f(r.ft_hi)}] | {r.variant} | {d43} | {d41} | {verd} |")
    L += ["", "## Train / validation / test and seed spread", "",
          "| task | train | validation | test | seeds | sd of per-fold validation score | best frozen 4.3 on test (any encoder; reference only) |", "|---|---|---|---|---|---|---|"]
    for r in head.itertuples():
        L.append(f"| {r.task} | {_f(r.train)} | {_f(r.val)} | {_f(r.finetuned)} | {r.n_seeds} | {_f(r.seed_val_sd)} | {_f(r.best_frozen_43_any_encoder)} ({r.best_frozen_43_encoder}) |")
    if not jobs.empty:
        g = jobs.groupby("task", sort=False).agg(jobs=("seed", "size"), best_epoch=("best_epoch", "mean"), epochs=("epochs_run", "mean"),
                                                 val_sd=("val_primary", "std"), minutes=("fit_s", lambda s: s.sum() / 60)).reset_index()
        L += ["", "## Jobs", "", "| task | jobs | mean best epoch | mean epochs run | sd of validation score across jobs | total minutes |", "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.task} | {r.jobs} | {r.best_epoch:.1f} | {r.epochs:.1f} | {_f(r.val_sd)} | {r.minutes:.0f} |")
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--frozen-root", default=None)
    ap.add_argument("--fix-root", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    head, jobs = build(a.tag, a.variant, Path(a.frozen_root) if a.frozen_root else None, Path(a.fix_root) if a.fix_root else None, a.n_boot)
    suffix = "" if a.tag == "base" else f"_{a.tag}"
    head.to_csv(OUT / f"us51_headline{suffix}.csv", index=False)
    jobs.to_csv(OUT / f"us51_jobs{suffix}.csv", index=False)
    write_md(head, jobs, OUT / f"us51_report{suffix}.md", a.tag)
    print((OUT / f"us51_report{suffix}.md").read_text())


if __name__ == "__main__":
    main()
