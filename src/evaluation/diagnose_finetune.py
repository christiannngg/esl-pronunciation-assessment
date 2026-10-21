"""
US 5.2 - weakness diagnosis of the fine-tuned encoder (Sprint 5).

    python -m src.evaluation.diagnose_finetune --tag base --seen-tag base__seen

Evidence, in the order of the user story:
  1. Train / validation loss curves (per job, from the history each run saved): gap at the best epoch, divergence after it, how early the best epoch
     is, and how much fine-tuning added over the job's own head-only epoch (the frozen-encoder state at the start of the same run).
  2. Seen vs held-out speakers (the `--holdout-seen 0.2` runs, tag `base__seen`): score on UNSEEN utterances of TRAINING speakers vs on held-out
     validation / test speakers, with independent speaker-bootstrap CIs, plus the fraction of evaluation sentences whose text also occurs in training.
  3. Fine-tuned vs frozen: which tasks fine-tuning helped, left unchanged or hurt (test, paired bootstrap, from US 5.1; validation difference), the age-group /
     L1 breakdown for the weak spots Sprint 4 found, and regression-to-the-mean slope.
Criteria C1-C5 and their thresholds are declared in configs/finetune.yaml (`diagnose52`) before the diagnostic runs were inspected. A task's overfitting
risk is high / medium / low by how many criteria fire. Decisions for US 5.3 use validation-side evidence only; the test-set group tables are descriptive.

Outputs: results/tables/finetune/us52_*.csv, us52_report.md, results/figures/us52/*.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation import report_frozen as RF
from src.evaluation import report_us43 as RU
from src.evaluation import report_us51 as R51
from src.features import ssl_embeddings as se
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models.baselines import build_tasks, load_config as load_baseline_config, split_masks

warnings.filterwarnings("ignore")
OUT = FT.REPO / "results" / "tables" / "finetune"
FIG = FT.REPO / "results" / "figures" / "us52"


# --------------------------------------------------------------------------- #
# 1. curves
# --------------------------------------------------------------------------- #
def rel_gap(kind: str, val: float, train: float) -> float:
    """Relative generalisation gap in natural metric units. Positive = better on train. RMSE: (val - train) / val;  macro-F1: (train - val) / train."""
    if kind == "reg":
        return (val - train) / val if val else float("nan")
    return (train - val) / train if train else float("nan")


def job_summary(task, d, thr: dict) -> dict:
    """One row of curve statistics for one finished job (npz-like mapping written by finetune.run_job)."""
    s = 1.0 if task.kind == "reg" else -1.0                       # history stores lower-is-better values (-macro-F1 for classification)
    vp, tp = s * np.asarray(d["hist_val_primary"], float), s * np.asarray(d["hist_trsub_primary"], float)
    vl, tl = np.asarray(d["hist_val_loss"], float), np.asarray(d["hist_train_loss"], float)
    b, last = int(d["best_epoch"]), len(vp) - 1
    spe = max(int(d["steps_per_epoch"]), 1)
    n_head = int(round(int(d["freeze_steps"]) / spe))
    ref = max(n_head - 1, 0)                                       # last head-only epoch: the frozen-encoder state of this very run
    gain = (vp[ref] - vp[b]) if task.kind == "reg" else (vp[b] - vp[ref])
    return {"task": task.name, "fold": int(d["fold"]) if "fold" in d else -1, "seed": int(d["seed"]), "best_epoch": b, "epochs_run": last + 1, "head_epochs": n_head,
            "val_at_best": vp[b], "trainsub_at_best": tp[b], "gap_best": rel_gap(task.kind, vp[b], tp[b]), "gap_last": rel_gap(task.kind, vp[last], tp[last]),
            "val_loss_rise": float(vl[-1] / vl.min() - 1.0) if vl.min() > 0 else float("nan"),
            "train_loss_drop": float((tl[b] - tl[-1]) / tl[b]) if b < last and tl[b] > 0 else 0.0,
            "gain_over_head_only": float(gain), "gain_rel": float(gain / abs(vp[ref])) if vp[ref] else float("nan"),
            "early_best": b <= int(thr["early_best_epoch"]), "head_only_best": b == 0}


def collect_job_summaries(cfg: dict, tag: str, tasks: dict) -> pd.DataFrame:
    thr = cfg["diagnose52"]
    rows = []
    for name in cfg["tasks"]:
        task = tasks[name]
        for fold in task.folds:
            for seed in FT.seeds_for(task, cfg, None):
                f = FT.job_file(cfg, tag, name, fold, seed)
                if f.exists():
                    d = dict(np.load(f, allow_pickle=True))
                    d["fold"] = fold
                    rows.append(job_summary(task, d, thr))
    return pd.DataFrame(rows)


def plot_curves(cfg: dict, tag: str, tasks: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    for kind_name, keys, ylab in (("primary", ("hist_trsub_primary", "hist_val_primary"), "metric"), ("loss", ("hist_train_loss", "hist_val_loss"), "loss")):
        n = len(cfg["tasks"])
        fig, axes = plt.subplots(3, (n + 2) // 3, figsize=(4.2 * ((n + 2) // 3), 9), squeeze=False)
        for ax, name in zip(axes.T.ravel(), cfg["tasks"]):
            task = tasks[name]
            s = 1.0 if task.kind == "reg" or kind_name == "loss" else -1.0
            for fold in task.folds:
                for seed in FT.seeds_for(task, cfg, None):
                    f = FT.job_file(cfg, tag, name, fold, seed)
                    if not f.exists():
                        continue
                    d = np.load(f, allow_pickle=True)
                    for k, color, lab in ((keys[0], "tab:blue", "train" if kind_name == "loss" else "train subset (eval mode)"), (keys[1], "tab:red", "validation")):
                        y = s * np.asarray(d[k], float)
                        ax.plot(np.arange(len(y)), y, color=color, alpha=0.5, lw=1, label=lab if (fold == task.folds[0] and seed == FT.seeds_for(task, cfg, None)[0]) else None)
                    ax.plot(int(d["best_epoch"]), s * np.asarray(d[keys[1]], float)[int(d["best_epoch"])], "k.", ms=5)
                    ax.axvline(max(int(round(int(d["freeze_steps"]) / max(int(d["steps_per_epoch"]), 1))), 0) - 0.5, color="gray", ls=":", lw=0.8)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("epoch")
            ax.set_ylabel(("macro-F1" if task.kind == "clf" and kind_name == "primary" else ("RMSE" if kind_name == "primary" else "loss")))
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"US 5.2 {kind_name} curves, tag {tag} (dotted = encoder unfrozen, black dot = best epoch)", fontsize=10)
        fig.tight_layout()
        p = FIG / f"curves_{kind_name}_{tag}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# 2. seen vs held-out speakers
# --------------------------------------------------------------------------- #
def indep_diff(spk_a, ya, pa, spk_b, yb, pb, fast, key, n_boot: int = 500, seed: int = 0) -> dict:
    """metric(a) - metric(b) where a and b are DIFFERENT speakers: independent speaker bootstrap of both sides."""
    na, ia = M._speaker_index(spk_a)
    nb, ib = M._speaker_index(spk_b)
    rng = np.random.default_rng(seed)
    ya, pa, yb, pb = map(np.asarray, (ya, pa, yb, pb))
    ds = []
    for _ in range(n_boot):
        ra = np.concatenate([ia[k] for k in rng.integers(0, na, na)])
        rb = np.concatenate([ib[k] for k in rng.integers(0, nb, nb)])
        try:
            ds.append(fast(ya[ra], pa[ra])[key] - fast(yb[rb], pb[rb])[key])
        except ValueError:
            continue
    d = np.array(ds)
    return {"diff": float(fast(ya, pa)[key] - fast(yb, pb)[key]), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def _split_pred(task, ds: list, key: str):
    """Seed-averaged (rows, prediction labels/values) of one split from several jobs of one fold; classification: argmax of the mean probability."""
    rows = ds[0][f"{key}_rows"]
    for d in ds[1:]:
        assert (d[f"{key}_rows"] == rows).all(), f"{task.name}: seeds disagree on the {key} rows"
    if task.kind == "reg":
        return rows, np.mean([d[f"{key}_pred"].astype(float) for d in ds], 0)
    return rows, FH.decode_labels(task, np.mean([d[f"{key}_proba"] for d in ds], 0).argmax(1))


def seen_vs_heldout(cfg: dict, tag: str, task, y_all, meta, n_boot: int) -> tuple[list[dict], dict] | None:
    """Rows (one per split) for the holdout runs of one task, and the seen-vs-validation gap with CI. None if the runs are missing."""
    jobs = R51.load_jobs(cfg, tag, task)
    if jobs is None or any("seen_rows" not in jobs[f][0].files for f in task.folds):
        return None
    fast, key = RF._fast(task), ("rmse" if task.kind == "reg" else "macro_f1")
    parts = {}
    for split in ("train", "seen", "val", "test"):
        rl, pl = [], []
        for f in task.folds:
            r, p = _split_pred(task, jobs[f], split)
            rl.append(r); pl.append(p)
        rows, pred = np.concatenate(rl), np.concatenate(pl)
        parts[split] = (rows, y_all[rows], pred, meta["speaker"].to_numpy()[rows])
    out = []
    for split, (rows, y, p, spk) in parts.items():
        ye, pe = RF._enc(task, y), RF._enc(task, p)
        ci = M.speaker_bootstrap(spk, ye, pe, fast, n_boot=n_boot, seed=0)
        out.append({"task": task.name, "split": split, "n": len(rows), "n_speakers": len(np.unique(spk)), "metric": key, "value": fast(ye, pe)[key],
                    "lo": ci[key][0], "hi": ci[key][1]})
    sa, va = parts["seen"], parts["val"]
    d = indep_diff(sa[3], RF._enc(task, sa[1]), RF._enc(task, sa[2]), va[3], RF._enc(task, va[1]), RF._enc(task, va[2]), fast, key, n_boot)
    seen_v, val_v = fast(RF._enc(task, sa[1]), RF._enc(task, sa[2]))[key], fast(RF._enc(task, va[1]), RF._enc(task, va[2]))[key]
    gap = {"task": task.name, "seen": seen_v, "heldout_val": val_v, "rel_gap": rel_gap(task.kind, val_v, seen_v), "abs_diff_val_minus_seen": -d["diff"] if task.kind == "reg" else d["diff"],
           "lo": (-d["hi"] if task.kind == "reg" else d["lo"]), "hi": (-d["lo"] if task.kind == "reg" else d["hi"])}
    gap["ci_excludes_0"] = bool(gap["lo"] > 0 or gap["hi"] < 0)
    return out, gap


def sentence_texts(task) -> dict:
    """utt_id -> lower-cased word sequence, from the Sprint 3 word meta tables (L2-ARCTIC: annotated word slots; additions are not words)."""
    name = "so762_word" if task.dataset == "so762" else "l2_word"
    w = pd.read_csv(se.tables_dir() / f"{name}_meta.csv.gz", dtype={"utt_id": str})
    if "word" not in w.columns:
        return {}
    w = w.sort_values(["utt_id", "word_idx"])
    return w.groupby("utt_id")["word"].apply(lambda s: " ".join(str(x).lower() for x in s)).to_dict()


def text_overlap(cfg: dict, tag: str, task, meta) -> list[dict]:
    """Share of evaluation utterances whose sentence also occurs among the TRAINING utterances of the same run (holdout utterances excluded)."""
    jobs = R51.load_jobs(cfg, tag, task)
    texts = sentence_texts(task)
    if jobs is None or not texts:
        return []
    rows = []
    agg = {s: [0, 0] for s in ("seen", "val", "test")}
    for f in task.folds:
        d = jobs[f][0]
        m = split_masks(task, meta, f)
        utt = meta["utt_id"].to_numpy()
        seen_utts = set(utt[d["seen_rows"]]) if "seen_rows" in d.files else set()
        train_texts = {texts.get(u) for u in set(utt[m["train"]]) - seen_utts} - {None}
        for s in agg:
            if f"{s}_rows" not in d.files:
                continue
            us = set(utt[d[f"{s}_rows"]])
            agg[s][0] += sum(texts.get(u) in train_texts for u in us)
            agg[s][1] += len(us)
    for s, (k, n) in agg.items():
        if n:
            rows.append({"task": task.name, "split": s, "utterances": n, "text_in_train_share": k / n})
    return rows


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #
def aggregate_flags(task_name: str, js: pd.DataFrame, thr: dict, seen_gap: dict | None) -> dict:
    f = {"task": task_name, "jobs": len(js), "mean_best_epoch": js.best_epoch.mean(), "mean_epochs_run": js.epochs_run.mean(),
         "gap_best": js.gap_best.mean(), "gap_last": js.gap_last.mean(), "val_loss_rise": js.val_loss_rise.mean(), "train_loss_drop": js.train_loss_drop.mean(),
         "share_diverging": float(((js.val_loss_rise >= thr["val_loss_rise"]) & (js.train_loss_drop >= thr["train_loss_drop"])).mean()),
         "share_early_best": float(js.early_best.mean()), "share_head_only_best": float(js.head_only_best.mean()),
         "gain_rel_mean": js.gain_rel.mean(), "share_low_gain": float((js.gain_rel < thr["min_gain_rel"]).mean())}
    f["C1_gap"] = bool(f["gap_best"] >= thr["gap_rel"])
    f["C2_diverge"] = bool(f["share_diverging"] >= thr["diverge_share"])
    f["C3_early_best"] = bool(f["share_early_best"] >= thr["early_best_share"])
    f["C4_low_gain"] = bool(f["share_low_gain"] >= thr["low_gain_share"])
    f["C5_seen_gap"] = None if seen_gap is None else bool(seen_gap["rel_gap"] >= thr["seen_gap_rel"] and seen_gap["ci_excludes_0"])
    fired = [k for k in ("C1_gap", "C2_diverge", "C3_early_best", "C4_low_gain", "C5_seen_gap") if f[k]]
    f["criteria_fired"] = ",".join(c.split("_")[0] for c in fired)
    f["n_fired"] = len(fired)
    f["overfit_risk"] = "high" if len(fired) >= thr["risk_high"] else ("medium" if len(fired) >= thr["risk_medium"] else "low")
    return f


# --------------------------------------------------------------------------- #
# 3. fine-tuned vs frozen
# --------------------------------------------------------------------------- #
def vs_frozen_table(head: pd.DataFrame, h43: pd.DataFrame, jobs: pd.DataFrame, encoder: str) -> pd.DataFrame:
    rows = []
    for r in head.itertuples():
        f43 = h43[(h43.encoder == encoder) & (h43.task == r.task)]
        val43 = float(f43.val_43.iloc[0]) if len(f43) else np.nan
        reg = r.metric == "rmse"
        val_delta = (r.val - val43) if reg else (r.val - val43)                       # fine-tuned minus frozen on validation (natural metric units)
        val_better = val_delta < 0 if reg else val_delta > 0
        sd = float(jobs[jobs.task == r.task].val_primary.std()) if (jobs.task == r.task).sum() > 1 else np.nan
        verdict = getattr(r, "verdict_vs_43", "n/a")
        status = {"better": "helped", "worse": "hurt"}.get(verdict, "unchanged" if verdict != "n/a" else "n/a")
        rows.append({"task": r.task, "metric": r.metric, "frozen_43_test": r.frozen_43, "finetuned_test": r.finetuned, "delta_test": getattr(r, "d_vs_43", np.nan),
                     "delta_lo": getattr(r, "d_vs_43_lo", np.nan), "delta_hi": getattr(r, "d_vs_43_hi", np.nan), "p": getattr(r, "d_vs_43_p", np.nan),
                     "val_frozen_43": val43, "val_finetuned": r.val, "val_delta": val_delta, "val_better": bool(val_better), "job_val_sd": sd,
                     "test_verdict": verdict, "status": status,
                     "val_regression": bool((not val_better) and (np.isnan(sd) or abs(val_delta) > sd))})
    return pd.DataFrame(rows)


def group_col(task, meta: pd.DataFrame) -> tuple[str, np.ndarray]:
    if task.dataset == "so762":
        return "age group", np.where(meta["age_group"].astype(str).str.lower().eq("adult"), "adult", "child/teen")
    return "L1", meta["l1"].astype(str).to_numpy()


def group_table(task, ft: dict, fr: dict, n_boot: int) -> list[dict]:
    """Per age group / L1: metric of the fine-tuned and the frozen 4.3 system on the same test rows and the paired speaker-bootstrap delta."""
    fast, key = RF._fast(task), ("rmse" if task.kind == "reg" else "macro_f1")
    name, g = group_col(task, ft["meta"])
    y, pf, pr = RF._enc(task, ft["y"]), RF._enc(task, ft["pred"]), RF._enc(task, fr["pred"])
    spk = ft["meta"]["speaker"].to_numpy()
    rows = []
    for v in sorted(set(g)):
        m = g == v
        if m.sum() < 30 or len(np.unique(spk[m])) < 3:
            continue
        t = M.paired_speaker_bootstrap(spk[m], y[m], pr[m], pf[m], fast, key, n_boot=n_boot)
        sig = t["lo"] > 0 or t["hi"] < 0
        good = t["delta"] < 0 if task.kind == "reg" else t["delta"] > 0
        rows.append({"task": task.name, "group_by": name, "group": v, "n": int(m.sum()), "n_speakers": len(np.unique(spk[m])), "frozen_43": fast(y[m], pr[m])[key],
                     "finetuned": fast(y[m], pf[m])[key], "delta": t["delta"], "lo": t["lo"], "hi": t["hi"], "p": t["p"],
                     "verdict": ("better" if good else "worse") if sig else "no significant change"})
    return rows


def calibration_rows(task, ft: dict, fr: dict) -> list[dict]:
    """Regression to the mean (word accuracy, phone score): slope of prediction on truth and mean bias in the lowest score bin, test rows."""
    lo_thr = 4.0 if task.name == "word_accuracy" else 0.5
    y = np.asarray(ft["y"], float)
    rows = []
    for label, p in (("frozen 4.3", fr["pred"]), ("fine-tuned", ft["pred"])):
        p = np.asarray(p, float)
        m = y <= lo_thr if task.name == "word_accuracy" else y < lo_thr
        rows.append({"task": task.name, "system": label, "slope_pred_on_true": float(np.polyfit(y, p, 1)[0]), "pcc": M.regression_metrics(y, p)["pcc"],
                     f"low_bin_share": float(m.mean()), "low_bin_mean_true": float(y[m].mean()), "low_bin_mean_pred": float(p[m].mean())})
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _f(v, nd=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))


def md(df: pd.DataFrame, cols: list[str], nd: int = 3) -> list[str]:
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        L.append("| " + " | ".join(_f(v, nd) if not isinstance(v, (bool, np.bool_)) else ("yes" if v else "no") for v in r) + " |")
    return L


def write_report(path: Path, thr: dict, flags: pd.DataFrame, seen: pd.DataFrame, gaps: pd.DataFrame, overlap: pd.DataFrame, vs: pd.DataFrame,
                 groups: pd.DataFrame, calib: pd.DataFrame, tag: str, seen_tag: str, figs: list[Path]) -> None:
    L = [f"# US 5.2 - fine-tuned encoder diagnosis (tags `{tag}`, `{seen_tag}`)", "",
         "Criteria (declared before the diagnostic runs were inspected; configs/finetune.yaml `diagnose52`): "
         f"**C1** relative train-vs-validation gap at the best epoch >= {thr['gap_rel']:.2f}; **C2** validation loss >= {thr['val_loss_rise']:.0%} above its minimum at the last epoch while "
         f"training loss fell >= {thr['train_loss_drop']:.0%} since the best epoch (in >= {thr['diverge_share']:.0%} of jobs); **C3** best epoch <= {thr['early_best_epoch']} in >= {thr['early_best_share']:.0%} of jobs; "
         f"**C4** fine-tuning improved validation by < {thr['min_gain_rel']:.0%} over the job's own head-only epoch in >= {thr['low_gain_share']:.0%} of jobs; **C5** seen-speaker (unseen utterances) score beats "
         f"held-out-speaker validation by >= {thr['seen_gap_rel']:.0%} with a CI that excludes 0. Risk: high >= {thr['risk_high']} criteria, medium {thr['risk_medium']}, else low. "
         "C1, C2 and C5 measure overfitting; C3 and C4 measure instability / low value of fine-tuning.", ""]
    L += ["## 1. Overfitting risk and flags per task", ""]
    L += md(flags, ["task", "jobs", "overfit_risk", "criteria_fired", "mean_best_epoch", "mean_epochs_run", "gap_best", "gap_last", "val_loss_rise", "train_loss_drop",
                    "share_early_best", "share_head_only_best", "gain_rel_mean"])
    L += ["", "gap = relative train-vs-validation gap (positive = better on training data); gain_rel = validation improvement over the job's own head-only epoch.", ""]
    if figs:
        L += ["Curves: " + ", ".join(f"`{p.relative_to(FT.REPO)}`" for p in figs), ""]
    L += ["## 2. Seen vs held-out speakers", ""]
    if seen.empty:
        L += ["Not available: run `bash run_52_local.sh seen` first.", ""]
    else:
        L += ["Runs trained on 80 % of each training speaker's utterances. `seen` = the held-back 20 % (same speakers, unseen utterances), `val` / `test` = held-out speakers, "
              "`train` = a subsample of training utterances in eval mode. RMSE (regression) or macro-F1 with the class-weighted argmax (classification); 95 % speaker-bootstrap CIs.", ""]
        L += md(seen, ["task", "split", "n", "n_speakers", "metric", "value", "lo", "hi"])
        L += ["", "Seen-speaker gap (validation minus seen for RMSE, seen minus validation for macro-F1; positive = seen speakers are easier):", ""]
        L += md(gaps, ["task", "seen", "heldout_val", "rel_gap", "abs_diff_val_minus_seen", "lo", "hi", "ci_excludes_0"])
        if not overlap.empty:
            L += ["", "Sentence-text overlap with the training utterances of the same run (if held-out sentences are also read by training speakers, the seen/held-out gap is not a text effect):", ""]
            L += md(overlap, ["task", "split", "utterances", "text_in_train_share"])
        L.append("")
    L += ["## 3. Fine-tuned vs frozen 4.3 (same encoder)", "",
          "Test: paired speaker bootstrap from US 5.1. Validation: difference of the validation primary metric (RMSE lower / macro-F1 higher is better); `job_val_sd` is the spread of the fine-tuned "
          "validation score across jobs. `status` follows the paired test verdict; `val_regression` marks tasks where validation is worse than frozen by more than the job spread.", ""]
    L += md(vs, ["task", "metric", "frozen_43_test", "finetuned_test", "delta_test", "delta_lo", "delta_hi", "test_verdict", "status", "val_frozen_43", "val_finetuned", "val_delta", "job_val_sd", "val_regression"])
    if not groups.empty:
        L += ["", "### Age-group / L1 breakdown, test rows (descriptive; not used to select fixes)", ""]
        L += md(groups, ["task", "group_by", "group", "n", "n_speakers", "frozen_43", "finetuned", "delta", "lo", "hi", "verdict"])
    if not calib.empty:
        L += ["", "### Regression to the mean (test)", ""]
        L += md(calib, ["task", "system", "slope_pred_on_true", "pcc", "low_bin_share", "low_bin_mean_true", "low_bin_mean_pred"])
    L.append("")
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seen-tag", default=None)
    ap.add_argument("--frozen-root", default=None)
    ap.add_argument("--fix-root", default=None)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    cfg = FT.load_config()
    thr = cfg["diagnose52"]
    seen_tag = a.seen_tag or thr["seen_tag"]
    tasks = build_tasks(load_baseline_config())
    OUT.mkdir(parents=True, exist_ok=True)
    nb = int(thr["n_boot"])

    js = collect_job_summaries(cfg, a.tag, tasks)
    if js.empty:
        raise SystemExit(f"no finished jobs for tag {a.tag!r}")
    figs = [] if a.no_plots else plot_curves(cfg, a.tag, tasks)

    seen_rows, gap_rows, over_rows = [], {}, []
    for name in cfg["tasks"]:
        task = tasks[name]
        y_all, meta = FH.load_labels(task)
        r = seen_vs_heldout(cfg, seen_tag, task, y_all, meta, nb)
        if r is not None:
            seen_rows += r[0]
            gap_rows[name] = r[1]
            over_rows += text_overlap(cfg, seen_tag, task, meta)
    flags = pd.DataFrame([aggregate_flags(n, js[js.task == n], thr, gap_rows.get(n)) for n in cfg["tasks"] if (js.task == n).any()])

    head = pd.read_csv(OUT / "us51_headline.csv") if (OUT / "us51_headline.csv").exists() else pd.DataFrame()
    jobs51 = pd.read_csv(OUT / "us51_jobs.csv") if (OUT / "us51_jobs.csv").exists() else pd.DataFrame()
    h43 = pd.read_csv(RF.OUT / "us43_headline.csv")
    vs = vs_frozen_table(head, h43, jobs51, cfg["encoder"]) if not head.empty else pd.DataFrame()

    groups, calib = [], []
    refs = R51.frozen_refs(cfg["encoder"], h43, Path(a.frozen_root) if a.frozen_root else None, Path(a.fix_root) if a.fix_root else None)
    for name in cfg["tasks"]:
        task = tasks[name]
        jobs = R51.load_jobs(cfg, a.tag, task)
        fr = (refs.get(name) or {}).get("4.3")
        if jobs is None or fr is None:
            continue
        y_all, meta = FH.load_labels(task)
        P = R51.pooled_ft(task, jobs, y_all, meta)
        _, pv, pt, _, _ = R51.headline_variant(task, P["val"], P["test"])
        ft = {"y": P["test"]["y"], "pred": pt, "meta": P["test"]["meta"]}
        try:
            groups += group_table(task, ft, fr, nb)
        except AssertionError as e:
            print(f"[us52] {name}: {e}")
        if name in ("word_accuracy", "phone_score"):
            calib += calibration_rows(task, ft, fr)

    seen_df, gaps_df, over_df = pd.DataFrame(seen_rows), pd.DataFrame(list(gap_rows.values())), pd.DataFrame(over_rows)
    groups_df, calib_df = pd.DataFrame(groups), pd.DataFrame(calib)
    for name, df in (("job_summaries", js), ("flags", flags), ("seen_vs_heldout", seen_df), ("seen_gaps", gaps_df), ("text_overlap", over_df),
                     ("vs_frozen", vs), ("groups", groups_df), ("calibration", calib_df)):
        df.to_csv(OUT / f"us52_{name}.csv", index=False)
    write_report(OUT / "us52_report.md", thr, flags, seen_df, gaps_df, over_df, vs, groups_df, calib_df, a.tag, seen_tag, figs)
    print((OUT / "us52_report.md").read_text())


if __name__ == "__main__":
    main()
