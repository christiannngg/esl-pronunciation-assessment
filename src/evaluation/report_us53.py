"""
US 5.3 - weakness fixes for the fine-tuned encoder: before / after metrics and the per-task selection.

    python -m src.evaluation.report_us53                       # base vs every variant under experiments/results/finetune/<variant>/
    python -m src.evaluation.report_us53 --variants lr reg     # only some variants

Inputs: the 5.1 jobs (tag ``base``) and the variant jobs (tag = variant name, run with ``python -m src.models.finetune run --tag <v> --variant <v>``).

Selection (declared before any variant was run: configs/finetune.yaml ``diagnose53`` and docs/weakness_memo_s5.md section 6). Per task, from the VALIDATION
side of the training jobs only (mean over jobs of the early-stopping metric at the best epoch):
  * target met  = relative train-validation gap at the best epoch <= gap_best_max AND last-epoch val loss / its minimum <= cap AND no training-loss
                  spike (an epoch above spike_ratio x the previous one) in any job;
  * eligible    = validation metric not worse than base by more than the margin, max(margin_floor, seed sd of the base metric within a fold);
  * base already meets the target -> nothing to fix: keep base, except the W6 tasks (cost_tasks), where the cheapest eligible frozen-bottom variant
                  (cost_variants, >= cost_saving faster per job) is preferred;
  * otherwise   -> the eligible variant that meets the target with the best validation metric; if there is none, keep base and say so.
Test is scored for every variant (paired speaker bootstrap against base, and against the frozen US 4.3 system) AFTER selection and never chooses.

Outputs (results/tables/finetune/): us53_variants.csv, us53_selection.csv, us53_headline.csv, us53_calibration.csv, us53_report.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import diagnose_finetune as D
from src.evaluation import metrics as M
from src.evaluation import report_frozen as RF
from src.evaluation import report_us43 as RU
from src.evaluation import report_us51 as R51
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models.baselines import build_tasks, load_config as load_baseline_config

warnings.filterwarnings("ignore")
OUT = R51.OUT
BASE = "base"


# --------------------------------------------------------------------------- #
# per-job and per-variant training-side summaries (numpy only)
# --------------------------------------------------------------------------- #
def max_spike(train_loss, first_ft_epoch: int) -> float:
    """Largest epoch-to-epoch ratio of the training loss after the head-only epochs (a blow-up shows as a ratio >> 1)."""
    tl = np.asarray(train_loss, float)
    r = [tl[e] / tl[e - 1] for e in range(max(int(first_ft_epoch), 0) + 1, len(tl)) if tl[e - 1] > 0]
    return float(max(r)) if r else 1.0


def job_rows(cfg: dict, tag: str, task, thr52: dict) -> list[dict]:
    rows = []
    for fold in task.folds:
        for seed in FT.seeds_for(task, cfg, None):
            f = FT.job_file(cfg, tag, task.name, fold, seed)
            if not f.exists():
                continue
            d = dict(np.load(f, allow_pickle=True))
            d["fold"] = fold
            s = D.job_summary(task, d, thr52)
            vl = np.asarray(d["hist_val_loss"], float)
            s.update(tag=tag, val_loss_ratio=float(vl[-1] / vl.min()) if vl.min() > 0 else float("nan"),
                     spike=max_spike(d["hist_train_loss"], s["head_epochs"]), fit_s=float(d["fit_s"]))
            rows.append(s)
    return rows


def within_fold_sd(rows: list[dict]) -> float:
    """Seed-to-seed sd of the validation metric inside a fold (mean over folds); NaN when every fold has a single job."""
    df = pd.DataFrame(rows)
    sds = [g.val_at_best.std(ddof=1) for _, g in df.groupby("fold") if len(g) > 1]
    return float(np.mean(sds)) if sds else float("nan")


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {"n_jobs": len(df), "val": float(df.val_at_best.mean()), "gap_best": float(df.gap_best.mean()), "loss_ratio": float(df.val_loss_ratio.mean()),
            "spike": float(df.spike.max()), "best_epoch": float(df.best_epoch.mean()), "epochs_run": float(df.epochs_run.mean()), "fit_s": float(df.fit_s.mean()),
            "seed_sd": within_fold_sd(rows)}


def target_met(kind: str, s: dict, t: dict) -> bool:
    cap = t["loss_ratio_max_reg"] if kind == "reg" else t["loss_ratio_max_clf"]
    return bool(s["gap_best"] <= t["gap_best_max"] and s["loss_ratio"] <= cap and s["spike"] < t["spike_ratio"])


def worse_by(kind: str, val: float, base_val: float) -> float:
    """Positive = worse than base (RMSE higher / macro-F1 lower)."""
    return (val - base_val) if kind == "reg" else (base_val - val)


def select(task, base: dict, variants: dict[str, dict], t: dict) -> dict:
    """The declared selection rule. ``base`` / ``variants[name]`` are summarize() dicts. Returns {selected, reason, margin, need_fix}."""
    sd = base["seed_sd"]
    margin = max(float(t["margin_floor"]), 0.0 if np.isnan(sd) else float(sd))
    complete = {v: s for v, s in variants.items() if s["n_jobs"] == base["n_jobs"]}
    eligible = {v: s for v, s in complete.items() if worse_by(task.kind, s["val"], base["val"]) <= margin}
    need_fix = not target_met(task.kind, base, t)
    out = {"margin": margin, "need_fix": need_fix, "n_complete": len(complete), "n_eligible": len(eligible)}
    if not complete:
        return {**out, "selected": BASE, "reason": "no complete variant runs yet"}
    if not need_fix:
        if task.name in t["cost_tasks"]:
            cheap = {v: s for v, s in eligible.items() if v in t["cost_variants"] and s["fit_s"] <= (1.0 - float(t["cost_saving"])) * base["fit_s"]}
            if cheap:
                v = min(cheap, key=lambda k: cheap[k]["fit_s"])
                return {**out, "selected": v, "reason": f"base meets the targets; {v} is non-inferior and {1 - cheap[v]['fit_s'] / base['fit_s']:.0%} faster per job"}
        return {**out, "selected": BASE, "reason": "base already meets the targets (no weakness to fix)"}
    ok = {v: s for v, s in eligible.items() if target_met(task.kind, s, t)}
    if ok:
        v = min(ok, key=lambda k: worse_by(task.kind, ok[k]["val"], base["val"]))
        return {**out, "selected": v, "reason": f"{v} meets the targets within the validation margin (best validation metric among {len(ok)} qualifying)"}
    if eligible:
        return {**out, "selected": BASE, "reason": "no non-inferior variant meets the targets; weakness not fixed by the tried regularisers"}
    return {**out, "selected": BASE, "reason": "every variant is worse than base on validation by more than the margin"}


# --------------------------------------------------------------------------- #
# test-side re-measurement (after selection)
# --------------------------------------------------------------------------- #
def tag_preds(cfg: dict, tag: str, task, y_all, meta):
    jobs = R51.load_jobs(cfg, tag, task)
    if jobs is None:
        return None
    P = R51.pooled_ft(task, jobs, y_all, meta)
    _, pv, pt, _, val_prim = R51.headline_variant(task, P["val"], P["test"])
    return {"y": P["test"]["y"], "pred": pt, "meta": P["test"]["meta"], "val_pred": pv, "val_primary": val_prim, "val": P["val"]}


def fmt(v, nd=3):
    return D._f(v, nd)


def build(variants: list[str], n_boot: int | None = None):
    cfg = FT.load_config()
    t, thr52 = cfg["diagnose53"], cfg["diagnose52"]
    bcfg = load_baseline_config()
    n_boot = n_boot or t["n_boot"]
    tasks = build_tasks(bcfg)
    h43 = pd.read_csv(RF.OUT / "us43_headline.csv")
    refs = R51.frozen_refs(cfg["encoder"], h43, None, None)
    vrows, srows, hrows, crows = [], [], [], []
    for name in cfg["tasks"]:
        task = tasks[name]
        base_rows = job_rows(cfg, BASE, task, thr52)
        if not base_rows:
            print(f"[us53] {name}: no base jobs; skipped")
            continue
        summ = {BASE: summarize(base_rows)}
        for v in variants:
            r = job_rows(FT.load_config(v), v, task, thr52)
            if r:
                summ[v] = summarize(r)
        sel = select(task, summ[BASE], {k: s for k, s in summ.items() if k != BASE}, t)
        for v, s in summ.items():
            vrows.append({"task": name, "variant": v, "n_jobs": s["n_jobs"], "val": s["val"], "val_vs_base": s["val"] - summ[BASE]["val"] if v != BASE else 0.0,
                          "worse_by": worse_by(task.kind, s["val"], summ[BASE]["val"]), "gap_best": s["gap_best"], "loss_ratio": s["loss_ratio"], "spike": s["spike"],
                          "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "fit_s": s["fit_s"], "target_met": target_met(task.kind, s, t),
                          "eligible": worse_by(task.kind, s["val"], summ[BASE]["val"]) <= sel["margin"], "selected": v == sel["selected"]})
        y_all, meta = FH.load_labels(task)
        base_p = tag_preds(cfg, BASE, task, y_all, meta)
        pv_p = {v: tag_preds(FT.load_config(v), v, task, y_all, meta) for v in summ if v != BASE}
        pv_p[BASE] = base_p
        key = "rmse" if task.kind == "reg" else "macro_f1"
        for r in vrows:                                                  # validation of the SEED-AVERAGED headline (same decision rule as the test headline); reported next to the job mean
            if r["task"] == name:
                p = pv_p.get(r["variant"])
                r["val_headline"] = float("nan") if p is None else float(p["val_primary"])
                r["val_headline_vs_base"] = float("nan") if p is None else float(p["val_primary"] - base_p["val_primary"])
        for v in summ:                                                   # test deltas for every variant (reported, not used to select)
            p = pv_p.get(v)
            if p is None or v == BASE:
                continue
            d = RU.pair(task, base_p, p, n_boot)
            vrows[[i for i, r in enumerate(vrows) if r["task"] == name and r["variant"] == v][0]].update(
                {"test": d["b"], "test_delta_vs_base": d["delta"], "test_lo": d["lo"], "test_hi": d["hi"], "test_verdict": RU.verdict(task.kind, d)})
        for r in vrows:
            if r["task"] == name and r["variant"] == BASE and base_p is not None:
                r["test"] = RF._score(task, base_p["y"], base_p["pred"])[key]
        sp = pv_p[sel["selected"]]
        b, s0 = summ[BASE], summ[sel["selected"]]
        row = {"task": name, "metric": key, "selected": sel["selected"], "reason": sel["reason"], "need_fix": sel["need_fix"], "margin": sel["margin"],
               "val_before": b["val"], "val_after": s0["val"], "val_headline_before": base_p["val_primary"], "val_headline_after": sp["val_primary"], "gap_before": b["gap_best"], "gap_after": s0["gap_best"],
               "loss_ratio_before": b["loss_ratio"], "loss_ratio_after": s0["loss_ratio"], "spike_before": b["spike"], "spike_after": s0["spike"],
               "epochs_before": b["epochs_run"], "epochs_after": s0["epochs_run"], "fit_s_before": b["fit_s"], "fit_s_after": s0["fit_s"],
               "test_before": RF._score(task, base_p["y"], base_p["pred"])[key], "test_after": RF._score(task, sp["y"], sp["pred"])[key]}
        if sel["selected"] != BASE:
            d = RU.pair(task, base_p, sp, n_boot)
            row.update({"d_vs_base": d["delta"], "d_vs_base_lo": d["lo"], "d_vs_base_hi": d["hi"], "verdict_vs_base": RU.verdict(task.kind, d)})
        else:
            row.update({"d_vs_base": 0.0, "d_vs_base_lo": 0.0, "d_vs_base_hi": 0.0, "verdict_vs_base": "same model"})
        ref = (refs.get(name) or {}).get("4.3")
        if ref is not None:
            for label, pp in (("before", base_p), ("after", sp)):
                d = RU.pair(task, ref, {"y": pp["y"], "pred": pp["pred"], "meta": pp["meta"]}, n_boot)
                row.update({f"d43_{label}": d["delta"], f"d43_{label}_lo": d["lo"], f"d43_{label}_hi": d["hi"], f"verdict43_{label}": RU.verdict(task.kind, d)})
        hrows.append(row)
        srows.append({"task": name, **{k: sel[k] for k in ("selected", "reason", "need_fix", "margin", "n_complete", "n_eligible")}})
        if name in ("word_accuracy", "phone_score"):                      # W8: regression to the mean, re-measured for the selected model
            cal = D.calibration_rows(task, {"y": sp["y"], "pred": sp["pred"]}, {"pred": base_p["pred"]})
            cal[0]["system"], cal[1]["system"] = "5.1 base", f"5.3 selected ({sel['selected']})"
            crows += cal
    return pd.DataFrame(vrows), pd.DataFrame(srows), pd.DataFrame(hrows), pd.DataFrame(crows)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_md(path: Path, t: dict, var: pd.DataFrame, sel: pd.DataFrame, head: pd.DataFrame, cal: pd.DataFrame, variants: list[str]) -> None:
    L = ["# US 5.3 - weakness fixes for the fine-tuned encoder (before / after)", "",
         "Variants (one knob group each, relative to the 5.1 recipe `base`): " + ", ".join(f"`{v}`" for v in variants) + ". Definitions in `configs/finetune.yaml` `variants`.", "",
         f"Targets and selection rule were fixed before any variant was run (`diagnose53`): relative train-validation gap at the best epoch <= {t['gap_best_max']}, last-epoch validation loss "
         f"<= {t['loss_ratio_max_reg']}x (regression) / {t['loss_ratio_max_clf']}x (classification) its minimum, no training-loss spike >= {t['spike_ratio']}x; a variant is eligible if its validation "
         f"metric is not worse than base by more than max({t['margin_floor']}, seed sd of base). Selection uses validation-side quantities only; test is reported afterwards.", "",
         "## 1. Selection per task", ""]
    L += D.md(sel, ["task", "selected", "need_fix", "n_complete", "n_eligible", "margin", "reason"])
    L += ["", "## 2. Before / after (selected variant vs base 5.1)", "",
          "Validation and training-side columns are job means (`val` = early-stopping metric at the best epoch, plain argmax); `val_headline` is the seed-averaged validation score with the same decision rule as the test headline (cw / cw+thr tuned on validation). Test is the seed-averaged headline (class-weight / threshold variant chosen on validation), "
          "`d_vs_base` a paired speaker bootstrap of the selected model against base on the same test rows.", ""]
    L += D.md(head, ["task", "selected", "val_before", "val_after", "val_headline_before", "val_headline_after", "gap_before", "gap_after", "loss_ratio_before", "loss_ratio_after", "spike_before", "spike_after", "test_before", "test_after",
                     "d_vs_base", "d_vs_base_lo", "d_vs_base_hi", "verdict_vs_base"])
    if "verdict43_after" in head.columns:
        L += ["", "Against the frozen US 4.3 system of the same encoder (paired, test):", ""]
        L += D.md(head, ["task", "d43_before", "verdict43_before", "d43_after", "d43_after_lo", "d43_after_hi", "verdict43_after"])
    L += ["", "Cost (mean seconds per job, epochs run):", ""]
    L += D.md(head, ["task", "selected", "fit_s_before", "fit_s_after", "epochs_before", "epochs_after"], nd=1)
    L += ["", "## 3. Every variant (validation side; test shown for reference only)", ""]
    cols = [c for c in ["task", "variant", "n_jobs", "val", "worse_by", "val_headline", "val_headline_vs_base", "gap_best", "loss_ratio", "spike", "best_epoch", "epochs_run", "fit_s", "target_met", "eligible", "selected",
                        "test", "test_delta_vs_base", "test_lo", "test_hi", "test_verdict"] if c in var.columns]
    L += D.md(var, cols)
    if len(cal):
        L += ["", "## 4. Regression to the mean (test), base vs selected", ""]
        L += D.md(cal, ["task", "system", "slope_pred_on_true", "pcc", "low_bin_share", "low_bin_mean_true", "low_bin_mean_pred"])
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = FT.load_config()
    variants = a.variants or list(cfg["variants"])
    OUT.mkdir(parents=True, exist_ok=True)
    var, sel, head, cal = build(variants, a.n_boot)
    var.to_csv(OUT / "us53_variants.csv", index=False)
    sel.to_csv(OUT / "us53_selection.csv", index=False)
    head.to_csv(OUT / "us53_headline.csv", index=False)
    cal.to_csv(OUT / "us53_calibration.csv", index=False)
    write_md(OUT / "us53_report.md", cfg["diagnose53"], var, sel, head, cal, variants)
    print((OUT / "us53_report.md").read_text())


if __name__ == "__main__":
    main()
