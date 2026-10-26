"""
US 6.1 - single-task vs multi-task comparison (and the multi-task recipe screen).

    python -m src.evaluation.report_us61 --tag mt_base                        # results/tables/multitask/us61_report_mt_base.md (+ .csv)
    python -m src.evaluation.report_us61 --recipe-screen --tags mt_base mt_lr # which multi-task recipe goes forward (validation only)

Comparators (configs/multitask.yaml `compare61`), always the SAME encoder, the SAME rows, splits and seeds:
    base        the US 5.1 single-task fine-tuned models (tag `base`)
    stabilized  the US 5.3 per-task selected variants (results/tables/finetune/us53_selection.csv; `base` where 5.3 kept it)
L2-ARCTIC single-task models have extra seeds under the tags `<tag>_l2x` (bash run_61_local.sh st-l2); both sides are restricted to the seeds they share.

For every head: test metric of both systems, paired speaker bootstrap of multi-task minus single-task (95 % CI, two-sided p) and a verdict;
validation difference against the margin max(margin_floor, job-to-job sd of the single-task validation metric). Choices (recipe, classification decision rule)
use validation only; test numbers are reported afterwards and never select.

Everything below the numpy line reads the same job files as src/evaluation/report_us51.py, because multi-task jobs are written in the single-task layout.
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
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models import multitask as MT
from src.models.baselines import build_tasks, load_config as load_baseline_config

OUT = FT.REPO / "results" / "tables" / "multitask"


# --------------------------------------------------------------------------- #
# pure helpers (numpy only, unit-tested)
# --------------------------------------------------------------------------- #
def error_of(kind: str, value: float) -> float:
    """RMSE for regression, 1 - macro-F1 for classification (``value`` as RF.primary returns it)."""
    return float(value) if kind == "reg" else 1.0 - float(value)


def val_flag(kind: str, st_val: float, mt_val: float, job_sd: float, floor: float) -> tuple[float, float, str]:
    """(MT-worse-by amount, margin, label). Positive amount = multi-task is worse on validation. label: worse | better | within margin."""
    worse_by = error_of(kind, mt_val) - error_of(kind, st_val)
    margin = max(float(floor), 0.0 if job_sd is None or not np.isfinite(job_sd) else float(job_sd))
    return worse_by, margin, ("worse" if worse_by > margin else "better" if worse_by < -margin else "within margin")


def job_val_sd(task, jobs: dict) -> float:
    """Mean over folds of the sd (across seeds) of the jobs' own validation score; NaN when no fold has two seeds."""
    sds = [np.std([float(d["val_score"]) for d in ds], ddof=1) for ds in jobs.values() if len(ds) >= 2]
    return float(np.mean(sds)) if sds else float("nan")


def common_seeds(avail: list[dict[int, set]]) -> list[int]:
    """avail: per side {fold: set(seeds)}. Seeds present in every fold of every side."""
    sets = [s for side in avail for s in side.values()]
    return sorted(set.intersection(*sets)) if sets else []


def recipe_choice(scores: dict[str, float], floor: float, default: str = "base") -> tuple[str, str]:
    """scores: tag-suffix -> mean (MT validation error / ST-base validation error), lower is better. Another recipe must beat ``default`` by more than ``floor``."""
    best = min(scores, key=scores.get)
    if default not in scores:
        return best, f"{best} (the default {default!r} was not run)"
    if best != default and scores[best] < scores[default] - floor:
        return best, f"{best} beats {default} by {scores[default] - scores[best]:.3f} (> {floor})"
    return default, f"no candidate beats {default} by more than {floor}"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _seed_files(cfg: dict, tag: str, task, fold: int, seeds) -> dict[int, Path]:
    return {s: FT.job_file(cfg, tag, task.name, fold, s) for s in seeds if FT.job_file(cfg, tag, task.name, fold, s).exists()}


def available(sources: list[tuple[dict, str]], task, seeds) -> dict[int, set]:
    """{fold: seeds found in any source}."""
    out = {}
    for fold in task.folds:
        found = set()
        for cfg, tag in sources:
            found |= set(_seed_files(cfg, tag, task, fold, seeds))
        out[fold] = found
    return out


def load_sources(sources: list[tuple[dict, str]], task, seeds) -> dict | None:
    """{fold: [npz, ...]} over the given seeds, pooling the sources (a seed lives in one source); None when a fold has none."""
    out = {}
    for fold in task.folds:
        ds = {}
        for cfg, tag in sources:
            for s, f in _seed_files(cfg, tag, task, fold, seeds).items():
                ds.setdefault(s, np.load(f, allow_pickle=True))
        if not ds:
            return None
        out[fold] = [ds[s] for s in sorted(ds)]
    return out


def build_system(task, jobs: dict, y_all, meta, n_boot: int, boot_seed: int = 0) -> dict:
    """Seed-averaged, fold-pooled system with the headline decision rule chosen on validation (as report_us51)."""
    P = R51.pooled_ft(task, jobs, y_all, meta)
    dv, dt = P["val"], P["test"]
    variant, pv, pt, train_prim, val_prim = R51.headline_variant(task, dv, dt)
    key = "rmse" if task.kind == "reg" else "macro_f1"
    sc = RF._score(task, dt["y"], pt)
    ci = M.speaker_bootstrap(dt["meta"]["speaker"].to_numpy(), RF._enc(task, dt["y"]), RF._enc(task, pt), RF._fast(task), n_boot=n_boot, seed=boot_seed)
    return {"variant": variant, "val": val_prim, "train": train_prim, "test": sc[key], "lo": ci[key][0], "hi": ci[key][1], "metric": key,
            "sys": {"y": dt["y"], "pred": pt, "meta": dt["meta"]}, "job_sd": job_val_sd(task, jobs), "n_seeds": min(len(v) for v in jobs.values()),
            "minutes": float(np.mean([np.mean([float(d["fit_s"]) for d in ds]) for ds in jobs.values()]) / 60.0)}


def compare_head(task, st: dict, mt: dict, n_boot: int, floor: float) -> dict:
    d = RU.pair(task, st["sys"], mt["sys"], n_boot)                  # delta = metric(mt) - metric(st)
    worse_by, margin, label = val_flag(task.kind, st["val"], mt["val"], st["job_sd"], floor)
    return {"st": st["test"], "mt": mt["test"], "delta": d["delta"], "lo": d["lo"], "hi": d["hi"], "p": d["p"], "verdict": RU.verdict(task.kind, d),
            "st_val": st["val"], "mt_val": mt["val"], "val_worse_by": worse_by, "val_margin": margin, "val_flag": label, "st_variant": st["variant"],
            "st_job_sd": st["job_sd"]}


def stabilized_tags(path: Path | None = None) -> dict[str, str]:
    """{task: tag of the US 5.3 selected variant}; {} when the selection file is not on this machine."""
    path = path or FT.REPO / "results" / "tables" / "finetune" / "us53_selection.csv"
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["task"], df["selected"]))


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(tag: str, variant: str | None = None, groups=None, n_boot: int | None = None, stab_path: Path | None = None):
    cfg = MT.load_config(variant)
    st_cfg = FT.load_config()
    c61 = cfg["compare61"]
    suffix = c61.get("extra_seed_tag_suffix", "_l2x")
    n_boot = n_boot or c61["n_boot"]
    floor = float(c61["margin_floor"])
    bcfg = load_baseline_config()
    registry = build_tasks(bcfg)
    stab = stabilized_tags(stab_path)
    seeds_all = sorted(set(cfg["seeds"]) | set(cfg["l2_seeds"]))
    rows, jobs_rows, eff_rows = [], [], []
    for group in groups or list(cfg["groups"]):
        for name in cfg["groups"][group]["tasks"]:
            task = registry[name]
            mt_src = [(cfg, tag)]
            jobs_mt_all = load_sources(mt_src, task, seeds_all)
            if jobs_mt_all is None:
                print(f"[us61] {name}: no finished multi-task jobs under tag {tag!r}; skipped")
                continue
            y_all, meta = FH.load_labels(task)
            r = {"tag": tag, "group": group, "task": name, "level": task.level, "dataset": task.dataset, "metric": "rmse" if task.kind == "reg" else "macro_f1"}
            comps = {"base": [(st_cfg, "base"), (st_cfg, "base" + suffix)]}
            if stab:
                v = stab.get(name, "base")
                comps["stabilized"] = [(st_cfg, v), (st_cfg, v + suffix)] if v != "base" else comps["base"]
                r["stabilized_variant"] = v
            first = True
            for comp, src in comps.items():
                seeds = common_seeds([available(mt_src, task, seeds_all), available(src, task, seeds_all)])
                if not seeds:
                    print(f"[us61] {name}: no seeds in common with the {comp} single-task jobs; skipped")
                    continue
                jobs_st, jobs_mt = load_sources(src, task, seeds), load_sources(mt_src, task, seeds)
                st = build_system(task, jobs_st, y_all, meta, n_boot, bcfg["bootstrap"]["seed"])
                mt = build_system(task, jobs_mt, y_all, meta, n_boot, bcfg["bootstrap"]["seed"])
                c = compare_head(task, st, mt, n_boot, floor)
                r.update({f"st_{comp}": c["st"], f"d_{comp}": c["delta"], f"d_{comp}_lo": c["lo"], f"d_{comp}_hi": c["hi"], f"d_{comp}_p": c["p"],
                          f"verdict_{comp}": c["verdict"], f"st_val_{comp}": c["st_val"], f"val_worse_by_{comp}": c["val_worse_by"],
                          f"val_margin_{comp}": c["val_margin"], f"val_flag_{comp}": c["val_flag"], f"st_rule_{comp}": c["st_variant"],
                          f"st_job_sd_{comp}": c["st_job_sd"], f"seeds_{comp}": len(seeds)})
                if first:
                    r.update({"mt": mt["test"], "mt_lo": mt["lo"], "mt_hi": mt["hi"], "mt_val": mt["val"], "mt_train": mt["train"], "mt_rule": mt["variant"],
                              "mt_job_sd": mt["job_sd"]})
                    first = False
            rows.append(r)
            for fold, ds in jobs_mt_all.items():
                for d in ds:
                    jobs_rows.append({"task": name, "group": group, "fold": fold, "seed": int(d["seed"]), "best_epoch": int(d["best_epoch"]),
                                      "head_best_epoch": int(d["mt_head_best_epoch"]), "epochs_run": int(d["epochs_run"]),
                                      "val_primary": float(d["val_score"]) * (1 if task.kind == "reg" else -1),
                                      "train_primary": float(d["train_score"]) * (1 if task.kind == "reg" else -1), "group_fit_s": float(d["mt_group_fit_s"])})
        eff = efficiency(cfg, st_cfg, registry, group, tag, seeds_all)
        if eff:
            eff_rows.append(eff)
    return pd.DataFrame(rows), pd.DataFrame(jobs_rows), pd.DataFrame(eff_rows)


def efficiency(cfg: dict, st_cfg: dict, registry: dict, group: str, tag: str, seeds) -> dict | None:
    """Training cost of one (fold, seed): one shared-encoder job vs the sum of the single-task jobs of the same heads (`base`)."""
    names = cfg["groups"][group]["tasks"]
    mt_min, st_min = [], []
    for fold in registry[names[0]].folds:
        for s in seeds:
            f_mt = [FT.job_file(cfg, tag, n, fold, s) for n in names]
            f_st = [FT.job_file(st_cfg, "base", n, fold, s) for n in names]
            if all(f.exists() for f in f_mt) and all(f.exists() for f in f_st):
                mt_min.append(float(np.load(f_mt[0], allow_pickle=True)["mt_group_fit_s"]) / 60.0)
                st_min.append(sum(float(np.load(f, allow_pickle=True)["fit_s"]) for f in f_st) / 60.0)
    if not mt_min:
        return None
    return {"group": group, "heads": len(names), "encoders_mt": 1, "encoders_st": len(names), "jobs": len(mt_min), "mt_minutes": float(np.mean(mt_min)),
            "st_minutes": float(np.mean(st_min)), "ratio": float(np.mean(mt_min) / max(np.mean(st_min), 1e-9))}


def recipe_screen(tags: list[str], groups=None, n_boot: int = 50) -> pd.DataFrame:
    """For each tag and group: mean over heads of MT validation error / single-task `base` validation error (same seeds). Validation only."""
    cfg = MT.load_config()
    st_cfg = FT.load_config()
    registry = build_tasks(load_baseline_config())
    seeds_all = sorted(set(cfg["seeds"]) | set(cfg["l2_seeds"]))
    suffix = cfg["compare61"].get("extra_seed_tag_suffix", "_l2x")
    rows = []
    for group in groups or list(cfg["groups"]):
        for tag in tags:
            ratios = {}
            for name in cfg["groups"][group]["tasks"]:
                task = registry[name]
                src_st = [(st_cfg, "base"), (st_cfg, "base" + suffix)]
                seeds = common_seeds([available([(cfg, tag)], task, seeds_all), available(src_st, task, seeds_all)])
                if not seeds:
                    continue
                y_all, meta = FH.load_labels(task)
                a = build_system(task, load_sources(src_st, task, seeds), y_all, meta, n_boot)
                b = build_system(task, load_sources([(cfg, tag)], task, seeds), y_all, meta, n_boot)
                ratios[name] = error_of(task.kind, b["val"]) / max(error_of(task.kind, a["val"]), 1e-9)
            if ratios:
                rows.append({"group": group, "tag": tag, "n_heads": len(ratios), "mean_val_error_ratio": float(np.mean(list(ratios.values()))),
                             **{f"ratio_{k}": v for k, v in ratios.items()}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _f(v, nd=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _cell(r, comp: str) -> str:
    if f"d_{comp}" not in r or pd.isna(r.get(f"d_{comp}")):
        return ""
    return f"{r[f'd_{comp}']:+.3f} [{r[f'd_{comp}_lo']:+.3f}, {r[f'd_{comp}_hi']:+.3f}] p={r[f'd_{comp}_p']:.3f}"


def write_md(head: pd.DataFrame, jobs: pd.DataFrame, eff: pd.DataFrame, path: Path, tag: str, floor: float) -> None:
    L = [f"# US 6.1 - single-task vs multi-task (tag `{tag}`, test set; every choice made on validation)", "",
         "Metric: RMSE (regression, lower is better) or macro-F1 (classification, higher is better). Both systems are fine-tuned WavLM-base-plus, seed-averaged, on the "
         "identical rows and the identical seeds; the only difference is that the multi-task encoder is shared by the heads of its group. Delta = multi-task minus "
         "single-task on test, paired speaker bootstrap (95 % CI, two-sided p); for RMSE a negative delta favours multi-task, for macro-F1 a positive one. "
         "The verdict says `better` / `worse` only when the interval excludes 0.", "",
         "| group | task | metric | single-task 5.1 | single-task 5.3 | **multi-task** [95% CI] | delta vs 5.1 [95% CI] | verdict vs 5.1 | delta vs 5.3 [95% CI] | verdict vs 5.3 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in head.to_dict("records"):
        L.append(f"| {r['group']} | {r['task']} | {r['metric']} | {_f(r.get('st_base'))} | {_f(r.get('st_stabilized'))} ({r.get('stabilized_variant', '')}) "
                 f"| **{_f(r.get('mt'))}** [{_f(r.get('mt_lo'))}, {_f(r.get('mt_hi'))}] | {_cell(r, 'base')} | {r.get('verdict_base', '')} | {_cell(r, 'stabilized')} | {r.get('verdict_stabilized', '')} |")
    for comp, label in (("base", "5.1"), ("stabilized", "5.3")):
        col = f"verdict_{comp}"
        if col in head:
            c = head[col].value_counts().to_dict()
            L.append(f"\nVs single-task {label}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + f" (of {int(head[col].notna().sum())} heads).")
    L += ["", f"## Validation side (margin = max({floor}, job-to-job sd of the single-task validation metric))", "",
          "| task | metric | single-task val (5.1) | multi-task val | multi-task worse by | margin | flag | seeds used |", "|---|---|---|---|---|---|---|---|"]
    for r in head.to_dict("records"):
        L.append(f"| {r['task']} | {r['metric']} | {_f(r.get('st_val_base'))} | {_f(r.get('mt_val'))} | {_f(r.get('val_worse_by_base'))} | {_f(r.get('val_margin_base'))} "
                 f"| {r.get('val_flag_base', '')} | {r.get('seeds_base', '')} |")
    L += ["", "Positive 'worse by' = multi-task is worse on validation (RMSE units for regression, 1 - macro-F1 for classification). Validation has 19 SO762 speakers: "
          "differences below the margin are not credited.", ""]
    if not eff.empty:
        L += ["## Training cost (one fold and seed)", "", "| group | heads | encoders (multi-task / single-task) | multi-task minutes | sum of single-task minutes | ratio |", "|---|---|---|---|---|---|"]
        for r in eff.to_dict("records"):
            L.append(f"| {r['group']} | {r['heads']} | {r['encoders_mt']} / {r['encoders_st']} | {r['mt_minutes']:.1f} | {r['st_minutes']:.1f} | {r['ratio']:.2f} |")
        L.append("\nMinutes are wall clock on a possibly shared GPU: indicative only.\n")
    if not jobs.empty:
        g = jobs.groupby(["group", "task"], sort=False).agg(jobs=("seed", "size"), best_epoch=("best_epoch", "mean"), head_best=("head_best_epoch", "mean"),
                                                            epochs=("epochs_run", "mean"), val_sd=("val_primary", "std")).reset_index()
        L += ["## Jobs", "", "One checkpoint per multi-task job (`best epoch`). `head's own best` = the epoch at which that head alone was best on validation: "
              "a large gap means one stopping epoch is a compromise (looked at in US 6.2).", "",
              "| group | task | jobs | shared best epoch | head's own best epoch | epochs run | sd of validation score across jobs |", "|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.group} | {r.task} | {r.jobs} | {r.best_epoch:.1f} | {r.head_best:.1f} | {r.epochs:.1f} | {_f(r.val_sd)} |")
    L += ["", "## Caveats", "",
          "- `phone_mispron_t1.0` is `phone_score < 1.0` (the same labels thresholded) and `sent_total` is highly correlated with the other sentence scores: heads on "
          "near-duplicate targets can make joint training look better than it is. US 6.2 separates this from real transfer.",
          "- L2-ARCTIC: the single-task `base` jobs of Sprint 5 used one seed per fold; seeds 1 and 2 were added for this sprint under the tag `<tag>_l2x`. "
          "Both sides are always restricted to the seeds they share.",
          "- One multi-task checkpoint per job is chosen on the mean relative validation error over the heads; single-task jobs each choose their own epoch.", ""]
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="mt_base")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--groups", nargs="*", choices=["so762", "l2"])
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--recipe-screen", action="store_true")
    ap.add_argument("--tags", nargs="*", default=["mt_base", "mt_lr"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = MT.load_config(a.variant)
    floor = float(cfg["compare61"]["margin_floor"])
    warnings.filterwarnings("ignore", category=UserWarning)
    if a.recipe_screen:
        df = recipe_screen(a.tags, a.groups)
        df.to_csv(OUT / "us61_recipe_screen.csv", index=False)
        L = ["# US 6.1 - multi-task recipe screen (validation only)", "",
             f"Rule (fixed before running): lowest mean of multi-task validation error / single-task `base` validation error over the heads; a recipe other than `base` must win by more than {floor}.", "",
             "| group | tag | heads | mean val-error ratio |", "|---|---|---|---|"]
        for r in df.to_dict("records"):
            L.append(f"| {r['group']} | {r['tag']} | {r['n_heads']} | {r['mean_val_error_ratio']:.4f} |")
        for g, sub in df.groupby("group"):
            scores = {t.replace("mt_", "", 1): v for t, v in zip(sub["tag"], sub["mean_val_error_ratio"])}
            pick, why = recipe_choice(scores, floor)
            L.append(f"\n**{g}: goes forward with `mt_{pick}`** ({why}).")
        (OUT / "us61_recipe_screen.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    head, jobs, eff = build(a.tag, a.variant, a.groups, a.n_boot)
    head.to_csv(OUT / f"us61_headline_{a.tag}.csv", index=False)
    jobs.to_csv(OUT / f"us61_jobs_{a.tag}.csv", index=False)
    eff.to_csv(OUT / f"us61_efficiency_{a.tag}.csv", index=False)
    write_md(head, jobs, eff, OUT / f"us61_report_{a.tag}.md", a.tag, floor)
    print((OUT / f"us61_report_{a.tag}.md").read_text())


if __name__ == "__main__":
    main()
