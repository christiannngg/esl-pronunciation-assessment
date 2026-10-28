"""
US 6.3 - fixing negative transfer / task imbalance in the SO762 multi-task model: candidates, validation-only adoption, test reporting.

    python -m src.evaluation.report_us63                # all candidates that have finished jobs -> results/tables/multitask/us63_report.md (+ us63_*.csv)
    python -m src.evaluation.report_us63 --n-boot 50    # quick look

Candidates (configs/multitask.yaml `compare63`, fixed before any US 6.3 job was run). Each is a set of multi-task training runs that together give every
SO762 head a prediction; every head is compared with the SAME-recipe single-task model (US 5.1 `base`) and with the full 7-head model of US 6.1:
    full          one shared encoder, 7 heads, fixed equal loss weights (US 6.1, tag mt_base)
    regroup       two encoders: sentence block (4 heads, tag mt63_sent4) + word / phone block (3 heads, tag mt62_word_phone from US 6.2)
    uncertainty   full model, learned per-head log-variance weights (tag mt63_uncertainty)
    gradnorm_a05 / gradnorm_a15   full model, GradNorm with alpha 0.5 / 1.5 (tags mt63_gradnorm_a05 / mt63_gradnorm_a15)

Adoption (validation only; see `compare63.adoption`): a candidate PASSES if (1) the target head phone_mispron_t1.0 is not worse than single-task on validation by more
than its margin, (2) no other head is, (3) the target head's validation F1 is at least that of the full model. Among passing candidates the lowest mean relative
validation error wins; candidates within `tie_margin` of it are tied and the fewest encoders wins. Test numbers (paired speaker bootstrap) are reported for every
candidate and never select.

Also reported: the loss weights each scheme actually applied (mean over the epochs after the head-only epoch, and at the last epoch), the gradient shares on the
last shared layer, the cost of ONE shared checkpoint (regret vs each head's own best epoch; the trigger for per-head checkpointing), and training cost / encoders.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.evaluation import report_us61 as R61
from src.evaluation import report_us62 as R62
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models import multitask as MT
from src.models.baselines import build_tasks
from src.models.baselines import load_config as load_baseline_config

OUT = FT.REPO / "results" / "tables" / "multitask"


# --------------------------------------------------------------------------- #
# pure helpers (numpy / pandas only, unit-tested)
# --------------------------------------------------------------------------- #
def candidate_criteria(heads: pd.DataFrame, target: str) -> dict:
    """heads: one row per head of ONE candidate with columns task, kind, st_val, cand_val, full_val, val_flag (R61.val_flag label vs single-task).
    st_val / cand_val / full_val are the validation primary values as R61.compare_head returns them (RMSE, or macro-F1).
    -> the three declared criteria, the mean relative validation error and the list of failing heads."""
    t = heads[heads["task"] == target]
    if t.empty:
        return {"c1_target_ok": False, "c2_others_ok": False, "c3_not_below_full": False, "mean_rel_err": float("nan"), "heads_flagged": [], "passes": False,
                "reason": f"target head {target!r} missing"}
    t = t.iloc[0]
    c1 = bool(t["val_flag"] != "worse")
    flagged = [r["task"] for _, r in heads.iterrows() if r["task"] != target and r["val_flag"] == "worse"]
    c2 = not flagged
    c3 = bool(R61.error_of(t["kind"], t["cand_val"]) <= R61.error_of(t["kind"], t["full_val"]) + 1e-12)
    rel = [R61.error_of(r["kind"], r["cand_val"]) / max(R61.error_of(r["kind"], r["st_val"]), 1e-12) for _, r in heads.iterrows()]
    fails = [name for name, ok in (("target head worse than single-task on validation", c1), ("other head(s) worse than single-task on validation: " + ", ".join(flagged), c2),
                                   ("target head below the full model on validation", c3)) if not ok]
    return {"c1_target_ok": c1, "c2_others_ok": c2, "c3_not_below_full": c3, "mean_rel_err": float(np.mean(rel)), "heads_flagged": flagged,
            "passes": bool(c1 and c2 and c3), "reason": "; ".join(fails) if fails else "meets all three"}


def choose_candidate(cands: pd.DataFrame, tie_margin: float) -> tuple[str | None, str]:
    """cands: columns name, passes, mean_rel_err, encoders. -> (adopted name or None, explanation). Lowest mean_rel_err among the passing; those within tie_margin
    of it are tied and the fewest encoders wins (then the lowest mean_rel_err)."""
    ok = cands[cands["passes"].astype(bool)]
    if ok.empty:
        return None, ("no candidate meets all three criteria on validation: report the full model and the per-level split side by side; sharing the encoder across "
                      "levels does not help in this data (declared rule)")
    best = float(ok["mean_rel_err"].min())
    tied = ok[ok["mean_rel_err"] <= best + float(tie_margin) + 1e-12]
    pick = tied.sort_values(["encoders", "mean_rel_err", "name"]).iloc[0]
    why = f"lowest mean relative validation error {best:.4f}" if len(tied) == 1 else \
        f"{len(tied)} candidates within {tie_margin} of the best ({best:.4f}): {', '.join(tied['name'])}; the fewest encoders ({int(pick['encoders'])}) wins"
    return str(pick["name"]), f"{pick['name']}: {why}"


def mean_weights(hist_w: np.ndarray, skip_epochs: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(mean over the epochs after the head-only epoch, last epoch) of the applied loss weights; hist_w is (epochs, K)."""
    w = np.asarray(hist_w, float)
    body = w[skip_epochs:] if len(w) > skip_epochs else w
    return body.mean(0), w[-1]


# --------------------------------------------------------------------------- #
# loading / building
# --------------------------------------------------------------------------- #
def part_sources(c63: dict, cand: dict) -> dict[str, tuple[dict, str]]:
    """{head: (config of the part that trains it, tag)} for one candidate."""
    out = {}
    for part in cand["parts"]:
        pcfg = MT.load_config(part["variant"])
        for h in pcfg["groups"][c63["group"]]["tasks"]:
            out[h] = (pcfg, part["tag"])
    return out


def head_table(c63: dict, n_boot: int) -> pd.DataFrame:
    """One row per (candidate, head): test + validation vs single-task and vs the full model (same seeds, paired speaker bootstrap)."""
    bcfg = load_baseline_config()
    registry = build_tasks(bcfg)
    st_cfg = FT.load_config()
    floor = float(c63["margin_floor"])
    ref = c63["reference"]
    full_cfg = MT.load_config(ref["variant"])
    seeds_all = sorted(set(full_cfg["seeds"]))
    target = c63["target_head"]
    rows = []
    for cname, cand in c63["candidates"].items():
        srcs_h = part_sources(c63, cand)
        for head, (pcfg, ptag) in srcs_h.items():
            task = registry[head]
            srcs = {"cand": [(pcfg, ptag)], "full": [(full_cfg, ref["tag"])], "st": [(st_cfg, c63["comparator"])]}
            seeds = R61.common_seeds([R61.available(s, task, seeds_all) for s in srcs.values()])
            if not seeds:
                print(f"[us63] {cname}/{head}: no finished jobs in common with the full model and the single-task model; skipped")
                continue
            y_all, meta = FH.load_labels(task)
            sysm = {k: R61.build_system(task, R61.load_sources(s, task, seeds), y_all, meta, n_boot, bcfg["bootstrap"]["seed"]) for k, s in srcs.items()}
            vs_st = R61.compare_head(task, sysm["st"], sysm["cand"], n_boot, floor)
            vs_full = R61.compare_head(task, sysm["full"], sysm["cand"], n_boot, floor) if cname != "full" else None
            rows.append({"candidate": cname, "encoders": cand["encoders"], "task": head, "kind": task.kind, "is_target": head == target, "metric": sysm["cand"]["metric"],
                         "seeds": len(seeds), "st_test": sysm["st"]["test"], "full_test": sysm["full"]["test"], "cand_test": sysm["cand"]["test"],
                         "d_vs_st": vs_st["delta"], "d_vs_st_lo": vs_st["lo"], "d_vs_st_hi": vs_st["hi"], "d_vs_st_p": vs_st["p"], "verdict_vs_st": vs_st["verdict"],
                         "d_vs_full": np.nan if vs_full is None else vs_full["delta"], "d_vs_full_lo": np.nan if vs_full is None else vs_full["lo"],
                         "d_vs_full_hi": np.nan if vs_full is None else vs_full["hi"], "verdict_vs_full": "-" if vs_full is None else vs_full["verdict"],
                         "reading": "-" if vs_full is None else R62.ablation_reading(vs_st["verdict"], vs_full["verdict"]),
                         "st_val": vs_st["st_val"], "cand_val": vs_st["mt_val"], "full_val": sysm["full"]["val"], "val_worse_by": vs_st["val_worse_by"],
                         "val_margin": vs_st["val_margin"], "val_flag": vs_st["val_flag"]})
    return pd.DataFrame(rows)


def candidate_table(heads: pd.DataFrame, c63: dict) -> pd.DataFrame:
    rows = []
    n_heads = len(MT.load_config()["groups"][c63["group"]]["tasks"])
    for cname, sub in heads.groupby("candidate", sort=False):
        crit = candidate_criteria(sub, c63["target_head"])
        t = sub[sub["task"] == c63["target_head"]]
        rows.append({"name": cname, "encoders": int(sub["encoders"].iloc[0]), "heads_found": len(sub), "heads_expected": n_heads, **{k: v for k, v in crit.items() if k != "heads_flagged"},
                     "heads_flagged": ", ".join(crit["heads_flagged"]) or "-",
                     "target_test": float(t["cand_test"].iloc[0]) if len(t) else np.nan, "target_vs_st": float(t["d_vs_st"].iloc[0]) if len(t) else np.nan,
                     "target_vs_st_verdict": t["verdict_vs_st"].iloc[0] if len(t) else "-", "target_vs_full_verdict": t["verdict_vs_full"].iloc[0] if len(t) else "-",
                     "n_vs_st_worse": int((sub["verdict_vs_st"] == "worse").sum()), "n_vs_st_better": int((sub["verdict_vs_st"] == "better").sum())})
    df = pd.DataFrame(rows)
    df["complete"] = df["heads_found"] == df["heads_expected"]
    df.loc[~df["complete"], "passes"] = False                      # a candidate with a head missing cannot be adopted
    return df


def weights_table(c63: dict, registry: dict) -> pd.DataFrame:
    """Per (candidate part, head): the loss weight the scheme applied (mean after the head-only epoch / last epoch) and the gradient share on the last shared layer."""
    rows = []
    for cname, cand in c63["candidates"].items():
        for part in cand["parts"]:
            pcfg = MT.load_config(part["variant"])
            names = pcfg["groups"][c63["group"]]["tasks"]
            first = registry[names[0]]
            W, L, S, E = [], [], [], []
            for fold in first.folds:
                for seed in MT.seeds_for_group(pcfg, registry, c63["group"]):
                    gf = MT.group_file(pcfg, part["tag"], c63["group"], fold, seed)
                    if not gf.exists():
                        continue
                    g = np.load(gf, allow_pickle=True)
                    w_mean, w_last = mean_weights(g["hist_w"])
                    W.append(w_mean); L.append(w_last)
                    d = R62.job_diag(g)
                    S.append(d["shares"] if d is not None else np.full(len(names), np.nan))
                    E.append(d["eff_shares"] if d is not None else np.full(len(names), np.nan))
            if not W:
                continue
            njobs = len(W)
            W, L, S, E = np.mean(W, 0), np.mean(L, 0), np.nanmean(np.array(S), 0), np.nanmean(np.array(E), 0)
            for k, n in enumerate(names):
                rows.append({"candidate": cname, "tag": part["tag"], "task": n, "jobs": njobs, "weight_mean": float(W[k]),
                             "weight_last": float(L[k]), "weight_x_balanced": float(W[k] * len(names)), "grad_share": float(S[k]), "grad_share_x_balanced": float(S[k] * len(names)),
                             "eff_share": float(E[k]), "eff_share_x_balanced": float(E[k] * len(names))})
    return pd.DataFrame(rows)


def stopping_regret_table(c63: dict, registry: dict, threshold: float) -> pd.DataFrame:
    """Cost of ONE shared checkpoint per head and candidate part (validation, optimistic bound); heads at or above ``threshold`` trigger per-head checkpointing (d)."""
    rows = []
    for cname, cand in c63["candidates"].items():
        for part in cand["parts"]:
            data = R62.collect_group(c63["group"], part["tag"], part["variant"], registry)
            if data is None:
                continue
            for k, n in enumerate(data["names"]):
                rs = [R62.stopping_regret(j["kinds"][k], j["heads"][n]["hist_val_primary"], j["shared_epoch"]) for j in data["jobs"]]
                rel = float(np.mean([r["regret_rel"] for r in rs]))
                rows.append({"candidate": cname, "tag": part["tag"], "task": n, "jobs": len(rs), "shared_epoch": float(np.mean([j["shared_epoch"] for j in data["jobs"]])),
                             "own_best_epoch": float(np.mean([r["own_epoch"] for r in rs])), "regret_rel": rel, "above_threshold": bool(rel >= threshold)})
    return pd.DataFrame(rows)


def cost_table(c63: dict, registry: dict) -> pd.DataFrame:
    rows = []
    for cname, cand in c63["candidates"].items():
        mins, found = 0.0, True
        for part in cand["parts"]:
            pcfg = MT.load_config(part["variant"])
            names = pcfg["groups"][c63["group"]]["tasks"]
            fits = []
            for fold in registry[names[0]].folds:
                for seed in MT.seeds_for_group(pcfg, registry, c63["group"]):
                    f = MT.job_file(pcfg, part["tag"], names[0], fold, seed)
                    if f.exists():
                        fits.append(float(np.load(f, allow_pickle=True)["mt_group_fit_s"]) / 60.0)
            if not fits:
                found = False
                break
            mins += float(np.mean(fits))
        if found:
            rows.append({"candidate": cname, "encoders": cand["encoders"], "minutes_per_seed": mins})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _t(df: pd.DataFrame, cols: list[tuple[str, str]], nd: int = 3) -> list[str]:
    return R62.md_table(df, cols, nd) if len(df) else ["(no rows)"]


def write_md(path, c63, heads, cands, adopted, why, weights, regret, cost) -> None:
    tgt = c63["target_head"]
    L = ["# US 6.3 - fixing negative transfer and task imbalance in the SO762 multi-task model", "",
         "Recipe `base`, 3 seeds, SO762 (19 validation speakers). Every head is compared with the single-task model of the SAME recipe (US 5.1 `base`) and with the full 7-head "
         "multi-task model of US 6.1. Delta = candidate minus comparator on test (RMSE: negative is better; macro-F1: positive is better), paired speaker bootstrap, 95 % CI; "
         "`worse` / `better` only when the interval excludes 0. **Adoption uses validation only; test numbers never select.**", "",
         "## 1. Adoption on validation (rule fixed before any US 6.3 job was run)", "",
         f"Declared: {' '.join(str(c63['adoption']).split())}", ""]
    L += _t(cands, [("name", "candidate"), ("encoders", "encoders"), ("complete", "all heads"), ("c1_target_ok", f"(1) {tgt} not worse (val)"),
                    ("c2_others_ok", "(2) no other head worse (val)"), ("c3_not_below_full", f"(3) {tgt} >= full (val)"), ("mean_rel_err", "mean rel. val. error"),
                    ("passes", "PASSES"), ("reason", "why not")])
    L += ["", f"**Adopted: {adopted if adopted else 'none'}.** {why}", "",
          "## 2. Test: the target head and everyone else", ""]
    L += _t(cands, [("name", "candidate"), ("target_test", f"{tgt} test F1"), ("target_vs_st", "delta vs single-task"), ("target_vs_st_verdict", "verdict vs single-task"),
                    ("target_vs_full_verdict", "verdict vs full MT"), ("n_vs_st_better", "heads better than single-task"), ("n_vs_st_worse", "heads worse than single-task")])
    L += ["", "### Per head (test) and validation flag", ""]
    L += _t(heads, [("candidate", "candidate"), ("task", "head"), ("st_test", "single-task"), ("full_test", "full MT"), ("cand_test", "candidate"),
                    ("d_vs_st", "delta vs ST"), ("d_vs_st_lo", "lo"), ("d_vs_st_hi", "hi"), ("verdict_vs_st", "verdict vs ST"), ("d_vs_full", "delta vs full"),
                    ("verdict_vs_full", "verdict vs full"), ("reading", "reading"), ("val_worse_by", "val worse by"), ("val_margin", "margin"), ("val_flag", "val flag"), ("seeds", "seeds")])
    L += ["", "`reading` (only meaningful for the target head): recovers = no longer worse than single-task AND better than the full model; partly = only the second; hurts / no effect otherwise.", ""]
    if len(weights):
        L += ["## 3. What the schemes did to the loss weights and the gradient balance", "",
              "Weight = applied loss weight (sum 1; `fixed` = 1/K); unweighted share = share of the summed gradient norms of the UNWEIGHTED head losses on the last shared transformer layer (it does not change when a scheme changes the weights); effective share = the same with each norm multiplied by the applied weight, i.e. the balance the encoder actually receives. 'x balanced' = value x K.", ""]
        L += _t(weights, [("candidate", "candidate"), ("task", "head"), ("weight_x_balanced", "weight (mean) x balanced"), ("weight_last", "weight (last epoch)"),
                          ("grad_share_x_balanced", "unweighted gradient share x balanced"),
                          ("eff_share_x_balanced", "effective share (weight x gradient norm) x balanced")])
        L += [""]
    if len(regret):
        L += ["## 4. One shared checkpoint (trigger for per-head checkpointing, plan item d)", "",
              f"Validation regret of the shared epoch vs the head's own best epoch (optimistic bound). Threshold 2 % (`compare62.stopping_regret_rel`): heads at or above it keep the regret that (d) would address.", ""]
        L += _t(regret, [("candidate", "candidate"), ("task", "head"), ("shared_epoch", "shared epoch"), ("own_best_epoch", "own best epoch"), ("regret_rel", "regret (rel.)"),
                         ("above_threshold", ">= 2 %")])
        L += [""]
    if len(cost):
        L += ["## 5. Cost (indicative: wall clock, possibly shared GPU)", ""] + _t(cost, [("candidate", "candidate"), ("encoders", "encoders"), ("minutes_per_seed", "minutes per seed (sum over parts)")], 1) + [""]
    L += ["## Not addressed in US 6.3", "",
          "- **W8 (low-score compression of word_accuracy / phone_score, Sprint 5): still open.** No loss that re-weights low scores was tried; slopes are unchanged from US 5.2 (0.168 / 0.275).",
          "- **M7 (l2_phone_error -0.005):** negligible, no fix planned; the L2 group was not rerun.", "",
          "## Caveats", "",
          "- The idea of regrouping comes from the US 6.2 ablations, which were read on test. Adoption here is on validation, and regrouping gets no credit for having been seen first.",
          "- Five candidates are screened on 19 validation speakers; the adopted one is partly the luckiest on validation (winner's curse). All candidates are reported.",
          "- GradNorm is the first-order form (targets held constant) and uses the last shared transformer layer as the gradient probe; weights updated every step after the head-only epoch.",
          "- `uncertainty` starts from weights c_k (1/2 for a regression head, 1 for a classification head, then normalised): the target head begins with twice the weight of "
          "each regression head, i.e. part of that candidate is a fixed x2 re-weighting of the target head (the US 6.2 ablation `w_mispron3`, x3, did not help).",
          "- Seeds are matched across a candidate, the full model and the single-task model for every head (column `seeds`).", ""]
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = MT.load_config()
    c63 = cfg["compare63"]
    registry = build_tasks(load_baseline_config())
    n_boot = a.n_boot or int(c63["n_boot"])
    OUT.mkdir(parents=True, exist_ok=True)
    heads = head_table(c63, n_boot)
    if heads.empty:
        raise SystemExit("no finished candidate jobs")
    cands = candidate_table(heads, c63)
    adopted, why = choose_candidate(cands, float(c63["tie_margin"]))
    weights = weights_table(c63, registry)
    regret = stopping_regret_table(c63, registry, float(cfg["compare62"]["stopping_regret_rel"]))
    cost = cost_table(c63, registry)
    for name, df in (("heads", heads), ("candidates", cands), ("weights", weights), ("regret", regret), ("cost", cost)):
        df.to_csv(OUT / f"us63_{name}.csv", index=False)
    write_md(OUT / "us63_report.md", c63, heads, cands, adopted, why, weights, regret, cost)
    print((OUT / "us63_report.md").read_text())


if __name__ == "__main__":
    main()
