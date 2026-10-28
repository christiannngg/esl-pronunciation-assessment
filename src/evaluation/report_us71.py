"""
US 7.1 - midterm comparison of every model built so far (aggregation only: nothing is retrained, no earlier file is modified).

    python -m src.evaluation.report_us71                 # -> results/tables/midterm/us71_*.{csv,md} + results/figures/midterm/us71_ladder.png
    python -m src.evaluation.report_us71 --no-fig        # tables only
    python -m src.evaluation.report_us71 --out /tmp/x    # write somewhere else (the inputs are only ever read)

Reads the saved headline tables of Sprints 3-6 (no predictions, so it runs on any machine):
    B          baseline after the US 3.3 fix           results/tables/baseline/us33_before_after.csv   (best model per task, chosen on validation)
    F          frozen WavLM-base-plus, US 4.3 system   results/tables/frozen/us43_headline.csv
    FT-base    fine-tuned, US 5.1 `base` recipe        results/tables/finetune/us51_headline.csv
    FT-stab    fine-tuned, US 5.3 stabilized set       results/tables/finetune/us53_headline.csv
    MT-full    multi-task, fixed equal weights         results/tables/multitask/us63_heads.csv (+ us61_headline_mt_{base,lr}.csv for the CI and L2)
    MT-GN15 / MT-GN05 / MT-regroup                     results/tables/multitask/us63_heads.csv (SO762 only)

Carry-forward rule (docs/us71_inventory.md section 4, fixed before the final table was looked at):
    candidates per task: FT-base, FT-stab, MT-full, MT-GN15 (SO762 tasks only). B and F are references.
    1. a candidate that is significantly worse than FT-base on test is ineligible (reported, never used to select anything else);
    2. the leader has the best VALIDATION error; every eligible candidate within the margin max(0.01, seed sd) of it is tied;
    3. ties: fewest encoders for the group, then fewest extra mechanisms (no GradNorm / no stabilizing variant), then the lower validation error;
    4. carried into Sprint 8: the single-task reference (FT-base or FT-stab, by the same rule) and MT-GN15; MT-full stays as the equal-weight reference;
    5. when the test leader is not among the validation-tied candidates both are shown and the validation leader is carried; test never overrides;
    6. L2-ARCTIC: FT-stab and MT-full are carried (they are within 0.005 macro-F1 of each other on test).
MT-GN15 is the best-supported candidate with the US 6.3 disclosure, NOT a rule-adopted fix (docs/us63_fix_summary.md section 6).

Three paired comparisons are not stored anywhere (B vs FT-base, B vs MT-GN15, F vs MT-GN15). They need the prediction files, which sit on different machines:
src/evaluation/report_us71_pairs.py computes them (export on each machine, then `pairs`). Until results/tables/midterm/us71_pairs_computed.csv exists they are
listed as MISSING in us71_pairs.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "results" / "tables"
OUT = TABLES / "midterm"
FIG = REPO / "results" / "figures" / "midterm"

SO762 = ["sent_accuracy", "sent_fluency", "sent_prosodic", "sent_total", "word_accuracy", "phone_score", "phone_mispron_t1.0"]
L2 = ["l2_phone_error", "l2_word_error"]
TASKS = SO762 + L2
CLF = {"phone_mispron_t1.0", "l2_phone_error", "l2_word_error"}
GROUP = {t: ("so762" if t in SO762 else "l2") for t in TASKS}
ENCODER = "wavlm_base_plus"

SYSTEMS = ["B", "F", "FT-base", "FT-stab", "MT-full", "MT-GN15", "MT-GN05", "MT-regroup"]
CANDIDATES = ["FT-base", "FT-stab", "MT-full", "MT-GN15"]
MECHANISMS = {"FT-base": 0, "MT-full": 0, "FT-stab": 1, "MT-GN15": 1}          # extra mechanisms beyond the plain recipe
US63_NAME = {"MT-full": "full", "MT-GN15": "gradnorm_a15", "MT-GN05": "gradnorm_a05", "MT-regroup": "regroup"}


# --------------------------------------------------------------------------- #
# pure helpers (numpy / pandas only, unit-tested)
# --------------------------------------------------------------------------- #
def kind_of(task: str) -> str:
    return "clf" if task in CLF else "reg"


def metric_of(task: str) -> str:
    return "macro_f1" if task in CLF else "rmse"


def error_of(task: str, value: float) -> float:
    """RMSE for regression, 1 - macro-F1 for classification: lower is better for both."""
    return float(value) if kind_of(task) == "reg" else 1.0 - float(value)


def verdict(task: str, delta: float, lo: float, hi: float) -> str:
    """Same reading as report_us43.verdict: a delta whose 95% CI excludes 0 is better / worse, otherwise no significant change."""
    if not np.isfinite(delta) or not np.isfinite(lo) or not np.isfinite(hi):
        return "n/a"
    if not (lo > 0 or hi < 0):
        return "no significant change"
    good = delta < 0 if kind_of(task) == "reg" else delta > 0
    return "better" if good else "worse"


def encoders_for(system: str, task: str) -> int | float:
    """Fine-tuned encoders needed to serve the task's group: single-task systems train one per task, multi-task one (two for the regroup)."""
    n_tasks = len(SO762) if GROUP[task] == "so762" else len(L2)
    if system in ("FT-base", "FT-stab"):
        return n_tasks
    if system == "MT-regroup":
        return 2
    if system in ("MT-full", "MT-GN15", "MT-GN05"):
        return 1
    return 0                                                          # B: no neural encoder; F: frozen encoder, nothing fine-tuned


def pick_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task from us33_before_after.csv: the model with the best validation score after the fix (lowest RMSE / highest macro-F1)."""
    rows = []
    for task in TASKS:
        g = df[df["task"] == task]
        if g.empty:
            continue
        col = "val_primary_after"
        g = g.sort_values(col, ascending=kind_of(task) == "reg", kind="stable")
        r = g.iloc[0]
        test = r["after_rmse"] if kind_of(task) == "reg" else r["after_macro_f1"]
        rows.append({"task": task, "system": "B", "test": float(test), "lo": np.nan, "hi": np.nan, "val": float(r[col]), "n_seeds": 1,
                     "note": f"{r['model']} + {r['fix']}"})
    return pd.DataFrame(rows)


def carry_forward(long: pd.DataFrame, margins: dict[str, float], worse_than_base: dict[tuple[str, str], bool]) -> pd.DataFrame:
    """Apply the declared rule. long: columns task, system, test, val, encoders. margins: task -> max(0.01, seed sd). worse_than_base[(task, system)]: significantly
    worse than FT-base on test (stored verdicts). One row per task."""
    out = []
    for task in TASKS:
        g = long[(long["task"] == task) & (long["system"].isin(CANDIDATES))].copy()
        if g.empty:
            continue
        margin = float(margins.get(task, 0.01))
        g["err"] = [error_of(task, v) for v in g["val"]]
        g["eligible"] = [not worse_than_base.get((task, s), False) for s in g["system"]]
        pool = g[g["eligible"]]
        if pool.empty:
            pool = g
        best = float(pool["err"].min())
        tied = pool[pool["err"] <= best + margin + 1e-12].copy()
        tied["mech"] = tied["system"].map(MECHANISMS)
        winner = tied.sort_values(["encoders", "mech", "err", "system"], kind="stable").iloc[0]
        st = pool[pool["system"].isin(["FT-base", "FT-stab"])]
        st_ref = None
        if not st.empty:
            st_tied = st[st["err"] <= float(st["err"].min()) + margin + 1e-12].copy()
            st_tied["mech"] = st_tied["system"].map(MECHANISMS)
            st_ref = st_tied.sort_values(["mech", "err", "system"], kind="stable").iloc[0]["system"]
        test_err = {r.system: error_of(task, r.test) for r in g.itertuples()}
        test_best = min(test_err, key=test_err.get)
        if GROUP[task] == "l2":
            carried = ["FT-stab", "MT-full"]
        else:
            carried = [s for s in (st_ref, "MT-GN15") if s is not None and s in set(g["system"])] + (["MT-full (reference)"] if "MT-full" in set(g["system"]) else [])
        out.append({"task": task, "group": GROUP[task], "margin": margin, "val_leader": winner["system"], "tied_with_leader": ", ".join(tied["system"]),
                    "single_task_ref": st_ref, "test_leader": test_best, "validation_test_disagree": test_best not in set(tied["system"]),
                    "ineligible": ", ".join(g.loc[~g["eligible"], "system"]), "carried": ", ".join(carried)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# loading the saved tables (read only)
# --------------------------------------------------------------------------- #
def _read(rel: str) -> pd.DataFrame:
    p = TABLES / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run the report of that sprint first (see the module docstring)")
    return pd.read_csv(p)


def load_systems() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """-> (long table, margins per task, worse-than-FT-base flags, stored paired deltas)."""
    rows, pairs = [], []
    worse: dict[tuple[str, str], bool] = {}

    b = pick_baseline(_read("baseline/us33_before_after.csv"))
    rows.append(b)

    f43 = _read("frozen/us43_headline.csv")
    fw = f43[f43["encoder"] == ENCODER].set_index("task")
    anyenc = f43.groupby("task")["frozen_43"].agg(["min", "max"])
    for t in TASKS:
        if t not in fw.index:
            continue
        r = fw.loc[t]
        rows.append(pd.DataFrame([{"task": t, "system": "F", "test": float(r["frozen_43"]), "lo": np.nan, "hi": np.nan, "val": float(r["val_43"]), "n_seeds": 1,
                                   "note": f"{r['frozen_43_system']}; best of 3 encoders {anyenc.loc[t, 'min' if kind_of(t) == 'reg' else 'max']:.3f}"}]))
        pairs.append(_pair(t, "B", "F", r["d_vs_33"], r["d_vs_33_lo"], r["d_vs_33_hi"], r["d_vs_33_p"]))

    ft51 = _read("finetune/us51_headline.csv")
    ft51 = ft51[ft51["tag"] == "base"].set_index("task")
    ft53 = _read("finetune/us53_headline.csv").set_index("task")
    margins = {}
    for t in TASKS:
        if t in ft51.index:
            r = ft51.loc[t]
            rows.append(pd.DataFrame([{"task": t, "system": "FT-base", "test": float(r["finetuned"]), "lo": float(r["ft_lo"]), "hi": float(r["ft_hi"]), "val": float(r["val"]),
                                       "n_seeds": int(r["n_seeds"]), "note": "5.1 base"}]))
            pairs.append(_pair(t, "F", "FT-base", r["d_vs_43"], r["d_vs_43_lo"], r["d_vs_43_hi"], r["d_vs_43_p"]))
        if t in ft53.index:
            r = ft53.loc[t]
            rows.append(pd.DataFrame([{"task": t, "system": "FT-stab", "test": float(r["test_after"]), "lo": np.nan, "hi": np.nan, "val": float(r["val_headline_after"]),
                                       "n_seeds": 1, "note": f"5.3 {r['selected']}"}]))
            margins[t] = float(r["margin"])
            if r["selected"] != "base":
                pairs.append(_pair(t, "FT-base", "FT-stab", r["d_vs_base"], r["d_vs_base_lo"], r["d_vs_base_hi"], np.nan))
                worse[(t, "FT-stab")] = r["verdict_vs_base"] == "worse"

    h63 = _read("multitask/us63_heads.csv")
    mt_base = _read("multitask/us61_headline_mt_base.csv").set_index("task")
    mt_lr = _read("multitask/us61_headline_mt_lr.csv").set_index("task")
    for sysname, cname in US63_NAME.items():
        g = h63[h63["candidate"] == cname].set_index("task")
        for t in SO762:
            if t not in g.index:
                continue
            r = g.loc[t]
            lo = hi = np.nan
            if sysname == "MT-full" and t in mt_base.index:
                lo, hi = float(mt_base.loc[t, "mt_lo"]), float(mt_base.loc[t, "mt_hi"])
            rows.append(pd.DataFrame([{"task": t, "system": sysname, "test": float(r["cand_test"]), "lo": lo, "hi": hi, "val": float(r["cand_val"]), "n_seeds": int(r["seeds"]),
                                       "note": cname}]))
            pairs.append(_pair(t, "FT-base", sysname, r["d_vs_st"], r["d_vs_st_lo"], r["d_vs_st_hi"], r["d_vs_st_p"]))
            if sysname != "MT-full":
                pairs.append(_pair(t, "MT-full", sysname, r["d_vs_full"], r["d_vs_full_lo"], r["d_vs_full_hi"], np.nan))
            worse[(t, sysname)] = r["verdict_vs_st"] == "worse"
    for t in L2:
        if t in mt_lr.index:
            r = mt_lr.loc[t]
            rows.append(pd.DataFrame([{"task": t, "system": "MT-full", "test": float(r["mt"]), "lo": float(r["mt_lo"]), "hi": float(r["mt_hi"]), "val": float(r["mt_val"]),
                                       "n_seeds": int(r["seeds_base"]), "note": "mt_lr, 4 folds x 3 seeds"}]))
            pairs.append(_pair(t, "FT-base", "MT-full", r["d_base"], r["d_base_lo"], r["d_base_hi"], r["d_base_p"]))
            pairs.append(_pair(t, "FT-stab", "MT-full", r["d_stabilized"], r["d_stabilized_lo"], r["d_stabilized_hi"], r["d_stabilized_p"]))
            worse[(t, "MT-full")] = r["verdict_base"] == "worse"

    long = pd.concat(rows, ignore_index=True)
    long["group"] = long["task"].map(GROUP)
    long["metric"] = long["task"].map(metric_of)
    long["encoders"] = [encoders_for(s, t) for s, t in zip(long["system"], long["task"])]
    for s, t in [("B", "FT-base"), ("B", "MT-GN15"), ("F", "MT-GN15")]:                    # not stored anywhere: need the prediction files
        for task in TASKS:
            if GROUP[task] == "l2" and "GN" in t:
                continue
            pairs.append({"task": task, "a": s, "b": t, "delta": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "verdict": "MISSING (needs prediction files on the lab machine)"})
    pairs = pd.DataFrame(pairs)
    computed = TABLES / "midterm" / "us71_pairs_computed.csv"
    if computed.exists():                                                  # written by report_us71_pairs on the machines that hold the predictions
        pairs = merge_computed(pairs, pd.read_csv(computed))
    return long, margins, worse, pairs


def merge_computed(pairs: pd.DataFrame, computed: pd.DataFrame) -> pd.DataFrame:
    """Replace the MISSING rows by the comparisons that report_us71_pairs computed from the prediction files (same task, a, b)."""
    if computed is None or computed.empty:
        return pairs
    key = set(zip(computed["task"], computed["a"], computed["b"]))
    keep = [not (str(v).startswith("MISSING") and (t, a, b) in key) for t, a, b, v in zip(pairs["task"], pairs["a"], pairs["b"], pairs["verdict"])]
    return pd.concat([pairs[keep], computed], ignore_index=True)


def _pair(task, a, b, delta, lo, hi, p) -> dict:
    d, l, h = (float(x) if x == x else np.nan for x in (delta, lo, hi))
    return {"task": task, "a": a, "b": b, "delta": d, "lo": l, "hi": h, "p": float(p) if p == p else np.nan, "verdict": verdict(task, d, l, h)}


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _f(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def wide_table(long: pd.DataFrame, systems=SYSTEMS) -> str:
    present = [s for s in systems if s in set(long["system"])]
    lines = ["| task | metric | " + " | ".join(present) + " |", "|---|---|" + "---|" * len(present)]
    for t in TASKS:
        g = long[long["task"] == t].set_index("system")
        vals = {s: float(g.loc[s, "test"]) for s in present if s in g.index}
        best = (min if kind_of(t) == "reg" else max)(vals.values()) if vals else None
        cells = []
        for s in present:
            if s not in vals:
                cells.append("-")
                continue
            txt = _f(vals[s])
            if np.isfinite(g.loc[s, "lo"]):
                txt += f" [{_f(g.loc[s, 'lo'])}, {_f(g.loc[s, 'hi'])}]"
            cells.append(f"**{txt}**" if vals[s] == best else txt)
        lines.append(f"| {t} | {metric_of(t)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md(long: pd.DataFrame, cf: pd.DataFrame, pairs: pd.DataFrame, path: Path) -> None:
    L = ["# US 7.1 midterm comparison (generated by `src.evaluation.report_us71`; aggregation of the saved Sprint 3-6 tables)", ""]
    L += ["## 1. Test results, all systems (RMSE lower is better; macro-F1 higher is better; **bold** = best of the row; [95% CI] where the saved table has one)", "",
          wide_table(long), ""]
    L += ["Notes: B = best baseline per task chosen on validation after the US 3.3 fix; F = frozen WavLM-base-plus (US 4.3 system per task); FT-base = US 5.1; FT-stab = US 5.3 "
          "rule-selected variants; MT-* = multi-task (SO762: 7 heads in one encoder, L2: 2 heads; GradNorm and regroup were run on SO762 only). "
          "Seeds: SO762 3, L2 FT 1 per fold, L2 MT 3 per fold. L2 FT-base / FT-stab here are the 1-seed Sprint 5 values; single- vs multi-task comparisons use the 3-seed "
          "Sprint 6 comparators (see section 3). MT-GN15 is the best-supported candidate with disclosure, not a rule-adopted fix.", ""]
    L += ["## 2. Carry-forward rule applied (validation decides; test is reported)", "",
          "| task | validation leader | tied with leader | single-task ref | test leader | val/test disagree | ineligible | carried |", "|---|---|---|---|---|---|---|---|"]
    for r in cf.itertuples():
        L.append(f"| {r.task} | {r.val_leader} | {r.tied_with_leader} | {r.single_task_ref or '-'} | {r.test_leader} | {'yes' if r.validation_test_disagree else 'no'} | "
                 f"{r.ineligible or '-'} | {r.carried} |")
    L += ["", "## 3. Paired comparisons (delta = metric(b) - metric(a); stored by the sprint reports; negative RMSE / positive F1 is better for b)", "",
          "| task | a -> b | delta [95% CI] | p | verdict |", "|---|---|---|---|---|"]
    for r in pairs.sort_values(["task", "a", "b"]).itertuples():
        d = "-" if not np.isfinite(r.delta) else _f(r.delta) + ("" if not np.isfinite(r.lo) else f" [{_f(r.lo)}, {_f(r.hi)}]")
        L.append(f"| {r.task} | {r.a} -> {r.b} | {d} | {_f(r.p)} | {r.verdict} |")
    path.write_text("\n".join(L) + "\n")


def figure(long: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ["B", "F", "FT-base", "FT-stab", "MT-full", "MT-GN15"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, t in zip(axes.ravel(), TASKS):
        g = long[(long["task"] == t) & (long["system"].isin(shown))].set_index("system").reindex([s for s in shown if s in set(long[long["task"] == t]["system"])])
        x = np.arange(len(g))
        ax.bar(x, g["test"], color=["#999999", "#8da0cb", "#66c2a5", "#a6d854", "#fc8d62", "#e78ac3"][:len(g)])
        err = np.vstack([(g["test"] - g["lo"]).to_numpy(float), (g["hi"] - g["test"]).to_numpy(float)])
        ok = np.isfinite(err).all(0)
        ax.errorbar(x[ok], g["test"].to_numpy(float)[ok], yerr=err[:, ok], fmt="none", ecolor="black", capsize=2, lw=1)
        ax.set_xticks(x, g.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{t} ({metric_of(t)}, {'lower' if kind_of(t) == 'reg' else 'higher'} is better)", fontsize=8)
        lo, hi = float(g["test"].min()), float(g["test"].max())
        pad = max(hi - lo, 0.02) * 0.6
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("US 7.1 - test results by system (error bars: 95% speaker-bootstrap CI where the saved table has one)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def main(argv=None) -> None:
    global OUT, FIG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    ap.add_argument("--out", default=None, help="directory for the tables (default results/tables/midterm)")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args(argv)
    OUT = Path(a.out) if a.out else OUT
    FIG = Path(a.out) if a.out else FIG
    long, margins, worse, pairs = load_systems()
    cf = carry_forward(long, margins, worse)
    OUT.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT / "us71_systems.csv", index=False)
    cf.to_csv(OUT / "us71_carry_forward.csv", index=False)
    pairs.to_csv(OUT / "us71_pairs.csv", index=False)
    write_md(long, cf, pairs, OUT / "us71_report.md")
    print(f"[us71] baseline picks (check against us33_before_after.md headline):")
    for r in long[long["system"] == "B"].itertuples():
        print(f"    {r.task:22s} {r.note}")
    if not a.no_fig:
        try:
            figure(long, FIG / "us71_ladder.png")
        except ImportError as e:                                       # pragma: no cover
            print(f"[us71] figure skipped ({e})")
    print(f"[us71] wrote {OUT}/us71_{{systems,carry_forward,pairs}}.csv and us71_report.md")


if __name__ == "__main__":
    main()
