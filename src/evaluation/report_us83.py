"""US 8.3: select the generalization fix by speaker-grouped cross-validation, then (later) evaluate winner and control once on test.

This file holds the SELECTION half (pure numpy / pandas; unit-tested in tests/test_report_us83.py). The rule is the one of docs/sprint8_plan.md section 5,
with two terms made precise BEFORE any CV job was run (logged in section 9 of the plan):

* margin of a head h:  m_h = max(0.01, sd over the 5 CV folds of F0's per-fold overall error of h)   (error = RMSE, or 1 - macro-F1)
* tie:  qualifying candidates whose child/teen improvement is within m_target of the best are tied; the one with the fewest extra mechanisms wins
  (order ORDER below: F1 strength 0.5, F1 strength 1.0, F2 weaker, F2 stronger, F3).

All comparisons are on POOLED OUT-OF-FOLD predictions of the 125 train + val speakers; the 125 official test speakers are not read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import report_us81 as S
from src.evaluation import report_us71 as R71

TARGET = "phone_mispron_t1.0"
HEADS = ["sent_accuracy", "sent_fluency", "sent_prosodic", "sent_total", "word_accuracy", "phone_score", TARGET]
CONTROL = "f0"
ORDER = ["f1_s05", "f1_s10", "f2_a", "f2_b", "f3"]
MARGIN_FLOOR = 0.01
OUT = S.REPO / "results" / "tables" / "fix"


def error(task: str, y, pred) -> float:
    """RMSE, or 1 - macro-F1, of one block of rows (one 'speaker' per row is enough: the statistics are additive)."""
    y, pred = np.asarray(y), np.asarray(pred)
    codes = np.zeros(len(y), np.int64)
    st = S.clf_stats(y, pred, codes, 1, S.N_CLASSES[task]) if task in S.N_CLASSES else S.reg_stats(y, pred, codes, 1)
    return float(S.adverse(task, S.metrics(task, st[0])[R71.metric_of(task)]))


def per_fold_sd(task: str, frame: pd.DataFrame) -> float:
    """Spread over the CV folds of one candidate's per-fold overall error (used for the margin; F0's frame)."""
    e = [error(task, f["y"], f["pred"]) for _, f in frame.groupby("fold")]
    return float(np.std(e, ddof=1)) if len(e) > 1 else 0.0


def margin(task: str, f0_frame: pd.DataFrame) -> float:
    return max(MARGIN_FLOOR, per_fold_sd(task, f0_frame))


def _stats(task: str, frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    return S.Cell(task, frame["y"].to_numpy(), frame["pred"].to_numpy(), frame["speaker"].to_numpy()).stats(mask)


def _aligned(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = a.sort_values("row").reset_index(drop=True), b.sort_values("row").reset_index(drop=True)
    if not (a["row"].to_numpy() == b["row"].to_numpy()).all() or not (a["speaker"].to_numpy() == b["speaker"].to_numpy()).all():
        raise ValueError("candidate and control do not predict the same rows")
    return b


def compare(oof: dict, cand: str, n_boot: int, seed: int) -> dict:
    """One candidate against the control. ``oof[cand][head]`` = DataFrame(row, speaker, age_group, fold, y, pred) of the pooled out-of-fold predictions.

    Returns child/teen and adult change of the TARGET head (error scale, negative = candidate better, with a paired speaker bootstrap for the child/teen change)
    and the change of every head's overall error, with its margin."""
    out = {"candidate": cand}
    f0 = {h: oof[CONTROL][h].sort_values("row").reset_index(drop=True) for h in HEADS}
    c = {h: _aligned(f0[h], oof[cand][h]) for h in HEADS}
    mt = margin(TARGET, f0[TARGET])
    out["margin_target"] = mt
    for grp, key in (("child/teen", "child"), ("adult", "adult")):
        m = (f0[TARGET]["age_group"] == grp).to_numpy()
        sa, sb = _stats(TARGET, f0[TARGET], m), _stats(TARGET, c[TARGET], m)
        W = S.weights(len(sa), n_boot, seed if grp == "child/teen" else seed + 1)
        key_m = R71.metric_of(TARGET)
        d = S.adverse(TARGET, S.metrics(TARGET, sb.sum(0))[key_m]) - S.adverse(TARGET, S.metrics(TARGET, sa.sum(0))[key_m])
        dr = S.adverse(TARGET, S.metrics(TARGET, W @ sb)[key_m]) - S.adverse(TARGET, S.metrics(TARGET, W @ sa)[key_m])
        lo, hi = S.interval(dr)
        out.update({f"{key}_delta": float(d), f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_n_speakers": int(len(sa))})
    worse = {}
    for h in HEADS:
        e0, e1 = error(h, f0[h]["y"], f0[h]["pred"]), error(h, c[h]["y"], c[h]["pred"])
        out[f"overall_delta_{h}"] = e1 - e0
        out[f"margin_{h}"] = margin(h, f0[h])
        if h != TARGET:
            worse[h] = (e1 - e0) > out[f"margin_{h}"]
    out["non_target_worse"] = ",".join(h for h, w in worse.items() if w)
    out["adult_ok"] = bool(out["adult_delta"] <= mt)
    out["eligible"] = bool(not any(worse.values()) and out["adult_ok"])
    out["beats_control"] = bool(out["child_delta"] < -mt and out["child_hi"] < 0)
    out["qualifies"] = bool(out["eligible"] and out["beats_control"])
    return out


def select(table: pd.DataFrame) -> dict:
    """Winner by the pre-declared rule. -> {winner, reason, evaluate (candidate to run once on test), tied}."""
    t = table.copy()
    q = t[t["qualifies"]]
    if len(q):
        best = q["child_delta"].min()
        tied = q[q["child_delta"] <= best + q["margin_target"].iloc[0]]
        pick = sorted(tied["candidate"], key=lambda c: ORDER.index(c) if c in ORDER else len(ORDER))[0]
        return {"winner": pick, "evaluate": pick, "tied": list(tied["candidate"]), "reason": "qualifies; fewest extra mechanisms among candidates within the margin of the best"}
    pool = t[t["eligible"]] if t["eligible"].any() else t
    pick = pool.sort_values("child_delta").iloc[0]["candidate"]
    return {"winner": None, "evaluate": pick, "tied": [],
            "reason": "no candidate qualifies; the single best (eligible if any, else overall) is still evaluated once on test and reported as inconclusive or negative"}


def run_selection(oof: dict, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rows = [compare(oof, c, n_boot, seed) for c in oof if c != CONTROL]
    table = pd.DataFrame(rows)
    return table, select(table)


# --------------------------------------------------------------------------- #
# reading the CV jobs (needs the lab outputs; not used by the tests)
# --------------------------------------------------------------------------- #
def tag_of(cand: str, k: int) -> str:
    return f"s8cv{k}_{cand}"


def load_oof(cands: list[str], folds_csv: Path | None = None, seed: int = 0) -> dict:
    """{cand: {head: DataFrame(row, speaker, age_group, fold, y, pred)}} from experiments/results/multitask/s8cv<k>_<cand>/<head>/fold0/seed<seed>.npz."""
    from src.models import fix_s8 as FS
    from src.models import frozen_heads as FH
    from src.models import multitask as MT
    from src.models.baselines import build_tasks, load_config as load_baseline_config
    cfg = MT.load_config("s8_f0")
    reg = build_tasks(load_baseline_config())
    folds = pd.read_csv(folds_csv or FS.CV_CSV, dtype={"speaker": str})
    out: dict = {c: {} for c in cands}
    for h in HEADS:
        task = reg[h]
        y_all, meta = FH.load_labels(task)
        grp = FS.speaker_groups(meta["speaker"])
        for c in cands:
            parts = []
            for k in range(FS.N_FOLDS):
                f = MT.job_file(cfg, tag_of(c, k), h, 0, seed)
                if not f.exists():
                    raise FileNotFoundError(f"missing CV job {f}")
                z = np.load(f, allow_pickle=True)
                rows = z["test_rows"]
                parts.append(pd.DataFrame({"row": rows, "speaker": meta["speaker"].to_numpy()[rows], "age_group": grp[rows], "fold": k,
                                           "y": FH.encode_y(task, y_all)[rows] if task.kind == "clf" else y_all[rows],
                                           "pred": FH.encode_y(task, z["test_pred"]) if task.kind == "clf" else z["test_pred"]}))
            out[c][h] = pd.concat(parts, ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# the single test evaluation: control vs the candidate chosen by the CV (3 seeds each, official split)
# --------------------------------------------------------------------------- #
def _err_delta(task: str, key: str, sa, sb, W) -> dict:
    """Change of the ERROR (RMSE, or 1 - macro-F1) from control a to fix b, with a paired speaker bootstrap: negative = the fix is better."""
    f = lambda st: S.adverse(task, S.metrics(task, st)[key])
    d = float(f(sb.sum(0)) - f(sa.sum(0)))
    dr = f(W @ sb) - f(W @ sa)
    lo, hi = S.interval(dr)
    return {"delta": d, "lo": lo, "hi": hi, "p": S.pvalue(dr)}


def final_table(control: str, fix: str, n_boot: int, seed: int, heads=HEADS) -> pd.DataFrame:
    """Per head: overall, adult and child/teen error change (fix - control) with paired intervals, and the change of the child/teen - adult gap."""
    rows = []
    for task in heads:
        frame = S.test_frame(task)
        m = S.group_masks(frame)["age"]
        cells = {}
        for sysn in (control, fix):
            b = S.load_bundle(sysn, task)
            if b is None:
                raise FileNotFoundError(f"missing bundle {sysn}__{task}: run `report_us71_pairs export --systems S8-F0 S8-FIX` on the lab first")
            S.check_bundle(task, frame, b)
            cells[sysn] = S.Cell(task, b["y"], b["pred"], b["speaker"])
        key = R71.metric_of(task)
        allm = np.ones(len(frame), bool)
        st = {(s, g): cells[s].stats(mk) for s in cells for g, mk in (("overall", allm), ("adult", m["adult"]), ("child/teen", m["child/teen"]))}
        Wd = {"overall": S.weights(len(st[(control, "overall")]), n_boot, seed + 2), "adult": S.weights(len(st[(control, "adult")]), n_boot, seed),
              "child/teen": S.weights(len(st[(control, "child/teen")]), n_boot, seed + 1)}
        row = {"task": task, "target": task == TARGET, "metric": key}
        for g in ("overall", "adult", "child/teen"):
            d = _err_delta(task, key, st[(control, g)], st[(fix, g)], Wd[g])
            row.update({f"{g}_{k}": v for k, v in d.items()})
            row[f"{g}_control_error"] = float(S.adverse(task, S.metrics(task, st[(control, g)].sum(0))[key]))
        ga, gb = (S.gap_draws(task, st[(s, "adult")], st[(s, "child/teen")], Wd["adult"], Wd["child/teen"]) for s in (control, fix))
        lo, hi = S.interval(gb[1] - ga[1])
        row.update({"gap_control": ga[0], "gap_fix": gb[0], "gap_delta": gb[0] - ga[0], "gap_lo": lo, "gap_hi": hi, "gap_p": S.pvalue(gb[1] - ga[1])})
        rows.append(row)
    return pd.DataFrame(rows)


def acceptance(t: pd.DataFrame) -> dict:
    """The three acceptance rules of docs/sprint8_plan.md section 5, on the error scale (negative delta = the fix is better)."""
    tgt = t[t["target"]]
    better = ((tgt["child/teen_delta"] < 0) & (tgt["child/teen_hi"] < 0)).any()
    sig_worse_target = ((tgt["child/teen_delta"] > 0) & (tgt["child/teen_lo"] > 0)).any()
    r1 = bool(better and not sig_worse_target)
    worse_heads = t.loc[(t["overall_delta"] > 0) & (t["overall_lo"] > 0), "task"].tolist()
    r2 = not worse_heads
    r3 = not ((tgt["adult_delta"] > 0) & (tgt["adult_lo"] > 0)).any()
    return {"rule_i_child_target_improves_and_none_worse": r1, "rule_ii_no_head_significantly_worse_overall": r2, "worse_heads": worse_heads,
            "rule_iii_adult_target_not_worse": r3, "accepted": bool(r1 and r2 and r3)}


# --------------------------------------------------------------------------- #
# reference analysis (NOT a candidate): per-age-group decision thresholds on F0's out-of-fold probabilities, fit by nested folds
# --------------------------------------------------------------------------- #
GRID = np.linspace(0.05, 0.95, 91)


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    f = []
    for c in (0, 1):
        tp, fp, fn = np.sum((pred == c) & (y == c)), np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    scores = [_macro_f1(y, (p >= t).astype(int)) for t in GRID]
    return float(GRID[int(np.argmax(scores))])


def threshold_reference(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """``frame``: DataFrame(row, speaker, age_group, fold, y, p) = F0's pooled out-of-fold positive-class probabilities. The thresholds applied to fold k
    are fit on the OTHER folds only: one pooled threshold, and one per age group. Rows of the result: default (0.5), pooled threshold, per-group thresholds."""
    f = frame.sort_values("row").reset_index(drop=True)
    y, p, g = f["y"].to_numpy(int), f["p"].to_numpy(float), f["age_group"].to_numpy()
    pred = {"default (0.5)": (p >= 0.5).astype(int), "pooled threshold": np.zeros(len(f), int), "per-group thresholds": np.zeros(len(f), int)}
    used = []
    for k in sorted(f["fold"].unique()):
        te, tr = (f["fold"] == k).to_numpy(), (f["fold"] != k).to_numpy()
        tp = _best_threshold(y[tr], p[tr])
        pred["pooled threshold"][te] = (p[te] >= tp).astype(int)
        tg = {grp: _best_threshold(y[tr & (g == grp)], p[tr & (g == grp)]) for grp in np.unique(g)}
        for grp, t in tg.items():
            m = te & (g == grp)
            pred["per-group thresholds"][m] = (p[m] >= t).astype(int)
        used.append({"fold": int(k), "pooled": tp, **{f"t_{grp}": t for grp, t in tg.items()}})
    rows = []
    cells = {name: S.Cell(TARGET, y, pr, f["speaker"].to_numpy()) for name, pr in pred.items()}
    masks = {"adult": g == "adult", "child/teen": g == "child/teen"}
    sts = {(name, grp): c.stats(mk) for name, c in cells.items() for grp, mk in masks.items()}
    base = "default (0.5)"
    for name in pred:
        row = {"rule": name}
        for grp, mk in masks.items():
            st = sts[(name, grp)]
            met = S.metrics(TARGET, st.sum(0))
            row[f"{grp}_error"] = float(1 - met["macro_f1"])
            row[f"{grp}_recall"] = float(met["recall"])
            W = S.weights(len(st), n_boot, seed if grp == "adult" else seed + 1)
            d = _err_delta(TARGET, "macro_f1", sts[(base, grp)], st, W)
            row.update({f"{grp}_delta_vs_default": d["delta"], f"{grp}_lo": d["lo"], f"{grp}_hi": d["hi"]})
        row["gap"] = row["child/teen_error"] - row["adult_error"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["thresholds"] = pd.DataFrame(used)
    return out


def load_oof_proba(cand: str = CONTROL, folds_csv: Path | None = None, seed: int = 0) -> pd.DataFrame:
    from src.models import fix_s8 as FS
    from src.models import frozen_heads as FH
    from src.models import multitask as MT
    from src.models.baselines import build_tasks, load_config as load_baseline_config
    cfg = MT.load_config("s8_f0")
    task = build_tasks(load_baseline_config())[TARGET]
    y_all, meta = FH.load_labels(task)
    grp = FS.speaker_groups(meta["speaker"])
    parts = []
    for k in range(FS.N_FOLDS):
        z = np.load(MT.job_file(cfg, tag_of(cand, k), TARGET, 0, seed), allow_pickle=True)
        rows, proba = z["test_rows"], z["test_proba"]
        pos = list(z["classes"]).index("1") if "classes" in z.files and "1" in list(z["classes"]) else 1
        parts.append(pd.DataFrame({"row": rows, "speaker": meta["speaker"].to_numpy()[rows], "age_group": grp[rows], "fold": k,
                                   "y": FH.encode_y(task, y_all)[rows], "p": proba[:, pos]}))
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", default="select", choices=["select", "reference", "final"])
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--seed-job", type=int, default=0)
    a = ap.parse_args(argv)
    import json
    if a.command == "reference":
        nb, seed = S.boot_settings()
        t = threshold_reference(load_oof_proba(seed=a.seed_job), nb, seed)
        OUT.mkdir(parents=True, exist_ok=True)
        t.to_csv(OUT / "us83_threshold_reference.csv", index=False)
        t.attrs["thresholds"].to_csv(OUT / "us83_threshold_reference_thresholds.csv", index=False)
        pd.set_option("display.width", 220)
        print(t.round(3).to_string(index=False))
        print(t.attrs["thresholds"].round(2).to_string(index=False))
        return
    if a.command == "final":
        nb, seed = S.boot_settings()
        pick = json.loads((OUT / "us83_cv_selection.json").read_text())
        tab = final_table("S8-F0", "S8-FIX", nb, seed)
        verdict = acceptance(tab)
        verdict["candidate"] = pick["evaluate"]
        verdict["cv_winner"] = pick["winner"]
        tab.to_csv(OUT / "us83_final_test.csv", index=False)
        (OUT / "us83_final_verdict.json").write_text(json.dumps(verdict, indent=2))
        pd.set_option("display.width", 250)
        print(tab[["task", "overall_delta", "overall_lo", "overall_hi", "adult_delta", "adult_lo", "adult_hi", "child/teen_delta", "child/teen_lo", "child/teen_hi",
                   "gap_delta", "gap_lo", "gap_hi"]].round(3).to_string(index=False))
        print(verdict)
        return
    import yaml
    cands = a.candidates or yaml.safe_load((S.REPO / "configs" / "multitask.yaml").read_text())["s8"]["candidates"]
    nb, seed = S.boot_settings()
    oof = load_oof(cands, seed=a.seed_job)
    table, pick = run_selection(oof, nb, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "us83_cv_selection.csv", index=False)
    (OUT / "us83_cv_selection.json").write_text(__import__("json").dumps(pick, indent=2))
    pd.set_option("display.width", 220)
    print(table[["candidate", "child_delta", "child_lo", "child_hi", "adult_delta", "non_target_worse", "eligible", "beats_control", "qualifies"]].round(3).to_string(index=False))
    print(pick)


if __name__ == "__main__":
    main()
