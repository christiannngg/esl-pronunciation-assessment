"""US 8.2: diagnosis of the child/teen weakness and the ablation table (numpy / pandas only; reads saved predictions, trains nothing).

Target head (rule fixed in docs/sprint8_plan.md, confirmed by Family A of US 8.1): phone_mispron_t1.0. The regression heads are diagnosed as a
secondary outcome only; they do not enter the target set.

  check 1  is the gap real?     per-group human ceiling (leave-one-out r of the 5 experts), model PCC / ceiling, and a LABEL-MATCHED comparison: adult rows are
                                reweighted to the child/teen label distribution, so a difference that survives is not range restriction.
  check 2  prior or ranking?    mispronunciation: AUC / AP per group and the recall gap split into a ranking part (adult recall - child recall at the SAME
                                false-positive rate) and a calibration part (child recall at that rate - child recall at the pooled operating point).
                                Needs results/analysis/us82/<system>__phone_mispron_t1.0.npz (lab: `report_us71_pairs export --proba-only`).
  check 3  bias and compression bias / RMSE per group x true-score bin for the regression heads (descriptive).
  check 4  is age decodable?    not here: experiments/scripts/probe_age_us82.py (needs the frozen-encoder cache and scikit-learn).
  ablation representation (B to F), fine-tuning (F to FT), sharing (FT to MT), stabilisation (FT to FT-stab): change of the metric inside each age group and of
           the age gap, on the SO762 heads; the L2-ARCTIC heads pooled and per L1 (exploratory).

    python -m src.evaluation.report_us82 [--n-boot N] [--skip-ceiling]

Conventions as US 8.1 (speaker-level bootstrap, percentile CI, p = 2 min(P(d <= 0), P(d >= 0)); group draws independent, seeds seed and seed + 1).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation import report_us71 as R71
from src.evaluation import report_us81 as S

OUT = S.OUT
PROBA = S.REPO / "results" / "analysis" / "us82"
TARGET = "phone_mispron_t1.0"
DIFF_KEYS = ("auc", "ap", "positive_rate", "fpr_at_pooled_threshold", "recall_at_pooled_threshold", "recall_child_at_adult_fpr")   # reported as child - adult
REG = [t for t in R71.SO762 if t not in S.N_CLASSES]
CHECK_SYSTEMS = ["B", "F", "FT-base", "MT-GN15"]
PROBA_SYSTEMS = ["FT-base", "FT-stab", "MT-full", "MT-GN15"]
CONTRASTS = [("B", "F", "representation"), ("F", "FT-base", "fine-tuning"), ("FT-base", "MT-GN15", "sharing (GradNorm)"),
             ("FT-base", "MT-full", "sharing (equal weights)"), ("FT-base", "FT-stab", "stabilisation (US 5.3)")]
CONTRASTS_L2 = [("B", "F", "representation"), ("F", "FT-stab", "fine-tuning"), ("FT-stab", "MT-full", "sharing")]
EXPERT_COLS = {"sent_accuracy": ("sentence", "accuracy"), "sent_fluency": ("sentence", "fluency"), "sent_prosodic": ("sentence", "prosodic"),
               "sent_total": ("sentence", "total"), "word_accuracy": ("word", "accuracy"), "phone_score": ("phone", "score")}


# --------------------------------------------------------------------------- #
# check 1: human ceiling and label-matched comparison
# --------------------------------------------------------------------------- #
def loo_r(wide) -> float:
    """Mean over raters of the correlation with the mean of the other raters (as eda_utils.leave_one_out_pearson); wide = items x raters."""
    x = np.asarray(wide, float)
    k, tot = x.shape[1], x.sum(1)
    return float(np.mean([np.corrcoef(x[:, j], (tot - x[:, j]) / (k - 1))[0, 1] for j in range(k)]))


def consensus_reliability(r: float, k: int = 5) -> float:
    """Reliability of the mean of k raters from the leave-one-out r (rater vs mean of the other k - 1), classical test theory with exchangeable raters:
    r^2 = rho_1 * rho_(k-1) with Spearman-Brown rho_m = m rho_1 / (1 + (m - 1) rho_1). Approximate: the published label is a consensus of five, not exactly their mean."""
    r2 = r * r
    rho1 = ((k - 2) * r2 + np.sqrt((k - 2) ** 2 * r2 * r2 + 4 * (k - 1) * r2)) / (2 * (k - 1))
    return float(k * rho1 / (1 + (k - 1) * rho1))


def expert_ceilings(experts=None) -> pd.DataFrame:
    """Leave-one-out human r per head and age group, on the test speakers' items."""
    if experts is None:
        from src.data.loaders import load_so762_experts
        experts = load_so762_experts()
    age = pd.read_csv(S.SPLITS / "so762_speakers.csv", dtype={"speaker": str}).set_index("speaker")["age_group"]
    sent = S._read_meta("so762_sentence")
    test = sent[S._so762_split(sent) == "test"]
    utt_age = test.set_index("utt_id")["speaker"].map(age)
    long = {"sentence": experts.sent, "word": experts.words, "phone": experts.phones}
    rows = []
    for head, (level, col) in EXPERT_COLS.items():
        df = long[level]
        key = df["utt_id"].astype(str)
        if level != "sentence":
            key = key + "|" + df["word_idx"].astype(str) + ("|" + df["phone_idx"].astype(str) if level == "phone" else "")
        wide = df.assign(_k=key).pivot(index="_k", columns="expert", values=col).dropna()
        grp = wide.index.str.split("|").str[0].map(utt_age)
        for g in ("adult", "child/teen"):
            sel = np.asarray(grp == g)
            rows.append({"task": head, "group": g, "n_items": int(sel.sum()), "human_loo_r": loo_r(wide.to_numpy()[sel])})
    return pd.DataFrame(rows)


def matched_weights(y, adult, child) -> tuple[np.ndarray, float]:
    """Row weights that make the label distribution of the adult rows equal to that of the child/teen rows (strata = distinct label values).
    Also returns the share of the child/teen label distribution that adults cover (1.0 = every child/teen label value occurs among adults)."""
    yr = np.round(np.asarray(y, float), 3)
    va, ia = np.unique(yr[adult], return_inverse=True)
    pa = np.bincount(ia) / len(ia)
    vc, ic = np.unique(yr[child], return_inverse=True)
    share = dict(zip(vc, np.bincount(ic) / len(ic)))
    target = np.array([share.get(v, 0.0) for v in va])
    w = np.zeros(len(yr))
    w[np.flatnonzero(adult)] = (target / pa)[ia]
    return w, float(target.sum())


def matched_compare(task, y, p, speaker, adult, child, n_boot, seed) -> dict:
    """PCC and slope: adult, child/teen, and adult with the child/teen label distribution; gaps with speaker-bootstrap CIs."""
    cell = S.Cell(task, y, p, speaker)
    w, cov = matched_weights(y, adult, child)
    sa, sc, sm = cell.stats(adult), cell.stats(child), cell.stats(adult, wt=w)
    Wa, Wc = S.weights(len(sa), n_boot, seed), S.weights(len(sc), n_boot, seed + 1)
    out = {"coverage": cov}
    for key in ("pcc", "slope"):
        va, vc, vm = (float(S.metrics(task, s.sum(0))[key]) for s in (sa, sc, sm))
        da, dc, dm = (S.metrics(task, W @ s)[key] for W, s in ((Wa, sa), (Wc, sc), (Wa, sm)))
        out[key] = {"adult": va, "child": vc, "adult_matched": vm, "raw_gap": va - vc, "raw_ci": S.interval(da - dc), "matched_gap": vm - vc,
                    "matched_ci": S.interval(dm - dc), "matched_p": S.pvalue(dm - dc)}
    return out


def check1(n_boot: int, seed: int, ceilings: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    ceil = {} if ceilings is None else {(r.task, r.group): r.human_loo_r for r in ceilings.itertuples()}
    for task in REG:
        frame = S.test_frame(task)
        m = S.group_masks(frame)["age"]
        for sysn in CHECK_SYSTEMS:
            b = S.load_bundle(sysn, task)
            if b is None:
                continue
            S.check_bundle(task, frame, b)
            r = matched_compare(task, b["y"], b["pred"], b["speaker"], m["adult"], m["child/teen"], n_boot, seed)
            ra, rc = consensus_reliability(ceil[(task, "adult")]) if (task, "adult") in ceil else np.nan, consensus_reliability(ceil[(task, "child/teen")]) if (task, "child/teen") in ceil else np.nan
            for key in ("pcc", "slope"):
                d = r[key]
                rows.append({"label_reliability_adult": ra, "label_reliability_child": rc,
                             "slope_ratio_expected_from_label_noise": rc / ra if key == "slope" else np.nan,
                             "slope_ratio_observed": d["child"] / d["adult_matched"] if key == "slope" else np.nan, "task": task, "system": sysn, "metric": key, "adult": d["adult"], "child": d["child"], "adult_matched": d["adult_matched"],
                             "raw_gap": d["raw_gap"], "raw_lo": d["raw_ci"][0], "raw_hi": d["raw_ci"][1], "matched_gap": d["matched_gap"],
                             "matched_lo": d["matched_ci"][0], "matched_hi": d["matched_ci"][1], "matched_p": d["matched_p"],
                             "share_left_after_matching": d["matched_gap"] / d["raw_gap"] if d["raw_gap"] else np.nan, "label_coverage": r["coverage"],
                             "human_ceiling_adult": ceil.get((task, "adult"), np.nan), "human_ceiling_child": ceil.get((task, "child/teen"), np.nan)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# check 2: prior shift or ranking (mispronunciation)
# --------------------------------------------------------------------------- #
def auc(y, s) -> float:
    """Rank-based AUC with average ranks for ties."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ss = s[np.argsort(s, kind="mergesort")]
    ys = y[np.argsort(s, kind="mergesort")]
    _, start, cnt = np.unique(ss, return_index=True, return_counts=True)
    ranks = np.repeat(start + (cnt + 1) / 2.0, cnt)
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    ys = y[np.argsort(-s, kind="mergesort")]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(prec[ys == 1].mean())


def recall_at_fpr(y, s, fpr: float) -> float:
    """Recall when the threshold is set (on this group's own scores) so that at most ``fpr`` of the negatives are flagged."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = neg.max() if fpr <= 0 else np.quantile(neg, 1.0 - fpr)
    return float((pos > thr).mean())


def op_metrics(y, s, pred, rows_a, rows_c) -> dict[str, float]:
    ya, sa, pa, yc, sc, pc = y[rows_a], s[rows_a], pred[rows_a], y[rows_c], s[rows_c], pred[rows_c]
    with np.errstate(all="ignore"):
        fpr_a = float((pa[ya == 0] == 1).mean())
        rec_a, rec_c = float((pa[ya == 1] == 1).mean()), float((pc[yc == 1] == 1).mean())
        rec_cm = recall_at_fpr(yc, sc, fpr_a)
        return {"auc": (auc(ya, sa), auc(yc, sc)), "ap": (average_precision(ya, sa), average_precision(yc, sc)), "positive_rate": (float(ya.mean()), float(yc.mean())),
                "fpr_at_pooled_threshold": (fpr_a, float((pc[yc == 0] == 1).mean())), "recall_at_pooled_threshold": (rec_a, rec_c),
                "recall_child_at_adult_fpr": (rec_a, rec_cm), "ranking_part": (rec_a - rec_cm, 0.0), "calibration_part": (rec_cm - rec_c, 0.0),
                "total_recall_gap": (rec_a - rec_c, 0.0)}


def _spk_rows(speaker, mask) -> list[np.ndarray]:
    rows = np.flatnonzero(mask)
    _, inv = np.unique(np.asarray(speaker)[mask], return_inverse=True)
    return [rows[inv == k] for k in range(int(inv.max()) + 1)]


def check2(n_boot: int, seed: int, proba_dir=None) -> pd.DataFrame:
    proba_dir = proba_dir or PROBA
    frame = S.test_frame(TARGET)
    m = S.group_masks(frame)["age"]
    rows = []
    for sysn in PROBA_SYSTEMS:
        f = proba_dir / f"{sysn}__{TARGET}.npz"
        b = S.load_bundle(sysn, TARGET)
        if not f.exists() or b is None:
            print(f"[us82] check 2: no probabilities for {sysn} ({f.name}); skipped")
            continue
        z = np.load(f, allow_pickle=False)
        S.check_bundle(TARGET, frame, b)
        if not (np.array_equal(z["speaker"].astype(str), b["speaker"]) and np.array_equal(z["y"], b["y"])):
            raise ValueError(f"{sysn}: probability rows do not match the bundle rows")
        y, s, pred = np.asarray(b["y"], int), np.asarray(z["proba"], float), np.asarray(b["pred"], int)
        ga, gc = _spk_rows(b["speaker"], m["adult"]), _spk_rows(b["speaker"], m["child/teen"])
        point = op_metrics(y, s, pred, np.concatenate(ga), np.concatenate(gc))
        ra, rc = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        draws = {k: [] for k in point}
        for _ in range(n_boot):
            da = np.concatenate([ga[i] for i in ra.integers(0, len(ga), len(ga))])
            dc = np.concatenate([gc[i] for i in rc.integers(0, len(gc), len(gc))])
            for k, v in op_metrics(y, s, pred, da, dc).items():
                draws[k].append(v[1] - v[0] if k in DIFF_KEYS else v[0])
        for k, (va, vc) in point.items():
            dr = np.array(draws[k], float)
            diff = (vc - va) if k in DIFF_KEYS else va
            lo, hi = S.interval(dr)
            rows.append({"system": sysn, "measure": k, "adult": va if k not in ("ranking_part", "calibration_part", "total_recall_gap") else np.nan,
                         "child": vc if k not in ("ranking_part", "calibration_part", "total_recall_gap") else np.nan, "child_minus_adult_or_part": diff,
                         "lo": lo, "hi": hi, "p": S.pvalue(dr)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# check 3: bias and compression by group x true-score bin (descriptive)
# --------------------------------------------------------------------------- #
def check3() -> pd.DataFrame:
    rows = []
    for task in REG:
        frame = S.test_frame(task)
        masks = S.group_masks(frame)
        for sysn in CHECK_SYSTEMS:
            b = S.load_bundle(sysn, task)
            if b is None:
                continue
            S.check_bundle(task, frame, b)
            cell = S.Cell(task, b["y"], b["pred"], b["speaker"])
            for g, gm in masks["age"].items():
                for bn, bm in masks["score_bin"].items():
                    mk = gm & bm
                    if not mk.any():
                        continue
                    st = cell.stats(mk)
                    mt = S.metrics(task, st.sum(0))
                    rows.append({"task": task, "system": sysn, "group": g, "score_bin": bn, "n_rows": int(mk.sum()), "n_speakers": len(st),
                                 "share_of_group": float(mk.sum() / gm.sum()), "bias": float(mt["bias"]), "rmse": float(mt["rmse"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ablation table
# --------------------------------------------------------------------------- #
def ablation(n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for task in R71.SO762:
        frame = S.test_frame(task)
        m = S.group_masks(frame)["age"]
        cells = {}
        for sysn in S.SYSTEMS:
            b = S.load_bundle(sysn, task)
            if b is not None:
                S.check_bundle(task, frame, b)
                cells[sysn] = S.Cell(task, b["y"], b["pred"], b["speaker"])
        key = R71.metric_of(task)
        sA, sC = {s: c.stats(m["adult"]) for s, c in cells.items()}, {s: c.stats(m["child/teen"]) for s, c in cells.items()}
        nA, nC = len(next(iter(sA.values()))), len(next(iter(sC.values())))
        Wa, Wc = S.weights(nA, n_boot, seed), S.weights(nC, n_boot, seed + 1)
        gaps = {s: S.gap_draws(task, sA[s], sC[s], Wa, Wc) for s in cells}
        for a, b, label in CONTRASTS:
            if a not in cells or b not in cells:
                continue
            da, dc = S.delta_ci(task, key, sA[a], sA[b], Wa), S.delta_ci(task, key, sC[a], sC[b], Wc)
            dg = gaps[b][1] - gaps[a][1]
            lo, hi = S.interval(dg)
            rows.append({"task": task, "contrast": f"{a} -> {b}", "cause": label, "metric": key, "adult_delta": da["delta"], "adult_lo": da["lo"], "adult_hi": da["hi"],
                         "child_delta": dc["delta"], "child_lo": dc["lo"], "child_hi": dc["hi"], "gap_a": gaps[a][0], "gap_b": gaps[b][0],
                         "delta_gap": gaps[b][0] - gaps[a][0], "gap_lo": lo, "gap_hi": hi, "p_raw": S.pvalue(dg),
                         "family": "B" if (a, b) in S.FAMILY_B else "exploratory"})
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    sel = df["family"] == "B"
    if sel.any():
        df.loc[sel, "p_holm"] = S.holm(df.loc[sel, "p_raw"].to_numpy())
    return df


def ablation_l2(n_boot: int, seed: int) -> pd.DataFrame:
    """L2-ARCTIC heads: the contrasts pooled over L1 and inside each L1 (exploratory; 4 speakers per L1)."""
    rows = []
    for task in R71.L2:
        frame = S.test_frame(task)
        masks = S.group_masks(frame)
        cells = {}
        for sysn in S.SYSTEMS:
            b = S.load_bundle(sysn, task)
            if b is not None:
                S.check_bundle(task, frame, b)
                cells[sysn] = S.Cell(task, b["y"], b["pred"], b["speaker"])
        key = R71.metric_of(task)
        for fam, g in [("all", "all")] + [("l1", x) for x in masks["l1"]]:
            mask = masks[fam][g]
            st = {s: c.stats(mask) for s, c in cells.items()}
            W = S.weights(len(next(iter(st.values()))), n_boot, seed)
            for a, b, label in CONTRASTS_L2:
                if a in st and b in st:
                    rows.append({"task": task, "group": g, "contrast": f"{a} -> {b}", "cause": label, "metric": key, **S.delta_ci(task, key, st[a], st[b], W)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--skip-ceiling", action="store_true", help="skip the per-group human ceiling (loads scores-detail.json)")
    a = ap.parse_args(argv)
    nb0, seed = S.boot_settings()
    nb = a.n_boot or nb0
    ceil = None if a.skip_ceiling else expert_ceilings()
    OUT.mkdir(parents=True, exist_ok=True)
    if ceil is not None:
        ceil.to_csv(OUT / "us82_human_ceiling.csv", index=False)
    pd.set_option("display.width", 220)
    c1 = check1(nb, seed, ceil)
    c1.to_csv(OUT / "us82_check1_matched.csv", index=False)
    print(c1[c1["system"] == "MT-GN15"].round(3).to_string(index=False))
    c2 = check2(nb, seed)
    if len(c2):
        c2.to_csv(OUT / "us82_check2_ranking.csv", index=False)
        print(c2.round(3).to_string(index=False))
    check3().to_csv(OUT / "us82_check3_bins.csv", index=False)
    ab = ablation(nb, seed)
    ab.to_csv(OUT / "us82_ablation.csv", index=False)
    ablation_l2(nb, seed).to_csv(OUT / "us82_ablation_l2.csv", index=False)
    print(f"[us82] wrote {OUT}")


if __name__ == "__main__":
    main()
