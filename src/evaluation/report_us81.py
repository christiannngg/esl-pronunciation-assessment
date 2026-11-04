"""US 8.1: subgroup generalization of the carried systems (numpy / pandas only; reads saved predictions, trains nothing).

Inputs:  results/analysis/us71/<system>__<task>.npz  (y, pred, speaker; test rows)  +  data/processed/features/tables/*_meta.csv.gz  (subgroup labels).
         The test metadata is rebuilt in bundle order (SO762: split == test in table order; L2-ARCTIC: test rows of folds 0..3) and every bundle is checked
         against it (same speakers, same labels) before it is used.
Outputs: results/tables/generalization/us81_*.csv  and  us81_slice_definitions.json.

    python -m src.evaluation.report_us81                 # all systems that have bundles, n_boot / seed from configs/baseline.yaml
    python -m src.evaluation.report_us81 --n-boot 200    # quicker

Design is fixed in docs/sprint8_plan.md (sections 3 and 4):
  * Family A (Holm over 7 tests): the age gap of MT-GN15 on the 7 SO762 heads.
  * Family B (Holm over 21 tests): change of the age gap for B -> F, F -> FT-base, FT-base -> MT-GN15 on the 7 SO762 heads.
  * Everything else (L1, gender, length, score bin, proficiency proxy, other systems, other contrasts) is exploratory.
The age gap is reported as an *adverse* gap = error(child/teen) - error(adult) with error = RMSE or 1 - macro-F1, so a positive value always means child/teen
speakers are served worse. Speaker-level bootstrap (percentile CI, p = 2 min(P(d <= 0), P(d >= 0)), as report_us43 / metrics.paired_speaker_bootstrap);
paired comparisons use the same speaker draws for both systems, group gaps resample the two groups independently.
AUC / AP need probabilities, which the bundles do not carry (hard predictions only): they belong to US 8.2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import report_us71 as R71

REPO = R71.REPO
BUNDLES = REPO / "results" / "analysis" / "us71"
FEATURES = REPO / "data" / "processed" / "features" / "tables"
SPLITS = REPO / "data" / "splits"
OUT = REPO / "results" / "tables" / "generalization"

SYSTEMS = ["B", "F", "FT-base", "FT-stab", "MT-full", "MT-GN15"]
ERROR_CLASSES = ["correct", "substitution", "deletion", "addition"]
SPEC = {                                   # task: (meta table, target column, encoding of the target in the bundles)
    "sent_accuracy": ("so762_sentence", "y_accuracy", None), "sent_fluency": ("so762_sentence", "y_fluency", None),
    "sent_prosodic": ("so762_sentence", "y_prosodic", None), "sent_total": ("so762_sentence", "y_total", None),
    "word_accuracy": ("so762_word", "y_word_accuracy", None), "phone_score": ("so762_phone", "y_phone_score", None),
    "phone_mispron_t1.0": ("so762_phone", "y_phone_score", 1.0),
    "l2_phone_error": ("l2_phone", "error_class", "ec"), "l2_word_error": ("l2_word", "y_has_error", "bin"),
}
N_CLASSES = {"phone_mispron_t1.0": 2, "l2_word_error": 2, "l2_phone_error": 4}
PAIRS = {"so762": [("B", "FT-base"), ("B", "MT-GN15"), ("F", "FT-base"), ("FT-base", "MT-GN15")],
         "l2": [("B", "FT-stab"), ("B", "MT-full"), ("F", "FT-stab"), ("FT-stab", "MT-full")]}
FAMILY_B = [("B", "F"), ("F", "FT-base"), ("FT-base", "MT-GN15")]                 # pre-declared
EXPLORATORY_B = [("FT-base", "MT-full"), ("FT-base", "FT-stab")]
SCORE_BINS = {"sentence": ([6.5, 8.5], ["<=6", "7-8", "9-10"]), "word_accuracy": ([4.5, 9.5], ["0-4", "5-9", "10"]),
              "phone_score": ([0.99, 1.99], ["<1.0", "1.0-1.9", "2.0"])}
LENGTH_LABELS = ["short", "medium", "long"]
PROF_LABELS = ["low", "mid", "high"]


def boot_settings() -> tuple[int, int]:
    try:
        import yaml
        b = yaml.safe_load((REPO / "configs" / "baseline.yaml").read_text())["bootstrap"]
        return int(b["n_boot"]), int(b["seed"])
    except Exception:                                                               # pragma: no cover - only without pyyaml
        return 500, 0


# --------------------------------------------------------------------------- #
# test metadata, rebuilt in bundle order
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def _read_meta(table: str) -> pd.DataFrame:
    key = (str(FEATURES), table)
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(FEATURES / f"{table}_meta.csv.gz", dtype={"utt_id": str, "speaker": str})
    return _CACHE[key]


def _so762_split(meta: pd.DataFrame) -> np.ndarray:
    sp = pd.read_csv(SPLITS / "so762_speakers.csv", dtype={"speaker": str}).set_index("speaker")["split"]
    s = meta["speaker"].map(sp)
    assert s.notna().all(), "speaker without a split"
    return s.to_numpy()


def test_rows(table: str) -> np.ndarray:
    m = _read_meta(table)
    if table.startswith("so762"):
        return np.flatnonzero(_so762_split(m) == "test")
    f = m["fold"].to_numpy()
    return np.concatenate([np.flatnonzero(f == k) for k in range(4)])


def utterance_durations(dataset: str) -> pd.Series:
    """Duration per utterance id: SO762 from the sentence side features, L2-ARCTIC as the sum of the annotated phone durations."""
    if dataset == "so762":
        m = _read_meta("so762_sentence")
        names = json.loads((FEATURES / "so762_sentence_side_names.json").read_text())
        X = np.load(FEATURES / "so762_sentence_side_X.npy", mmap_mode="r")
        assert len(X) == len(m), "side features and sentence meta differ in length"
        return pd.Series(np.asarray(X[:, names.index("dur")], float), index=m["utt_id"].to_numpy())
    return _read_meta("l2_phone").groupby("utt_id")["ann_dur"].sum()


def speaker_proficiency(dataset: str) -> pd.Series:
    """Accent-strength proxy per speaker, higher = more proficient. SO762: mean human sentence accuracy; L2-ARCTIC: minus the phone error rate."""
    if dataset == "so762":
        m = _read_meta("so762_sentence")
        t = m[_so762_split(m) == "test"]
        return t.groupby("speaker")["y_accuracy"].mean()
    m = _read_meta("l2_phone")
    return -(m["error_class"] != "correct").groupby(m["speaker"]).mean()


def terciles(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 3, labels=PROF_LABELS).astype(str)


def encode_truth(raw: np.ndarray, mode) -> np.ndarray:
    if mode is None:
        return np.asarray(raw, float)
    if mode == "ec":
        return np.array([ERROR_CLASSES.index(v) for v in raw])
    if mode == "bin":
        return np.asarray(raw).astype(int)
    return (np.asarray(raw, float) < mode).astype(int)


def score_bins_of(task: str) -> tuple[list[float], list[str]]:
    return SCORE_BINS["sentence" if task.startswith("sent_") else task]


def score_bin(task: str, y: np.ndarray) -> np.ndarray:
    edges, labels = score_bins_of(task)
    return np.array(labels)[np.digitize(y, edges)]


def test_frame(task: str) -> pd.DataFrame:
    """One row per test row in bundle order: speaker, gender, utt_id, y_enc, age | l1, length, prof, (score_bin for regression)."""
    table, col, mode = SPEC[task]
    dataset = "so762" if table.startswith("so762") else "l2"
    d = _read_meta(table).iloc[test_rows(table)].reset_index(drop=True)
    f = pd.DataFrame({"speaker": d["speaker"].to_numpy(), "gender": d["gender"].astype(str).str.lower().to_numpy(), "utt_id": d["utt_id"].to_numpy()})
    raw = d[col].to_numpy()
    f["y_enc"] = encode_truth(raw, mode)
    if dataset == "so762":
        f["age"] = d["age_group"].to_numpy()
    else:
        f["l1"] = d["l1"].to_numpy()
    dur = utterance_durations(dataset)
    uniq = pd.unique(f["utt_id"])
    miss = set(uniq) - set(dur.index)
    assert not miss, f"{task}: {len(miss)} test utterances without a duration"
    cuts = np.quantile(dur.reindex(uniq).to_numpy(), [1 / 3, 2 / 3])
    f["length"] = np.array(LENGTH_LABELS)[np.digitize(dur.reindex(f["utt_id"]).to_numpy(), cuts)]
    f["prof"] = terciles(speaker_proficiency(dataset)).reindex(f["speaker"]).to_numpy()
    assert f["prof"].notna().all(), f"{task}: speaker without a proficiency tercile"
    if N_CLASSES.get(task) is None:
        f["score_bin"] = score_bin(task, np.asarray(raw, float))
    f.attrs["length_cuts"] = [float(c) for c in cuts]
    f.attrs["score_labels"] = score_bins_of(task)[1] if "score_bin" in f else None
    return f


def group_masks(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    fams = {"all": {"all": np.ones(len(frame), bool)}}
    if "age" in frame:
        fams["age"] = {g: (frame["age"] == g).to_numpy() for g in ["adult", "child/teen"]}
    if "l1" in frame:
        fams["l1"] = {g: (frame["l1"] == g).to_numpy() for g in sorted(frame["l1"].unique())}
    fams["gender"] = {g: (frame["gender"] == g).to_numpy() for g in sorted(frame["gender"].unique())}
    fams["length"] = {g: (frame["length"] == g).to_numpy() for g in LENGTH_LABELS}
    fams["proficiency"] = {g: (frame["prof"] == g).to_numpy() for g in PROF_LABELS}
    if "score_bin" in frame:
        fams["score_bin"] = {g: (frame["score_bin"] == g).to_numpy() for g in frame.attrs["score_labels"]}
    return fams


def load_bundle(system: str, task: str) -> dict | None:
    p = BUNDLES / f"{system}__{task}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {"y": z["y"], "pred": z["pred"], "speaker": z["speaker"].astype(str)}


def check_bundle(task: str, frame: pd.DataFrame, b: dict) -> None:
    ok = (len(b["y"]) == len(frame) and np.array_equal(b["speaker"], frame["speaker"].to_numpy())
          and np.allclose(np.asarray(b["y"], float), frame["y_enc"].to_numpy(float)))
    if not ok:
        raise ValueError(f"{task}: bundle rows do not match the rebuilt test manifest")


# --------------------------------------------------------------------------- #
# metrics from per-speaker sufficient statistics (so a bootstrap draw is one matrix product)
# --------------------------------------------------------------------------- #
def reg_stats(y, p, codes, n_spk, wt=None) -> np.ndarray:
    """Per-speaker sums n, sy, sp, syy, spp, syp; ``wt`` (frequency weights per row) turns them into weighted sums, the metrics formulas are unchanged."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(y) if wt is None else np.asarray(wt, float)
    cols = [w, w * y, w * p, w * y * y, w * p * p, w * y * p]
    return np.stack([np.bincount(codes, weights=c, minlength=n_spk) for c in cols], axis=1)


def clf_stats(y, p, codes, n_spk, k) -> np.ndarray:
    y, p = np.asarray(y, int), np.asarray(p, int)
    return np.bincount(codes * k * k + y * k + p, minlength=n_spk * k * k).reshape(n_spk, k * k).astype(float)


def reg_metrics(s) -> dict[str, np.ndarray]:
    s = np.asarray(s, float)
    n = s[..., 0]
    with np.errstate(all="ignore"):
        my, mp = s[..., 1] / n, s[..., 2] / n
        vy, vp = s[..., 3] / n - my ** 2, s[..., 4] / n - mp ** 2
        cov = s[..., 5] / n - my * mp
        mse = (s[..., 4] - 2 * s[..., 5] + s[..., 3]) / n
        oy, op = vy > 1e-12, vp > 1e-12
        pcc = np.where(oy & op, cov / np.sqrt(np.where(oy & op, vy * vp, 1.0)), np.nan)
        slope = np.where(oy, cov / np.where(oy, vy, 1.0), np.nan)
        return {"rmse": np.sqrt(np.maximum(mse, 0)), "pcc": pcc, "bias": mp - my, "slope": slope, "y_sd": np.sqrt(np.maximum(vy, 0))}


def clf_metrics(s, k) -> dict[str, np.ndarray]:
    """macro-F1 over the classes that occur in y (as metrics.fast_clf_factory); binary tasks add precision / recall / positive rate of class 1."""
    s = np.asarray(s, float)
    cm = s.reshape(s.shape[:-1] + (k, k))
    tp, sup, pred = np.einsum("...ii->...i", cm), cm.sum(-1), cm.sum(-2)
    with np.errstate(all="ignore"):
        rec = np.where(sup > 0, tp / np.where(sup > 0, sup, 1), 0.0)
        prec = np.where(pred > 0, tp / np.where(pred > 0, pred, 1), 0.0)
        den = prec + rec
        f1 = np.where(den > 0, 2 * prec * rec / np.where(den > 0, den, 1), 0.0)
    present = sup > 0
    out = {"macro_f1": (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)}
    if k == 2:
        out.update(precision=prec[..., 1], recall=rec[..., 1], pos_rate=sup[..., 1] / np.maximum(sup.sum(-1), 1))
    return out


def metrics(task: str, s) -> dict[str, np.ndarray]:
    return clf_metrics(s, N_CLASSES[task]) if task in N_CLASSES else reg_metrics(s)


def adverse(task: str, value):
    """Error on a lower-is-better scale: RMSE, or 1 - macro-F1."""
    return value if task not in N_CLASSES else 1.0 - np.asarray(value)


class Cell:
    """One system's test predictions on one task."""

    def __init__(self, task: str, y, pred, speaker):
        self.task, self.y, self.p = task, np.asarray(y), np.asarray(pred)
        _, self.codes = np.unique(np.asarray(speaker).astype(str), return_inverse=True)

    def stats(self, mask: np.ndarray, wt=None) -> np.ndarray:
        """Per-speaker statistics of the masked rows (speakers without a row in the mask are dropped), in sorted speaker order."""
        _, inv = np.unique(self.codes[mask], return_inverse=True)
        n = int(inv.max()) + 1 if len(inv) else 0
        if self.task in N_CLASSES:
            return clf_stats(self.y[mask], self.p[mask], inv, n, N_CLASSES[self.task])
        return reg_stats(self.y[mask], self.p[mask], inv, n, None if wt is None else np.asarray(wt)[mask])


# --------------------------------------------------------------------------- #
# bootstrap pieces
# --------------------------------------------------------------------------- #
def weights(n_spk: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_spk) counts of how often each speaker is drawn; identical for identical (n_spk, n_boot, seed), which is what pairs two systems."""
    idx = np.random.default_rng(seed).integers(0, n_spk, size=(n_boot, n_spk))
    w = np.zeros((n_boot, n_spk))
    np.add.at(w, (np.arange(n_boot)[:, None], idx), 1.0)
    return w


def interval(draws) -> tuple[float, float]:
    d = np.asarray(draws, float)
    return (float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))) if np.isfinite(d).any() else (np.nan, np.nan)


def pvalue(draws) -> float:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    return float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)) if len(d) else np.nan


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def point_ci(task: str, st: np.ndarray, w: np.ndarray) -> dict[str, tuple[float, float, float]]:
    pt, dr = metrics(task, st.sum(0)), metrics(task, w @ st)
    return {k: (float(pt[k]), *interval(dr[k])) for k in pt}


def delta_ci(task: str, key: str, st_a, st_b, w) -> dict:
    d = float(metrics(task, st_b.sum(0))[key] - metrics(task, st_a.sum(0))[key])
    dr = metrics(task, w @ st_b)[key] - metrics(task, w @ st_a)[key]
    lo, hi = interval(dr)
    return {"delta": d, "lo": lo, "hi": hi, "p": pvalue(dr)}


def gap_draws(task: str, st_adult, st_child, w_adult, w_child) -> tuple[float, np.ndarray, float, float]:
    """Adverse gap error(child/teen) - error(adult): point, bootstrap draws, child value, adult value (primary metric)."""
    key = R71.metric_of(task)
    vc, va = float(metrics(task, st_child.sum(0))[key]), float(metrics(task, st_adult.sum(0))[key])
    dr = adverse(task, metrics(task, w_child @ st_child)[key]) - adverse(task, metrics(task, w_adult @ st_adult)[key])
    return float(adverse(task, vc) - adverse(task, va)), dr, vc, va


def reading_gap(adv: float, lo: float, hi: float, p_adj: float) -> str:
    sig = (p_adj < 0.05) if np.isfinite(p_adj) else (lo > 0 or hi < 0)
    tag = "" if np.isfinite(p_adj) else " (unadjusted)"
    return ("no significant gap" if not sig else ("child/teen worse" if adv > 0 else "child/teen better")) + tag


def reading_change(d: float, lo: float, hi: float, p_adj: float) -> str:
    sig = (p_adj < 0.05) if np.isfinite(p_adj) else (lo > 0 or hi < 0)
    tag = "" if np.isfinite(p_adj) else " (unadjusted)"
    return ("no significant change" if not sig else ("gap wider" if d > 0 else "gap narrower")) + tag


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(systems=SYSTEMS, n_boot: int | None = None, seed: int | None = None, tasks=None) -> dict[str, pd.DataFrame]:
    nb0, sd0 = boot_settings()
    nb, sd = n_boot or nb0, sd0 if seed is None else seed
    slices, pairs, gaps, changes, spk_rows, defs = [], [], [], [], [], {}
    gap_store: dict[tuple[str, str], tuple[float, np.ndarray]] = {}
    for task in tasks or R71.TASKS:
        frame = test_frame(task)
        masks = group_masks(frame)
        group_of = {"so762": "age", "l2": "l1"}[R71.GROUP[task]]
        defs[task] = {"length_cuts": frame.attrs["length_cuts"]}
        cells = {}
        for s in systems:
            b = load_bundle(s, task)
            if b is not None:
                check_bundle(task, frame, b)
                cells[s] = Cell(task, b["y"], b["pred"], b["speaker"])
        if not cells:
            continue
        key = R71.metric_of(task)
        for fam, groups in masks.items():
            for g, mask in groups.items():
                if not mask.any():
                    continue
                st = {s: c.stats(mask) for s, c in cells.items()}
                n_sel = len(next(iter(st.values())))
                w = weights(n_sel, nb, sd)
                for s, a in st.items():
                    for m, (v, lo, hi) in point_ci(task, a, w).items():
                        slices.append({"task": task, "system": s, "family": fam, "group": g, "n_rows": int(mask.sum()), "n_speakers": n_sel, "metric": m,
                                       "value": v, "lo": lo, "hi": hi})
                for a, b in PAIRS[R71.GROUP[task]]:
                    if a in st and b in st:
                        r = delta_ci(task, key, st[a], st[b], w)
                        pairs.append({"task": task, "family": fam, "group": g, "a": a, "b": b, "metric": key, "n_speakers": n_sel, **r,
                                      "verdict": R71.verdict(task, r["delta"], r["lo"], r["hi"])})
        mask_all = masks["all"]["all"]
        for s, c in cells.items():                                              # per-speaker values: the L1 groups have only 4 speakers each
            st = c.stats(mask_all)
            spk = np.unique(frame["speaker"].to_numpy())
            grp = frame.groupby("speaker")[group_of].first().reindex(spk).to_numpy()
            nrow = frame.groupby("speaker").size().reindex(spk).to_numpy()
            val = metrics(task, st)[key]
            for i in range(len(spk)):
                spk_rows.append({"task": task, "system": s, "speaker": spk[i], "group": grp[i], "n_rows": int(nrow[i]), "metric": key, "value": float(val[i])})
        if R71.GROUP[task] != "so762":
            continue
        adult, child = masks["age"]["adult"], masks["age"]["child/teen"]
        stA = {s: c.stats(adult) for s, c in cells.items()}
        stC = {s: c.stats(child) for s, c in cells.items()}
        nA, nC = len(next(iter(stA.values()))), len(next(iter(stC.values())))
        wA, wC = weights(nA, nb, sd), weights(nC, nb, sd + 1)
        for s in cells:
            adv, dr, vc, va = gap_draws(task, stA[s], stC[s], wA, wC)
            gap_store[(task, s)] = (adv, dr)
            lo, hi = interval(dr)
            gaps.append({"task": task, "system": s, "metric": key, "child_value": vc, "adult_value": va, "adverse_gap": adv, "lo": lo, "hi": hi, "p_raw": pvalue(dr),
                         "family": "A" if s == "MT-GN15" else "exploratory"})
        for a, b in FAMILY_B + EXPLORATORY_B:
            if (task, a) in gap_store and (task, b) in gap_store:
                (pa, da), (pb, db) = gap_store[(task, a)], gap_store[(task, b)]
                lo, hi = interval(db - da)
                changes.append({"task": task, "a": a, "b": b, "metric": key, "gap_a": pa, "gap_b": pb, "delta_gap": pb - pa, "lo": lo, "hi": hi, "p_raw": pvalue(db - da),
                                "family": "B" if (a, b) in FAMILY_B else "exploratory"})
    g, c = pd.DataFrame(gaps), pd.DataFrame(changes)
    for df, fam, col in [(g, "A", "adverse_gap"), (c, "B", "delta_gap")]:
        df["p_holm"] = np.nan
        sel = df["family"] == fam
        if sel.any():
            df.loc[sel, "p_holm"] = holm(df.loc[sel, "p_raw"].to_numpy())
    if len(g):
        g["reading"] = [reading_gap(r.adverse_gap, r.lo, r.hi, r.p_holm) for r in g.itertuples()]
    if len(c):
        c["reading"] = [reading_change(r.delta_gap, r.lo, r.hi, r.p_holm) for r in c.itertuples()]
    return {"slices": pd.DataFrame(slices), "pairs": pd.DataFrame(pairs), "age_gap": g, "gap_change": c, "speakers": pd.DataFrame(spk_rows), "defs": defs}


def sanity(slices: pd.DataFrame, tol: float = 5e-4) -> list[str]:
    """Overall metric recomputed from the bundles must equal the US 7.1 table (the numbers the whole report rests on)."""
    long, *_ = R71.load_systems()
    exp = {(r.system, r.task): float(r.test) for r in long.itertuples()}
    out = []
    for r in slices[slices["family"] == "all"].itertuples():
        if r.metric != R71.metric_of(r.task) or (r.system, r.task) not in exp:
            continue
        if abs(r.value - exp[(r.system, r.task)]) > tol:
            out.append(f"{r.system}/{r.task}: bundles give {r.value:.4f}, US 7.1 table {exp[(r.system, r.task)]:.4f}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--systems", nargs="*", default=SYSTEMS)
    ap.add_argument("--no-sanity", action="store_true")
    a = ap.parse_args(argv)
    res = build(a.systems, a.n_boot)
    if not a.no_sanity:
        bad = sanity(res["slices"])
        if bad:
            print("[us81] SANITY CHECK FAILED, nothing written:\n  " + "\n  ".join(bad))
            raise SystemExit(1)
        print("[us81] sanity check passed: overall metrics from the bundles equal the US 7.1 table")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("slices", "pairs", "age_gap", "gap_change", "speakers"):
        res[name].to_csv(OUT / f"us81_{name}.csv", index=False)
    (OUT / "us81_slice_definitions.json").write_text(json.dumps({
        "length": "terciles of utterance duration over the unique test utterances of each dataset (cuts per task below)",
        "proficiency": "terciles of test speakers: SO762 by mean human sentence accuracy, L2-ARCTIC by minus the phone error rate (label-derived proxy, descriptive only)",
        "score_bin": {k: {"edges": v[0], "labels": v[1]} for k, v in SCORE_BINS.items()}, "tasks": res["defs"]}, indent=2))
    pd.set_option("display.width", 200)
    print(res["age_gap"][res["age_gap"]["family"] == "A"].round(3).to_string(index=False))
    print(res["gap_change"][res["gap_change"]["family"] == "B"].round(3).to_string(index=False))
    print(f"[us81] wrote {OUT}")


if __name__ == "__main__":
    main()
