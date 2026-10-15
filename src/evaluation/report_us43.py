"""
US 4.3 - updated frozen-encoder results and the evidence for each fix (Sprint 4).

    python -m src.evaluation.report_us43 --encoders wavlm_base_plus hubert_base wav2vec2_base
    python -m src.evaluation.report_us43 --encoders wavlm_base_plus --frozen-root experiments/results/frozen_outputs/frozen_results \\
                                         --fix-root experiments/results/frozen_outputs/frozen43

Inputs: the US 4.1 / 4.2 head results (experiments/results/frozen/<encoder>/L<layer>/emb+side), the US 4.3 results written by
``python -m src.models.frozen_fix`` (experiments/results/frozen43/<encoder>/<recipe>) and the out-of-fold file of fix 5.

Systems compared per (encoder, task); each system = the validation-selected head / config / class-weight variant of that input recipe:
    "4.1 L12"            the US 4.1 benchmark (layer 12, mean pooling, emb+side)
    "L6" ... "L11"       fix 1, the layers that got the full head protocol
    "ws"                 fix 2, learned layer-weighted sum
    "sent_*" recipes     fix 3 (sentence tasks);  "word_*" recipes   fix 4 (word accuracy)
    "mispron-grp"        fix 5, SO762 mispronunciation with decision weights tuned per age group on out-of-fold train probabilities
Everything is selected on VALIDATION only; the test set is scored once for the selected systems. Deltas are paired speaker-level bootstraps
of the primary metric (RMSE for regression, macro-F1 for classification) on the identical test rows.

Outputs (results/tables/frozen/):
    us43_report.md               the readable report (headline table, fix effects, layer weights, layer choice, age-group table)
    us43_headline.csv            per encoder x task: 4.1 vs 4.3 vs US 3.1 / US 3.3 (test, CIs, deltas)
    us43_fix_effects.csv         every control-vs-fix comparison with paired bootstrap
    us43_candidates.csv          every system x head x variant with validation and test scores (screening, coarse CIs)
    us43_layer_weights.csv       learned layer weights of the ws system per task
    us43_mispron_groups.csv      fix 5: strategy x age group on validation and test
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import metrics as M
from src.evaluation import report_frozen as RF
from src.evaluation.report_us33 import fit_weights
from src.models import frozen_fix as FF
from src.models import frozen_heads as FH
from src.models.baselines import build_tasks, load_config as load_baseline_config

warnings.filterwarnings("ignore")
OUT = RF.OUT
RECIPES = ["ws", "sent_ws", "sent_ws_pros", "sent_avgpca", "sent_avgpca_pros", "word_mean_pros", "word_max_pros", "word_meanmax_pros",
           "word_meanmaxstd_pros"]
LAYER_SYSTEMS = [f"L{l}" for l in range(6, 12)]
SENT = ["sent_accuracy", "sent_fluency", "sent_prosodic", "sent_total"]
WORD_POOL = ["word_max_pros", "word_meanmax_pros", "word_meanmaxstd_pros"]
MISPRON = "phone_mispron_t1.0"
MIN_POS_FOR_GROUP_WEIGHTS = 20


# --------------------------------------------------------------------------- #
# collecting the systems
# --------------------------------------------------------------------------- #
def collect_systems(enc: str, frozen_root=None, fix_root=None, recipes=RECIPES, n_boot: int = 20):
    """({system: {(task, head): headline dict}}, long candidate table) for one encoder. Systems without results are skipped."""
    cfg = FH.load_config()
    systems, longs = {}, []

    def add(name, df, preds):
        if df.empty:
            return
        systems[name] = preds
        longs.append(df.assign(system=name))

    add("4.1 L12", *RF.evaluate(enc, 12, "emb+side", n_boot=n_boot, root=frozen_root))
    for sysname in LAYER_SYSTEMS:
        df, preds = RF.evaluate(enc, int(sysname[1:]), "emb+side", n_boot=n_boot, root=frozen_root)
        # layers 6-11 also hold the US 4.2 single-config probe files; only layers that got the full head protocol (a `final` stage) count
        preds = {k: v for k, v in preds.items() if v["stage"] == "final"}
        add(sysname, df[df["stage"] == "final"] if not df.empty else df, preds)
    for r in recipes:
        hcfg, spec = FF.recipe_cfg(cfg, r)
        base = FF.recipe_base(enc, r, fix_root)
        if base.exists():
            add(r, *RF.evaluate(enc, spec["layer"], r, n_boot=n_boot, hcfg=hcfg, base=base))
    return systems, (pd.concat(longs, ignore_index=True) if longs else pd.DataFrame())


def best_of(systems: dict, names, task_name: str, kind: str):
    """(system, head, preds) with the best VALIDATION score among ``names`` x heads, or None. Ties keep the earlier name."""
    best = None
    for s in names:
        for h in FH.HEADS:
            p = systems.get(s, {}).get((task_name, h))
            if p is None:
                continue
            if best is None or (p["val_primary"] < best[2]["val_primary"] if kind == "reg" else p["val_primary"] > best[2]["val_primary"]):
                best = (s, h, p)
    return best


def _enc(task, arr):
    return RF._enc(task, arr)


def pair(task, a: dict, b: dict, n_boot: int) -> dict:
    """Metric of a and b on the identical test rows and the paired speaker bootstrap of metric(b) - metric(a)."""
    y = _enc(task, a["y"])
    assert (y == _enc(task, b["y"])).all(), f"{task.name}: systems were scored on different test rows"
    assert (a["meta"]["speaker"].to_numpy() == b["meta"]["speaker"].to_numpy()).all()
    fast, key = RF._fast(task), "rmse" if task.kind == "reg" else "macro_f1"
    pa, pb = _enc(task, a["pred"]), _enc(task, b["pred"])
    t = M.paired_speaker_bootstrap(a["meta"]["speaker"].to_numpy(), y, pa, pb, fast, key, n_boot=n_boot)
    ma, mb = fast(y, pa), fast(y, pb)
    return {"metric": key, "a": ma[key], "b": mb[key], "delta": t["delta"], "lo": t["lo"], "hi": t["hi"], "p": t["p"],
            "a_aux": ma["pcc"] if task.kind == "reg" else ma["balanced_acc"], "b_aux": mb["pcc"] if task.kind == "reg" else mb["balanced_acc"]}


def verdict(kind: str, d: dict) -> str:
    sig = d["lo"] > 0 or d["hi"] < 0
    good = d["delta"] < 0 if kind == "reg" else d["delta"] > 0
    if not sig:
        return "no significant change"
    return "better" if good else "worse"


# --------------------------------------------------------------------------- #
# fix 5: decision weights per age group from out-of-fold train probabilities
# --------------------------------------------------------------------------- #
def _group_of(meta: pd.DataFrame) -> np.ndarray:
    return np.where(meta["age_group"].astype(str).str.lower().eq("adult"), 0, 1)       # 0 adult, 1 child/teen


def source_base(src: dict, enc: str, frozen_root=None, fix_root=None) -> Path:
    if src["kind"] == "ws":
        return FF.recipe_base(enc, "ws", fix_root)
    return FH.results_root(enc, int(src["tag"]), "emb+side", frozen_root)


def _binary_metrics(y, pred, proba1=None) -> dict:
    f = M.fast_clf_factory(2, 1)(np.asarray(y, int), np.asarray(pred, int))
    out = {"macro_f1": f["macro_f1"], "precision": f["precision"], "recall": f["recall"], "f1": f["f1"], "n": int(len(y)),
           "n_pos": int(np.sum(y)), "pred_pos": int(np.sum(pred))}
    if proba1 is not None and 0 < np.sum(y) < len(y):
        out["auc"], out["ap"] = float(roc_auc_score(y, proba1)), float(average_precision_score(y, proba1))
    return out


def _weights(task, proba, y, groups, per_group: bool) -> dict:
    """{'global': w} or {group: w}. Groups with fewer than MIN_POS_FOR_GROUP_WEIGHTS positives fall back to the global weights."""
    w_all = fit_weights(task, proba, y)
    if not per_group:
        return {0: w_all, 1: w_all}
    out = {}
    for g in (0, 1):
        m = groups == g
        out[g] = fit_weights(task, proba[m], y[m]) if y[m].sum() >= MIN_POS_FOR_GROUP_WEIGHTS and (1 - y[m]).sum() > 0 else w_all
    return out


def _apply(proba, groups, w) -> np.ndarray:
    pred = np.empty(len(proba), int)
    for g in (0, 1):
        m = groups == g
        pred[m] = (proba[m] * w[g]).argmax(1)
    return pred


def mispron_groups(enc: str, frozen_root=None, fix_root=None, tol: float | None = None):
    """Fix 5. Returns (table, {system name: preds}) or (None, None) when the out-of-fold file is missing.
    Systems: 'mispron-grp' (the strategy kept by the rule) and 'mispron-ctrl' (the SAME source head with the 4.1 / 3.3 cw+thr rule).

    Strategies, scored per age group (adult / child-teen) on validation and test:
      cw+thr         the US 3.3 / 4.1 rule: weights tuned on validation (cross-fitted for the validation estimate)
      global-oof     one weight vector tuned on the out-of-fold TRAIN probabilities (validation is untouched when scoring validation;
                     for test the pool is OOF-train + validation)
      group-oof      one weight vector per age group, same data
    Rule (declared in configs/frozen.yaml): group-oof is kept if pooled validation macro-F1 >= global-oof - tolerance."""
    cfg = FH.load_config()
    tol = cfg["fix43"]["mispron"]["keep_if_pooled_val_drop_at_most"] if tol is None else tol
    oof_path = FF.fix_root(cfg, fix_root) / enc / "mispron" / "oof.npz"
    if not oof_path.exists():
        return None, None
    oof = np.load(oof_path, allow_pickle=True)
    src = json.loads(str(oof["source"]))
    task = FF.mispron_task()
    y_all, meta = FH.load_labels(task)
    base, head, config = source_base(src, enc, frozen_root, fix_root), src["head"], src["config"]
    stage = "final" if all(FH.job_path(base, task.name, head, config, f, "final").exists() for f in task.folds) else "search"
    dv = RF.pooled(base, task, head, config, stage, "val", y_all, meta, proba=True)
    dt = RF.pooled(base, task, head, config, stage, "test", y_all, meta, proba=True)
    yv, yt = dv["y"].astype(int), dt["y"].astype(int)
    gv, gt = _group_of(dv["meta"]), _group_of(dt["meta"])
    go = _group_of(meta.iloc[oof["rows"]])
    po, yo = oof["proba"], oof["y"].astype(int)

    strategies = {}
    cwthr = RF.apply_variants(task, "balanced", dv, dt)[1]                                    # (variant, val_pred, test_pred, train_primary)
    strategies["cw+thr"] = (_enc(task, cwthr[1]), _enc(task, cwthr[2]))
    for name, per_group in (("global-oof", False), ("group-oof", True)):
        w_val = _weights(task, po, yo, go, per_group)                                           # train OOF only -> honest validation estimate
        pool_p, pool_y, pool_g = np.vstack([po, dv["proba"]]), np.concatenate([yo, yv]), np.concatenate([go, gv])
        w_test = _weights(task, pool_p, pool_y, pool_g, per_group)                              # train OOF + validation -> test
        strategies[name] = (_apply(dv["proba"], gv, w_val), _apply(dt["proba"], gt, w_test))
        strategies[name + "_w"] = (w_val, w_test)

    rows = []
    for name in ("cw+thr", "global-oof", "group-oof"):
        pv, pt = strategies[name]
        for split, y, p, g, proba in (("val", yv, pv, gv, dv["proba"][:, 1]), ("test", yt, pt, gt, dt["proba"][:, 1])):
            for label, m in (("all", np.ones(len(y), bool)), ("adult", g == 0), ("child/teen", g == 1)):
                rows.append({"encoder": enc, "strategy": name, "split": split, "group": label, "source": f"{src['kind']}:{src['tag']}/{head}",
                             **_binary_metrics(y[m], p[m], proba[m])})
    table = pd.DataFrame(rows)
    v = table[(table.split == "val") & (table.group == "all")].set_index("strategy")["macro_f1"]
    chosen = "group-oof" if v["group-oof"] >= v["global-oof"] - tol else "global-oof"
    table["chosen"] = table["strategy"].eq(chosen)
    labels = np.array(task.labels)
    common = {"y": dt["y"], "meta": dt["meta"], "config": config, "stage": stage, "base": base}
    systems = {"mispron-grp": {(task.name, head): {**common, "pred": labels[strategies[chosen][1]], "variant": chosen, "val_primary": float(v[chosen])}},
               "mispron-ctrl": {(task.name, head): {**common, "pred": labels[strategies["cw+thr"][1]], "variant": "cw+thr", "val_primary": float(v["cw+thr"])}}}
    return table, systems


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fix_comparisons(task_name: str, kind: str):
    """[(label, control systems, treatment systems)] for one task. Each side = best of its listed systems by validation."""
    layer_sys = ["4.1 L12"] + LAYER_SYSTEMS
    c = [("fix 1: layer re-selection (6-12) vs layer 12", ["4.1 L12"], layer_sys),
         ("fix 2: learned layer-weighted sum vs best single layer", layer_sys, ["ws"])]
    if task_name in SENT:
        c += [("fix 3: sentence regularisation (best of 4 recipes) vs ws", ["ws"], ["sent_ws", "sent_ws_pros", "sent_avgpca", "sent_avgpca_pros"]),
              ("fix 3a: + US 3.3 prosody columns (ws input)", ["sent_ws"], ["sent_ws_pros"]),
              ("fix 3b: + US 3.3 prosody columns (PCA input)", ["sent_avgpca"], ["sent_avgpca_pros"]),
              ("fix 3c: stronger regularisation alone (ws input)", ["ws"], ["sent_ws"]),
              ("fix 3d: PCA on the layer-averaged input vs regularised ws", ["sent_ws"], ["sent_avgpca"])]
    if task_name == "word_accuracy":
        c += [("fix 4a: prosody columns alone (mean pooling) vs ws", ["ws"], ["word_mean_pros"]),
              ("fix 4: max / mean+max / mean+max+std pooling vs mean pooling (same inputs)", ["word_mean_pros"], WORD_POOL)]
        c += [(f"fix 4 ({w[5:-5]}): pooling vs mean pooling", ["word_mean_pros"], [w]) for w in WORD_POOL]
    if task_name == MISPRON:
        c += [("fix 5: age-group decision weights vs the 4.1 / 3.3 cw+thr rule (same head)", ["mispron-ctrl"], ["mispron-grp"])]
    return c


def build(encoders, frozen_root=None, fix_root=None, n_boot: int | None = None, screen_boot: int = 20):
    bcfg = load_baseline_config()
    n_boot = n_boot or bcfg["bootstrap"]["n_boot"]
    tasks = build_tasks(bcfg)
    cand, headline, effects, weights, groups = [], [], [], [], []
    for enc in encoders:
        systems, long = collect_systems(enc, frozen_root, fix_root, n_boot=screen_boot)
        if "4.1 L12" not in systems:
            raise SystemExit(f"[us43] no US 4.1 layer-12 results for {enc}; pass --frozen-root (folder holding <encoder>/L12/emb+side)")
        gtab, gsys = mispron_groups(enc, frozen_root, fix_root)
        if gsys is not None:
            systems.update(gsys)
            groups.append(gtab)
        cand.append(long.assign(encoder=enc))
        for tname in FH.load_config()["heads"]["tasks"]:
            task = tasks[tname]
            b41 = best_of(systems, ["4.1 L12"], tname, task.kind)
            if b41 is None:
                continue
            allsys = [s for s in systems if s != "4.1 L12"] + ["4.1 L12"]
            b43 = best_of(systems, allsys, tname, task.kind)
            r31, r33 = RF.baseline_refs(task, bcfg)
            ref = {"US 3.1": {"y": r31["y"], "pred": r31["pred"], "meta": r31["meta"]}, "US 3.3": {"y": r33["y"], "pred": r33["pred"], "meta": r33["meta"]}}
            d41 = pair(task, b41[2], b43[2], n_boot)
            d33 = pair(task, ref["US 3.3"], b43[2], n_boot)
            d31 = pair(task, ref["US 3.1"], b43[2], n_boot)
            headline.append({"encoder": enc, "task": tname, "metric": d41["metric"],
                             "us31": d31["a"], "us33": d33["a"], "frozen_41": d41["a"], "frozen_41_detail": f"{b41[1]} ({b41[2]['variant']})",
                             "frozen_43": d41["b"], "frozen_43_system": b43[0], "frozen_43_detail": f"{b43[1]} ({b43[2]['variant']}) {b43[2]['config']}",
                             "val_43": b43[2]["val_primary"], "val_41": b41[2]["val_primary"],
                             "d_vs_41": d41["delta"], "d_vs_41_lo": d41["lo"], "d_vs_41_hi": d41["hi"], "d_vs_41_p": d41["p"],
                             "d_vs_33": d33["delta"], "d_vs_33_lo": d33["lo"], "d_vs_33_hi": d33["hi"], "d_vs_33_p": d33["p"],
                             "verdict_vs_41": verdict(task.kind, d41), "aux_41": d41["a_aux"], "aux_43": d41["b_aux"]})
            for label, ctrl, treat in fix_comparisons(tname, task.kind):
                c, t = best_of(systems, ctrl, tname, task.kind), best_of(systems, treat, tname, task.kind)
                if c is None or t is None:
                    continue
                d = pair(task, c[2], t[2], n_boot)
                effects.append({"encoder": enc, "task": tname, "fix": label, "control": f"{c[0]} / {c[1]} ({c[2]['variant']})",
                                "treatment": f"{t[0]} / {t[1]} ({t[2]['variant']})", "metric": d["metric"], "control_test": d["a"], "treatment_test": d["b"],
                                "delta": d["delta"], "lo": d["lo"], "hi": d["hi"], "p": d["p"], "verdict": verdict(task.kind, d),
                                "val_control": c[2]["val_primary"], "val_treatment": t[2]["val_primary"]})
            w = ws_weights(systems, tname, task.kind)
            if w is not None:
                weights.append({"encoder": enc, "task": tname, **w})
    return (pd.concat(cand, ignore_index=True) if cand else pd.DataFrame(), pd.DataFrame(headline), pd.DataFrame(effects), pd.DataFrame(weights),
            pd.concat(groups, ignore_index=True) if groups else pd.DataFrame())


def ws_weights(systems: dict, task_name: str, kind: str):
    """Mean learned layer weights (over folds) of the ws system's selected head for this task."""
    best = best_of(systems, ["ws"], task_name, kind)
    if best is None:
        return None
    _, head, p = best
    base, config, stage = p["base"], p["config"], p["stage"]
    ws = []
    for fold in range(4):
        f = FH.job_path(base, task_name, head, config, fold, stage)
        if f.exists():
            d = np.load(f, allow_pickle=True)
            if "ws_weights" in d.files:
                ws.append(d["ws_weights"])
    if not ws:
        return None
    w = np.mean(ws, axis=0)
    order = np.argsort(-w)
    ent = float(-(w * np.log(np.maximum(w, 1e-12))).sum() / np.log(len(w)))
    return {"head": head, "top1": int(order[0]), "w1": float(w[order[0]]), "top2": int(order[1]), "w2": float(w[order[1]]),
            "top3": int(order[2]), "w3": float(w[order[2]]), "norm_entropy": ent, "weights": " ".join(f"{x:.3f}" for x in w)}


def _fmt(v, nd=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def write_md(cand, headline, effects, weights, groups, path: Path, encoders) -> None:
    L = ["# US 4.3 - updated frozen-encoder results and fix evidence (test set; every choice made on validation)", "",
         "Metric: RMSE for regression (lower is better), macro-F1 for classification (higher is better). Delta = 4.3 minus reference on the identical test "
         "rows, paired speaker bootstrap (95% CI, two-sided p). `verdict` is 'better' / 'worse' only when the CI excludes 0.", ""]
    if not headline.empty:
        L += ["## 1. Updated frozen-encoder results (per encoder, best 4.3 system chosen on validation)", "",
              "| encoder | task | US 3.1 | US 3.3 | 4.1 (L12) | 4.3 | system | delta vs 4.1 [95% CI] | delta vs US 3.3 [95% CI] | verdict vs 4.1 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in headline.itertuples():
            L.append(f"| {r.encoder} | {r.task} | {_fmt(r.us31)} | {_fmt(r.us33)} | {_fmt(r.frozen_41)} | **{_fmt(r.frozen_43)}** | {r.frozen_43_system} "
                     f"| {r.d_vs_41:+.3f} [{r.d_vs_41_lo:+.3f}, {r.d_vs_41_hi:+.3f}] p={r.d_vs_41_p:.3f} "
                     f"| {r.d_vs_33:+.3f} [{r.d_vs_33_lo:+.3f}, {r.d_vs_33_hi:+.3f}] p={r.d_vs_33_p:.3f} | {r.verdict_vs_41} |")
        L.append("")
        L += ["Best encoder per task by validation (the frozen baseline carried into Sprint 5):", "", "| task | encoder | system | 4.3 test | 4.1 test | delta vs 4.1 | verdict |", "|---|---|---|---|---|---|---|"]
        tasks = build_tasks(load_baseline_config())
        for tname, g in headline.groupby("task", sort=False):
            asc = tasks[tname].kind == "reg"
            b = g.sort_values("val_43", ascending=asc).iloc[0]
            L.append(f"| {tname} | {b.encoder} | {b.frozen_43_system} | {_fmt(b.frozen_43)} | {_fmt(b.frozen_41)} | {b.d_vs_41:+.3f} | {b.verdict_vs_41} |")
        L.append("")
    if not effects.empty:
        L += ["## 2. What each fix did (control vs fix, each side = best system of its group by validation)", "",
              "| encoder | task | fix | control | treatment | control | treatment | delta [95% CI] | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for r in effects.itertuples():
            L.append(f"| {r.encoder} | {r.task} | {r.fix} | {r.control} | {r.treatment} | {_fmt(r.control_test)} | {_fmt(r.treatment_test)} "
                     f"| {r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] p={r.p:.3f} | {r.verdict} |")
        L.append("")
    if not weights.empty:
        L += ["## 3. Layers the learned weighted sum relies on (softmax weights, mean over folds; uniform = 0.077)", "",
              "| encoder | task | top layers (weight) | normalised entropy (1 = uniform) |", "|---|---|---|---|"]
        for r in weights.itertuples():
            L.append(f"| {r.encoder} | {r.task} | L{r.top1} ({r.w1:.2f}), L{r.top2} ({r.w2:.2f}), L{r.top3} ({r.w3:.2f}) | {r.norm_entropy:.2f} |")
        L.append("")
    if not groups.empty:
        L += ["## 4. Fix 5: mispronunciation decision weights by age group", "",
              "Weights for the `*-oof` strategies are tuned on speaker-grouped out-of-fold TRAIN probabilities (validation scored with weights that never saw "
              "validation; test weights use OOF-train + validation). Precision / recall / F1 are for the mispronounced class.", "",
              "| encoder | strategy | split | group | n (pos) | macro-F1 | precision | recall | F1 | AUC | AP |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in groups.itertuples():
            star = " *" if r.chosen else ""
            L.append(f"| {r.encoder} | {r.strategy}{star} | {r.split} | {r.group} | {r.n} ({r.n_pos}) | {_fmt(r.macro_f1)} | {_fmt(r.precision)} | {_fmt(r.recall)} "
                     f"| {_fmt(r.f1)} | {_fmt(getattr(r, 'auc', np.nan))} | {_fmt(getattr(r, 'ap', np.nan))} |")
        L += ["", "`*` = strategy kept by the declared rule (group-specific weights if pooled validation macro-F1 is within the tolerance of the global OOF weights).", ""]
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--frozen-root", default=None, help="folder holding <encoder>/L<layer>/emb+side (default experiments/results/frozen)")
    ap.add_argument("--fix-root", default=None, help="folder holding <encoder>/<recipe> (default fix43.root)")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cand, headline, effects, weights, groups = build(a.encoders, Path(a.frozen_root) if a.frozen_root else None,
                                                     Path(a.fix_root) if a.fix_root else None, a.n_boot)
    cand.to_csv(OUT / "us43_candidates.csv", index=False)
    headline.to_csv(OUT / "us43_headline.csv", index=False)
    effects.to_csv(OUT / "us43_fix_effects.csv", index=False)
    weights.to_csv(OUT / "us43_layer_weights.csv", index=False)
    groups.to_csv(OUT / "us43_mispron_groups.csv", index=False)
    write_md(cand, headline, effects, weights, groups, OUT / "us43_report.md", a.encoders)
    print(f"[us43] {len(headline)} headline rows, {len(effects)} fix comparisons -> {OUT}/us43_report.md")


if __name__ == "__main__":
    main()
