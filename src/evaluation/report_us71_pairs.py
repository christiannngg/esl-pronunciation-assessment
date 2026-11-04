"""
US 7.1 add-on: the three paired comparisons that no sprint report stored (they need the prediction files, which sit on different machines).

    B  -> FT-base    baseline (US 3.3) vs fine-tuned US 5.1 base, 9 tasks
    B  -> MT-GN15    baseline vs multi-task GradNorm alpha 1.5, 7 SO762 tasks
    F  -> MT-GN15    frozen WavLM (US 4.3) vs multi-task GradNorm alpha 1.5, 7 SO762 tasks

Two steps, so that each machine only needs the files it has. Both only READ existing results.

  1. export (on every machine that holds some of the prediction files; skips what it cannot find):
         python -m src.evaluation.report_us71_pairs export                 # all systems this machine can build
         python -m src.evaluation.report_us71_pairs export --systems B F   # or a subset (B, F, FT-base, MT-GN15)
     Each system's test predictions go to results/analysis/us71/<system>__<task>.npz (encoded labels / scores + speaker, a few hundred KB in total). Before a
     bundle is written, the metric recomputed from it must equal the number already in the saved sprint table (tolerance 1e-4): this proves it is the same
     system that the report shows. A mismatch is printed and the bundle is NOT written (--force writes it anyway).
         B        needs experiments/results/{us31,cw,pros,...} (the Sprint 3 job files)
         F        needs experiments/results/frozen43 (+ the fix outputs) as report_us51.frozen_refs does
         FT-base  needs experiments/results/finetune/base           (lab machine)
         MT-GN15  needs experiments/results/multitask/mt63_gradnorm_a15  (lab machine)
         FT-stab  needs the tag per task in results/tables/finetune/us53_selection.csv under experiments/results/finetune  (lab machine; Sprint 8 add-on)
         MT-full  needs experiments/results/multitask/mt_base (SO762 heads) and mt_lr (L2 heads)  (lab machine; Sprint 8 add-on)
     FT-stab and MT-full are exported for the Sprint 8 subgroup analysis only; WANTED (the US 7.1 pairs) is unchanged.
     Copy / commit the small npz files so that all bundles end up in one checkout.

  2. pairs (anywhere, numpy only):
         python -m src.evaluation.report_us71_pairs pairs
     Paired speaker bootstrap exactly as report_us43.pair (same procedure, n_boot and seed as configs/baseline.yaml `bootstrap`) on every pair whose two bundles
     exist -> results/tables/midterm/us71_pairs_computed.csv. python -m src.evaluation.report_us71 picks that file up and replaces the MISSING rows.

delta = metric(b) - metric(a): negative RMSE or positive macro-F1 is better for b.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation import report_us71 as R71

BUNDLES = R71.REPO / "results" / "analysis" / "us71"
COMPUTED = R71.TABLES / "midterm" / "us71_pairs_computed.csv"
WANTED = [("B", "FT-base", R71.TASKS), ("B", "MT-GN15", R71.SO762), ("F", "MT-GN15", R71.SO762)]
SYSTEM_TASKS = {"B": R71.TASKS, "F": R71.TASKS, "FT-base": R71.TASKS, "MT-GN15": R71.SO762, "FT-stab": R71.TASKS, "MT-full": R71.TASKS}
TOL = 1e-4
PROBA_DIR = R71.REPO / "results" / "analysis" / "us82"
PROBA_TASKS: set[str] = set()          # filled by `export --proba`: tasks whose positive-class probability is saved next to the bundle
PROBA_ONLY = False                     # `export --proba-only`: do not rewrite the hard-prediction bundles


# --------------------------------------------------------------------------- #
# pure helpers (numpy / pandas only, unit-tested)
# --------------------------------------------------------------------------- #
def bundle_path(system: str, task: str) -> Path:
    return BUNDLES / f"{system}__{task}.npz"


def save_bundle(system: str, task: str, y, pred, speaker) -> Path:
    p = bundle_path(system, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, y=np.asarray(y), pred=np.asarray(pred), speaker=np.asarray(speaker).astype(str))
    return p


def load_bundle(system: str, task: str) -> dict | None:
    p = bundle_path(system, task)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {"y": z["y"], "pred": z["pred"], "speaker": z["speaker"]}


def metric_fn(task: str, y, pred):
    """(fast metric function, key) for integer-coded classification or float regression; the classes are inferred from the data (macro-F1 averages the
    classes that occur in y, so the number of classes only has to cover the largest label)."""
    if R71.kind_of(task) == "reg":
        return M.fast_reg, "rmse"
    return M.fast_clf_factory(int(max(np.max(y), np.max(pred))) + 1), "macro_f1"


def score_bundle(task: str, b: dict) -> float:
    fn, key = metric_fn(task, b["y"], b["pred"])
    return float(fn(b["y"], b["pred"])[key])


def paired(task: str, a_name: str, b_name: str, a: dict, b: dict, n_boot: int, seed: int) -> dict | None:
    """Paired speaker bootstrap of metric(b) - metric(a); None when the two bundles were not scored on the identical test rows."""
    if len(a["y"]) != len(b["y"]) or not np.array_equal(a["y"], b["y"]) or not np.array_equal(a["speaker"], b["speaker"]):
        print(f"[us71-pairs] {task}: {a_name} and {b_name} were scored on different test rows; skipped")
        return None
    y = a["y"]
    allp = np.concatenate([a["pred"], b["pred"]])
    fn, key = metric_fn(task, y, allp)
    t = M.paired_speaker_bootstrap(a["speaker"], y, a["pred"], b["pred"], fn, key, n_boot=n_boot, seed=seed)
    return {"task": task, "a": a_name, "b": b_name, "delta": t["delta"], "lo": t["lo"], "hi": t["hi"], "p": t["p"],
            "verdict": R71.verdict(task, t["delta"], t["lo"], t["hi"])}


def compute_pairs(n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for a_name, b_name, tasks in WANTED:
        for t in tasks:
            a, b = load_bundle(a_name, t), load_bundle(b_name, t)
            if a is None or b is None:
                continue
            r = paired(t, a_name, b_name, a, b, n_boot, seed)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows, columns=["task", "a", "b", "delta", "lo", "hi", "p", "verdict"])


def missing_report() -> list[str]:
    out = []
    for a_name, b_name, tasks in WANTED:
        for t in tasks:
            gone = [s for s in (a_name, b_name) if not bundle_path(s, t).exists()]
            if gone:
                out.append(f"{t}: {a_name} -> {b_name} waits for the {', '.join(gone)} bundle")
    return out


# --------------------------------------------------------------------------- #
# exporting the systems (needs the prediction files of that system; heavy imports only here)
# --------------------------------------------------------------------------- #
def _expected() -> dict[tuple[str, str], float]:
    long, *_ = R71.load_systems()
    return {(r.system, r.task): float(r.test) for r in long.itertuples()}


def save_proba(system: str, task: str, y, proba, speaker) -> Path:
    """Positive-class probability on the test rows, in bundle order: results/analysis/us82/<system>__<task>.npz (y, proba, speaker)."""
    p = PROBA_DIR / f"{system}__{task}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    proba = np.asarray(proba, float)
    np.savez_compressed(p, y=np.asarray(y), proba=proba[:, 1] if proba.ndim == 2 else proba, speaker=np.asarray(speaker).astype(str))
    print(f"[us71-pairs] {system}/{task}: probabilities written ({len(proba)} rows)")
    return p


def _save_checked(system, task, y, pred, speaker, expected, force) -> bool:
    if PROBA_ONLY:
        return True
    b = {"y": np.asarray(y), "pred": np.asarray(pred), "speaker": np.asarray(speaker).astype(str)}
    got, want = score_bundle(task, b), expected.get((system, task))
    if want is not None and abs(got - want) > TOL and not force:
        print(f"[us71-pairs] {system}/{task}: MISMATCH recomputed {got:.6f} vs saved table {want:.6f}; bundle NOT written (use --force to write it anyway)")
        return False
    save_bundle(system, task, b["y"], b["pred"], b["speaker"])
    print(f"[us71-pairs] {system}/{task}: written ({len(b['y'])} rows, metric {got:.4f}" + ("" if want is None else f", matches the table {want:.4f}") + ")")
    return True


def export_B(tasks, registry, expected, force):
    from src.evaluation import report_frozen as RF
    from src.evaluation import report_us33 as R33
    from src.models.baselines import load_config

    cfg = load_config()
    sel = R71.pick_baseline(pd.read_csv(R71.TABLES / "baseline" / "us33_before_after.csv")).set_index("task")
    for t in tasks:
        if t not in sel.index:
            continue
        model, fix = sel.loc[t, "note"].split(" + ")
        variant, thr = fix.replace("+thr", ""), fix.endswith("+thr")
        try:
            d = R33._test_preds(registry[t], model, variant, thr, cfg)
        except Exception as e:                                                          # pragma: no cover - depends on the machine
            print(f"[us71-pairs] B/{t}: baseline prediction files not usable here ({type(e).__name__}: {e}); skipped")
            continue
        if d is None:
            print(f"[us71-pairs] B/{t}: no job files for {model} + {fix} on this machine; skipped")
            continue
        _save_checked("B", t, RF._enc(registry[t], d["y"]), RF._enc(registry[t], d["pred"]), d["meta"]["speaker"].to_numpy(), expected, force)


def export_F(tasks, registry, expected, force):
    from src.evaluation import report_frozen as RF
    from src.evaluation import report_us51 as R51

    h43 = pd.read_csv(R71.TABLES / "frozen" / "us43_headline.csv")
    refs = R51.frozen_refs(R71.ENCODER, h43, None, None)
    for t in tasks:
        s = (refs.get(t) or {}).get("4.3")
        if s is None:
            print(f"[us71-pairs] F/{t}: frozen predictions not available on this machine; skipped")
            continue
        _save_checked("F", t, RF._enc(registry[t], s["y"]), RF._enc(registry[t], s["pred"]), s["meta"]["speaker"].to_numpy(), expected, force)


def _export_ft(system, tag_of, tasks, registry, expected, force):
    from src.evaluation import report_frozen as RF
    from src.evaluation import report_us61 as R61
    from src.models import finetune as FT
    from src.models import frozen_heads as FH
    from src.models.baselines import load_config as load_baseline_config

    cfg, seed = FT.load_config(), load_baseline_config()["bootstrap"]["seed"]
    for t in tasks:
        task, tag = registry[t], tag_of(t)
        jobs = R61.load_sources([(cfg, tag)], task, FT.seeds_for(task, cfg, None))
        if jobs is None:
            print(f"[us71-pairs] {system}/{t}: no finished jobs under tag {tag!r} on this machine; skipped")
            continue
        y_all, meta = FH.load_labels(task)
        s = R61.build_system(task, jobs, y_all, meta, 20, seed)["sys"]
        _save_checked(system, t, RF._enc(task, s["y"]), RF._enc(task, s["pred"]), s["meta"]["speaker"].to_numpy(), expected, force)
        _maybe_proba(system, t, task, jobs, y_all, meta, s)


def _maybe_proba(system, t, task, jobs, y_all, meta, s) -> None:
    """Same seed-averaged probabilities that the headline decision rule used (report_us51.pooled_ft), checked to sit on the same rows."""
    if t not in PROBA_TASKS:
        return
    from src.evaluation import report_frozen as RF
    from src.evaluation import report_us51 as R51

    d = R51.pooled_ft(task, jobs, y_all, meta)["test"]
    assert np.array_equal(RF._enc(task, d["y"]), RF._enc(task, s["y"])) and np.array_equal(d["meta"]["speaker"].to_numpy(), s["meta"]["speaker"].to_numpy()), "probability rows differ from the bundle rows"
    save_proba(system, t, RF._enc(task, d["y"]), d["proba"], d["meta"]["speaker"].to_numpy())


def export_FT(tasks, registry, expected, force):
    _export_ft("FT-base", lambda t: "base", tasks, registry, expected, force)


def export_FTstab(tasks, registry, expected, force):
    """US 5.3 stabilized set: the variant tag per task comes from the selection file (tasks that kept `base` reuse the base jobs)."""
    from src.evaluation import report_us61 as R61

    stab = R61.stabilized_tags()
    if not stab:
        print("[us71-pairs] FT-stab: results/tables/finetune/us53_selection.csv not found; skipped")
        return
    _export_ft("FT-stab", lambda t: stab.get(t, "base"), tasks, registry, expected, force)


def _export_mt(system, srcs, seeds_of, tasks, registry, expected, force):
    from src.evaluation import report_frozen as RF
    from src.evaluation import report_us61 as R61
    from src.models import frozen_heads as FH
    from src.models.baselines import load_config as load_baseline_config

    seed = load_baseline_config()["bootstrap"]["seed"]
    for t in tasks:
        if t not in srcs:
            continue
        task, source = registry[t], [srcs[t]]
        seeds = R61.common_seeds([R61.available(source, task, seeds_of(task))])
        jobs = R61.load_sources(source, task, seeds) if seeds else None
        if jobs is None:
            print(f"[us71-pairs] {system}/{t}: no finished multi-task jobs for tag {srcs[t][1]!r} on this machine; skipped")
            continue
        y_all, meta = FH.load_labels(task)
        s = R61.build_system(task, jobs, y_all, meta, 20, seed)["sys"]
        _save_checked(system, t, RF._enc(task, s["y"]), RF._enc(task, s["pred"]), s["meta"]["speaker"].to_numpy(), expected, force)
        _maybe_proba(system, t, task, jobs, y_all, meta, s)


def export_GN15(tasks, registry, expected, force):
    from src.evaluation import report_us63 as R63
    from src.models import multitask as MT

    cfg = MT.load_config()
    c63 = cfg["compare63"]
    srcs = R63.part_sources(c63, c63["candidates"]["gradnorm_a15"])
    seeds_all = sorted(set(MT.load_config(c63["reference"]["variant"])["seeds"]))
    _export_mt("MT-GN15", srcs, lambda task: seeds_all, tasks, registry, expected, force)


def export_MTfull(tasks, registry, expected, force):
    """Equal-weight multi-task model: SO762 heads = US 6.3 candidate `full` (tag mt_base); L2 heads = the US 6.1 `lr` recipe (tag mt_lr, 4 folds x 3 seeds)."""
    from src.evaluation import report_us63 as R63
    from src.models import multitask as MT

    cfg = MT.load_config()
    c63 = cfg["compare63"]
    srcs = dict(R63.part_sources(c63, c63["candidates"]["full"]))
    lr = MT.load_config("lr")
    srcs.update({h: (lr, "mt_lr") for h in lr["groups"]["l2"]["tasks"]})
    so_seeds, l2_seeds = sorted(set(cfg["seeds"])), sorted(set(lr["l2_seeds"]))
    _export_mt("MT-full", srcs, lambda task: so_seeds if task.dataset == "so762" else l2_seeds, tasks, registry, expected, force)


EXPORTERS = {"B": export_B, "F": export_F, "FT-base": export_FT, "MT-GN15": export_GN15, "FT-stab": export_FTstab, "MT-full": export_MTfull}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write the prediction bundles this machine can build")
    ex.add_argument("--systems", nargs="*", default=list(EXPORTERS), choices=list(EXPORTERS))
    ex.add_argument("--force", action="store_true", help="write a bundle even when its metric differs from the saved table")
    ex.add_argument("--proba", action="store_true", help="also save the positive-class probabilities of phone_mispron_t1.0 (US 8.2) to results/analysis/us82/")
    ex.add_argument("--proba-only", action="store_true", help="like --proba, without rewriting the hard-prediction bundles")
    pr = sub.add_parser("pairs", help="compute the paired comparisons from the bundles that exist")
    pr.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    if a.cmd == "export":
        from src.models.baselines import build_tasks, load_config
        registry = build_tasks(load_config())
        expected = _expected()
        if a.proba or a.proba_only:
            PROBA_TASKS.add("phone_mispron_t1.0")
        if a.proba_only:
            globals()["PROBA_ONLY"] = True
        for s in a.systems:
            EXPORTERS[s](SYSTEM_TASKS[s], registry, expected, a.force)
        print("[us71-pairs] bundles now present: " + ", ".join(sorted(p.stem for p in BUNDLES.glob("*.npz"))))
    else:
        from src.models.baselines import load_config
        b = load_config()["bootstrap"]
        df = compute_pairs(a.n_boot or int(b["n_boot"]), int(b["seed"]))
        COMPUTED.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(COMPUTED, index=False)
        print(df.to_string(index=False) if len(df) else "[us71-pairs] no pair has both bundles yet")
        for line in missing_report():
            print("[us71-pairs] still missing: " + line)
        print(f"[us71-pairs] wrote {COMPUTED} ({len(df)} pairs); rerun python -m src.evaluation.report_us71 to put them in the report")


if __name__ == "__main__":
    main()
