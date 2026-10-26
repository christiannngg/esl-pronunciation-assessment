"""Sprint 6 (US 6.1) tests for src/evaluation/report_us61.py: validation margins, seed matching, recipe rule, and the full comparison on synthetic job files."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.evaluation import report_us61 as R
from src.models import finetune as ft
from src.models import multitask as mt

from test_finetune import _fake_jobs, _task  # noqa: F401


# ---- pure helpers ------------------------------------------------------------ #
def test_error_of_and_val_flag_directions():
    assert R.error_of("reg", 0.9) == 0.9 and R.error_of("clf", 0.7) == pytest.approx(0.3)
    w, m, lab = R.val_flag("reg", st_val=0.90, mt_val=0.95, job_sd=0.02, floor=0.01)
    assert w == pytest.approx(0.05) and m == pytest.approx(0.02) and lab == "worse"
    assert R.val_flag("reg", 0.90, 0.91, 0.02, 0.01)[2] == "within margin"
    assert R.val_flag("reg", 0.90, 0.85, 0.02, 0.01)[2] == "better"
    assert R.val_flag("clf", 0.70, 0.65, 0.01, 0.01)[2] == "worse"                    # F1 is higher-better: lower F1 = worse
    assert R.val_flag("clf", 0.70, 0.75, 0.01, 0.01)[2] == "better"
    assert R.val_flag("reg", 0.9, 0.915, float("nan"), 0.01)[1] == 0.01                # no seed sd -> the floor


def test_job_val_sd_averages_over_folds_with_two_or_more_seeds():
    task = _task("sent_accuracy")
    jobs = {0: [{"val_score": 0.5}, {"val_score": 0.7}], 1: [{"val_score": 0.5}]}
    assert R.job_val_sd(task, jobs) == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert np.isnan(R.job_val_sd(task, {0: [{"val_score": 0.5}]}))


def test_common_seeds_needs_every_fold_of_every_side():
    a = {0: {0, 1, 2}, 1: {0, 1, 2}}
    b = {0: {0}, 1: {0, 1}}
    assert R.common_seeds([a, b]) == [0] and R.common_seeds([a, a]) == [0, 1, 2] and R.common_seeds([a, {0: set(), 1: {0}}]) == []


def test_recipe_choice_keeps_base_unless_another_wins_by_more_than_the_floor():
    assert R.recipe_choice({"base": 0.990, "lr": 0.985}, 0.01)[0] == "base"
    assert R.recipe_choice({"base": 0.990, "lr": 0.960}, 0.01)[0] == "lr"
    assert R.recipe_choice({"base": 0.95, "lr": 0.99}, 0.01)[0] == "base"
    assert R.recipe_choice({"lr": 0.99}, 0.01)[0] == "lr"


# ---- loading / systems / comparison ------------------------------------------ #
def _save(cfg, tag, task, jobs):
    for fold, ds in jobs.items():
        for d in ds:
            f = ft.job_file(cfg, tag, task.name, fold, int(d["seed"]))
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, **{**d, "mt_group_fit_s": 30.0, "mt_head_best_epoch": 2})


def _world(tmp_path, name="sent_accuracy", seeds=(0, 1)):
    n = 200
    meta = pd.DataFrame({"speaker": np.repeat([f"s{i}" for i in range(20)], 10), "age_group": "adult", "utt_id": [f"u{i}" for i in range(n)]})
    task = _task(name)
    rng = np.random.default_rng(0)
    y_all = rng.normal(size=n) if task.kind == "reg" else (rng.random(n) < 0.3).astype(int)
    st = _fake_jobs(task, np.random.default_rng(1), n, meta.speaker.to_numpy(), seeds)
    for ds in st.values():
        for j in ds:
            for k in ("val", "test", "train"):
                y = y_all[j[f"{k}_rows"]]
                if task.kind == "reg":
                    j[f"{k}_pred"] = (y + rng.normal(0, 0.8, len(y))).astype(np.float32)
                else:
                    p1 = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, len(y)), 0.01, 0.99)
                    j[f"{k}_proba"] = np.stack([1 - p1, p1], 1).astype(np.float32)
                    j[f"{k}_pred"] = (p1 > 0.5).astype(int)
            if task.kind == "clf":
                j["train_y"] = y_all[j["train_rows"]]
    mtj = copy.deepcopy(st)
    for ds in mtj.values():
        for j in ds:
            if task.kind == "reg":
                for k in ("val", "test", "train"):
                    y = y_all[j[f"{k}_rows"]]
                    j[f"{k}_pred"] = (y + rng.normal(0, 0.4, len(y))).astype(np.float32)             # clearly better than single-task
    return task, meta, y_all, st, mtj


def test_build_system_and_compare_head_report_the_direction_and_validation_margin(tmp_path):
    task, meta, y_all, st, mtj = _world(tmp_path)
    s = R.build_system(task, st, y_all, meta, n_boot=100)
    m = R.build_system(task, mtj, y_all, meta, n_boot=100)
    assert s["n_seeds"] == m["n_seeds"] == 2 and s["variant"] == "plain" and s["lo"] <= s["test"] <= s["hi"]
    c = R.compare_head(task, s, m, 100, 0.01)
    assert c["mt"] < c["st"] and c["delta"] < 0 and c["verdict"] == "better" and c["val_flag"] in ("better", "within margin")
    assert c["val_worse_by"] == pytest.approx(m["val"] - s["val"])                    # regression: error = RMSE
    c2 = R.compare_head(task, m, s, 100, 0.01)                                         # swapped roles
    assert c2["delta"] > 0 and c2["verdict"] == "worse" and c2["val_flag"] in ("worse", "within margin")


def test_classification_comparison_uses_macro_f1_directions(tmp_path):
    task, meta, y_all, st, _ = _world(tmp_path, "phone_mispron_t1.0")
    s = R.build_system(task, st, y_all, meta, n_boot=100)
    c = R.compare_head(task, s, s, 100, 0.01)
    assert c["delta"] == 0 and c["verdict"] == "no significant change" and c["val_flag"] == "within margin" and s["variant"] in ("cw", "cw+thr")


def test_load_sources_pools_extra_seed_tags_and_respects_the_seed_list(tmp_path):
    task, meta, y_all, st, _ = _world(tmp_path, seeds=(0, 1, 2))
    cfg = ft.load_config()
    cfg["root"] = str(tmp_path / "res")
    for d in st[0]:
        _save(cfg, "base" if d["seed"] == 0 else "base_l2x", task, {0: [d]})
    srcs = [(cfg, "base"), (cfg, "base_l2x")]
    assert R.available(srcs, task, [0, 1, 2]) == {0: {0, 1, 2}}
    jobs = R.load_sources(srcs, task, [0, 2])
    assert [int(d["seed"]) for d in jobs[0]] == [0, 2]
    assert R.load_sources([(cfg, "missing")], task, [0, 1]) is None


def test_build_end_to_end_on_synthetic_jobs_writes_a_report(tmp_path, monkeypatch):
    task, meta, y_all, st, mtj = _world(tmp_path)
    mcfg = mt.load_config()
    mcfg["groups"] = {"g": {"tasks": ["sent_accuracy"]}}
    mcfg["root"] = str(tmp_path / "mres")
    scfg = ft.load_config()
    scfg["root"] = str(tmp_path / "sres")
    _save(scfg, "base", task, st)
    _save(mcfg, "mt_x", task, mtj)
    monkeypatch.setattr(R.MT, "load_config", lambda variant=None, path=None: mcfg)
    monkeypatch.setattr(R.FT, "load_config", lambda variant=None: scfg)
    monkeypatch.setattr(R.FH, "load_labels", lambda t, *a, **k: (y_all, meta))
    head, jobs, eff = R.build("mt_x", n_boot=60, stab_path=tmp_path / "no_selection.csv")
    assert len(head) == 1 and head.loc[0, "verdict_base"] == "better" and head.loc[0, "seeds_base"] == 2 and "d_stabilized" not in head
    assert len(jobs) == 2 and set(jobs["group"]) == {"g"}
    assert len(eff) == 1 and eff.loc[0, "mt_minutes"] == pytest.approx(0.5) and eff.loc[0, "st_minutes"] == pytest.approx(1 / 60)
    out = tmp_path / "r.md"
    R.write_md(head, jobs, eff, out, "mt_x", 0.01)
    txt = out.read_text()
    assert "sent_accuracy" in txt and "better" in txt and "Validation side" in txt and "Caveats" in txt


def test_build_with_the_stabilized_comparator(tmp_path, monkeypatch):
    task, meta, y_all, st, mtj = _world(tmp_path)
    mcfg = mt.load_config()
    mcfg["groups"] = {"g": {"tasks": ["sent_accuracy"]}}
    mcfg["root"] = str(tmp_path / "mres")
    scfg = ft.load_config()
    scfg["root"] = str(tmp_path / "sres")
    stab = copy.deepcopy(st)
    for d in stab[0]:
        d["test_pred"] = (d["test_pred"] + 0.3).astype(np.float32)                     # a worse `lr` single-task system
    _save(scfg, "base", task, st)
    _save(scfg, "lr", task, stab)
    _save(mcfg, "mt_x", task, mtj)
    sel = tmp_path / "sel.csv"
    pd.DataFrame({"task": ["sent_accuracy"], "selected": ["lr"]}).to_csv(sel, index=False)
    monkeypatch.setattr(R.MT, "load_config", lambda variant=None, path=None: mcfg)
    monkeypatch.setattr(R.FT, "load_config", lambda variant=None: scfg)
    monkeypatch.setattr(R.FH, "load_labels", lambda t, *a, **k: (y_all, meta))
    head, _, _ = R.build("mt_x", n_boot=60, stab_path=sel)
    r = head.iloc[0]
    assert r["stabilized_variant"] == "lr" and r["st_stabilized"] > r["st_base"] and r["verdict_stabilized"] == "better"
