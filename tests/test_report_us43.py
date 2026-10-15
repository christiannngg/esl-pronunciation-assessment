"""Sprint 4 (US 4.3) report tests on the synthetic world of test_frozen_fix.py: system collection, validation-only selection, paired
deltas, learned layer weights, age-group decision weights (out-of-fold train vs validation), and the markdown/CSV outputs."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from test_frozen_fix import ENC, SIG, NL, task, world  # noqa: F401  (fixtures)

from src.evaluation import report_frozen as RF
from src.evaluation import report_us43 as R
from src.models import frozen_fix as FF
from src.models import frozen_heads as FH

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed; these run on Colab")

SMALL = {"reg": {"linear": {"wd": [1e-4]}, "mlp": {"dropout": [0.1], "wd": [1e-2]}},
         "clf": {"linear": {"wd": [1e-3], "cw": ["none", "balanced"]}, "mlp": {"dropout": [0.1], "wd": [1e-2], "cw": ["balanced"]}}}


def _shrink(world):
    for name, r in world["cfg"]["fix43"]["recipes"].items():
        r["grids"] = SMALL if name != "sent_avgpca" and name != "sent_avgpca_pros" else {**SMALL, "reg": {"linear": {"wd": [1e-2], "pca": [8]}, "mlp": {"dropout": [0.1], "wd": [1e-2], "pca": [8]}}}


@pytest.fixture
def results(world, monkeypatch):
    _shrink(world)
    cache, root, frozen = world["cache"], world["root"], world["frozen"]
    tasks = ["sent_accuracy", "word_accuracy", "phone_mispron_t1.0"]
    FH.run(ENC, 12, "emb+side", "search", only_tasks=tasks, cache=cache, base=frozen)
    FH.run(ENC, 7, "emb+side", "search", only_tasks=tasks, cache=cache, base=frozen)
    FH.run(ENC, 7, "emb+side", "final", only_tasks=tasks, cache=cache, base=frozen)
    for r, ts in (("ws", tasks), ("sent_ws", ["sent_accuracy"]), ("sent_ws_pros", ["sent_accuracy"]), ("sent_avgpca", ["sent_accuracy"]),
                  ("sent_avgpca_pros", ["sent_accuracy"]), ("word_mean_pros", ["word_accuracy"]), ("word_max_pros", ["word_accuracy"]),
                  ("word_meanmax_pros", ["word_accuracy"]), ("word_meanmaxstd_pros", ["word_accuracy"])):
        FF.run_recipe(ENC, r, "search", only_tasks=ts, cache=cache, root=root)
    FF.run_recipe(ENC, "ws", "final", only_tasks=["phone_mispron_t1.0"], cache=cache, root=root)
    FF.run_mispron_oof(ENC, n_folds=3, cache=cache, frozen_root=frozen, root=root)

    def fake_refs(t, bcfg):
        y_all, meta = FH.load_labels(t)
        m = FH.split_masks(t, meta, 0)
        te, tr = m["test"], m["train"]
        if t.kind == "reg":
            pred = np.full(te.sum(), y_all[tr].mean())
        else:
            pred = np.full(te.sum(), np.bincount(y_all[tr].astype(int)).argmax())
        ref = {"model": "dummy", "fix": "none", "y": y_all[te], "pred": pred, "meta": meta[te].reset_index(drop=True)}
        return ref, ref
    monkeypatch.setattr(RF, "baseline_refs", fake_refs)
    monkeypatch.setattr(R, "LAYER_SYSTEMS", ["L7"])
    return {"frozen": frozen, "root": root}


def test_systems_are_collected_and_selection_never_looks_at_test(results):
    systems, long = R.collect_systems(ENC, results["frozen"], results["root"], n_boot=5)
    assert {"4.1 L12", "L7", "ws", "sent_ws", "word_max_pros"} <= set(systems)
    assert ("sent_accuracy", "linear") in systems["ws"] and "base" in systems["ws"][("sent_accuracy", "linear")]
    tasks = FH.build_tasks(FH.load_baseline_config())
    best = R.best_of(systems, list(systems), "sent_accuracy", "reg")
    vals = [p["val_primary"] for s in systems.values() for (t, h), p in s.items() if t == "sent_accuracy"]
    assert best[2]["val_primary"] == min(vals)
    assert set(long["system"]) >= {"4.1 L12", "L7", "ws"} and {"val", "test"} <= set(long["split"])


def test_layers_without_a_final_stage_are_not_layer_systems(results, world):
    FH.run(ENC, 9, "emb+side", "search", only_tasks=["sent_accuracy"], cache=world["cache"], base=results["frozen"], probe=True)    # US 4.2 probe file
    import src.evaluation.report_us43 as RR
    old = RR.LAYER_SYSTEMS
    RR.LAYER_SYSTEMS = ["L7", "L9"]
    try:
        systems, long = RR.collect_systems(ENC, results["frozen"], results["root"], n_boot=5)
    finally:
        RR.LAYER_SYSTEMS = old
    assert "L9" not in systems                                               # probe-only layer: not a candidate
    assert "L7" in systems and all(p["stage"] == "final" for p in systems["L7"].values()) and set(long[long.system == "L7"].stage) == {"final"}


def test_build_finds_the_informative_layer_and_pooling_and_reports_paired_deltas(results):
    cand, headline, effects, weights, groups = R.build([ENC], results["frozen"], results["root"], n_boot=60, screen_boot=5)
    h = headline.set_index("task")
    assert h.loc["sent_accuracy", "frozen_43_system"] != "4.1 L12" and h.loc["sent_accuracy", "d_vs_41"] < -0.5
    assert h.loc["sent_accuracy", "verdict_vs_41"] == "better"
    assert h.loc["word_accuracy", "frozen_43_system"] in ("word_max_pros", "word_meanmax_pros", "word_meanmaxstd_pros", "ws", "L7")
    assert h.loc["word_accuracy", "d_vs_41"] < 0
    # fix effects: the layer fix sees layer 7, the pooling fix sees the max block
    e = effects.set_index(["task", "fix"])
    assert e.loc[("sent_accuracy", "fix 1: layer re-selection (6-12) vs layer 12"), "delta"] < -0.5
    k = [i for i in e.index if i[0] == "word_accuracy" and i[1].startswith("fix 4: max")][0]
    assert e.loc[k, "delta"] < 0 and e.loc[k, "lo"] <= e.loc[k, "delta"] <= e.loc[k, "hi"]
    # learned weights concentrate on the informative layer
    w = weights.set_index("task")
    assert w.loc["sent_accuracy", "top1"] == SIG and len(w.loc["sent_accuracy", "weights"].split()) == NL
    assert 0 <= w.loc["sent_accuracy", "norm_entropy"] <= 1
    # the grid-search CSV has both splits for every system
    assert not cand.empty and set(cand["encoder"]) == {ENC}


def test_mispron_age_group_table_has_every_strategy_group_and_one_chosen_rule(results):
    table, systems = R.mispron_groups(ENC, results["frozen"], results["root"])
    assert set(table["strategy"]) == {"cw+thr", "global-oof", "group-oof"} and set(table["group"]) == {"all", "adult", "child/teen"}
    assert set(table["split"]) == {"val", "test"}
    ch = table[table.chosen].strategy.unique()
    assert len(ch) == 1 and ch[0] in ("global-oof", "group-oof")
    row = table[(table.strategy == "cw+thr") & (table.split == "val") & (table.group == "all")].iloc[0]
    assert row.n == table[(table.strategy == "group-oof") & (table.split == "val") & (table.group == "all")].iloc[0].n
    assert 0 <= row.macro_f1 <= 1 and row.auc > 0.5                                    # synthetic signal is ranked above chance
    ctrl, grp = systems["mispron-ctrl"][("phone_mispron_t1.0", next(iter(systems["mispron-ctrl"]))[1])], systems["mispron-grp"][("phone_mispron_t1.0", next(iter(systems["mispron-grp"]))[1])]
    assert len(ctrl["pred"]) == len(grp["pred"]) == len(ctrl["y"]) and grp["variant"] == ch[0]


def test_group_rule_keeps_group_weights_only_inside_the_tolerance(results):
    t_loose, _ = R.mispron_groups(ENC, results["frozen"], results["root"], tol=1.0)       # always within tolerance -> group-specific
    t_strict, _ = R.mispron_groups(ENC, results["frozen"], results["root"], tol=-1.0)     # never within tolerance -> global
    assert t_loose[t_loose.chosen].strategy.unique().tolist() == ["group-oof"]
    assert t_strict[t_strict.chosen].strategy.unique().tolist() == ["global-oof"]


def test_group_weights_fall_back_to_global_when_a_group_has_too_few_positives(world):
    t = task("phone_mispron_t1.0")
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    groups = np.r_[np.zeros(n - 30, int), np.ones(30, int)]
    y[groups == 1] = 0                                            # no positives at all in group 1
    p1 = np.clip(0.5 * y + 0.25 + 0.2 * rng.normal(size=n), 0.01, 0.99)
    proba = np.c_[1 - p1, p1]
    w = R._weights(t, proba, y, groups, per_group=True)
    wg = R._weights(t, proba, y, groups, per_group=False)
    assert np.allclose(w[1], wg[0]) and np.allclose(wg[0], wg[1])            # group 1 falls back; global uses one vector for both
    pred = R._apply(proba, groups, w)
    assert pred.shape == (n,) and set(np.unique(pred)) <= {0, 1}


def test_markdown_and_csv_are_written(results, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    cand, headline, effects, weights, groups = R.build([ENC], results["frozen"], results["root"], n_boot=40, screen_boot=5)
    R.write_md(cand, headline, effects, weights, groups, tmp_path / "r.md", [ENC])
    txt = (tmp_path / "r.md").read_text()
    for part in ("## 1. Updated frozen-encoder results", "## 2. What each fix did", "## 3. Layers the learned weighted sum", "## 4. Fix 5"):
        assert part in txt
    assert "sent_accuracy" in txt and "L7" in txt


def test_verdict_uses_the_ci_and_the_metric_direction():
    assert R.verdict("reg", {"delta": -0.1, "lo": -0.2, "hi": -0.01}) == "better"
    assert R.verdict("reg", {"delta": 0.1, "lo": 0.01, "hi": 0.2}) == "worse"
    assert R.verdict("clf", {"delta": 0.05, "lo": 0.01, "hi": 0.09}) == "better"
    assert R.verdict("clf", {"delta": 0.05, "lo": -0.01, "hi": 0.09}) == "no significant change"
