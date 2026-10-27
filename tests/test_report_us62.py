"""Sprint 6 (US 6.2) tests for src/evaluation/report_us62.py: gradient diagnostics, conflict / dominance criteria, stopping regret, tables on synthetic files."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation import report_us62 as R
from src.models import multitask as mt


# ---- pure helpers ------------------------------------------------------------ #
def _group_hist(E=4, K=3, gn=(0, 10, 10, 10), norms=None, cos=None, w=None):
    norms = np.tile([1.0, 1.0, 1.0], (E, 1)) if norms is None else np.asarray(norms, float)
    cos = np.tile(np.eye(K), (E, 1, 1)) if cos is None else np.asarray(cos, float)
    w = np.full((E, K), 1.0 / K) if w is None else np.asarray(w, float)
    return {"hist_gn": np.array(gn), "hist_gnorm": norms, "hist_gcos": cos, "hist_w": w}


def test_job_diag_ignores_epochs_without_samples_and_weights_by_steps():
    norms = np.array([[np.nan] * 3, [1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
    d = R.job_diag(_group_hist(norms=norms, gn=(0, 10, 10, 20)))
    assert d["norms"] == pytest.approx([(10 + 10 + 60) / 40, 1.0, 2.0])
    assert d["shares"].sum() == pytest.approx(1.0) and np.isnan(d["shares_epoch"][0]).all()
    assert R.job_diag(_group_hist(gn=(0, 0, 0, 0))) is None
    assert d["eff_shares"].sum() == pytest.approx(1.0)


def test_effective_share_includes_the_loss_weight():
    w = np.tile([0.5, 0.25, 0.25], (4, 1))
    d = R.job_diag(_group_hist(w=w))                                    # equal norms, unequal weights
    assert d["shares"] == pytest.approx([1 / 3] * 3) and d["eff_shares"] == pytest.approx([0.5, 0.25, 0.25])


def test_effective_share_equals_share_when_weights_are_equal_even_if_norms_vary_by_epoch():
    norms = np.array([[0, 0, 0], [1.0, 3.0, 1.0], [5.0, 1.0, 2.0], [2.0, 2.0, 8.0]])
    d = R.job_diag(_group_hist(norms=norms, gn=(0, 5, 20, 10)))
    assert d["eff_shares"] == pytest.approx(d["shares"])


def test_dominance_label_thresholds():
    assert R.dominance_label(0.70, 4, 2.0) == "dominant"                # > 2/4 = 0.5
    assert R.dominance_label(0.50, 4, 2.0) == "balanced"                # not strictly greater
    assert R.dominance_label(0.10, 4, 2.0) == "starved"                 # < 1/8 = 0.125
    assert R.dominance_label(0.125, 4, 2.0) == "balanced"
    assert R.dominance_label(float("nan"), 4, 2.0) == "n/a"


def test_conflict_pairs_need_mean_below_threshold_in_enough_jobs():
    J = np.zeros((3, 3, 3))
    for j in range(3):
        J[j] = np.eye(3)
    J[:, 0, 1] = J[:, 1, 0] = [-0.3, -0.2, -0.4]                         # consistent conflict
    J[:, 0, 2] = J[:, 2, 0] = [-0.5, 0.2, 0.3]                           # one job only
    J[:, 1, 2] = J[:, 2, 1] = [0.5, 0.4, 0.6]                            # aligned
    names = ["a", "b", "c"]
    pairs = {(p["head_a"], p["head_b"]): p for p in R.conflict_pairs(J, names, -0.1, 0.67, 0.3)}
    assert pairs[("a", "b")]["conflict"] and pairs[("a", "b")]["frac_jobs_below_thr"] == 1.0
    assert not pairs[("a", "c")]["conflict"] and pairs[("a", "c")]["frac_jobs_below_thr"] == pytest.approx(1 / 3)
    assert pairs[("b", "c")]["aligned"] and not pairs[("b", "c")]["conflict"]
    assert len(pairs) == 3


def test_alignment_with_rest_matches_explicit_vectors():
    V = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])                  # three explicit gradients
    n = np.linalg.norm(V, axis=1)
    cos = (V @ V.T) / np.outer(n, n)
    a = R.alignment_with_rest(n, cos)
    for k in range(3):
        rest = V[[j for j in range(3) if j != k]].sum(0)
        assert a[k] == pytest.approx(V[k] @ rest / (np.linalg.norm(V[k]) * np.linalg.norm(rest)))
    assert np.isnan(R.alignment_with_rest(np.array([1.0]), np.eye(1))).all()
    two = R.alignment_with_rest(np.array([1.0, 3.0]), np.array([[1, -0.4], [-0.4, 1]]))
    assert two == pytest.approx([-0.4, -0.4])                            # K=2: the cosine itself


def test_stopping_regret_regression_and_classification():
    r = R.stopping_regret("reg", [1.2, 1.0, 0.9, 0.95], shared_epoch=3)
    assert r["own_epoch"] == 2 and r["regret"] == pytest.approx(0.05) and r["regret_rel"] == pytest.approx(0.05 / 0.95)
    c = R.stopping_regret("clf", [-0.60, -0.70, -0.65], shared_epoch=2)   # primary = -macro-F1, error = 1 - F1
    assert c["own_epoch"] == 1 and c["err_shared"] == pytest.approx(0.35) and c["regret"] == pytest.approx(0.05)
    assert R.stopping_regret("reg", [1.0, 0.9], shared_epoch=1)["regret"] == 0.0
    assert R.stopping_regret("reg", [1.0, 0.9], shared_epoch=9)["err_shared"] == 0.9     # epoch index is clipped


def test_matched_comparator_follows_the_recipe():
    assert R.matched_comparator("base", "lr") == "base"
    assert R.matched_comparator("lr", "lr") == "stabilized"
    assert R.matched_comparator("lr", "freeze6") == "base"
    assert R.matched_comparator("lr", None) == "base" and R.matched_comparator("lr", float("nan")) == "base"


def test_labels_and_ablation_reading():
    assert R.transfer_label("worse") == "negative transfer" and R.transfer_label("better") == "positive transfer"
    assert R.transfer_label("no significant change") == "no clear transfer"
    assert R.ablation_reading("no significant change", "better") == "recovers"
    assert R.ablation_reading("worse", "better") == "partly"
    assert R.ablation_reading("worse", "no significant change") == "no effect"
    assert R.ablation_reading("worse", "worse") == "hurts"


def test_pad_stack_pads_unequal_epoch_counts_with_nan():
    out = R.pad_stack([np.ones((2, 3)), np.ones((4, 3))])
    assert out.shape == (2, 4, 3) and np.isnan(out[0, 2:]).all() and not np.isnan(out[1]).any()


# ---- tables on synthetic files ------------------------------------------------ #
HEADS = ["h_reg", "h_clf", "h_other"]


def _fake_world(tmp_path, monkeypatch, seeds=(0, 1)):
    cfg = mt.load_config()
    cfg["root"] = str(tmp_path / "res")
    cfg["groups"] = {"g": {"tasks": HEADS}}
    cfg["seeds"] = list(seeds)
    monkeypatch.setattr(R.MT, "load_config", lambda variant=None, path=None: cfg)
    reg = {n: SimpleNamespace(folds=[0], dataset="so762", kind="reg") for n in HEADS}
    rng = np.random.default_rng(0)
    for s in seeds:
        E = 6
        cos = np.tile(np.eye(3), (E, 1, 1))
        cos[:, 0, 1] = cos[:, 1, 0] = -0.4                                  # h_reg and h_clf conflict in every job
        norms = np.tile([6.0, 1.0, 1.0], (E, 1))                            # h_reg dominates the gradients (share 0.75 > 2/3)
        gf = mt.group_file(cfg, "mt_x", "g", 0, s)
        gf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gf, heads=np.array(HEADS), kinds=np.array(["reg", "clf", "reg"]), best_epoch=3, fit_s=60.0, seed=s, fold=0, ref_err=np.array([1.0, 0.4, 1.0]),
                            hist_mean_rel=np.linspace(1, 0.8, E), hist_gnorm=norms, hist_gcos=cos, hist_gn=np.array([0, 10, 10, 10, 10, 10]), hist_w=np.full((E, 3), 1 / 3))
        for k, n in enumerate(HEADS):
            if n == "h_reg":
                vp = np.array([1.0, 0.9, 0.8, 0.85, 0.9, 0.95])             # own best at epoch 2, shared 3: regret 0.05 (5.9 %)
            elif n == "h_clf":
                vp = -np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.68])        # macro-F1 still improving after the shared epoch
            else:
                vp = np.array([1.0, 0.9, 0.85, 0.84, 0.86, 0.87])
            f = mt.job_file(cfg, "mt_x", n, 0, s)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, best_epoch=3, hist_val_primary=vp, hist_val_loss=np.full(6, 1.2), hist_trsub_loss=np.full(6, 0.6), n_train=1000 + k)
    return cfg, reg


def test_collect_group_and_tables_flag_the_planted_effects(tmp_path, monkeypatch):
    cfg, reg = _fake_world(tmp_path, monkeypatch)
    data = R.collect_group("g", "mt_x", None, reg)
    assert data["names"] == HEADS and len(data["jobs"]) == 2
    c62 = cfg["compare62"]
    gr, pairs, share_ep = R.gradient_tables(data, c62)
    assert share_ep.shape == (2, 6, 3)
    dom = dict(zip(gr["task"], gr["dominance"]))
    assert dom["h_reg"] == "dominant" and dom["h_clf"] == "starved"                                    # 0.75 > 2/3 ; 0.125 < 1/6
    assert gr.loc[gr.task == "h_reg", "share"].iloc[0] == pytest.approx(0.75)
    conflicts = {(r.head_a, r.head_b): r.conflict for r in pairs.itertuples()}
    assert conflicts[("h_reg", "h_clf")] and not conflicts[("h_reg", "h_other")]
    stop = R.stopping_table(data, c62)
    s = stop.set_index("task")
    assert s.loc["h_reg", "stopping_compromise"] and s.loc["h_reg", "own_best_epoch"] == 2
    assert s.loc["h_other", "regret"] == pytest.approx(0.0) and not s.loc["h_other", "stopping_compromise"]        # best epoch 3 = shared epoch
    assert s.loc["h_clf", "val_err_shared"] == pytest.approx(0.34) and s.loc["h_clf", "regret"] == pytest.approx(0.02) and s.loc["h_clf", "own_best_epoch"] == 4
    assert s.loc["h_reg", "n_train_units"] == 1000 and s.loc["h_reg", "val_loss_over_train_loss"] == pytest.approx(2.0)
    assert s.loc["h_reg", "progress_vs_head_only"] == pytest.approx(0.85)
    tr = pd.DataFrame({"group": "g", "task": HEADS, "transfer": ["negative transfer", "no clear transfer", "no clear transfer"], "delta": [0.05, 0.0, 0.0],
                       "val_flag": ["worse", "within margin", "within margin"]})
    ca = R.cause_table(tr, gr, pairs, stop).set_index("task")
    assert "gradient conflict" in ca.loc["h_reg", "criteria_fired"] and "stopping compromise" in ca.loc["h_reg", "criteria_fired"]
    assert ca.loc["h_reg", "conflicts_with"] == "h_clf"


def test_collect_group_skips_unfinished_jobs_and_rejects_changed_heads(tmp_path, monkeypatch):
    cfg, reg = _fake_world(tmp_path, monkeypatch)
    mt.job_file(cfg, "mt_x", "h_other", 0, 1).unlink()                     # seed 1 incomplete: dropped
    data = R.collect_group("g", "mt_x", None, reg)
    assert [j["seed"] for j in data["jobs"]] == [0]
    assert R.collect_group("g", "mt_missing", None, reg) is None
    cfg["groups"]["g"]["tasks"] = ["h_clf", "h_reg", "h_other"]
    with pytest.raises(ValueError):
        R.collect_group("g", "mt_x", None, reg)


def test_transfer_table_reads_the_matched_comparator(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    row = {"tag": "mt_lr", "group": "l2", "task": "t1", "metric": "macro_f1", "mt": 0.57, "stabilized_variant": "lr",
           "st_base": 0.55, "d_base": 0.02, "d_base_lo": 0.01, "d_base_hi": 0.03, "d_base_p": 0.0, "verdict_base": "better", "val_flag_base": "better",
           "val_worse_by_base": -0.01, "val_margin_base": 0.01, "seeds_base": 3, "st_rule_base": "plain",
           "st_stabilized": 0.58, "d_stabilized": -0.01, "d_stabilized_lo": -0.02, "d_stabilized_hi": -0.001, "d_stabilized_p": 0.01, "verdict_stabilized": "worse",
           "val_flag_stabilized": "within margin", "val_worse_by_stabilized": 0.0, "val_margin_stabilized": 0.01, "seeds_stabilized": 3, "st_rule_stabilized": "plain"}
    pd.DataFrame([row]).to_csv(tmp_path / "us61_headline_mt_lr.csv", index=False)
    t = R.transfer_table("l2", {"tag": "mt_lr", "variant": "lr"}, ["t1"])
    assert t.loc[0, "comparator"] == "stabilized" and t.loc[0, "transfer"] == "negative transfer" and t.loc[0, "st"] == 0.58
    with pytest.raises(FileNotFoundError):
        R.transfer_table("so762", {"tag": "mt_base", "variant": None}, ["t1"])


def test_report_writes_every_section(tmp_path, monkeypatch):
    cfg, reg = _fake_world(tmp_path, monkeypatch)
    data = R.collect_group("g", "mt_x", None, reg)
    c62 = cfg["compare62"]
    tr = pd.DataFrame({"group": "g", "task": HEADS, "metric": "rmse", "comparator": "base", "st_recipe": "plain", "st": 1.0, "mt": 1.05, "delta": 0.05, "lo": 0.01, "hi": 0.09, "p": 0.01,
                       "transfer": "negative transfer", "val_flag": "worse", "val_worse_by": 0.02, "val_margin": 0.01, "seeds": 2})
    gr, pairs, _ = R.gradient_tables(data, c62)
    stop = R.stopping_table(data, c62)
    ca = R.cause_table(tr, gr, pairs, stop)
    out = tmp_path / "r.md"
    R.write_md(out, c62, {"g": {"tag": "mt_x", "transfer": tr, "grads": gr, "pairs": pairs, "stop": stop, "causes": ca}}, None)
    txt = out.read_text()
    for s in ("Transfer", "gradient shares", "Gradient conflict", "own best epoch", "criteria fired", "Caveats", "h_reg"):
        assert s in txt


def test_config_declares_the_criteria_and_the_ablation_variants_load():
    cfg = mt.load_config()
    c62 = cfg["compare62"]
    for k in ("group_tags", "grad_dominance_ratio", "conflict_cos", "conflict_min_job_frac", "stopping_regret_rel", "lag_margin", "ablations", "ablation_tag_prefix"):
        assert k in c62
    assert set(c62["group_tags"]) == set(cfg["groups"])
    for v in c62["ablations"]:
        a = mt.load_config(v)
        assert a["train"]["max_epochs"] == cfg["train"]["max_epochs"] and a["encoder"] == cfg["encoder"]
        assert all(t in cfg["tasks"] for g in a["groups"].values() for t in g["tasks"])
    assert mt.load_config("phone_pair")["groups"]["so762"]["tasks"] == ["phone_score", "phone_mispron_t1.0"]
    assert mt.load_config("word_phone")["groups"]["so762"]["tasks"] == ["word_accuracy", "phone_score", "phone_mispron_t1.0"]
    w = mt.normalise_weights(cfg["groups"]["so762"]["tasks"], mt.load_config("w_mispron3")["train"]["loss"]["weights"])
    assert w["phone_mispron_t1.0"] == pytest.approx(3 / 9) and w["sent_accuracy"] == pytest.approx(1 / 9)
    assert mt.load_config()["groups"] == mt.load_config("sharedalpha")["groups"]               # the full model's groups are untouched by the ablation variants
