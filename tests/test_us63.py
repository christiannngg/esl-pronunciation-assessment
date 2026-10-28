"""Sprint 6 (US 6.3) tests: the uncertainty / GradNorm loss weighters, the declared adoption rule of src/evaluation/report_us63.py, and the config.

Numpy / pandas tests run anywhere. The torch tests (uncertainty weights, a full tiny multi-task job with each new scheme) use the tiny random WavLM of
test_finetune.py and run on a laptop CPU; they are skipped when torch is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import report_us63 as R63
from src.models import finetune as ft
from src.models import multitask as mt

from test_finetune import _tiny_encoder, world  # noqa: F401
from test_multitask import _group, _mt_cfg


class _L:
    """Stand-in for a loss tensor in the numpy-only tests."""
    def __init__(self, v):
        self.v = float(v)

    def detach(self):
        return self

    def __float__(self):
        return self.v


NAMES = ["a", "b"]


def _warm(gn, a=1.0, b=1.0, steps=4):
    for _ in range(steps):
        gn.observe({"a": _L(a), "b": _L(b)})


# ---- GradNorm (numpy) --------------------------------------------------------- #
def test_gradnorm_pulls_weighted_gradient_norms_together_keeps_mean_one_and_respects_the_floor():
    gn = mt.GradNormWeights(NAMES, alpha=0.5, lr=0.025, w_min=0.05)
    _warm(gn)
    norms = np.array([10.0, 1.0])                                    # head a has a 10x larger gradient on the probe layer
    for _ in range(300):
        gn.observe({"a": _L(1.0), "b": _L(1.0)})
        gn.update({"a": _L(1.0), "b": _L(1.0)}, norms)
        assert gn.w.mean() == pytest.approx(1.0) and (gn.w >= 0.05 - 1e-12).all()
    assert gn.w[0] < gn.w[1]
    G = gn.w * norms
    assert G[0] / G[1] == pytest.approx(1.0, rel=0.35)               # weighted gradient norms are (nearly) equal
    assert (gn.w / len(NAMES)).sum() == pytest.approx(1.0)                                            # what combine() applies sums to 1, as `fixed`


def test_gradnorm_alpha_zero_leaves_balanced_heads_alone_and_alpha_slows_the_fast_learner():
    gn0 = mt.GradNormWeights(NAMES, alpha=0.0)
    _warm(gn0)
    gn0.observe({"a": _L(0.2), "b": _L(1.0)})
    for _ in range(20):
        gn0.update({"a": _L(0.2), "b": _L(1.0)}, np.array([1.0, 1.0]))
    np.testing.assert_allclose(gn0.w, [1.0, 1.0])                    # equal norms, alpha 0 -> target = mean(G) -> nothing to correct

    gn = mt.GradNormWeights(NAMES, alpha=1.5, ema=0.0)               # ema 0 -> the smoothed loss is the latest loss
    _warm(gn)
    gn.update({"a": _L(1.0), "b": _L(1.0)}, np.array([1.0, 1.0]))    # fixes L0 = 1 for both heads
    for _ in range(30):
        gn.observe({"a": _L(0.2), "b": _L(1.0)})                     # head a trains fast (relative inverse training rate low) ...
        gn.update({"a": _L(0.2), "b": _L(1.0)}, np.array([1.0, 1.0]))
    assert gn.w[0] < gn.w[1]                                         # ... so its weight shrinks and the slow head's grows (GradNorm's restoring force)


def test_gradnorm_reference_loss_is_the_second_half_of_the_head_only_phase_and_updates_follow_update_every():
    gn = mt.GradNormWeights(NAMES, update_every=5)
    for v in (4.0, 3.0, 2.0, 1.0):
        gn.observe({"a": _L(v), "b": _L(2 * v)})
    gn.update({"a": _L(1.0), "b": _L(2.0)}, np.array([1.0, 1.0]))
    np.testing.assert_allclose(gn.L0, [1.5, 3.0])                    # mean of the last two warm-up losses
    assert [gn.wants_norms(k) for k in range(7)] == [True, False, False, False, False, True, False]
    assert mt.GradNormWeights(NAMES).wants_norms(3) and mt.GradNormWeights(NAMES).parameters() == [] and mt.GradNormWeights(NAMES).needs_grad_norms


def test_gradnorm_survives_a_zero_gradient_head():
    gn = mt.GradNormWeights(NAMES)
    _warm(gn)
    gn.update({"a": _L(1.0), "b": _L(1.0)}, np.array([0.0, 0.0]))
    assert np.isfinite(gn.w).all() and gn.w.mean() == pytest.approx(1.0)
    gn.update({"a": _L(1.0), "b": _L(1.0)}, np.array([2.0, 0.0]))
    assert np.isfinite(gn.w).all() and gn.w.mean() == pytest.approx(1.0)


def test_make_weighter_builds_every_scheme_and_fixed_still_has_the_loop_interface():
    f = mt.make_weighter({"scheme": "fixed"}, NAMES, {})
    assert not f.needs_grad_norms and f.observe({}) is None and f.wants_norms(0) is False and f.update({}, None) is None
    g = mt.make_weighter({"scheme": "gradnorm", "alpha": 1.5, "lr_w": 0.01, "update_every": 3, "ema": 0.5, "w_min": 0.1}, NAMES, {})
    assert (g.alpha, g.lr, g.every, g.ema, g.w_min) == (1.5, 0.01, 3, 0.5, 0.1)
    with pytest.raises(ValueError):
        mt.make_weighter({"scheme": "pcgrad"}, NAMES, {})


# ---- uncertainty weights (torch) ----------------------------------------------- #
def test_uncertainty_weights_learn_a_smaller_weight_for_the_high_loss_head_and_stay_normalised():
    torch = pytest.importorskip("torch")
    uw = mt.make_weighter({"scheme": "uncertainty"}, NAMES, {"a": "reg", "b": "clf"})
    assert len(uw.parameters()) == 1 and not uw.needs_grad_norms
    opt = torch.optim.SGD(uw.parameters(), lr=0.1)
    for _ in range(50):
        losses = {"a": torch.tensor(5.0, requires_grad=True), "b": torch.tensor(0.1, requires_grad=True)}      # a stays hard, b is already fit
        total, w = uw.combine(losses)
        opt.zero_grad()
        total.backward()
        opt.step()
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert w["a"] < w["b"]                                                                                      # high loss -> large variance -> small weight
    assert float(uw.s[0]) > float(uw.s[1])
    # the model-side gradient uses the normalised weights, exactly like `fixed`
    l1, l2 = torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)
    total, w = uw.combine({"a": l1, "b": l2})
    g1, g2 = torch.autograd.grad(total, [l1, l2])
    assert float(g1) == pytest.approx(w["a"], abs=1e-6) and float(g2) == pytest.approx(w["b"], abs=1e-6)


def test_uncertainty_initial_weights_use_the_regression_half_factor():
    pytest.importorskip("torch")
    uw = mt.make_weighter({"scheme": "uncertainty"}, NAMES, {"a": "reg", "b": "clf"})
    import torch
    _, w = uw.combine({"a": torch.tensor(1.0), "b": torch.tensor(1.0)})
    assert w["a"] == pytest.approx(1 / 3) and w["b"] == pytest.approx(2 / 3)                                   # c = 0.5 (regression) vs 1 (classification), s = 0


# ---- tiny multi-task jobs with each scheme (torch) --------------------------------- #
@pytest.mark.parametrize("scheme", [{"scheme": "uncertainty"}, {"scheme": "gradnorm", "alpha": 0.5, "update_every": 1}])
def test_run_group_with_a_loss_scheme_writes_weights_that_sum_to_one_and_move_and_is_deterministic(world, scheme):
    pytest.importorskip("torch")
    cfg = _mt_cfg(world, loss={**mt.load_config()["train"]["loss"], **scheme})
    md = _group(world)
    cache = {"load": lambda mcfg: (_tiny_encoder(0), False)}
    r = mt.run_group("so762", md, cache, cfg, "w", 0, 0, "cpu", log=lambda *a: None)
    assert np.isfinite(list(r["val"].values())).all()
    g = np.load(mt.group_file(cfg, "w", "so762", 0, 0), allow_pickle=True)
    assert str(g["scheme"]) == scheme["scheme"]
    hw = g["hist_w"]
    np.testing.assert_allclose(hw.sum(1), 1.0, atol=1e-6)
    np.testing.assert_allclose(hw[0], 0.25 if scheme["scheme"] == "gradnorm" else hw[0], atol=1e-9)             # head-only epoch: GradNorm weights untouched
    assert len(hw) > 1 and not np.allclose(hw[1:], 0.25, atol=1e-7)                                              # the scheme actually changed the weights
    mt.run_group("so762", md, cache, cfg, "w2", 0, 0, "cpu", log=lambda *a: None)
    for n in md.names:
        a, b = np.load(mt.job_file(cfg, "w", n, 0, 0), allow_pickle=True), np.load(mt.job_file(cfg, "w2", n, 0, 0), allow_pickle=True)
        key = "val_pred" if md.tasks[n].kind == "reg" else "val_proba"
        np.testing.assert_allclose(a[key].astype(float), b[key].astype(float), atol=1e-4)


def test_gradnorm_works_without_the_us62_diagnostics(world):
    pytest.importorskip("torch")
    cfg = _mt_cfg(world, loss={**mt.load_config()["train"]["loss"], "scheme": "gradnorm"}, diag={"every_steps": 0, "probe": "none"})
    md = _group(world)
    r = mt.run_group("so762", md, {"load": lambda mcfg: (_tiny_encoder(0), False)}, cfg, "nd", 0, 0, "cpu", log=lambda *a: None)
    g = np.load(mt.group_file(cfg, "nd", "so762", 0, 0), allow_pickle=True)
    assert np.isfinite(list(r["val"].values())).all() and (g["hist_gn"] == 0).all()                              # no diagnostics sampled, GradNorm still ran
    assert not np.allclose(g["hist_w"][1:], 0.25, atol=1e-7)


# ---- adoption rule (pandas) ------------------------------------------------------------ #
def _heads(target_flag="within margin", other_flags=("within margin",) * 6, cand_f1=0.69, full_f1=0.654, st_f1=0.68):
    rows = [{"task": "phone_mispron_t1.0", "kind": "clf", "st_val": st_f1, "cand_val": cand_f1, "full_val": full_f1, "val_flag": target_flag}]
    for i, f in enumerate(other_flags):
        rows.append({"task": f"h{i}", "kind": "reg", "st_val": 1.0, "cand_val": 1.0, "full_val": 1.0, "val_flag": f})
    return pd.DataFrame(rows)


def test_candidate_passes_only_when_all_three_declared_criteria_hold():
    ok = R63.candidate_criteria(_heads(), "phone_mispron_t1.0")
    assert ok["passes"] and ok["reason"] == "meets all three"
    assert not R63.candidate_criteria(_heads(target_flag="worse"), "phone_mispron_t1.0")["c1_target_ok"]
    bad = R63.candidate_criteria(_heads(other_flags=("worse", "within margin", "better", "within margin", "within margin", "within margin")), "phone_mispron_t1.0")
    assert not bad["c2_others_ok"] and bad["heads_flagged"] == ["h0"] and not bad["passes"]
    below = R63.candidate_criteria(_heads(cand_f1=0.64, full_f1=0.654), "phone_mispron_t1.0")                     # within margin of single-task but below the full model
    assert below["c1_target_ok"] is True and not below["c3_not_below_full"] and not below["passes"]
    assert not R63.candidate_criteria(_heads().iloc[1:], "phone_mispron_t1.0")["passes"]                         # target head missing


def test_the_full_model_meets_criterion_three_trivially_and_mean_rel_err_uses_error_not_score():
    h = _heads(cand_f1=0.654, full_f1=0.654, st_f1=0.68)
    c = R63.candidate_criteria(h, "phone_mispron_t1.0")
    assert c["c3_not_below_full"]
    # target: error 0.346 vs 0.32 -> ratio 1.08125; six regression heads ratio 1.0
    assert c["mean_rel_err"] == pytest.approx((0.346 / 0.32 + 6) / 7)


def _cands(rows):
    return pd.DataFrame(rows, columns=["name", "passes", "mean_rel_err", "encoders"])


def test_choice_picks_lowest_error_then_fewest_encoders_within_the_tie_margin_or_none():
    pick, _ = R63.choose_candidate(_cands([("regroup", True, 0.95, 2), ("gradnorm", True, 0.99, 1), ("full", False, 0.90, 1)]), 0.01)
    assert pick == "regroup"                                                                                      # failing candidates never win; 0.04 apart is not a tie
    pick, why = R63.choose_candidate(_cands([("regroup", True, 0.950, 2), ("gradnorm", True, 0.955, 1)]), 0.01)
    assert pick == "gradnorm" and "fewest encoders" in why                                                       # tied -> cheaper
    pick, _ = R63.choose_candidate(_cands([("regroup", True, 0.950, 2), ("gradnorm", True, 0.962, 1)]), 0.01)
    assert pick == "regroup"                                                                                      # outside the tie margin
    pick, why = R63.choose_candidate(_cands([("regroup", False, 0.95, 2), ("full", False, 0.99, 1)]), 0.01)
    assert pick is None and "side by side" in why


def test_candidate_table_never_adopts_an_incomplete_candidate():
    extra = dict(candidate="x", encoders=1, d_vs_st=0.0, verdict_vs_st="no significant change", verdict_vs_full="-", cand_test=0.7)
    c63 = {"group": "so762", "target_head": "phone_mispron_t1.0"}
    n_expected = len(mt.load_config()["groups"]["so762"]["tasks"])                                                 # 7
    full = _heads(other_flags=("within margin",) * (n_expected - 1)).assign(**extra)
    ct = R63.candidate_table(full, c63)
    assert bool(ct.loc[0, "complete"]) and bool(ct.loc[0, "passes"]) and ct.loc[0, "heads_found"] == n_expected
    part = R63.candidate_table(full.iloc[:3], c63)                                                                 # criteria hold on the 3 heads present, but 4 are missing
    assert not bool(part.loc[0, "complete"]) and not bool(part.loc[0, "passes"])


def test_mean_weights_skip_the_head_only_epoch():
    hw = np.array([[0.5, 0.5], [0.2, 0.8], [0.4, 0.6]])
    m, last = R63.mean_weights(hw)
    np.testing.assert_allclose(m, [0.3, 0.7])
    np.testing.assert_allclose(last, [0.4, 0.6])
    np.testing.assert_allclose(R63.mean_weights(hw[:1])[0], [0.5, 0.5])


# ---- configuration ----------------------------------------------------------------------- #
def test_us63_variants_candidates_and_defaults():
    base = mt.load_config()
    c63 = base["compare63"]
    assert base["train"]["loss"]["scheme"] == "fixed"                                                              # the full model and US 6.1/6.2 are unchanged
    so = base["groups"]["so762"]["tasks"]
    sent4, wp = mt.load_config("sent4")["groups"]["so762"]["tasks"], mt.load_config("word_phone")["groups"]["so762"]["tasks"]
    assert sorted(sent4 + wp) == sorted(so) and not set(sent4) & set(wp) and len(sent4) == 4 and len(wp) == 3
    assert c63["target_head"] in wp and c63["group"] == "so762"
    assert mt.load_config("uncertainty")["train"]["loss"]["scheme"] == "uncertainty"
    g05, g15 = mt.load_config("gradnorm_a05")["train"]["loss"], mt.load_config("gradnorm_a15")["train"]["loss"]
    assert (g05["scheme"], g05["alpha"], g15["scheme"], g15["alpha"]) == ("gradnorm", 0.5, "gradnorm", 1.5)
    for v in ("sent4", "uncertainty", "gradnorm_a05", "gradnorm_a15"):                                              # same recipe as the full model, only the declared change
        c = mt.load_config(v)
        assert c["train"]["lr_encoder"] == base["train"]["lr_encoder"] and c["model"] == base["model"] and c["l2_seeds"] == base["l2_seeds"]
    for name, cand in c63["candidates"].items():
        heads = sorted(h for part in cand["parts"] for h in mt.load_config(part["variant"])["groups"]["so762"]["tasks"])
        assert heads == sorted(so), name                                                                           # every candidate gives every SO762 head a prediction exactly once
        assert cand["encoders"] == len(cand["parts"])
        for part in cand["parts"]:
            assert part["tag"] == "mt_base" or part["tag"].startswith(("mt63_", "mt62_"))
    assert R63.part_sources(c63, c63["candidates"]["regroup"]).keys() == set(so)
    assert c63["margin_floor"] == base["compare61"]["margin_floor"] and c63["tie_margin"] == 0.01
