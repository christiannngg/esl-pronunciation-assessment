"""Sprint 5 (US 5.3) tests: noise augmentation, spike detector, declared selection rule. Numpy only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation import report_us53 as R
from src.models import finetune as ft

T = ft.load_config()["diagnose53"]


def test_add_noise_hits_the_requested_snr_and_does_not_mutate():
    rng = np.random.default_rng(0)
    w = np.random.default_rng(1).normal(size=64000).astype(np.float32)
    w0 = w.copy()
    out = ft.add_noise(w, rng, (20.0, 20.0))
    assert (w == w0).all() and out.shape == w.shape and out.dtype == np.float32
    snr = 10 * np.log10(np.mean(w.astype(float) ** 2) / np.mean((out - w).astype(float) ** 2))
    assert snr == pytest.approx(20.0, abs=0.3)


def test_max_spike_ignores_the_head_epoch_and_finds_blowups():
    assert R.max_spike([1.5, 0.6, 0.4, 0.3], 1) == pytest.approx(0.75)                  # decreasing loss: largest ratio is 0.3 / 0.4
    assert R.max_spike([1.5, 0.3, 0.2, 0.95, 0.6], 1) == pytest.approx(0.95 / 0.2)      # the sent_total pattern of 5.1
    assert R.max_spike([1.0], 1) == 1.0


def _s(val, gap=0.2, ratio=1.1, spike=1.0, fit=100.0, n=3, sd=0.01):
    return {"n_jobs": n, "val": val, "gap_best": gap, "loss_ratio": ratio, "spike": spike, "fit_s": fit, "seed_sd": sd}


REG, CLF = SimpleNamespace(kind="reg", name="sent_accuracy"), SimpleNamespace(kind="clf", name="phone_mispron_t1.0")


def test_target_met_uses_the_declared_caps():
    assert R.target_met("reg", _s(1.0, gap=0.29, ratio=1.24), T)
    assert not R.target_met("reg", _s(1.0, gap=0.31), T) and not R.target_met("reg", _s(1.0, ratio=1.26), T) and not R.target_met("reg", _s(1.0, spike=4.7), T)
    assert R.target_met("clf", _s(0.7, ratio=1.49), T) and not R.target_met("clf", _s(0.7, ratio=1.51), T)


def test_select_picks_the_best_qualifying_variant_and_respects_the_margin():
    base = _s(1.00, gap=0.35, ratio=1.3)
    v = {"lr": _s(1.005, gap=0.25, ratio=1.1), "reg": _s(0.99, gap=0.28, ratio=1.2), "mask": _s(1.05, gap=0.1, ratio=1.0), "noise": _s(0.98, gap=0.4, ratio=1.1)}
    r = R.select(REG, base, v, T)
    assert r["need_fix"] and r["selected"] == "reg"                       # mask is worse than the margin, noise misses the gap target, reg has the best validation of the qualifying
    assert r["margin"] == pytest.approx(0.01) and r["n_eligible"] == 3


def test_select_direction_for_f1_and_margin_from_seed_sd():
    base = _s(0.70, ratio=2.6, sd=0.02)
    v = {"lr": _s(0.685, ratio=1.4), "reg": _s(0.67, ratio=1.1)}          # F1: 0.685 is 0.015 worse (inside 0.02 margin), 0.67 is 0.03 worse (outside)
    r = R.select(CLF, base, v, T)
    assert r["selected"] == "lr" and r["margin"] == pytest.approx(0.02)


def test_select_keeps_base_when_nothing_qualifies_or_nothing_to_fix_or_incomplete():
    base = _s(1.00, gap=0.35)
    assert R.select(REG, base, {"lr": _s(1.0, gap=0.33)}, T)["selected"] == "base"                      # no variant meets the target
    ok_base = _s(1.00, gap=0.2)
    r = R.select(REG, ok_base, {"lr": _s(0.9, gap=0.1)}, T)
    assert r["selected"] == "base" and not r["need_fix"]                                                  # base already meets the targets: do not chase validation noise
    assert R.select(REG, base, {"lr": _s(0.9, gap=0.1, n=2)}, T)["selected"] == "base"                  # incomplete runs are not eligible
    assert R.select(REG, base, {}, T)["selected"] == "base"


def test_cost_tasks_prefer_a_cheaper_non_inferior_variant():
    task = SimpleNamespace(kind="reg", name="phone_score")
    base = _s(0.31, gap=0.25, ratio=1.1, fit=100.0, sd=0.005)
    r = R.select(task, base, {"freeze6": _s(0.312, fit=70.0), "freeze9": _s(0.312, fit=50.0), "lr": _s(0.30, fit=90.0)}, T)
    assert r["selected"] == "freeze9"                                                                       # cheapest eligible frozen-bottom variant, lr is not a cost variant
    assert R.select(task, base, {"freeze6": _s(0.312, fit=90.0)}, T)["selected"] == "base"                  # only 10 % faster
    assert R.select(task, base, {"freeze9": _s(0.33, fit=40.0)}, T)["selected"] == "base"                   # faster but worse than the margin


def test_within_fold_sd_ignores_single_seed_folds():
    rows = [{"fold": 0, "val_at_best": 1.0}, {"fold": 0, "val_at_best": 1.2}, {"fold": 0, "val_at_best": 1.1}]
    assert R.within_fold_sd(rows) == pytest.approx(0.1)
    assert np.isnan(R.within_fold_sd([{"fold": 0, "val_at_best": 1.0}, {"fold": 1, "val_at_best": 2.0}]))
