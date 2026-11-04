"""US 8.2: ceiling, label-matched comparison, ranking / calibration split, ablation helpers (numpy / pandas only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import report_us82 as D


def test_auc_matches_brute_force_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)                              # rounding creates ties
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert D.auc(y, s) == pytest.approx(brute)
    assert np.isnan(D.auc(np.zeros(5, int), np.arange(5.0)))


def test_average_precision_perfect_and_random_ordering():
    y = np.array([1, 1, 0, 0, 0])
    assert D.average_precision(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(1.0)
    assert D.average_precision(y, np.array([0.1, 0.2, 0.9, 0.8, 0.3])) < 0.5


def test_recall_at_fpr():
    y = np.r_[np.zeros(100, int), np.ones(10, int)]
    s = np.r_[np.arange(100) / 100.0, np.linspace(0.5, 1.5, 10)]
    r = D.recall_at_fpr(y, s, 0.10)                                         # threshold near 0.90: positives above it
    assert r == pytest.approx(np.mean(np.linspace(0.5, 1.5, 10) > np.quantile(np.arange(100) / 100.0, 0.9)))
    assert D.recall_at_fpr(y, s, 0.0) == pytest.approx(np.mean(np.linspace(0.5, 1.5, 10) > 0.99))


def test_ranking_and_calibration_parts_add_up_to_the_recall_gap():
    rng = np.random.default_rng(1)
    n = 4000
    grp = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]              # True = child
    y = (rng.random(n) < np.where(grp, 0.03, 0.06)).astype(int)
    s = rng.normal(size=n) + np.where(grp, 1.0, 1.8) * y                     # children are ranked worse
    pred = (s > np.where(grp, 1.6, 1.0)).astype(int)                         # and the pooled-style threshold is stricter for them
    m = D.op_metrics(y, s, pred, np.flatnonzero(~grp), np.flatnonzero(grp))
    assert m["ranking_part"][0] + m["calibration_part"][0] == pytest.approx(m["total_recall_gap"][0])
    assert m["auc"][0] > m["auc"][1] and m["ranking_part"][0] > 0 and m["calibration_part"][0] > 0


def test_matched_weights_reproduce_the_child_label_distribution():
    rng = np.random.default_rng(2)
    n = 6000
    child = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    y = np.where(child, rng.integers(5, 9, n), rng.integers(0, 11, n)).astype(float)
    w, cov = D.matched_weights(y, ~child, child)
    assert cov == pytest.approx(1.0) and np.all(w[child] == 0)
    for v in range(5, 9):
        assert (w[(~child) & (y == v)].sum() / w[~child].sum()) == pytest.approx(np.mean(y[child] == v), abs=1e-9)
    assert w[(~child) & (y == 0)].sum() == 0                                  # adult values the children never take get no weight


def test_label_matching_removes_a_pure_range_restriction_but_keeps_a_real_compression():
    rng = np.random.default_rng(3)
    spk_n, per = 30, 80
    spk = np.repeat([f"s{i:02d}" for i in range(2 * spk_n)], per)
    child = np.repeat(np.arange(2 * spk_n) >= spk_n, per)
    y = np.where(child, rng.integers(5, 9, len(spk)), rng.integers(0, 11, len(spk))).astype(float)
    same = 0.6 * y + rng.normal(0, 0.8, len(y))                              # same relation in both groups; children only have a narrower label range
    r = D.matched_compare("sent_accuracy", y, same, spk, ~child, child, 200, 0)
    assert r["pcc"]["raw_gap"] > 0.15 and abs(r["pcc"]["matched_gap"]) < 0.06 and r["pcc"]["matched_ci"][0] < 0 < r["pcc"]["matched_ci"][1]
    assert abs(r["slope"]["raw_gap"]) < 0.06                                  # slope was never reduced by the narrower range
    compressed = np.where(child, 0.2 * y, 0.6 * y) + rng.normal(0, 0.8, len(y))
    r2 = D.matched_compare("sent_accuracy", y, compressed, spk, ~child, child, 200, 0)
    assert r2["slope"]["matched_gap"] > 0.3 and r2["slope"]["matched_ci"][0] > 0 and r2["pcc"]["matched_gap"] > 0.1


def test_leave_one_out_ceiling():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=2000)
    assert D.loo_r(np.column_stack([truth] * 5)) == pytest.approx(1.0)
    noisy = np.column_stack([truth + rng.normal(0, 1.0, 2000) for _ in range(5)])
    r = D.loo_r(noisy)
    assert 0.6 < r < 0.8                                                      # a rater against the mean of four noisy raters
    assert D.loo_r(noisy[:, :2]) < r                                          # fewer raters, lower ceiling


def test_consensus_reliability_recovers_the_simulated_reliability():
    rng = np.random.default_rng(6)
    n, k, noise_sd = 20000, 5, 1.0
    truth = rng.normal(size=n)
    raters = np.column_stack([truth + rng.normal(0, noise_sd, n) for _ in range(k)])
    consensus = raters.mean(1)
    true_rel = np.corrcoef(truth, consensus)[0, 1] ** 2                        # reliability of the mean of five
    assert D.consensus_reliability(D.loo_r(raters), k) == pytest.approx(true_rel, abs=0.02)
    assert D.consensus_reliability(0.6) < D.consensus_reliability(0.8) < 1.0
