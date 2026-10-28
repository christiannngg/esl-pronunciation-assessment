"""US 7.1 aggregation: the declared carry-forward rule and the small pure helpers of src/evaluation/report_us71.py (numpy / pandas only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import report_us71 as R


def _row(task, system, test, val, enc=None):
    return {"task": task, "system": system, "test": test, "val": val, "encoders": R.encoders_for(system, task) if enc is None else enc}


def test_error_and_verdict():
    assert R.error_of("sent_accuracy", 1.1) == 1.1
    assert abs(R.error_of("phone_mispron_t1.0", 0.7) - 0.3) < 1e-12
    assert R.verdict("sent_accuracy", -0.05, -0.08, -0.02) == "better"
    assert R.verdict("sent_accuracy", 0.05, 0.02, 0.08) == "worse"
    assert R.verdict("l2_word_error", 0.05, 0.02, 0.08) == "better"
    assert R.verdict("l2_word_error", 0.01, -0.02, 0.04) == "no significant change"
    assert R.verdict("l2_word_error", np.nan, np.nan, np.nan) == "n/a"


def test_encoders():
    assert R.encoders_for("FT-base", "sent_total") == 7 and R.encoders_for("FT-stab", "l2_word_error") == 2
    assert R.encoders_for("MT-GN15", "sent_total") == 1 and R.encoders_for("MT-regroup", "phone_score") == 2 and R.encoders_for("B", "phone_score") == 0


def test_tie_goes_to_fewest_encoders_then_fewest_mechanisms():
    t = "sent_total"
    long = pd.DataFrame([_row(t, "FT-base", 1.07, 0.920), _row(t, "FT-stab", 1.08, 0.915), _row(t, "MT-full", 1.05, 0.925), _row(t, "MT-GN15", 1.06, 0.918)])
    cf = R.carry_forward(long, {t: 0.02}, {}).iloc[0]
    assert cf["val_leader"] == "MT-full"                      # all within the margin: one encoder beats seven, plain beats GradNorm
    assert cf["single_task_ref"] == "FT-base"                 # base and stab tied: no extra mechanism wins
    assert "MT-GN15" in cf["carried"] and "FT-base" in cf["carried"]


def test_clear_validation_leader_wins_and_disagreement_is_flagged():
    t = "sent_fluency"
    long = pd.DataFrame([_row(t, "FT-base", 0.870, 0.700), _row(t, "FT-stab", 0.861, 0.655), _row(t, "MT-full", 0.887, 0.680), _row(t, "MT-GN15", 0.850, 0.690)])
    cf = R.carry_forward(long, {t: 0.01}, {}).iloc[0]
    assert cf["val_leader"] == "FT-stab" and cf["single_task_ref"] == "FT-stab"
    assert cf["test_leader"] == "MT-GN15" and bool(cf["validation_test_disagree"])       # test never overrides validation


def test_ineligible_candidate_is_excluded():
    t = "phone_mispron_t1.0"
    long = pd.DataFrame([_row(t, "FT-base", 0.680, 0.697), _row(t, "FT-stab", 0.680, 0.697), _row(t, "MT-full", 0.654, 0.700)])
    cf = R.carry_forward(long, {t: 0.0}, {(t, "MT-full"): True}).iloc[0]
    assert cf["val_leader"] != "MT-full" and cf["ineligible"] == "MT-full"


def test_l2_carries_ft_stab_and_mt_full():
    t = "l2_phone_error"
    long = pd.DataFrame([_row(t, "FT-base", 0.559, 0.566), _row(t, "FT-stab", 0.576, 0.576), _row(t, "MT-full", 0.577, 0.576)])
    assert R.carry_forward(long, {t: 0.01}, {}).iloc[0]["carried"] == "FT-stab, MT-full"


def test_pick_baseline_uses_validation_not_test():
    df = pd.DataFrame([{"task": "sent_total", "model": "rf", "fix": "pros", "val_primary_after": 1.00, "after_rmse": 1.18, "after_macro_f1": np.nan},
                       {"task": "sent_total", "model": "svr", "fix": "pros", "val_primary_after": 1.10, "after_rmse": 1.10, "after_macro_f1": np.nan},
                       {"task": "l2_word_error", "model": "logreg", "fix": "cw", "val_primary_after": 0.62, "after_rmse": np.nan, "after_macro_f1": 0.615},
                       {"task": "l2_word_error", "model": "svm", "fix": "cw", "val_primary_after": 0.60, "after_rmse": np.nan, "after_macro_f1": 0.700}])
    b = R.pick_baseline(df).set_index("task")
    assert b.loc["sent_total", "test"] == 1.18 and b.loc["l2_word_error", "test"] == 0.615
