"""US 8.3 selection rule on synthetic out-of-fold predictions (no GPU, no lab files)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import report_us83 as R


def make_frames(rng, n_spk=40, per=30, flip_child=0.30, flip_adult=0.10, reg_noise=0.3):
    """Pooled OOF frames for every head: 20 adult + 20 child/teen speakers over 5 folds. Mispronunciation errors are flip rates per group."""
    spk = [f"s{i:03d}" for i in range(n_spk)]
    grp = np.array(["adult"] * (n_spk // 2) + ["child/teen"] * (n_spk // 2))
    fold = np.tile(np.arange(5), n_spk // 5 + 1)[:n_spk]
    rows = np.arange(n_spk * per)
    sp_row, g_row, f_row = np.repeat(spk, per), np.repeat(grp, per), np.repeat(fold, per)
    frames = {}
    for h in R.HEADS:
        if h == R.TARGET:
            y = (rng.random(len(rows)) < 0.3).astype(int)
            frames[h] = {"y": y}
        else:
            y = rng.normal(8, 1.5, len(rows))
            frames[h] = {"y": y}
    base = pd.DataFrame({"row": rows, "speaker": sp_row, "age_group": g_row, "fold": f_row})
    return base, frames


def build(rng, base, frames, child_flip, adult_flip, nt_noise=0.3, seed_shift=0):
    out = {}
    for h, d in frames.items():
        y = d["y"]
        if h == R.TARGET:
            p = y.copy()
            r = np.random.default_rng(seed_shift + 1).random(len(y))
            flip = np.where(base["age_group"] == "child/teen", child_flip, adult_flip)
            p = np.where(r < flip, 1 - y, y)
        else:
            p = y + np.random.default_rng(seed_shift + 2).normal(0, nt_noise, len(y))
        out[h] = base.assign(y=y, pred=p)
    return out


def test_error_is_rmse_or_one_minus_f1():
    assert R.error("sent_accuracy", [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(np.sqrt(4 / 3))
    assert R.error(R.TARGET, [0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert R.error(R.TARGET, [0, 1], [1, 0]) == pytest.approx(1.0)


def test_margin_has_the_floor():
    rng = np.random.default_rng(0)
    base, frames = make_frames(rng)
    f0 = build(rng, base, frames, 0.3, 0.1)
    assert R.margin("sent_accuracy", f0["sent_accuracy"]) >= R.MARGIN_FLOOR


def test_clear_winner_qualifies_and_is_selected():
    rng = np.random.default_rng(1)
    base, frames = make_frames(rng, n_spk=60, per=60)
    oof = {"f0": build(rng, base, frames, 0.30, 0.10, seed_shift=0),
           "f1_s10": build(rng, base, frames, 0.10, 0.10, seed_shift=10)}        # child/teen errors fall to the adult level, rest unchanged
    table, pick = R.run_selection(oof, 200, 0)
    r = table.iloc[0]
    assert r["child_delta"] < -0.05 and r["child_hi"] < 0 and r["eligible"] and r["qualifies"]
    assert pick["winner"] == "f1_s10" and pick["evaluate"] == "f1_s10"


def test_a_candidate_that_hurts_a_non_target_head_is_not_eligible():
    rng = np.random.default_rng(2)
    base, frames = make_frames(rng, n_spk=60, per=60)
    oof = {"f0": build(rng, base, frames, 0.30, 0.10, nt_noise=0.3, seed_shift=0),
           "f1_s10": build(rng, base, frames, 0.10, 0.10, nt_noise=0.6, seed_shift=10)}    # a much noisier regression head
    table, pick = R.run_selection(oof, 200, 0)
    assert not table.iloc[0]["eligible"] and "sent_accuracy" in table.iloc[0]["non_target_worse"]
    assert pick["winner"] is None and pick["evaluate"] == "f1_s10"                       # still evaluated once, reported as negative


def test_candidate_that_hurts_adults_is_not_eligible():
    rng = np.random.default_rng(3)
    base, frames = make_frames(rng, n_spk=60, per=60)
    oof = {"f0": build(rng, base, frames, 0.30, 0.10, seed_shift=0),
           "f2_a": build(rng, base, frames, 0.10, 0.30, seed_shift=10)}                  # trades adult accuracy for child/teen accuracy
    table, pick = R.run_selection(oof, 200, 0)
    assert table.iloc[0]["child_delta"] < -0.05 and not table.iloc[0]["adult_ok"] and pick["winner"] is None


def test_no_change_is_inconclusive():
    rng = np.random.default_rng(4)
    base, frames = make_frames(rng, n_spk=60, per=60)
    oof = {"f0": build(rng, base, frames, 0.20, 0.10, seed_shift=0), "f1_s05": build(rng, base, frames, 0.20, 0.10, seed_shift=0)}
    table, pick = R.run_selection(oof, 200, 0)
    assert abs(table.iloc[0]["child_delta"]) < 1e-9 and not table.iloc[0]["qualifies"] and pick["winner"] is None


def test_tie_goes_to_the_fewest_extra_mechanisms():
    table = pd.DataFrame([
        {"candidate": "f2_b", "child_delta": -0.100, "margin_target": 0.02, "qualifies": True, "eligible": True},
        {"candidate": "f1_s05", "child_delta": -0.095, "margin_target": 0.02, "qualifies": True, "eligible": True},
        {"candidate": "f2_a", "child_delta": -0.020, "margin_target": 0.02, "qualifies": True, "eligible": True}])
    pick = R.select(table)
    assert pick["winner"] == "f1_s05" and set(pick["tied"]) == {"f2_b", "f1_s05"}


def test_misaligned_rows_are_refused():
    rng = np.random.default_rng(5)
    base, frames = make_frames(rng, n_spk=40, per=20)
    a = build(rng, base, frames, 0.3, 0.1)
    b = {h: d.assign(row=d["row"] + 1) for h, d in a.items()}
    with pytest.raises(ValueError):
        R.run_selection({"f0": a, "f1_s05": b}, 50, 0)


def _row(task, target, od, olo, ohi, ad, alo, ahi, cd, clo, chi):
    return {"task": task, "target": target, "overall_delta": od, "overall_lo": olo, "overall_hi": ohi, "adult_delta": ad, "adult_lo": alo, "adult_hi": ahi,
            "child/teen_delta": cd, "child/teen_lo": clo, "child/teen_hi": chi}


def test_acceptance_rules():
    ok = pd.DataFrame([_row("sent_total", False, 0.0, -0.02, 0.02, 0, -0.02, 0.02, 0, -0.03, 0.03),
                       _row(R.TARGET, True, -0.02, -0.04, 0.0, 0.0, -0.02, 0.02, -0.06, -0.10, -0.02)])
    v = R.acceptance(ok)
    assert v["accepted"] and v["rule_i_child_target_improves_and_none_worse"]
    # child/teen interval touching 0: not an improvement
    nope = ok.copy(); nope.loc[1, "child/teen_hi"] = 0.01
    assert not R.acceptance(nope)["accepted"]
    # a head significantly worse overall
    bad = ok.copy(); bad.loc[0, ["overall_delta", "overall_lo", "overall_hi"]] = [0.05, 0.02, 0.08]
    v = R.acceptance(bad)
    assert not v["rule_ii_no_head_significantly_worse_overall"] and v["worse_heads"] == ["sent_total"] and not v["accepted"]
    # adult target significantly worse
    ad = ok.copy(); ad.loc[1, ["adult_delta", "adult_lo", "adult_hi"]] = [0.05, 0.02, 0.08]
    assert not R.acceptance(ad)["rule_iii_adult_target_not_worse"]


def test_threshold_reference_removes_a_pure_calibration_gap():
    rng = np.random.default_rng(0)
    n_spk, per = 40, 80
    spk = np.repeat([f"s{i}" for i in range(n_spk)], per)
    grp = np.repeat(np.array(["adult"] * 20 + ["child/teen"] * 20), per)
    fold = np.repeat(np.tile(np.arange(5), 8), per)
    y = (rng.random(n_spk * per) < 0.3).astype(int)
    score = np.clip(0.5 + 0.35 * (2 * y - 1) + rng.normal(0, 0.12, len(y)), 0, 1)             # same ranking quality in both groups ...
    p = np.where(grp == "child/teen", score - 0.25, score)                                    # ... but child/teen scores sit lower (calibration gap)
    f = pd.DataFrame({"row": np.arange(len(y)), "speaker": spk, "age_group": grp, "fold": fold, "y": y, "p": np.clip(p, 0, 1)})
    t = R.threshold_reference(f, 100, 0).set_index("rule")
    assert t.loc["default (0.5)", "child/teen_error"] > t.loc["per-group thresholds", "child/teen_error"] + 0.05
    assert abs(t.loc["per-group thresholds", "adult_delta_vs_default"]) < 0.05                # adults unaffected
    assert t.loc["per-group thresholds", "gap"] < t.loc["default (0.5)", "gap"]
