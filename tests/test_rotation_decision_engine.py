import math

import numpy as np

from rotation_decision_engine import (
    SignalBundle,
    align_score,
    apply_crowding_penalty,
    arrow_for,
    compute_signals,
    crowding_zscore,
    recommendation_plan,
)


def test_signal_alignment_mapping():
    momentums = {"4W": 12.0, "12W": 10.0, "6M": 6.0, "12M": 4.0, "2Y": 2.0, "3Y": 3.0}
    st, mt, lt = compute_signals(momentums)
    align = align_score(st, mt, lt)
    assert align > 0.15
    assert arrow_for(align) == "↑"

    momentums_neg = {"4W": -6.0, "12W": -5.0, "6M": -4.0, "12M": -3.0, "2Y": -2.0, "3Y": -1.5}
    st_n, mt_n, lt_n = compute_signals(momentums_neg)
    align_n = align_score(st_n, mt_n, lt_n)
    assert align_n < -0.15
    assert arrow_for(align_n) == "↓"


def test_crowding_zscore_computation():
    history = [10.0, 12.0, 11.0, 13.0, 12.5]
    z = crowding_zscore(14.0, history)
    expected = (14.0 - np.mean(history)) / np.std(history)
    assert math.isclose(z, expected)


def test_recommendations_balance_and_caps():
    bundles = [
        SignalBundle("A", 0, 0, 0, 0.6, arrow_for(0.6), None, False, 0.6, 10.0),
        SignalBundle("B", 0, 0, 0, 0.4, arrow_for(0.4), None, False, 0.4, 10.0),
        SignalBundle("C", 0, 0, 0, -0.3, arrow_for(-0.3), None, False, -0.3, 10.0),
        SignalBundle("D", 0, 0, 0, -0.6, arrow_for(-0.6), None, False, -0.6, 10.0),
        SignalBundle("E", 0, 0, 0, 0.05, arrow_for(0.05), None, False, 0.05, 10.0),
    ]
    plan = recommendation_plan(bundles)
    deltas = [delta for _, delta in (plan["INCREASE"] + plan["HOLD"] + plan["REDUCE"])]
    assert abs(sum(deltas)) < 0.5
    assert all(-1.5 <= d <= 1.5 for d in deltas)

    align_penalized, crowded = apply_crowding_penalty(0.5, 1.2)
    assert crowded is True
    assert align_penalized < 0.5
