import numpy as np
import pandas as pd
from study import study_basket


def _panel(corr_sign):
    """Build a panel where signal rank perfectly (anti)correlates with fwd return rank."""
    months = pd.period_range("2024-01", "2024-06", freq="M")
    rows = []
    for m in months:
        for i, b in enumerate(["A", "B", "C"]):
            mom = float(i)
            fwd = corr_sign * float(i)
            rows.append({"month": m, "basket": b, "coverage_momentum": mom,
                         "fwd_rel_1m": fwd, "fwd_rel_3m": fwd})
    return pd.DataFrame(rows)


def test_ic_positive_when_signal_predicts():
    ic = study_basket.information_coefficient(_panel(+1), "coverage_momentum", "fwd_rel_1m")
    assert ic["ic_mean"] > 0.99
    assert ic["n"] == 6


def test_ic_negative_when_signal_anticorrelated():
    ic = study_basket.information_coefficient(_panel(-1), "coverage_momentum", "fwd_rel_1m")
    assert ic["ic_mean"] < -0.99


def test_verdict_pulse_on_positive_consistent():
    panel = _panel(+1)
    res = study_basket.run_study_a(panel)
    assert res["verdict"] == "pulse"
