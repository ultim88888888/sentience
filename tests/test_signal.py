import numpy as np
import pandas as pd
from study import signal


def _toy_inputs():
    months = pd.period_range("2024-01", "2024-04", freq="M")
    coverage = pd.DataFrame([
        {"month": m, "basket": b, "coverage_share": 0.0,
         "coverage_momentum": v}
        for m, vs in zip(months, [(0.1, -0.1), (0.2, -0.2), (0.0, 0.0), (0.0, 0.0)])
        for b, v in zip(["A", "B"], vs)
    ])
    basket_rets = pd.DataFrame([
        {"month": m, "basket": b, "ret": v, "n_live": 2}
        for m, vs in zip(months, [(0.05, 0.01), (0.06, 0.00), (0.02, 0.03), (0.01, 0.02)])
        for b, v in zip(["A", "B"], vs)
    ])
    return coverage, basket_rets


def test_forward_relative_return_no_lookahead():
    coverage, basket_rets = _toy_inputs()
    panel = signal.basket_signal_panel(coverage, basket_rets, benchmark="L1 Majors")
    # fwd_rel_1m at month=2024-01 must use month=2024-02 returns, demeaned across baskets.
    jan = panel[panel["month"] == pd.Period("2024-01", "M")].set_index("basket")
    feb_mean = (0.06 + 0.00) / 2
    assert abs(jan.loc["A", "fwd_rel_1m"] - (0.06 - feb_mean)) < 1e-9
    assert abs(jan.loc["B", "fwd_rel_1m"] - (0.00 - feb_mean)) < 1e-9


def test_last_month_forward_return_is_nan():
    coverage, basket_rets = _toy_inputs()
    panel = signal.basket_signal_panel(coverage, basket_rets, benchmark="L1 Majors")
    apr = panel[panel["month"] == pd.Period("2024-04", "M")]
    assert apr["fwd_rel_1m"].isna().all()  # no month after April
