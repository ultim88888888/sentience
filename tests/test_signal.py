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


def test_forward_3m_relative_return_no_lookahead():
    # 6-month series so 2024-01 has a complete 3-month forward window (Feb,Mar,Apr).
    months = pd.period_range("2024-01", "2024-06", freq="M")
    a_rets = [0.05, 0.06, 0.02, 0.01, 0.03, 0.04]
    b_rets = [0.01, 0.00, 0.03, 0.02, 0.01, 0.02]
    coverage = pd.DataFrame([
        {"month": m, "basket": b, "coverage_share": 0.0, "coverage_momentum": 0.0}
        for m in months for b in ["A", "B"]
    ])
    basket_rets = pd.DataFrame(
        [{"month": m, "basket": "A", "ret": a_rets[i], "n_live": 2} for i, m in enumerate(months)]
        + [{"month": m, "basket": "B", "ret": b_rets[i], "n_live": 2} for i, m in enumerate(months)]
    )
    panel = signal.basket_signal_panel(coverage, basket_rets, benchmark="L1 Majors")
    jan = panel[panel["month"] == pd.Period("2024-01", "M")].set_index("basket")
    # window covers Feb,Mar,Apr -> compounded
    a_cum = (1.06 * 1.02 * 1.01) - 1
    b_cum = (1.00 * 1.03 * 1.02) - 1
    mean_cum = (a_cum + b_cum) / 2
    assert abs(jan.loc["A", "fwd_rel_3m"] - (a_cum - mean_cum)) < 1e-9
    assert abs(jan.loc["B", "fwd_rel_3m"] - (b_cum - mean_cum)) < 1e-9


def test_three_basket_demean_is_not_symmetric():
    # With 3 baskets the cross-sectional demean is NOT a simple sign flip; this catches
    # any axis inversion in fwd.sub(fwd.mean(axis=1), axis=0).
    months = pd.period_range("2024-01", "2024-02", freq="M")
    feb = {"A": 0.10, "B": 0.04, "C": 0.01}   # mean 0.05
    coverage = pd.DataFrame([
        {"month": m, "basket": b, "coverage_share": 0.0, "coverage_momentum": 0.0}
        for m in months for b in ["A", "B", "C"]
    ])
    rows = []
    for m in months:
        for b in ["A", "B", "C"]:
            r = feb[b] if m == pd.Period("2024-02", "M") else 0.0
            rows.append({"month": m, "basket": b, "ret": r, "n_live": 2})
    basket_rets = pd.DataFrame(rows)
    panel = signal.basket_signal_panel(coverage, basket_rets, benchmark="L1 Majors")
    jan = panel[panel["month"] == pd.Period("2024-01", "M")].set_index("basket")
    assert abs(jan.loc["A", "fwd_rel_1m"] - 0.05) < 1e-9   # 0.10 - 0.05
    assert abs(jan.loc["B", "fwd_rel_1m"] - (-0.01)) < 1e-9  # 0.04 - 0.05
    assert abs(jan.loc["C", "fwd_rel_1m"] - (-0.04)) < 1e-9  # 0.01 - 0.05
