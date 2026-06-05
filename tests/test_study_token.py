import pytest
import pandas as pd
from study import study_token


def test_sum_aggregation_rewards_multi_basket_membership(baskets_cfg):
    # STRK is in ZK/Privacy AND L2/Scaling; ARB only in L2. With both baskets hot,
    # STRK conviction (sum) must exceed ARB conviction.
    coverage = pd.DataFrame([
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_momentum": 0.3},
        {"month": pd.Period("2024-02", "M"), "basket": "L2/Scaling", "coverage_momentum": 0.2},
        {"month": pd.Period("2024-02", "M"), "basket": "DeFi", "coverage_momentum": 0.0},
    ])
    conv = study_token.token_conviction(coverage, baskets_cfg, agg="sum")
    feb = conv[conv["month"] == pd.Period("2024-02", "M")].set_index("token")["conviction"]
    assert feb["STRK"] == 0.5   # 0.3 + 0.2
    assert feb["ARB"] == 0.2
    assert feb["STRK"] > feb["ARB"]


def test_mean_aggregation_normalizes(baskets_cfg):
    coverage = pd.DataFrame([
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_momentum": 0.3},
        {"month": pd.Period("2024-02", "M"), "basket": "L2/Scaling", "coverage_momentum": 0.2},
    ])
    conv = study_token.token_conviction(coverage, baskets_cfg, agg="mean")
    feb = conv[conv["month"] == pd.Period("2024-02", "M")].set_index("token")["conviction"]
    assert abs(feb["STRK"] - 0.25) < 1e-9   # mean(0.3, 0.2)


def test_benchmark_tokens_never_in_conviction(baskets_cfg):
    coverage = pd.DataFrame([
        {"month": pd.Period("2024-02", "M"), "basket": "L1 Majors", "coverage_momentum": 0.9},
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_momentum": 0.1},
    ])
    conv = study_token.token_conviction(coverage, baskets_cfg, agg="sum")
    assert "BTC" not in conv["token"].values
    assert "ETH" not in conv["token"].values


def test_bogus_agg_raises(baskets_cfg):
    coverage = pd.DataFrame([
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_momentum": 0.1},
    ])
    with pytest.raises(ValueError):
        study_token.token_conviction(coverage, baskets_cfg, agg="bogus")


def test_run_study_b_end_to_end_pulse(baskets_cfg):
    # Conviction (sum) ranks STRK(0.2) > ARB(0.1) > UNI(0.05); forward returns rank the
    # same (STRK>ARB>UNI every month) -> per-month spearman +1 -> IC ~ +1 -> pulse.
    # Only 3 tokens have returns, so the quartile backtest is empty (need >=4) and the
    # verdict rests on IC alone via the bypass.
    months = pd.period_range("2024-01", "2024-06", freq="M")
    cov = pd.DataFrame([
        {"month": m, "basket": b, "coverage_momentum": v}
        for m in months
        for b, v in [("ZK/Privacy", 0.1), ("L2/Scaling", 0.1), ("DeFi", 0.05)]
    ])
    rets = pd.DataFrame([
        {"month": m, "token": t, "ret": r}
        for m in months
        for t, r in [("STRK", 0.10), ("ARB", 0.05), ("UNI", 0.00)]
    ])
    res = study_token.run_study_b(cov, rets, baskets_cfg, agg="sum")
    assert res["agg"] == "sum"
    assert res["ic"][1]["ic_mean"] > 0.99
    assert res["verdict"] == "pulse"
