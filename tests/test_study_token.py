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
