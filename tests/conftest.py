"""Synthetic fixtures: a tiny corpus and tiny price history, fully deterministic."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def baskets_cfg():
    return {
        "baskets": {
            "ZK/Privacy": ["ZK", "STRK"],
            "L2/Scaling": ["ARB", "STRK"],   # STRK overlaps ZK on purpose
            "DeFi": ["UNI", "AAVE"],
            "L1 Majors": ["BTC", "ETH"],
        },
        "benchmark": "L1 Majors",
        "tag_map": {
            "ZK/Privacy": ["SNARKs", "cryptography"],
            "L2/Scaling": ["rollups"],
            "DeFi": ["DeFi"],
        },
    }


@pytest.fixture
def synthetic_corpus():
    # 6 monthly posts; tags chosen so attribution is hand-checkable.
    rows = [
        ("2024-01-10", ["SNARKs"]),               # ZK only
        ("2024-01-20", ["SNARKs", "rollups"]),    # ZK + L2 (multi-tag)
        ("2024-02-05", ["DeFi"]),                 # DeFi only
        ("2024-02-15", ["rollups"]),              # L2 only
        ("2024-03-01", ["cryptography", "DeFi"]), # ZK + DeFi
        ("2024-03-10", ["unmapped tag"]),         # attributes to nothing
    ]
    return pd.DataFrame(
        {"post_date": [r[0] for r in rows],
         "tags": [np.array(r[1], dtype=object) for r in rows]}
    )


@pytest.fixture
def synthetic_token_returns():
    # month x token monthly returns, long form.
    months = pd.period_range("2024-01", "2024-04", freq="M")
    data = {"ZK": [0.10, 0.20, -0.05, 0.00], "STRK": [0.15, 0.25, 0.00, 0.10],
            "ARB": [0.05, 0.10, 0.05, 0.05], "UNI": [-0.10, -0.05, 0.10, 0.00],
            "AAVE": [-0.05, 0.00, 0.05, 0.05], "BTC": [0.02, 0.03, 0.01, 0.02],
            "ETH": [0.03, 0.04, 0.00, 0.01]}
    recs = [{"month": m, "token": t, "ret": data[t][i]}
            for i, m in enumerate(months) for t in data]
    return pd.DataFrame.from_records(recs)
