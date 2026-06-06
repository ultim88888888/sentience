"""TDD tests for doppelganger.score."""
from __future__ import annotations

from doppelganger.score import discrimination, coverage_trajectory


def test_discrimination_overlap():
    a = {"sectors_excited": [{"name": "ZK"}, {"name": "Games"}], "sectors_concerned": [],
         "tokens_excited": [{"name": "ETH"}], "tokens_concerned": []}
    b = {"sectors_excited": [{"name": "zk"}, {"name": "DAOs"}], "sectors_concerned": [],
         "tokens_excited": [{"name": "SOL"}], "tokens_concerned": []}
    d = discrimination(a, b)
    # sectors: {zk,games} vs {zk,daos} -> intersection {zk}=1, union 3 -> 1/3
    assert round(d["sector_overlap"], 2) == 0.33
    assert d["token_overlap"] == 0.0          # eth vs sol, no overlap
    assert "zk" in d["shared_sectors"]


def test_coverage_trajectory():
    rows = [
        {"date": "2022-12-31", "variant": "full", "grounded": 13, "persisted": 0, "extrapolated": 0},
        {"date": "2022-12-31", "variant": "ablation", "grounded": 5, "persisted": 2, "extrapolated": 2},
        {"date": "2023-03-31", "variant": "full", "grounded": 11, "persisted": 0, "extrapolated": 0},
    ]
    cov = coverage_trajectory(rows)
    assert cov == [{"date": "2022-12-31", "grounded": 13, "persisted": 0, "extrapolated": 0},
                   {"date": "2023-03-31", "grounded": 11, "persisted": 0, "extrapolated": 0}]
