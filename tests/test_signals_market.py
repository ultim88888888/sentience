"""
TDD tests for signals.market — universe, LLM sector-classify, BTC beta, baskets.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from signals.market import (
    build_universe,
    classify_sectors,
    compute_beta,
    sector_basket,
    OI_FLOOR_USD,
)


# ──────────────────────────────────────────────────────────────────────────────
# build_universe
# ──────────────────────────────────────────────────────────────────────────────

def test_universe_filters_by_oi_floor():
    cm = [
        {"symbol": "BTC", "open_interest_usd": 4e10},
        {"symbol": "TINY", "open_interest_usd": 5e5},
    ]
    assert build_universe(cm) == ["BTC"]  # TINY ($500k) below $2M floor


def test_universe_includes_at_floor():
    cm = [{"symbol": "EDGE", "open_interest_usd": OI_FLOOR_USD}]
    assert build_universe(cm) == ["EDGE"]


def test_universe_excludes_missing_oi():
    cm = [{"symbol": "NO_OI"}]
    assert build_universe(cm) == []


def test_universe_excludes_zero_oi():
    cm = [{"symbol": "ZERO", "open_interest_usd": 0}]
    assert build_universe(cm) == []


def test_universe_skips_non_dict_entries():
    cm = [
        {"symbol": "BTC", "open_interest_usd": 4e10},
        "bad_entry",
        None,
    ]
    assert build_universe(cm) == ["BTC"]


def test_universe_custom_floor():
    cm = [
        {"symbol": "A", "open_interest_usd": 5e6},
        {"symbol": "B", "open_interest_usd": 1e6},
    ]
    assert build_universe(cm, oi_floor=3e6) == ["A"]


# ──────────────────────────────────────────────────────────────────────────────
# classify_sectors
# ──────────────────────────────────────────────────────────────────────────────

def test_classify_sectors_maps_and_defaults_other():
    resp = json.dumps({"map": [{"ticker": "ARB", "sector": "l2-scaling"}]})
    # OP is omitted by the LLM → defaults to "other"
    with patch("signals.market.run_claude", return_value=resp):
        m = classify_sectors(["ARB", "OP"], ["l2-scaling", "defi"])
    assert m["ARB"] == "l2-scaling"
    assert m["OP"] == "other"


def test_classify_sectors_all_mapped():
    resp = json.dumps({"map": [
        {"ticker": "UNI", "sector": "defi"},
        {"ticker": "AAVE", "sector": "defi"},
    ]})
    with patch("signals.market.run_claude", return_value=resp):
        m = classify_sectors(["UNI", "AAVE"], ["defi", "l2-scaling"])
    assert m["UNI"] == "defi"
    assert m["AAVE"] == "defi"


def test_classify_sectors_empty_tickers():
    with patch("signals.market.run_claude", return_value='{"map":[]}') as mock_rc:
        m = classify_sectors([], ["defi"])
    assert m == {}
    mock_rc.assert_not_called()


def test_classify_sectors_batching():
    """With batch=2, three tickers trigger two LLM calls."""
    tickers = ["A", "B", "C"]
    resp = json.dumps({"map": [{"ticker": t, "sector": "other"} for t in tickers]})
    with patch("signals.market.run_claude", return_value=resp) as mock_rc:
        m = classify_sectors(tickers, ["other"], batch=2)
    assert mock_rc.call_count == 2
    assert set(m.keys()) == set(tickers)


def test_classify_sectors_none_sector_defaults_other():
    resp = json.dumps({"map": [{"ticker": "X", "sector": None}]})
    with patch("signals.market.run_claude", return_value=resp):
        m = classify_sectors(["X"], ["defi"])
    assert m["X"] == "other"


def test_classify_sectors_uses_low_effort():
    """classify_sectors must pass effort='low' to run_claude."""
    resp = json.dumps({"map": []})
    with patch("signals.market.run_claude", return_value=resp) as mock_rc:
        classify_sectors(["BTC"], ["pos-l1"])
    _, kwargs = mock_rc.call_args
    assert kwargs.get("effort") == "low"


# ──────────────────────────────────────────────────────────────────────────────
# compute_beta
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_beta_is_rolling_and_btc_is_one():
    np.random.seed(0)
    btc = pd.Series(np.random.normal(0, 0.02, 200))
    df = pd.DataFrame(
        {"BTC": btc, "X2": btc * 2 + np.random.normal(0, 0.001, 200)},
        index=pd.RangeIndex(200),
    )
    b = compute_beta(df, window=60)
    assert abs(b["BTC"].dropna().iloc[-1] - 1.0) < 1e-6   # market beta to itself = 1
    assert abs(b["X2"].dropna().iloc[-1] - 2.0) < 0.2     # ~2x mover → beta ~2
    assert b["X2"].iloc[:59].isna().all()                  # no lookahead before window fills


def test_compute_beta_no_data_before_window():
    """First window-1 values must be NaN (rolling requires full window)."""
    np.random.seed(42)
    n = 100
    btc = pd.Series(np.random.normal(0, 0.02, n))
    df = pd.DataFrame({"BTC": btc, "Y": btc * 1.5}, index=pd.RangeIndex(n))
    b = compute_beta(df, window=30)
    assert b["Y"].iloc[:29].isna().all()
    assert b["Y"].iloc[29:].notna().all()


def test_compute_beta_preserves_index():
    """Output DataFrame shares the same index as the input."""
    n = 50
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    btc = pd.Series(np.random.normal(0, 0.02, n), index=idx)
    df = pd.DataFrame({"BTC": btc, "ETH": btc * 0.9}, index=idx)
    b = compute_beta(df, window=20)
    assert list(b.index) == list(df.index)


# ──────────────────────────────────────────────────────────────────────────────
# sector_basket
# ──────────────────────────────────────────────────────────────────────────────

def test_sector_basket_equal_weight_liquid_only():
    smap = {"ARB": "l2-scaling", "OP": "l2-scaling", "DEAD": "l2-scaling", "UNI": "defi"}
    oi = {"ARB": 5e8, "OP": 3e8, "DEAD": 1e5, "UNI": 4e8}
    result = sector_basket(smap, oi, "l2-scaling")
    assert result == ["ARB", "OP"]   # DEAD below OI floor; sorted alphabetically


def test_sector_basket_empty_when_none_liquid():
    smap = {"X": "defi", "Y": "defi"}
    oi = {"X": 1e4, "Y": 1e4}
    assert sector_basket(smap, oi, "defi") == []


def test_sector_basket_missing_oi_excluded():
    smap = {"A": "pos-l1", "B": "pos-l1"}
    oi = {"A": 1e9}
    # B has no OI entry → treated as 0 → excluded
    result = sector_basket(smap, oi, "pos-l1")
    assert result == ["A"]


def test_sector_basket_sorted_output():
    smap = {"ZZZ": "defi", "AAA": "defi", "MMM": "defi"}
    oi = {k: 1e9 for k in smap}
    result = sector_basket(smap, oi, "defi")
    assert result == sorted(result)


def test_sector_basket_custom_floor():
    smap = {"A": "defi", "B": "defi"}
    oi = {"A": 5e6, "B": 1e6}
    assert sector_basket(smap, oi, "defi", oi_floor=3e6) == ["A"]
