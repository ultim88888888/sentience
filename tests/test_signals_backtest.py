import pandas as pd
import numpy as np
import pytest
from signals.backtest import (beta_neutralize, sector_ls_targets, intra_sector_targets,
                               realize_period, period_funding, metrics,
                               walk_forward, benchmark_returns)
from signals.informativeness import load_close_panel


# ── helpers ──────────────────────────────────────────────────────────────────

def _price_parquet(tmp_path, sym, dates, closes):
    """Write a minimal OHLCV parquet for load_close_panel."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume_usd": [1.0] * len(closes),
    })
    df.to_parquet(tmp_path / f"{sym}_ohlcv_1d.parquet")


def _funding_parquet(tmp_path, sym, dates, rates):
    """Write a minimal funding parquet for load_close_panel (suffix='funding')."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": rates, "high": rates, "low": rates,
        "close": rates, "volume_usd": [0.0] * len(rates),
    })
    df.to_parquet(tmp_path / f"{sym}_funding_1d.parquet")


# ── beta_neutralize ───────────────────────────────────────────────────────────

def test_beta_neutralize_hedges_with_btc():
    w = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.5, "B": 0.5, "BTC": 1.0}
    out = beta_neutralize(w, betas)
    assert abs(sum(out[s] * betas[s] for s in out)) < 1e-9   # net beta ~ 0
    assert out["A"] == 0.5 and out["B"] == -0.5              # L/S book UNCHANGED (no leg scaling)
    assert abs(out["BTC"] - (-(0.5 * 1.5 - 0.5 * 0.5))) < 1e-9  # BTC hedge = -net_beta = -0.5


def test_beta_neutralize_bounded_gross_no_explosion():
    """Low-beta short book must NOT blow up leverage (the bug we fixed)."""
    w = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.5, "B": 0.05, "BTC": 1.0}   # tiny short beta
    out = beta_neutralize(w, betas)
    assert abs(sum(out[s] * betas[s] for s in out)) < 1e-9
    assert sum(abs(v) for v in out.values()) < 2.0   # bounded gross (old code exploded here)


def test_beta_neutralize_no_betas_unit_default():
    # no betas -> each treated as beta 1.0; net = 0.5-0.5 = 0 -> no BTC overlay added
    w = {"A": 0.5, "B": -0.5}
    out = beta_neutralize(w, {})
    assert "BTC" not in out and out == w


def test_beta_neutralize_net_long_adds_btc_short():
    w = {"A": 1.0}                       # net-long book
    betas = {"A": 2.0, "BTC": 1.0}
    out = beta_neutralize(w, betas)
    assert abs(out["BTC"] - (-2.0)) < 1e-9   # hedge the +2.0 net beta with -2.0 BTC
    assert abs(sum(out[s] * betas[s] for s in out)) < 1e-9


# ── sector_ls_targets ─────────────────────────────────────────────────────────

def test_sector_ls_momentum_vs_fade():
    live = pd.DataFrame([{
        "item": "l2-scaling", "item_type": "sector",
        "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW",
    }])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling"}
    oi = {"ARB": 5e8, "OP": 5e8}
    mom = sector_ls_targets(live, sm, oi, bet_sign="momentum")
    fade = sector_ls_targets(live, sm, oi, bet_sign="fade")
    assert mom["ARB"] > 0 and fade["ARB"] < 0   # momentum longs bullish, fade shorts it
    assert abs(sum(abs(v) for v in mom.values()) - 1.0) < 1e-9   # gross-normalized


def test_sector_ls_bearish_sector_shorted_momentum():
    live = pd.DataFrame([{
        "item": "defi", "item_type": "sector",
        "stance": "bearish", "conviction": 60, "lifecycle_state": "NEW",
    }])
    sm = {"UNI": "defi", "AAVE": "defi"}
    oi = {"UNI": 5e8, "AAVE": 5e8}
    out = sector_ls_targets(live, sm, oi, bet_sign="momentum")
    assert out["UNI"] < 0 and out["AAVE"] < 0


def test_sector_ls_neutral_excluded():
    live = pd.DataFrame([{
        "item": "defi", "item_type": "sector",
        "stance": "neutral", "conviction": 50, "lifecycle_state": "NEW",
    }])
    sm = {"UNI": "defi"}
    oi = {"UNI": 5e8}
    out = sector_ls_targets(live, sm, oi)
    assert out == {}


def test_sector_ls_empty_basket_excluded():
    """Sectors with no OI-liquid members should be silently dropped."""
    live = pd.DataFrame([{
        "item": "zk", "item_type": "sector",
        "stance": "bullish", "conviction": 70, "lifecycle_state": "NEW",
    }])
    sm = {"ZKS": "zk"}
    oi = {"ZKS": 0.0}   # below floor
    out = sector_ls_targets(live, sm, oi)
    assert out == {}


def test_sector_ls_token_rows_ignored():
    live = pd.DataFrame([{
        "item": "ETH", "item_type": "token",
        "stance": "bullish", "conviction": 90, "lifecycle_state": "NEW",
    }])
    sm = {"ETH": "pos-l1"}
    oi = {"ETH": 5e8}
    out = sector_ls_targets(live, sm, oi)
    assert out == {}


def test_sector_ls_gross_normalized_multi_sector():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "defi",       "item_type": "sector", "stance": "bearish", "conviction": 60, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "UNI": "defi"}
    oi = {"ARB": 5e8, "UNI": 5e8}
    out = sector_ls_targets(live, sm, oi)
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9


# ── intra_sector_targets ──────────────────────────────────────────────────────

def test_intra_sector_standout_is_long():
    live = pd.DataFrame([{
        "item": "l2-scaling", "item_type": "sector",
        "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW",
    }])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling", "STRK": "l2-scaling"}
    oi = {s: 5e8 for s in sm}
    conv = {"ARB": 90, "OP": 50, "STRK": 30}
    out = intra_sector_targets(live, sm, oi, conv)
    assert out["ARB"] > 0                  # highest conviction → long
    assert out["OP"] < 0 and out["STRK"] < 0  # laggards → short


def test_intra_sector_gross_normalized():
    live = pd.DataFrame([{
        "item": "l2-scaling", "item_type": "sector",
        "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW",
    }])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling"}
    oi = {s: 5e8 for s in sm}
    out = intra_sector_targets(live, sm, oi, {})
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9


def test_intra_sector_bearish_sector_excluded():
    live = pd.DataFrame([{
        "item": "defi", "item_type": "sector",
        "stance": "bearish", "conviction": 70, "lifecycle_state": "NEW",
    }])
    sm = {"UNI": "defi", "AAVE": "defi"}
    oi = {s: 5e8 for s in sm}
    assert intra_sector_targets(live, sm, oi, {}) == {}


def test_intra_sector_single_member_excluded():
    """A basket with only one member can't go L/S — must be skipped."""
    live = pd.DataFrame([{
        "item": "zk", "item_type": "sector",
        "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW",
    }])
    sm = {"ZKS": "zk"}
    oi = {"ZKS": 5e8}
    assert intra_sector_targets(live, sm, oi, {}) == {}


# ── period_funding ────────────────────────────────────────────────────────────

def test_period_funding_sums_correctly(tmp_path):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    rates = [0.001, 0.002, 0.003]
    _funding_parquet(tmp_path, "ETH", dates, rates)
    funding = load_close_panel(str(tmp_path), "funding")
    total = period_funding(funding, "ETH", "2024-01-01", "2024-01-03")
    assert abs(total - 0.005) < 1e-9   # day1 excluded (>t0), day2+day3 summed


def test_period_funding_missing_symbol():
    funding = pd.DataFrame()
    assert period_funding(funding, "BTC", "2024-01-01", "2024-03-31") == 0.0


def test_period_funding_exclusive_lower_bound(tmp_path):
    """Funding at exactly t0 should NOT be included (> not >=)."""
    dates = ["2024-01-01", "2024-01-02"]
    _funding_parquet(tmp_path, "ETH", dates, [0.01, 0.02])
    funding = load_close_panel(str(tmp_path), "funding")
    total = period_funding(funding, "ETH", "2024-01-01", "2024-01-02")
    assert abs(total - 0.02) < 1e-9   # only 2024-01-02 included


# ── realize_period ────────────────────────────────────────────────────────────

def test_realize_period_pnl_and_costs(tmp_path):
    d = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "A", d, [100.0, 120.0])
    _price_parquet(tmp_path, "B", d, [100.0, 90.0])
    prices = load_close_panel(str(tmp_path))
    funding = pd.DataFrame()
    # long A (+0.5): A +20% → +0.10; short B (-0.5): B -10% → -0.5*(-0.10) = +0.05 → total 0.15
    pnl = realize_period({"A": 0.5, "B": -0.5}, prices, funding, "2024-03-31", "2024-06-30", cost_bps=0)
    assert abs(pnl - (0.5 * 0.2 + -0.5 * -0.1)) < 1e-9   # = 0.15
    pnl_c = realize_period({"A": 0.5, "B": -0.5}, prices, funding, "2024-03-31", "2024-06-30", cost_bps=100)
    assert pnl_c < pnl   # cost reduces P&L


def test_realize_period_funding_reduces_long_pnl(tmp_path):
    d = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "A", d, [100.0, 110.0])
    prices = load_close_panel(str(tmp_path))
    # funding panel with a positive rate (cost for longs)
    _funding_parquet(tmp_path, "A", ["2024-04-01", "2024-06-30"], [0.005, 0.005])
    funding = load_close_panel(str(tmp_path), "funding")
    pnl_no_f = realize_period({"A": 1.0}, prices, pd.DataFrame(), "2024-03-31", "2024-06-30", cost_bps=0)
    pnl_f = realize_period({"A": 1.0}, prices, funding, "2024-03-31", "2024-06-30", cost_bps=0)
    assert pnl_f < pnl_no_f   # funding cost reduces long P&L


def test_realize_period_missing_price_skipped(tmp_path):
    d = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "A", d, [100.0, 120.0])
    prices = load_close_panel(str(tmp_path))
    # B has no price data → skipped silently
    pnl = realize_period({"A": 0.5, "B": 0.5}, prices, pd.DataFrame(), "2024-03-31", "2024-06-30", cost_bps=0)
    assert abs(pnl - 0.5 * 0.2) < 1e-9


# ── metrics ───────────────────────────────────────────────────────────────────

def test_metrics_basic():
    m = metrics(pd.Series([0.1, -0.05, 0.08, 0.02]))
    assert m["n"] == 4 and m["total"] > 0 and "sharpe" in m


def test_metrics_empty():
    m = metrics(pd.Series([], dtype=float))
    assert m["n"] == 0 and m["total"] == 0.0 and m["max_dd"] == 0.0


def test_metrics_max_dd_negative():
    # Declining series → max drawdown should be negative
    m = metrics(pd.Series([-0.1, -0.2, -0.05]))
    assert m["max_dd"] < 0


def test_metrics_all_positive_no_drawdown():
    m = metrics(pd.Series([0.05, 0.05, 0.05]))
    assert abs(m["max_dd"]) < 1e-9


def test_metrics_flat_series_nan_sharpe():
    m = metrics(pd.Series([0.0, 0.0, 0.0]))
    assert np.isnan(m["sharpe"])


def test_metrics_annualization():
    # 4 periods of 0.1 each, ppy=4 → annualized ≈ (1.1^4)^(4/4) - 1 = 1.1^4 - 1 ≈ 0.4641
    m = metrics(pd.Series([0.1, 0.1, 0.1, 0.1]), periods_per_year=4.0)
    assert abs(m["ann"] - (1.1 ** 4 - 1)) < 1e-9


# ── walk_forward ──────────────────────────────────────────────────────────────

def test_walk_forward_two_periods(tmp_path):
    dates = ["2024-03-31", "2024-06-30", "2024-09-30"]
    _price_parquet(tmp_path, "ARB", dates, [10.0, 12.0, 11.0])
    _price_parquet(tmp_path, "OP",  dates, [5.0,  6.0,  5.5])
    _price_parquet(tmp_path, "BTC", dates, [60000.0, 65000.0, 63000.0])

    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()  # reuse prices as synthetic OI (values >> OI_FLOOR)

    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
        {"as_of": "2024-06-30", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "ACTIVE"},
    ])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling"}

    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm,
                          dates, beta_neutral=False, cost_bps=0)
    assert len(result) == 2
    assert list(result.columns) == ["as_of", "ret"]
    assert all(isinstance(v, float) for v in result["ret"])


def test_walk_forward_exited_rows_excluded(tmp_path):
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "ARB", dates, [10.0, 12.0])

    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()

    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "EXITED"},
    ])
    sm = {"ARB": "l2-scaling"}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm, dates, beta_neutral=False)
    # EXITED rows → no weights → period skipped → empty result
    assert len(result) == 0


def test_walk_forward_intra_sector_strategy(tmp_path):
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "ARB",  dates, [10.0, 13.0])
    _price_parquet(tmp_path, "OP",   dates, [5.0,  5.5])
    _price_parquet(tmp_path, "STRK", dates, [2.0,  1.8])

    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()

    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
        {"as_of": "2024-03-31", "item": "ARB", "item_type": "token",
         "stance": "bullish", "conviction": 90, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling", "STRK": "l2-scaling"}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm,
                          dates, strategy="intra_sector", beta_neutral=False, cost_bps=0)
    assert len(result) == 1
    assert isinstance(result.iloc[0]["ret"], float)


# ── benchmark_returns ─────────────────────────────────────────────────────────

def test_benchmark_returns_btc(tmp_path):
    dates = ["2024-03-31", "2024-06-30", "2024-09-30"]
    _price_parquet(tmp_path, "BTC", dates, [60000.0, 66000.0, 63000.0])
    prices = load_close_panel(str(tmp_path))
    bm = benchmark_returns(prices, dates, mode="btc")
    assert len(bm) == 2
    assert abs(bm.iloc[0]["ret"] - (66000 / 60000 - 1)) < 1e-9
    assert abs(bm.iloc[1]["ret"] - (63000 / 66000 - 1)) < 1e-9


def test_benchmark_returns_eqw(tmp_path):
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "BTC", dates, [60000.0, 66000.0])
    _price_parquet(tmp_path, "ETH", dates, [3000.0,  3300.0])
    prices = load_close_panel(str(tmp_path))
    bm = benchmark_returns(prices, dates, mode="eqw")
    assert len(bm) == 1
    # both up 10% → eqw = 10%
    assert abs(bm.iloc[0]["ret"] - 0.1) < 1e-9


def test_benchmark_returns_btc_missing(tmp_path):
    """If BTC not in prices, forward_return returns None → falls back to 0.0."""
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "ETH", dates, [3000.0, 3300.0])
    prices = load_close_panel(str(tmp_path))
    bm = benchmark_returns(prices, dates, mode="btc")
    assert bm.iloc[0]["ret"] == 0.0
