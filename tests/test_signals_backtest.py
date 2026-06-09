import pandas as pd
import numpy as np
import pytest
from signals.backtest import (beta_neutralize, sector_ls_targets, intra_sector_targets,
                               realize_period, period_funding, metrics,
                               walk_forward, benchmark_returns,
                               _finalize, long_only_targets, token_ls_targets,
                               hedge_to_target_beta, regime_target_beta,
                               token_vs_sector_targets, both_targets)
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
    assert abs(total - 0.00005) < 1e-9   # (day2+day3) summed then /100 (percent->fraction)


def test_period_funding_missing_symbol():
    funding = pd.DataFrame()
    assert period_funding(funding, "BTC", "2024-01-01", "2024-03-31") == 0.0


def test_period_funding_exclusive_lower_bound(tmp_path):
    """Funding at exactly t0 should NOT be included (> not >=)."""
    dates = ["2024-01-01", "2024-01-02"]
    _funding_parquet(tmp_path, "ETH", dates, [0.01, 0.02])
    funding = load_close_panel(str(tmp_path), "funding")
    total = period_funding(funding, "ETH", "2024-01-01", "2024-01-02")
    assert abs(total - 0.0002) < 1e-9   # only 2024-01-02, /100 (percent->fraction)


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


# ── _finalize ─────────────────────────────────────────────────────────────────

def test_finalize_caps_and_normalizes():
    out = _finalize({"A": 0.1, "B": -0.5, "C": 0.2, "D": -0.05}, cap=2)
    assert set(out) == {"B", "C"}                      # top-2 by |w|
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9


def test_finalize_empty_returns_empty():
    assert _finalize({}) == {}


def test_finalize_no_cap_gross_normalizes():
    out = _finalize({"A": 0.3, "B": -0.9})
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9
    assert out["B"] < 0 and out["A"] > 0


def test_finalize_custom_gross():
    out = _finalize({"A": 1.0, "B": -1.0}, gross=2.0)
    assert abs(sum(abs(v) for v in out.values()) - 2.0) < 1e-9


# ── long_only_targets ─────────────────────────────────────────────────────────

def test_long_only_no_shorts():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "stance": "bullish",
         "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "defi", "item_type": "sector", "stance": "bearish",
         "conviction": 70, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "UNI": "defi"}
    oi = {"ARB": 5e8, "UNI": 5e8}
    w = long_only_targets(live, sm, oi)
    assert all(v > 0 for v in w.values()) and "ARB" in w and "UNI" not in w


def test_long_only_fade_longs_bearish():
    live = pd.DataFrame([
        {"item": "defi", "item_type": "sector", "stance": "bearish",
         "conviction": 70, "lifecycle_state": "NEW"},
    ])
    sm = {"UNI": "defi", "AAVE": "defi"}
    oi = {"UNI": 5e8, "AAVE": 5e8}
    w = long_only_targets(live, sm, oi, bet_sign="fade")
    assert all(v > 0 for v in w.values())
    assert "UNI" in w and "AAVE" in w


def test_long_only_gross_normalized():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "stance": "bullish",
         "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "pos-l1", "item_type": "sector", "stance": "bullish",
         "conviction": 60, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "ETH": "pos-l1"}
    oi = {"ARB": 5e8, "ETH": 5e8}
    w = long_only_targets(live, sm, oi)
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-9


def test_long_only_equal_weight_mode():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "stance": "bullish",
         "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "pos-l1", "item_type": "sector", "stance": "bullish",
         "conviction": 40, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "ETH": "pos-l1"}
    oi = {"ARB": 5e8, "ETH": 5e8}
    w_cv = long_only_targets(live, sm, oi, conviction_weighted=True)
    w_eq = long_only_targets(live, sm, oi, conviction_weighted=False)
    # equal-weight: both sectors contribute equally regardless of conviction
    assert abs(w_eq["ARB"] - w_eq["ETH"]) < 1e-9
    # conviction-weighted: higher conviction → higher weight
    assert w_cv["ARB"] > w_cv["ETH"]


def test_long_only_cap():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "stance": "bullish",
         "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "pos-l1", "item_type": "sector", "stance": "bullish",
         "conviction": 60, "lifecycle_state": "NEW"},
        {"item": "defi", "item_type": "sector", "stance": "bullish",
         "conviction": 40, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "ETH": "pos-l1", "UNI": "defi"}
    oi = {s: 5e8 for s in sm}
    w = long_only_targets(live, sm, oi, cap=2)
    assert len(w) == 2


# ── token_ls_targets ──────────────────────────────────────────────────────────

def test_token_ls_longs_bullish_shorts_bearish():
    live = pd.DataFrame([
        {"item": "SOL", "item_type": "token", "stance": "bullish",
         "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "XRP", "item_type": "token", "stance": "bearish",
         "conviction": 60, "lifecycle_state": "NEW"},
    ])
    w = token_ls_targets(live)
    assert w["SOL"] > 0 and w["XRP"] < 0


def test_token_ls_neutral_excluded():
    live = pd.DataFrame([
        {"item": "SOL", "item_type": "token", "stance": "neutral",
         "conviction": 50, "lifecycle_state": "NEW"},
    ])
    assert token_ls_targets(live) == {}


def test_token_ls_sector_rows_ignored():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "stance": "bullish",
         "conviction": 80, "lifecycle_state": "NEW"},
    ])
    assert token_ls_targets(live) == {}


def test_token_ls_gross_normalized():
    live = pd.DataFrame([
        {"item": "SOL", "item_type": "token", "stance": "bullish",
         "conviction": 90, "lifecycle_state": "NEW"},
        {"item": "XRP", "item_type": "token", "stance": "bearish",
         "conviction": 40, "lifecycle_state": "NEW"},
    ])
    w = token_ls_targets(live)
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-9


def test_token_ls_equal_weight_mode():
    live = pd.DataFrame([
        {"item": "SOL", "item_type": "token", "stance": "bullish",
         "conviction": 90, "lifecycle_state": "NEW"},
        {"item": "XRP", "item_type": "token", "stance": "bearish",
         "conviction": 40, "lifecycle_state": "NEW"},
    ])
    w = token_ls_targets(live, conviction_weighted=False)
    assert abs(abs(w["SOL"]) - abs(w["XRP"])) < 1e-9  # equal magnitude


def test_token_ls_cap():
    live = pd.DataFrame([
        {"item": "SOL", "item_type": "token", "stance": "bullish",
         "conviction": 90, "lifecycle_state": "NEW"},
        {"item": "ETH", "item_type": "token", "stance": "bullish",
         "conviction": 70, "lifecycle_state": "NEW"},
        {"item": "XRP", "item_type": "token", "stance": "bearish",
         "conviction": 40, "lifecycle_state": "NEW"},
    ])
    w = token_ls_targets(live, cap=2)
    assert len(w) == 2
    assert "SOL" in w  # highest |weight| survives


# ── hedge_to_target_beta ──────────────────────────────────────────────────────

def test_hedge_to_target_beta():
    out = hedge_to_target_beta({"A": 1.0}, {"A": 2.0, "BTC": 1.0}, target_beta=0.5)
    assert abs(sum(out[s] * {"A": 2.0, "BTC": 1.0}[s] for s in out) - 0.5) < 1e-9


def test_hedge_to_target_beta_zero_is_neutral():
    """target_beta=0 should behave like beta_neutralize."""
    w = {"A": 0.5, "B": -0.5}
    betas = {"A": 1.5, "B": 0.5, "BTC": 1.0}
    out_hedge = hedge_to_target_beta(w, betas, target_beta=0.0)
    out_neutral = beta_neutralize(w, betas)
    for s in set(out_hedge) | set(out_neutral):
        assert abs(out_hedge.get(s, 0.0) - out_neutral.get(s, 0.0)) < 1e-9


def test_hedge_to_target_beta_removes_zero_btc():
    """If the hedge needed is ~0, BTC should not appear."""
    w = {"A": 0.5}
    betas = {"A": 1.0, "BTC": 1.0}
    # net beta = 0.5, target = 0.5 → BTC hedge = 0
    out = hedge_to_target_beta(w, betas, target_beta=0.5)
    assert "BTC" not in out


# ── regime_target_beta ────────────────────────────────────────────────────────

def test_regime_target_beta_modes():
    pr = pd.DataFrame({"BTC": [1.0] * 210},
                      index=pd.date_range("2023-01-01", periods=210))
    assert regime_target_beta(pr, "2024-01-01", "risk_off", mode="consensus") == -0.5
    # quant: flat series → last == mean → uptrend (+1) → +0.5
    assert regime_target_beta(pr, "2024-01-01", "risk_off", mode="quant") == 0.5


def test_regime_target_beta_combined():
    pr = pd.DataFrame({"BTC": [1.0] * 210},
                      index=pd.date_range("2023-01-01", periods=210))
    # quant=+0.5, consensus(risk_on)=+0.5 → combined = 0.5
    result = regime_target_beta(pr, "2024-01-01", "risk_on", mode="combined")
    assert abs(result - 0.5) < 1e-9


def test_regime_target_beta_insufficient_history():
    pr = pd.DataFrame({"BTC": [1.0] * 10},
                      index=pd.date_range("2023-01-01", periods=10))
    # fewer than lookback=200 → _btc_trend returns 0 → quant = 0
    result = regime_target_beta(pr, "2023-01-15", "risk_on", mode="quant")
    assert result == 0.0


def test_regime_target_beta_downtrend():
    # Price that is below its 200-day mean → risk-off
    prices = list(range(210, 0, -1))  # descending: last value = 1 << mean ~= 105
    pr = pd.DataFrame({"BTC": prices},
                      index=pd.date_range("2023-01-01", periods=210))
    result = regime_target_beta(pr, "2023-07-31", "neutral", mode="quant")
    assert result == -0.5


# ── walk_forward extended strategies / hedge modes ────────────────────────────

def test_walk_forward_long_only_strategy(tmp_path):
    dates = ["2024-03-31", "2024-06-30", "2024-09-30"]
    _price_parquet(tmp_path, "ARB", dates, [10.0, 12.0, 11.0])
    _price_parquet(tmp_path, "OP",  dates, [5.0, 6.0, 5.5])
    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
        {"as_of": "2024-06-30", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "ACTIVE"},
    ])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling"}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm, dates,
                          strategy="long_only", beta_neutral=False, cost_bps=0)
    assert len(result) == 2


def test_walk_forward_token_ls_strategy(tmp_path):
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "SOL", dates, [100.0, 130.0])
    _price_parquet(tmp_path, "XRP", dates, [0.5, 0.4])
    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "SOL", "item_type": "token",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
        {"as_of": "2024-03-31", "item": "XRP", "item_type": "token",
         "stance": "bearish", "conviction": 60, "lifecycle_state": "NEW"},
    ])
    sm = {}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm, dates,
                          strategy="token_ls", beta_neutral=False, cost_bps=0)
    assert len(result) == 1
    # SOL up 30%, XRP down 20% → long SOL, short XRP → positive PnL
    assert result.iloc[0]["ret"] > 0


def test_walk_forward_hedge_none(tmp_path):
    """hedge_mode='none' skips all hedging; portfolio is unhedged."""
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "ARB", dates, [10.0, 12.0])
    _price_parquet(tmp_path, "OP",  dates, [5.0, 4.0])
    _price_parquet(tmp_path, "BTC", dates, [60000.0, 65000.0])
    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling"}
    betas = {"ARB": 1.2, "OP": 0.8, "BTC": 1.0}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm, dates,
                          strategy="sector_ls", betas=betas, hedge_mode="none", cost_bps=0)
    assert len(result) == 1


def test_walk_forward_cap_limits_positions(tmp_path):
    """cap param should flow through walk_forward into the strategy."""
    dates = ["2024-03-31", "2024-06-30"]
    for sym, p0, p1 in [("ARB", 10.0, 12.0), ("OP", 5.0, 6.0), ("STRK", 2.0, 2.2)]:
        _price_parquet(tmp_path, sym, dates, [p0, p1])
    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "OP": "l2-scaling", "STRK": "l2-scaling"}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm, dates,
                          strategy="sector_ls", beta_neutral=False, cost_bps=0, cap=2)
    assert len(result) == 1  # period ran; cap applied internally


def test_walk_forward_risk_by_date_consensus(tmp_path):
    """consensus hedge with risk_by_date wires through correctly."""
    dates = ["2024-03-31", "2024-06-30"]
    _price_parquet(tmp_path, "ARB", dates, [10.0, 12.0])
    # 210 BTC price points so _btc_trend has history (not used for consensus, but needed for prices)
    btc_dates = pd.date_range("2023-06-04", periods=210).strftime("%Y-%m-%d").tolist()
    btc_prices = [60000.0] * 210
    _price_parquet(tmp_path, "BTC", btc_dates + dates, btc_prices + [60000.0, 65000.0])
    prices = load_close_panel(str(tmp_path))
    oi_panel = prices.copy()
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling"}
    betas = {"ARB": 1.5, "BTC": 1.0}
    result = walk_forward(panel, prices, pd.DataFrame(), oi_panel, sm, dates,
                          strategy="sector_ls", betas=betas, hedge_mode="consensus",
                          risk_by_date={"2024-03-31": "risk_on"}, cost_bps=0)
    assert len(result) == 1


# ── token_vs_sector_targets ───────────────────────────────────────────────────

def test_token_vs_sector_is_sector_neutral():
    live = pd.DataFrame([{"item": "SOL", "item_type": "token", "parent_sector": "pos-l1",
                          "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"}])
    sm = {"SOL": "pos-l1", "ADA": "pos-l1", "NEAR": "pos-l1"}
    oi = {"SOL": 5e8, "ADA": 5e8, "NEAR": 5e8}
    w = token_vs_sector_targets(live, sm, oi)
    assert w["SOL"] > 0                                   # long the token
    assert w["ADA"] < 0 and w["NEAR"] < 0                 # short its sector peers
    assert abs(sum(w.values())) < 1e-9                    # dollar-neutral (long token == short peers)


def test_token_vs_sector_skips_token_with_no_sector():
    live = pd.DataFrame([{"item": "BTC", "item_type": "token", "parent_sector": None,
                          "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"}])
    assert token_vs_sector_targets(live, {}, {}) == {}    # no parent_sector -> skipped


# ── both_targets ──────────────────────────────────────────────────────────────

def test_both_merges_sector_and_token_books():
    live = pd.DataFrame([
        {"item": "l2-scaling", "item_type": "sector", "parent_sector": None,
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW"},
        {"item": "SOL", "item_type": "token", "parent_sector": "pos-l1",
         "stance": "bullish", "conviction": 70, "lifecycle_state": "NEW"},
    ])
    sm = {"ARB": "l2-scaling", "SOL": "pos-l1"}
    oi = {"ARB": 5e8, "SOL": 5e8}
    w = both_targets(live, sm, oi)
    assert "ARB" in w and "SOL" in w                      # both sector-basket member and token present
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-9
