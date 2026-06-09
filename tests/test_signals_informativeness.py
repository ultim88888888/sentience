import pandas as pd
import numpy as np
from signals.informativeness import (load_close_panel, forward_return, basket_forward_return,
                                      build_eval_table, compute_ic)


def _prices(tmp_path, sym, dates, closes):
    pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes,
                  "volume_usd": [1] * len(closes)}).to_parquet(tmp_path / f"{sym}_ohlcv_1d.parquet")


def test_forward_return_uses_asof_and_next(tmp_path):
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ARB", d, [100.0, 120.0])
    p = load_close_panel(str(tmp_path))
    assert abs(forward_return(p, "ARB", "2024-03-31", "2024-06-30") - 0.2) < 1e-9


def test_basket_is_equal_weight_and_liquidity_filtered(tmp_path):
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ARB", d, [100.0, 110.0])   # +10%
    _prices(tmp_path, "OP", d, [100.0, 130.0])    # +30%
    _prices(tmp_path, "DEAD", d, [100.0, 200.0])  # +100% but illiquid -> excluded
    p = load_close_panel(str(tmp_path))
    smap = {"ARB": "l2-scaling", "OP": "l2-scaling", "DEAD": "l2-scaling"}
    oi = {"ARB": 5e8, "OP": 5e8, "DEAD": 1e5}
    r = basket_forward_return("l2-scaling", "2024-03-31", "2024-06-30", smap, p, oi)
    assert abs(r - 0.20) < 1e-9                    # mean(10%, 30%) = 20%, DEAD excluded


def test_compute_ic_detects_positive_relationship():
    # signal perfectly rank-correlated with forward return
    t = pd.DataFrame({"stance_sign": [1, 1, -1, -1, 0], "fwd_return": [0.3, 0.2, -0.1, -0.2, 0.0]})
    r = compute_ic(t, "stance_sign")
    assert r["ic"] > 0.8 and r["n"] == 5


def test_build_eval_table_drops_exited_and_no_return(tmp_path):
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ARB", d, [100., 120.])
    p = load_close_panel(str(tmp_path))
    oip = p.copy()  # reuse as oi (values large enough)
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "ARB", "item_type": "token", "parent_sector": "l2-scaling",
         "stance": "bullish", "conviction": 80, "lifecycle_state": "NEW", "delta_stance": 0},
        {"as_of": "2024-03-31", "item": "GONE", "item_type": "token", "parent_sector": None,
         "stance": "bearish", "conviction": 50, "lifecycle_state": "EXITED", "delta_stance": 0}])
    tbl = build_eval_table(panel, ["2024-03-31", "2024-06-30"], {"ARB": "l2-scaling"}, p, oip)
    assert list(tbl["item"]) == ["ARB"]          # EXITED dropped; GONE has no price anyway
    assert abs(tbl.iloc[0]["fwd_return"] - 0.2) < 1e-9


# ---- additional edge-case tests ----

def test_forward_return_missing_symbol(tmp_path):
    """Returns None when symbol is not in price panel."""
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ARB", d, [100.0, 120.0])
    p = load_close_panel(str(tmp_path))
    assert forward_return(p, "MISSING", "2024-03-31", "2024-06-30") is None


def test_forward_return_zero_price_returns_none(tmp_path):
    """Returns None when t0 price is zero (div-by-zero guard)."""
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ZRO", d, [0.0, 100.0])
    p = load_close_panel(str(tmp_path))
    assert forward_return(p, "ZRO", "2024-03-31", "2024-06-30") is None


def test_compute_ic_too_few_rows():
    """Returns nan IC when fewer than 5 non-null rows."""
    t = pd.DataFrame({"stance_sign": [1, -1, 0, 1], "fwd_return": [0.1, -0.1, 0.0, 0.2]})
    r = compute_ic(t, "stance_sign")
    assert np.isnan(r["ic"])


def test_compute_ic_constant_signal():
    """Returns nan IC when signal has no variance (all same value)."""
    t = pd.DataFrame({"stance_sign": [1, 1, 1, 1, 1], "fwd_return": [0.1, 0.2, 0.3, 0.4, 0.5]})
    r = compute_ic(t, "stance_sign")
    assert np.isnan(r["ic"])


def test_basket_returns_none_when_no_liquid_members(tmp_path):
    """Returns None when all basket members fail the OI floor."""
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ARB", d, [100.0, 110.0])
    p = load_close_panel(str(tmp_path))
    smap = {"ARB": "l2-scaling"}
    oi = {"ARB": 1e3}  # below floor
    assert basket_forward_return("l2-scaling", "2024-03-31", "2024-06-30", smap, p, oi) is None


def test_load_close_panel_empty_dir(tmp_path):
    """Returns empty DataFrame when no parquet files present."""
    p = load_close_panel(str(tmp_path))
    assert isinstance(p, pd.DataFrame)
    assert p.empty


def test_build_eval_table_sector_item(tmp_path):
    """Sector items use basket forward return, not direct token price."""
    d = pd.to_datetime(["2024-03-31", "2024-06-30"])
    _prices(tmp_path, "ARB", d, [100.0, 110.0])  # +10%
    _prices(tmp_path, "OP", d, [100.0, 130.0])   # +30%
    p = load_close_panel(str(tmp_path))
    # build OI parquet files for oi_panel
    pd.DataFrame({"date": d, "open": [5e8, 5e8], "high": [5e8, 5e8], "low": [5e8, 5e8],
                  "close": [5e8, 5e8], "volume_usd": [1, 1]}).to_parquet(
        tmp_path / "ARB_oi_1d.parquet")
    pd.DataFrame({"date": d, "open": [5e8, 5e8], "high": [5e8, 5e8], "low": [5e8, 5e8],
                  "close": [5e8, 5e8], "volume_usd": [1, 1]}).to_parquet(
        tmp_path / "OP_oi_1d.parquet")
    oip = load_close_panel(str(tmp_path), suffix="oi")
    panel = pd.DataFrame([
        {"as_of": "2024-03-31", "item": "l2-scaling", "item_type": "sector",
         "parent_sector": None, "stance": "bullish", "conviction": 80,
         "lifecycle_state": "NEW", "delta_stance": 0}])
    smap = {"ARB": "l2-scaling", "OP": "l2-scaling"}
    tbl = build_eval_table(panel, ["2024-03-31", "2024-06-30"], smap, p, oip)
    assert len(tbl) == 1
    assert abs(tbl.iloc[0]["fwd_return"] - 0.20) < 1e-9  # mean(10%, 30%) = 20%
