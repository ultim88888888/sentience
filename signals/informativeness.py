"""Informativeness eval: does the signal (stance/conviction/Δ-stance) predict forward returns?
Spearman IC of signal vs next-period return, sectors via equal-weight baskets, tokens directly.
The A1-vs-A2a scientific comparison (NOT the strategy backtest)."""
from __future__ import annotations
import glob, os
import pandas as pd
import numpy as np
from signals.config import STANCE_SIGN
from signals.market import sector_basket, OI_FLOOR_USD


def load_close_panel(market_dir: str, suffix: str = "ohlcv") -> pd.DataFrame:
    """Wide DataFrame: DatetimeIndex (UTC, daily) × symbol -> close. From *_<suffix>_1d.parquet."""
    cols = {}
    for f in glob.glob(os.path.join(market_dir, f"*_{suffix}_1d.parquet")):
        sym = os.path.basename(f).replace(f"_{suffix}_1d.parquet", "")
        df = pd.read_parquet(f)
        s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]).dt.tz_localize(None))
        cols[sym] = s
    panel = pd.DataFrame(cols).sort_index()
    return panel


def _asof_price(panel: pd.DataFrame, sym: str, t) -> float | None:
    """Close at/just-before date t (nearest trading day <= t), or None."""
    if sym not in panel.columns:
        return None
    s = panel[sym].dropna()
    s = s[s.index <= pd.Timestamp(t)]
    return float(s.iloc[-1]) if len(s) else None


def forward_return(panel: pd.DataFrame, sym: str, t, t_next) -> float | None:
    p0 = _asof_price(panel, sym, t)
    p1 = _asof_price(panel, sym, t_next)
    if p0 and p1 and p0 > 0:
        return p1 / p0 - 1.0
    return None


def oi_at(oi_panel: pd.DataFrame, t) -> dict:
    """{symbol: OI close at/just-before t} for the OI floor at time t."""
    out = {}
    for sym in oi_panel.columns:
        v = _asof_price(oi_panel, sym, t)
        if v is not None:
            out[sym] = v
    return out


def basket_forward_return(sector: str, t, t_next, sector_map: dict, prices: pd.DataFrame,
                          oi_now: dict, *, oi_floor: float = OI_FLOOR_USD) -> float | None:
    """Equal-weight mean forward return of the sector's liquid constituents at t."""
    basket = sector_basket(sector_map, oi_now, sector, oi_floor=oi_floor)
    rets = [r for s in basket if (r := forward_return(prices, s, t, t_next)) is not None]
    return float(np.mean(rets)) if rets else None


def build_eval_table(panel: pd.DataFrame, dates: list, sector_map: dict, prices: pd.DataFrame,
                     oi_panel: pd.DataFrame, *, oi_floor: float = OI_FLOOR_USD) -> pd.DataFrame:
    """One row per (item, as_of) with signal features + forward return. Drops rows with no return
    (untradeable / no price). EXITED rows are dropped (no live position)."""
    dates = sorted(pd.Timestamp(d) for d in dates)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    rows = []
    for _, r in panel.iterrows():
        t = pd.Timestamp(r["as_of"])
        if t not in nxt or r["lifecycle_state"] == "EXITED":
            continue
        t_next = nxt[t]
        if r["item_type"] == "token":
            fr = forward_return(prices, r["item"], t, t_next)
        else:
            fr = basket_forward_return(r["item"], t, t_next, sector_map, prices,
                                       oi_at(oi_panel, t), oi_floor=oi_floor)
        if fr is None:
            continue
        sign = STANCE_SIGN.get(r["stance"], 0)
        rows.append({"as_of": r["as_of"], "item": r["item"], "item_type": r["item_type"],
                     "stance_sign": sign, "conviction_signed": sign * r["conviction"],
                     "delta_stance": r.get("delta_stance", 0), "fwd_return": fr})
    return pd.DataFrame(rows)


def compute_ic(table: pd.DataFrame, signal_col: str, ret_col: str = "fwd_return") -> dict:
    """Spearman rank IC between a signal column and forward return."""
    t = table[[signal_col, ret_col]].dropna()
    if len(t) < 5 or t[signal_col].nunique() < 2:
        return {"ic": float("nan"), "n": len(t)}
    ic = t[signal_col].corr(t[ret_col], method="spearman")
    return {"ic": float(ic), "n": int(len(t))}


def run_informativeness(panel_path: str, sector_map: dict, market_dir: str, dates: list,
                        *, label: str = "") -> dict:
    """Assemble the eval table for a panel and report IC for each signal feature."""
    panel = pd.read_parquet(panel_path)
    prices = load_close_panel(market_dir, "ohlcv")
    oi_panel = load_close_panel(market_dir, "oi")
    table = build_eval_table(panel, dates, sector_map, prices, oi_panel)
    return {"label": label, "n_rows": len(table),
            "ic_stance": compute_ic(table, "stance_sign"),
            "ic_conviction_signed": compute_ic(table, "conviction_signed"),
            "ic_delta_stance": compute_ic(table, "delta_stance")}
