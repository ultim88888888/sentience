"""Walk-forward beta-neutral backtest. Panel signal -> target weights -> realized period P&L
(price + funding - costs), tracked vs benchmarks. Beta-neutral construction is first-class."""
from __future__ import annotations
import numpy as np
import pandas as pd
from signals.config import STANCE_SIGN
from signals.market import sector_basket, OI_FLOOR_USD
from signals.informativeness import forward_return, oi_at, basket_forward_return


def period_funding(funding: pd.DataFrame, sym: str, t0, t1) -> float:
    """Sum of funding-rate 'close' over (t0, t1] for a symbol (cost a LONG pays; a short receives).
    funding panel = load_close_panel(dir,'funding'). Returns total fraction (e.g. 0.01 = 1%)."""
    if sym not in funding.columns:
        return 0.0
    s = funding[sym]
    s = s[(s.index > pd.Timestamp(t0)) & (s.index <= pd.Timestamp(t1))]
    return float(s.dropna().sum())


def beta_neutralize(weights: dict, betas: dict) -> dict:
    """Scale the SHORT book so net portfolio beta ≈ 0. weights: {sym: signed weight} (long>0, short<0),
    betas: {sym: beta_to_BTC}. Returns adjusted weights. If no shorts or no betas, returns input."""
    longs = {s: w for s, w in weights.items() if w > 0}
    shorts = {s: w for s, w in weights.items() if w < 0}
    bl = sum(w * betas.get(s, 1.0) for s, w in longs.items())
    bs = sum(w * betas.get(s, 1.0) for s, w in shorts.items())   # negative
    if not shorts or bs == 0:
        return weights
    k = -bl / bs                                                 # scale shorts to offset long beta
    return {**longs, **{s: w * k for s, w in shorts.items()}}


def sector_ls_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, *,
                      bet_sign: str = "momentum", oi_floor: float = OI_FLOOR_USD) -> dict:
    """Sector long/short from the live signal rows at T. bet_sign 'momentum': long bullish sectors,
    short bearish. 'fade': reverse. Sized by conviction. Each sector -> equal-weight basket members.
    Returns {symbol: signed weight} (gross-normalized to sum |w| = 1)."""
    raw = {}
    for _, r in live.iterrows():
        if r["item_type"] != "sector":
            continue
        sign = STANCE_SIGN.get(r["stance"], 0)
        if sign == 0:
            continue
        if bet_sign == "fade":
            sign = -sign
        basket = sector_basket(sector_map, oi_now, r["item"], oi_floor=oi_floor)
        if not basket:
            continue
        w = sign * (r["conviction"] / 100.0) / len(basket)
        for s in basket:
            raw[s] = raw.get(s, 0.0) + w
    gross = sum(abs(v) for v in raw.values())
    return {s: v / gross for s, v in raw.items()} if gross else {}


def intra_sector_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, conv_by_token: dict, *,
                         oi_floor: float = OI_FLOOR_USD) -> dict:
    """Within each BULLISH sector: long the standout (highest token conviction, else first member),
    short the rest equally. Market-neutral by construction (strips style tilt). conv_by_token:
    {token: conviction} from the signal's token rows (fallback 50)."""
    raw = {}
    for _, r in live.iterrows():
        if r["item_type"] != "sector" or STANCE_SIGN.get(r["stance"], 0) <= 0:
            continue
        basket = sector_basket(sector_map, oi_now, r["item"], oi_floor=oi_floor)
        if len(basket) < 2:
            continue
        standout = max(basket, key=lambda s: conv_by_token.get(s, 50))
        laggards = [s for s in basket if s != standout]
        raw[standout] = raw.get(standout, 0.0) + 1.0
        for s in laggards:
            raw[s] = raw.get(s, 0.0) - 1.0 / len(laggards)
    gross = sum(abs(v) for v in raw.values())
    return {s: v / gross for s, v in raw.items()} if gross else {}


def realize_period(weights: dict, prices: pd.DataFrame, funding: pd.DataFrame, t0, t1, *,
                   cost_bps: float = 10.0) -> float:
    """Period P&L: sum_w [ w * (price_return - sign(w)*funding) ] - turnover cost.
    cost_bps charged on gross at entry (one-way, simple)."""
    pnl = 0.0
    for s, w in weights.items():
        r = forward_return(prices, s, t0, t1)
        if r is None:
            continue
        f = period_funding(funding, s, t0, t1)
        pnl += w * (r - np.sign(w) * f)
    cost = (cost_bps / 1e4) * sum(abs(w) for w in weights.values())
    return pnl - cost


def walk_forward(panel: pd.DataFrame, prices: pd.DataFrame, funding: pd.DataFrame, oi_panel: pd.DataFrame,
                 sector_map: dict, dates: list, *, strategy: str = "sector_ls", bet_sign: str = "momentum",
                 beta_neutral: bool = True, betas: dict | None = None, cost_bps: float = 10.0) -> pd.DataFrame:
    """Run the strategy across rebalance dates. Returns DataFrame [as_of, ret] of per-period returns."""
    dates = sorted(pd.Timestamp(d) for d in dates)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    out = []
    for t in dates:
        if t not in nxt:
            continue
        live = panel[(pd.to_datetime(panel["as_of"]) == t) & (panel["lifecycle_state"] != "EXITED")]
        if live.empty:
            continue
        oi_now = oi_at(oi_panel, t)
        if strategy == "intra_sector":
            conv = {r["item"]: r["conviction"] for _, r in live[live["item_type"] == "token"].iterrows()}
            w = intra_sector_targets(live, sector_map, oi_now, conv)
        else:
            w = sector_ls_targets(live, sector_map, oi_now, bet_sign=bet_sign)
        if beta_neutral and betas:
            w = beta_neutralize(w, betas)
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, prices, funding, t, nxt[t], cost_bps=cost_bps)})
    return pd.DataFrame(out)


def benchmark_returns(prices: pd.DataFrame, dates: list, *, mode: str = "btc") -> pd.DataFrame:
    """'btc' = BTC buy-hold per period; 'eqw' = equal-weight all-symbol mean per period."""
    dates = sorted(pd.Timestamp(d) for d in dates)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    out = []
    for t in dates:
        if t not in nxt:
            continue
        if mode == "btc":
            r = forward_return(prices, "BTC", t, nxt[t]) or 0.0
        else:
            rs = [x for s in prices.columns if (x := forward_return(prices, s, t, nxt[t])) is not None]
            r = float(np.mean(rs)) if rs else 0.0
        out.append({"as_of": t.isoformat(), "ret": r})
    return pd.DataFrame(out)


def metrics(rets: pd.Series, *, periods_per_year: float = 4.0) -> dict:
    """Total/annualized return, Sharpe (per-period * sqrt(ppy)), max drawdown."""
    r = rets.dropna()
    if len(r) == 0:
        return {"total": 0.0, "ann": 0.0, "sharpe": float("nan"), "max_dd": 0.0, "n": 0}
    cum = (1 + r).cumprod()
    total = float(cum.iloc[-1] - 1)
    ann = float(cum.iloc[-1] ** (periods_per_year / len(r)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")
    dd = float((cum / cum.cummax() - 1).min())
    return {"total": total, "ann": ann, "sharpe": sharpe, "max_dd": dd, "n": int(len(r))}
