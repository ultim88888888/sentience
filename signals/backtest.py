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
    # Coinglass funding 'close' is in PERCENT (0.0089 = 0.0089%); convert to fraction.
    return float(s.dropna().sum()) / 100.0


def beta_neutralize(weights: dict, betas: dict, *, hedge: str = "BTC") -> dict:
    """Zero the portfolio's net BTC-beta with a BTC OVERLAY (bounded leverage), not by scaling legs.
    Net beta of the book = Σ w_i·β_i; add a BTC position of −net_beta (β_BTC=1) to cancel it. This
    keeps the L/S book intact and gross bounded (~1 + |net_beta|), avoiding the leverage explosion
    that scaling a low-beta short book causes."""
    net_beta = sum(w * betas.get(s, 1.0) for s, w in weights.items() if s != hedge)
    out = dict(weights)
    out[hedge] = out.get(hedge, 0.0) - net_beta
    if abs(out[hedge]) < 1e-12:
        out.pop(hedge, None)
    return out


def _finalize(raw: dict, *, gross: float = 1.0, cap: int | None = None) -> dict:
    """Cap to the top-`cap` positions by |weight|, then gross-normalize to sum|w|=gross.
    raw: {sym: signed weight}. Empty -> {}."""
    if not raw:
        return {}
    items = sorted(raw.items(), key=lambda kv: -abs(kv[1]))
    if cap:
        items = items[:cap]
    g = sum(abs(w) for _, w in items)
    return {s: w / g * gross for s, w in items} if g else {}


def sector_ls_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, *,
                      bet_sign: str = "momentum", oi_floor: float = OI_FLOOR_USD,
                      conviction_weighted: bool = True, cap: int | None = None) -> dict:
    """Sector long/short from the live signal rows at T. bet_sign 'momentum': long bullish sectors,
    short bearish. 'fade': reverse. Sized by conviction (or equal-weight when conviction_weighted=False).
    Each sector -> equal-weight basket members. Returns {symbol: signed weight} (gross-normalized to
    sum |w| = 1)."""
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
        mag = (r["conviction"] / 100.0) if conviction_weighted else 1.0
        w = sign * mag / len(basket)
        for s in basket:
            raw[s] = raw.get(s, 0.0) + w
    return _finalize(raw, cap=cap)


def intra_sector_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, conv_by_token: dict, *,
                         oi_floor: float = OI_FLOOR_USD, conviction_weighted: bool = True,
                         cap: int | None = None) -> dict:
    """Within each BULLISH sector: long the standout (highest token conviction, else first member),
    short the rest equally. Market-neutral by construction (strips style tilt). conv_by_token:
    {token: conviction} from the signal's token rows (fallback 50). When conviction_weighted=False,
    each position contributes equal magnitude (sign only, pre-basket-split)."""
    raw = {}
    for _, r in live.iterrows():
        if r["item_type"] != "sector" or STANCE_SIGN.get(r["stance"], 0) <= 0:
            continue
        basket = sector_basket(sector_map, oi_now, r["item"], oi_floor=oi_floor)
        if len(basket) < 2:
            continue
        standout = max(basket, key=lambda s: conv_by_token.get(s, 50))
        laggards = [s for s in basket if s != standout]
        mag = 1.0  # intra-sector: equal magnitude per sector regardless of conviction_weighted
        raw[standout] = raw.get(standout, 0.0) + mag
        for s in laggards:
            raw[s] = raw.get(s, 0.0) - mag / len(laggards)
    return _finalize(raw, cap=cap)


def long_only_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, *,
                      bet_sign: str = "momentum", conviction_weighted: bool = True,
                      cap: int | None = None, oi_floor: float = OI_FLOOR_USD) -> dict:
    """Long the favored side only (no opposite leg). momentum: long BULLISH sector baskets;
    fade: long BEARISH. Net-long book, gross-normalized. (Beta hedging handled separately.)"""
    raw = {}
    want = "bullish" if bet_sign == "momentum" else "bearish"
    for _, r in live.iterrows():
        if r["item_type"] != "sector" or r["stance"] != want:
            continue
        basket = sector_basket(sector_map, oi_now, r["item"], oi_floor=oi_floor)
        if not basket:
            continue
        w = (r["conviction"] / 100.0 if conviction_weighted else 1.0) / len(basket)
        for s in basket:
            raw[s] = raw.get(s, 0.0) + w
    return _finalize(raw, cap=cap)


def token_ls_targets(live: pd.DataFrame, *, conviction_weighted: bool = True,
                     cap: int | None = None) -> dict:
    """Rank the signal's TOKEN rows by signed conviction; long bullish tokens, short bearish.
    Cross-sector token L/S (distinct from intra_sector which is within-basket)."""
    raw = {}
    for _, r in live.iterrows():
        if r["item_type"] != "token":
            continue
        sign = STANCE_SIGN.get(r["stance"], 0)
        if sign == 0:
            continue
        raw[r["item"]] = raw.get(r["item"], 0.0) + sign * (r["conviction"] / 100.0 if conviction_weighted else 1.0)
    return _finalize(raw, cap=cap)


def token_vs_sector_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, *,
                             conviction_weighted: bool = True, cap: int | None = None,
                             oi_floor: float = OI_FLOOR_USD) -> dict:
    """Long the named token / short its parent-sector basket (idiosyncratic, sector-neutral).
    For each token row with a parent_sector: sign*mag on the token, -sign*mag spread across the
    sector basket (excluding the token itself). mag = conviction/100 if conviction_weighted else 1."""
    raw = {}
    for _, r in live.iterrows():
        if r["item_type"] != "token":
            continue
        sign = STANCE_SIGN.get(r["stance"], 0)
        sector = r.get("parent_sector")
        if sign == 0 or not sector or (isinstance(sector, float)):
            continue
        basket = [s for s in sector_basket(sector_map, oi_now, sector, oi_floor=oi_floor) if s != r["item"]]
        if not basket:
            continue
        mag = (r["conviction"] / 100.0) if conviction_weighted else 1.0
        raw[r["item"]] = raw.get(r["item"], 0.0) + sign * mag                 # long the token
        for s in basket:
            raw[s] = raw.get(s, 0.0) - sign * mag / len(basket)               # short its sector
    return _finalize(raw, cap=cap)


def both_targets(live: pd.DataFrame, sector_map: dict, oi_now: dict, *,
                 bet_sign: str = "momentum", conviction_weighted: bool = True,
                 cap: int | None = None, oi_floor: float = OI_FLOOR_USD) -> dict:
    """Combined: sector L/S book + token L/S book, merged and gross-normalized."""
    s = sector_ls_targets(live, sector_map, oi_now, bet_sign=bet_sign,
                          conviction_weighted=conviction_weighted, oi_floor=oi_floor)
    t = token_ls_targets(live, conviction_weighted=conviction_weighted)
    raw = {}
    for d in (s, t):
        for k, v in d.items():
            raw[k] = raw.get(k, 0.0) + v
    return _finalize(raw, cap=cap)


def _btc_trend(prices: pd.DataFrame, t, *, lookback: int = 200) -> int:
    """+1 if BTC close at t >= its trailing `lookback`-day mean (risk-on), else -1 (risk-off).
    0 if no data."""
    if "BTC" not in prices.columns:
        return 0
    s = prices["BTC"].dropna()
    s = s[s.index <= pd.Timestamp(t)]
    if len(s) < lookback:
        return 0
    return 1 if s.iloc[-1] >= s.iloc[-lookback:].mean() else -1


def regime_target_beta(prices: pd.DataFrame, t, risk_stance: str, *, mode: str) -> float:
    """Target NET portfolio beta given the regime. mode:
      'quant'     -> +0.5 if BTC uptrend else -0.5
      'consensus' -> from signal risk_regime: risk_on->+0.5, risk_off->-0.5, else 0
      'combined'  -> average of the two
    (Magnitude 0.5 = take half-beta in the favored direction; 0 = fully neutral.)"""
    q = 0.5 * _btc_trend(prices, t)
    c = {"risk_on": 0.5, "risk_off": -0.5}.get(risk_stance, 0.0)
    return {"quant": q, "consensus": c, "combined": (q + c) / 2}[mode]


def hedge_to_target_beta(weights: dict, betas: dict, target_beta: float, *,
                         hedge: str = "BTC") -> dict:
    """Add a BTC overlay so net portfolio beta == target_beta (target_beta=0 -> fully neutral)."""
    net = sum(w * betas.get(s, 1.0) for s, w in weights.items() if s != hedge)
    out = dict(weights)
    out[hedge] = out.get(hedge, 0.0) + (target_beta - net)
    if abs(out[hedge]) < 1e-12:
        out.pop(hedge, None)
    return out


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
                 beta_neutral: bool = True, betas: dict | None = None, cost_bps: float = 10.0,
                 conviction_weighted: bool = True, cap: int | None = None,
                 hedge_mode: str = "neutral",
                 risk_by_date: dict | None = None) -> pd.DataFrame:
    """Run the strategy across rebalance dates. Returns DataFrame [as_of, ret] of per-period returns.

    strategy: 'sector_ls' | 'long_only' | 'token_ls' | 'intra_sector'
    hedge_mode: 'neutral' -> beta_neutralize (target 0); 'none' -> no hedge;
                'quant'|'consensus'|'combined' -> hedge_to_target_beta via regime_target_beta.
    risk_by_date: {iso_date_str: risk_stance_str} for regime-aware hedge modes (default {}).
    """
    if risk_by_date is None:
        risk_by_date = {}
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
            w = intra_sector_targets(live, sector_map, oi_now, conv,
                                     conviction_weighted=conviction_weighted, cap=cap)
        elif strategy == "long_only":
            w = long_only_targets(live, sector_map, oi_now, bet_sign=bet_sign,
                                  conviction_weighted=conviction_weighted, cap=cap)
        elif strategy == "token_ls":
            w = token_ls_targets(live, conviction_weighted=conviction_weighted, cap=cap)
        elif strategy == "token_vs_sector":
            w = token_vs_sector_targets(live, sector_map, oi_now,
                                        conviction_weighted=conviction_weighted, cap=cap)
        elif strategy == "both":
            w = both_targets(live, sector_map, oi_now, bet_sign=bet_sign,
                             conviction_weighted=conviction_weighted, cap=cap)
        else:  # sector_ls (default)
            w = sector_ls_targets(live, sector_map, oi_now, bet_sign=bet_sign,
                                  conviction_weighted=conviction_weighted, cap=cap)

        # Apply hedge
        if hedge_mode == "neutral" and beta_neutral and betas:
            w = beta_neutralize(w, betas)
        elif hedge_mode == "none":
            pass  # no hedge
        elif hedge_mode in ("quant", "consensus", "combined") and betas:
            t_iso = t.isoformat()
            risk_stance = risk_by_date.get(t_iso, "neutral")
            target = regime_target_beta(prices, t, risk_stance, mode=hedge_mode)
            w = hedge_to_target_beta(w, betas, target)
        elif hedge_mode == "neutral" and beta_neutral and not betas:
            pass  # no betas provided, skip hedge

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
