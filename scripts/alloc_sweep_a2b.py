"""Allocation sweep (Jax: 'a big one will be allocation'). Holds the SIGNAL fixed (A2b council, long the
top-K conviction sectors) and varies only HOW capital is allocated across the chosen sectors:
  - ew         equal weight (baseline)
  - cw         conviction-weighted (proportional to conviction)
  - cw2        conviction^2 (concentrate harder on high conviction)
  - invvol     risk-parity / inverse trailing-vol (equal risk contribution) — theory says this lifts Sharpe
  - cw_invvol  conviction / vol (conviction tilt, risk-normalized)
  - rankdecay  linearly decreasing by conviction rank (1st gets K, last gets 1)
Gross is normalized to 1 in every scheme — leverage scales return AND vol together so it is NOT a Sharpe
lever; allocation ACROSS names is. Reports the landscape vs BTC; reusable allocation fns for A3 later.
Honest: n=13, post-hoc — report the whole grid, no cherry-pick; OOS (monthly) is the arbiter."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at
from signals.market import sector_basket, OI_FLOOR_USD
from signals.backtest import realize_period, benchmark_returns, metrics
from signals.reconcile import apply_map
from signals.run import rebalance_dates

DATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "quarterly")]
prices = load_close_panel("data/market_data", "ohlcv")
funding = load_close_panel("data/market_data", "funding")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
id_map = json.load(open("data/signal/reconciled/reconciliation_map.json"))
tc = json.load(open("data/signal/reconciled/type_corrections.json"))
a2b = apply_map(pd.read_parquet("data/signal/a2b_council/signal_panel.parquet"),
                id_map, {d["id"]: d["new_type"] for d in tc})
btc = benchmark_returns(prices, DATES, mode="btc")
btcm = metrics(btc.set_index("as_of")["ret"])
NXT = {d: DATES[i + 1] for i, d in enumerate(DATES[:-1])}


def basket_vol(basket, t, lookback=90):
    """Trailing daily-return vol of the equal-weight basket as of T (<= T only)."""
    px = prices[[s for s in basket if s in prices.columns]]
    px = px[px.index <= pd.Timestamp(t)].tail(lookback + 1)
    if len(px) < 20:
        return None
    rets = px.pct_change().dropna(how="all").mean(axis=1)   # EW basket daily return
    v = float(rets.std())
    return v if v and v > 0 else None


def sector_weights(sel, scheme):
    """sel: list of (conviction, sector, basket, vol). Return {sector: weight} per allocation scheme."""
    convs = np.array([c for c, _, _, _ in sel], float)
    vols = np.array([(v if v else np.nan) for _, _, _, v in sel])
    n = len(sel)
    if scheme == "ew":
        w = np.ones(n)
    elif scheme == "cw":
        w = convs
    elif scheme == "cw2":
        w = convs ** 2
    elif scheme == "invvol":
        w = np.where(np.isnan(vols), np.nanmedian(vols), 1.0 / vols)
    elif scheme == "cw_invvol":
        iv = np.where(np.isnan(vols), np.nanmedian(vols), 1.0 / vols)
        w = convs * iv
    elif scheme == "rankdecay":
        order = convs.argsort()[::-1]
        w = np.zeros(n)
        for rank, idx in enumerate(order):
            w[idx] = n - rank
    else:
        raise ValueError(scheme)
    w = np.nan_to_num(w, nan=0.0)
    s = w.sum()
    w = w / s if s else np.ones(n) / n
    return {sel[i][1]: w[i] for i in range(n)}


def run(top_k, scheme):
    out = []
    for t in DATES:
        if t not in NXT:
            continue
        live = a2b[(pd.to_datetime(a2b["as_of"]) == t) & (a2b["lifecycle_state"] != "EXITED")]
        oin = oi_at(oi, t)
        sel = []
        for _, r in live.iterrows():
            if r["item_type"] != "sector" or r["stance"] != "bullish":
                continue
            basket = sector_basket(sm, oin, r["item"], oi_floor=OI_FLOOR_USD)
            if basket:
                sel.append((r["conviction"], r["item"], basket, basket_vol(basket, t)))
        sel.sort(key=lambda x: -x[0])
        if top_k:
            sel = sel[:top_k]
        if not sel:
            out.append({"as_of": t.isoformat(), "ret": 0.0}); continue
        sw = sector_weights(sel, scheme)
        raw = {}
        for conv, sec, basket, _ in sel:
            per = sw[sec] / len(basket)
            for s in basket:
                raw[s] = raw.get(s, 0.0) + per
        g = sum(abs(v) for v in raw.values())
        w = {s: v / g for s, v in raw.items()} if g else {}
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, prices, funding, t, NXT[t], cost_bps=10.0)})
    return pd.DataFrame(out)


def ab(r):
    m = r.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 3:
        return float("nan"), float("nan")
    b, a = np.polyfit(m["ret_b"].values, m["ret_s"].values, 1)
    return float(a) * 4, float(b)


print(f"BTC bench: Sharpe={btcm['sharpe']:.2f} total={btcm['total']:.0%}  <- beat this\n")
print(f"{'K':>4} {'scheme':>10} {'sharpe':>7} {'total':>8} {'maxDD':>7} {'alphaA':>8} {'beta':>6}")
rows = []
for K in [3, 5, 8, None]:
    for scheme in ["ew", "cw", "cw2", "invvol", "cw_invvol", "rankdecay"]:
        r = run(K, scheme)
        m = metrics(r.set_index("as_of")["ret"])
        a, bt = ab(r)
        rows.append((m["sharpe"], K, scheme, m, a, bt))
        print(f"{str(K):>4} {scheme:>10} {m['sharpe']:>7.2f} {m['total']:>8.0%} {m['max_dd']:>7.0%} {a:>8.1%} {bt:>6.2f}")
rows.sort(key=lambda x: -x[0])
b = rows[0]
print(f"\nBEST: K={b[1]} {b[2]} -> Sharpe {b[0]:.2f} vs BTC {btcm['sharpe']:.2f} (beats: {b[0] > btcm['sharpe']})")
