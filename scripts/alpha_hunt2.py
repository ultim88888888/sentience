"""Alpha hunt R2 — does A2b SELECTION beat dumb benchmarks, and can momentum TIME it?

R1: blend dead; A2b-monthly top-3 = Sharpe 0.98. R2 asks the rigor questions:
  1. EW-all-tradeable-sectors basket benchmark (the hm lesson: most edge evaporates vs EW).
  2. A2b top-3 vs that benchmark — is the SELECTION real?
  3. Momentum as TIMING overlay (not blend): A2b selects sectors, momentum gates entry / de-risks.
  4. Avoid-side: does dropping A4b-monthly-bearish sectors from an EW book help?
  5. Long-short proxy: long A2b top-3, short lowest-conviction tradeable sectors.
All monthly. Jackknife winners."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at, forward_return
from signals.market import sector_basket, OI_FLOOR_USD
from signals.backtest import realize_period, benchmark_returns, metrics
from signals.digest import _basket_trailing
from signals.reconcile import apply_map
from signals.run import rebalance_dates

prices = load_close_panel("data/market_data", "ohlcv")
funding = load_close_panel("data/market_data", "funding")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
ALL_SECTORS = sorted(set(sm.values()))
ID = json.load(open("data/signal/reconciled/reconciliation_map.json"))
TC = {d["id"]: d["new_type"] for d in json.load(open("data/signal/reconciled/type_corrections.json"))}
MDATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")]
NXT = {d: MDATES[i + 1] for i, d in enumerate(MDATES[:-1])}
btc = benchmark_returns(prices, MDATES, mode="btc")
BTCM = metrics(btc.set_index("as_of")["ret"], periods_per_year=12)


def load(p):
    return apply_map(pd.read_parquet(p), ID, TC)


def tradeable(t):
    oin = oi_at(oi, t)
    return [s for s in ALL_SECTORS if sector_basket(sm, oin, s, oi_floor=OI_FLOOR_USD)]


def sector_conv(panel, t):
    dts = sorted(pd.Timestamp(x) for x in panel["as_of"].unique())
    asof = [d for d in dts if d <= t]
    if not asof:
        return {}
    live = panel[(pd.to_datetime(panel["as_of"]) == asof[-1]) & (panel["item_type"] == "sector")]
    out = {}
    for _, r in live.iterrows():
        s = {"bullish": 1, "bearish": -1}.get(r["stance"], 0)
        if s:
            out[r["item"]] = out.get(r["item"], 0.0) + s * r["conviction"]
    return out


def mom(t, sec):
    b = sector_basket(sm, oi_at(oi, t), sec, oi_floor=OI_FLOOR_USD)
    return _basket_trailing(b, prices, t, 30) if b else None


def book_ret(weights_fn):
    out = []
    for t in MDATES:
        if t not in NXT:
            continue
        w = weights_fn(t)
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, prices, funding, t, NXT[t], cost_bps=10)})
    return pd.DataFrame(out)


def ew_book(sectors_fn, short=False):
    def f(t):
        oin = oi_at(oi, t)
        secs = sectors_fn(t)
        raw = {}
        for sec in secs:
            b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
            sgn = -1.0 if short else 1.0
            for s in b:
                raw[s] = raw.get(s, 0.0) + sgn / len(b)
        g = sum(abs(x) for x in raw.values())
        return {s: x / g for s, x in raw.items()} if g else {}
    return f


def topk(panel, t, k, require_mom=None):
    sc = {s: v for s, v in sector_conv(panel, t).items() if v > 0 and s in tradeable(t)}
    if require_mom is not None:
        sc = {s: v for s, v in sc.items() if (mom(t, s) or -9) >= require_mom}
    return [s for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]


def ab(r):
    m = r.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 3:
        return float("nan"), float("nan")
    b, a = np.polyfit(m["ret_b"].values, m["ret_s"].values, 1)
    return float(a) * 12, float(b)


def rep(name, r):
    m = metrics(r.set_index("as_of")["ret"], periods_per_year=12)
    a, beta = ab(r)
    print(f"{name:<40} Sharpe={m['sharpe']:>5.2f} tot={m['total']:>7.0%} DD={m['max_dd']:>5.0%} alphaA={a:>6.1%} beta={beta:>4.2f}")
    return m["sharpe"]


def jk(r):
    a = r["ret"].dropna().values
    def sh(x): return float(np.mean(x)/np.std(x,ddof=1)*np.sqrt(12)) if len(x)>1 and np.std(x)>0 else float('nan')
    v=[sh(np.delete(a,i)) for i in range(len(a))]; return min(v),max(v)


def main():
    A2B = load("data/signal/a2b_council/signal_panel.parquet")
    A4BM = load("data/signal/a4b_council_m/signal_panel.parquet")
    print(f"BTC monthly: Sharpe={BTCM['sharpe']:.2f} tot={BTCM['total']:.0%}\n")

    print("=== benchmarks (the real bar) ===")
    rep("EW all tradeable sectors", book_ret(ew_book(lambda t: tradeable(t))))
    rep("EW A2b-bullish sectors (no rank)", book_ret(ew_book(
        lambda t: [s for s, v in sector_conv(A2B, t).items() if v > 0 and s in tradeable(t)])))

    print("\n=== A2b selection (does ranking/concentration add over EW?) ===")
    a2b3 = book_ret(lambda t: _ew_named(t, topk(A2B, t, 3))); rep("A2b top-3 EW", a2b3)

    print("\n=== momentum as TIMING overlay on A2b top-3 ===")
    for thr in [0.0, 0.05, 0.10]:
        rep(f"A2b top-3, require 1mo-mom>={thr:.0%}",
            book_ret(lambda t, thr=thr: _ew_named(t, topk(A2B, t, 3, require_mom=thr))))

    print("\n=== avoid-side: EW A2b-bull MINUS A4b-monthly-bearish sectors ===")
    def avoid(t):
        bull = [s for s, v in sector_conv(A2B, t).items() if v > 0 and s in tradeable(t)]
        a4b_bear = {s for s, v in sector_conv(A4BM, t).items() if v < 0}
        return [s for s in bull if s not in a4b_bear]
    rep("EW A2b-bull minus A4b-bearish", book_ret(ew_book(avoid)))

    print("\n=== long-short proxy: long A2b top-3, short lowest-conviction tradeable ===")
    rep("LS A2b top3 / short bottom-3", book_ret(lambda t: ls_weights(A2B, t, 3)))

    print(f"\nBTC Sharpe to beat: {BTCM['sharpe']:.2f}")
    print(f"A2b top-3 jackknife Sharpe: {jk(a2b3)[0]:.2f}..{jk(a2b3)[1]:.2f}")


# --- weight builders that take t (book_ret passes t) ---
def _ew_named(t, secs):
    oin = oi_at(oi, t); raw = {}
    for sec in secs:
        b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
        for s in b:
            raw[s] = raw.get(s, 0.0) + 1.0 / len(b)
    g = sum(abs(x) for x in raw.values()); return {s: x/g for s, x in raw.items()} if g else {}


def ls_weights(panel, t, k):
    sc = {s: v for s, v in sector_conv(panel, t).items() if s in tradeable(t)}
    longs = [s for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]
    shorts = [s for s, _ in sorted(sc.items(), key=lambda kv: kv[1])[:k]]
    oin = oi_at(oi, t); raw = {}
    for sec in longs:
        b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
        for s in b: raw[s] = raw.get(s, 0.0) + 0.5/len(b)
    for sec in shorts:
        b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
        for s in b: raw[s] = raw.get(s, 0.0) - 0.5/len(b)
    g = sum(abs(x) for x in raw.values()); return {s: x/g for s, x in raw.items()} if g else {}


if __name__ == "__main__":
    main()
