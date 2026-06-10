"""Overnight alpha hunt (Jax's mandate: sweep, simulate, invent, find alpha).

Central thesis from finding #12: A2b (quarterly static) is clean momentum-INDEPENDENT alpha; A4b-monthly is
momentum-LADEN alpha. Near-orthogonal → a BLEND should be additive. This harness:
  - monthly backtests each signal (long top-K sectors by conviction) vs BTC
  - builds the BLEND: at each monthly T, z-score the as-of A2b structural conviction + the A4b-monthly
    conviction per sector, combine with weight w, long top-K of the blend
  - sweeps top-K, blend weight, allocation; reports Sharpe / total / alpha / beta, with jackknife on winners

All monthly cadence (40 periods). Honest: report the landscape; jackknife anything that beats BTC."""
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

prices = load_close_panel("data/market_data", "ohlcv")
funding = load_close_panel("data/market_data", "funding")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
ID = json.load(open("data/signal/reconciled/reconciliation_map.json"))
TC = {d["id"]: d["new_type"] for d in json.load(open("data/signal/reconciled/type_corrections.json"))}
MDATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")]
NXT = {d: MDATES[i + 1] for i, d in enumerate(MDATES[:-1])}
btc = benchmark_returns(prices, MDATES, mode="btc")
BTCM = metrics(btc.set_index("as_of")["ret"])


def load(p):
    return apply_map(pd.read_parquet(p), ID, TC)


def sector_conv(panel, t):
    """{sector: signed conviction} for bullish/bearish sector rows as-of the most recent panel date <= t."""
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


def _z(d):
    if len(d) < 2:
        return {k: 0.0 for k in d}
    v = np.array(list(d.values()), float)
    m, s = v.mean(), v.std()
    return {k: ((val - m) / s if s > 0 else 0.0) for k, val in d.items()}


def long_topk(score_fn, k=3, alloc="ew"):
    """score_fn(t) -> {sector: score}; long top-k by score, weight ew/score. Returns monthly ret series."""
    out = []
    for t in MDATES:
        if t not in NXT:
            continue
        oin = oi_at(oi, t)
        sc = {s: v for s, v in score_fn(t).items()
              if v > 0 and sector_basket(sm, oin, s, oi_floor=OI_FLOOR_USD)}
        top = sorted(sc.items(), key=lambda kv: -kv[1])[:k]
        raw = {}
        for sec, v in top:
            b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
            wsec = v if alloc == "score" else 1.0
            for s in b:
                raw[s] = raw.get(s, 0.0) + wsec / len(b)
        g = sum(abs(x) for x in raw.values())
        w = {s: x / g for s, x in raw.items()} if g else {}
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, prices, funding, t, NXT[t], cost_bps=10)})
    return pd.DataFrame(out)


def ab(r):
    m = r.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 3:
        return float("nan"), float("nan")
    b, a = np.polyfit(m["ret_b"].values, m["ret_s"].values, 1)
    return float(a) * 12, float(b)


def jk_sharpe(r):
    a = r["ret"].dropna().values
    def sh(x): return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(12)) if len(x) > 1 and np.std(x) > 0 else float("nan")
    return [round(sh(np.delete(a, i)), 2) for i in range(len(a))]


def report(name, r):
    m = metrics(r.set_index("as_of")["ret"], periods_per_year=12)
    a, beta = ab(r)
    print(f"{name:<34} Sharpe={m['sharpe']:>5.2f} tot={m['total']:>7.0%} DD={m['max_dd']:>5.0%} "
          f"alphaA={a:>6.1%} beta={beta:>4.2f}")
    return m["sharpe"]


def main():
    A2B = load("data/signal/a2b_council/signal_panel.parquet")     # quarterly clean
    A4BM = load("data/signal/a4b_council_m/signal_panel.parquet")   # monthly momentum-laden
    A4AM = load("data/signal/a4a_consensus_m/signal_panel.parquet")
    print(f"BTC monthly: Sharpe={BTCM['sharpe']:.2f} tot={BTCM['total']:.0%}\n")

    print("=== individual signals, long top-3 EW, monthly rebalance ===")
    report("A2b (quarterly conv, monthly rebal)", long_topk(lambda t: sector_conv(A2B, t), 3))
    report("A4b-monthly", long_topk(lambda t: sector_conv(A4BM, t), 3))
    report("A4a-monthly", long_topk(lambda t: sector_conv(A4AM, t), 3))

    print("\n=== THE BLEND: z(A2b structural) + w*z(A4b monthly), long top-3 ===")
    best = (None, -9)
    for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        def blend(t, w=w):
            za = _z(sector_conv(A2B, t)); zb = _z(sector_conv(A4BM, t))
            keys = set(za) | set(zb)
            return {k: za.get(k, 0.0) + w * zb.get(k, 0.0) for k in keys}
        r = long_topk(blend, 3)
        s = report(f"blend w={w}", r)
        if s > best[1]:
            best = ((f"blend w={w}", r), s)

    print("\n=== top-K sweep on the best blend ===")
    bestname, bestr = best[0]
    bw = float(bestname.split("=")[1])
    for k in [2, 3, 4, 5, 8]:
        def blend(t, w=bw):
            za = _z(sector_conv(A2B, t)); zb = _z(sector_conv(A4BM, t))
            keys = set(za) | set(zb)
            return {k2: za.get(k2, 0.0) + w * zb.get(k2, 0.0) for k2 in keys}
        report(f"{bestname} K={k}", long_topk(blend, k))

    print(f"\nBTC Sharpe to beat: {BTCM['sharpe']:.2f}")
    print(f"best blend jackknife Sharpe range: {min(jk_sharpe(bestr)):.2f}..{max(jk_sharpe(bestr)):.2f}")


if __name__ == "__main__":
    main()
