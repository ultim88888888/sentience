"""THE DELIVERABLE — signal-gated RELATIVE-momentum sector rotation (case-study capstone).

Long the top-3 a16z-corpus (A2b council) BULLISH sectors whose basket is OUTPERFORMING BTC over both 1mo
AND 3mo (relative momentum, regime-neutral — #30/#31); equal-weight liquid baskets; hold BTC when none
qualify; beta-neutralize in bear regimes (#20). Monthly, $10mm.

Why RELATIVE momentum (the key refinement): the absolute-momentum gate only fired in uptrends, so it just
rode BTC down in the 2022 bear. The relative gate stays engaged in downturns by rotating into sectors that
fall LESS than BTC. The council provides the narrative-backed candidate SET; relative momentum times entry.
Council ∩ relative beats pure relative momentum in BOTH regimes (in-sample 2x; OOS bear -9% vs -69%).

Profile in-sample (2022-12..2026-03): $10mm->$149mm, Sharpe 1.26 (BTC 1.09), alpha +75%/yr, DD-45%.
**FULL CYCLE 2021-09..2026-03 incl the 2022 bear (53 mo): +1260% vs BTC +62%, Sharpe 1.03 (BTC 0.46),
alpha +71%/yr, t=1.92 (study best), DD-54% (BTC-72%).** WFV OOS +360% vs BTC +61%, t=1.22.
HONEST: t still <1.96; the council's raw conviction RANKING inverts OOS (#29) but its bullish CLASSIFICATION
+ relative-momentum timing is regime-robust; ~half P&L is beta; small-AUM (fades >$50mm)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at
from signals.market import sector_basket, compute_beta, OI_FLOOR_USD
from signals.backtest import realize_period, benchmark_returns, metrics, beta_neutralize
from signals.digest import _basket_trailing
from signals.reconcile import apply_map
from signals.run import rebalance_dates

CAPITAL = 10_000_000
# RELATIVE-momentum gate (#30/#31): sector basket must OUTPERFORM BTC over both 1mo and 3mo. Regime-neutral
# (the absolute gate only fired in uptrends → rode BTC down in the 2022 bear). thr=0 = "just beat BTC",
# parameter-free. Full-cycle 2021-26 (incl bear): Sharpe 1.03 vs BTC 0.46, alpha t=1.92 (study best).
REL_GATE_1M = 0.0   # (basket 1mo return) − (BTC 1mo return) >= this
REL_GATE_3M = 0.0   # (basket 3mo return) − (BTC 3mo return) >= this
TOP_K = 3
BEAR_HEDGE = -0.10  # beta-neutralize when trailing-3mo BTC < this (regime hedge — finding #20)

prices = load_close_panel("data/market_data", "ohlcv")
funding = load_close_panel("data/market_data", "funding")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
_b90 = compute_beta(prices.pct_change(), window=90, market="BTC")
BETAS = {c: (float(_b90[c].dropna().iloc[-1]) if len(_b90[c].dropna()) else 1.0) for c in _b90.columns}
BETAS["BTC"] = 1.0
ALL = sorted(set(sm.values()))
ID = json.load(open("data/signal/reconciled/reconciliation_map.json"))
TC = {d["id"]: d["new_type"] for d in json.load(open("data/signal/reconciled/type_corrections.json"))}
MD = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")]
NXT = {d: MD[i + 1] for i, d in enumerate(MD[:-1])}
btc = benchmark_returns(prices, MD, mode="btc")


def a2b_bullish(t):
    panel = apply_map(pd.read_parquet("data/signal/a2b_council/signal_panel.parquet"), ID, TC)
    dts = sorted(pd.Timestamp(x) for x in panel["as_of"].unique())
    asof = [d for d in dts if d <= t]
    if not asof:
        return {}
    live = panel[(pd.to_datetime(panel["as_of"]) == asof[-1]) & (panel["item_type"] == "sector")
                 & (panel["stance"] == "bullish")]
    return {r["item"]: r["conviction"] for _, r in live.iterrows()}


def weights(t):
    oin = oi_at(oi, t)
    cand = []
    for sec, conv in a2b_bullish(t).items():
        b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
        if not b:
            continue
        r1 = _rel_mom(b, t, 30)   # basket return minus BTC return, trailing 1mo
        r3 = _rel_mom(b, t, 91)   # ... trailing 3mo
        if r1 is not None and r1 >= REL_GATE_1M and r3 is not None and r3 >= REL_GATE_3M:
            cand.append((conv, sec, b))
    cand.sort(key=lambda x: -x[0])
    cand = cand[:TOP_K]
    if not cand:
        return {"BTC": 1.0}, ["BTC (fallback)"]
    raw = {}
    for conv, sec, b in cand:
        for s in b:
            raw[s] = raw.get(s, 0.0) + 1.0 / len(b)
    g = sum(abs(x) for x in raw.values())
    w = {s: x / g for s, x in raw.items()}
    if _trailing_btc(t) < BEAR_HEDGE:                 # regime bear-hedge
        w = beta_neutralize(w, BETAS)
    return w, [c[1] for c in cand]


def _trailing_btc(t, days=91):
    s = prices["BTC"].dropna()
    a = s[s.index <= pd.Timestamp(t) - pd.Timedelta(days=days)]
    b = s[s.index <= pd.Timestamp(t)]
    return (b.iloc[-1] / a.iloc[-1] - 1) if len(a) and len(b) else 0.0


def _rel_mom(basket, t, days):
    """Basket trailing return minus BTC trailing return over the same window (relative momentum vs BTC)."""
    bm = _basket_trailing(basket, prices, t, days)
    bt = _trailing_btc(t, days)
    return (bm - bt) if (bm is not None and bt is not None) else None


def main():
    rows, picks_log = [], []
    for t in MD:
        if t not in NXT:
            continue
        w, picks = weights(t)
        r = realize_period(w, prices, funding, t, NXT[t], cost_bps=10)
        rows.append({"as_of": t.isoformat(), "ret": r})
        picks_log.append(f"{t.date()}: {', '.join(picks)}  ({r:+.1%})")
    df = pd.DataFrame(rows)
    m = metrics(df.set_index("as_of")["ret"], periods_per_year=12)
    bm = metrics(btc.set_index("as_of")["ret"], periods_per_year=12)
    mm = df.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    beta, alpha = np.polyfit(mm["ret_b"], mm["ret_s"], 1)
    eq = CAPITAL * (1 + df["ret"]).cumprod().iloc[-1]

    print("=" * 64)
    print("  SIGNAL-GATED MOMENTUM SECTOR ROTATION — tearsheet")
    print(f"  (top-{TOP_K} A2b-bullish ∩ relative-mom>BTC (1M&3M), BTC fallback, bear-hedge, monthly)")
    print("=" * 64)
    print(f"  period:        {df['as_of'].iloc[0]} .. {df['as_of'].iloc[-1]}  ({len(df)} months)")
    print(f"  ${CAPITAL/1e6:.0f}mm → ${eq/1e6:.1f}mm   ({m['total']:+.0%} total)")
    print(f"  Sharpe:        {m['sharpe']:.2f}   (BTC {bm['sharpe']:.2f})")
    print(f"  ann. return:   {m['ann']:+.0%}")
    print(f"  Jensen alpha:  {alpha*12:+.0%}/yr   beta {beta:.2f}")
    print(f"  max drawdown:  {m['max_dd']:.0%}   (BTC {bm['max_dd']:.0%})")
    print(f"  win rate:      {(df['ret']>0).mean():.0%}")
    print("=" * 64)
    print("  monthly picks:")
    for line in picks_log:
        print("   ", line)
    df.to_parquet("data/signal/strategy_final_returns.parquet")
    print("\n  returns saved -> data/signal/strategy_final_returns.parquet")


if __name__ == "__main__":
    main()
