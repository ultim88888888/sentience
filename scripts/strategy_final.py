"""THE DELIVERABLE — signal-gated momentum sector rotation (case-study capstone).

Long the top-3 a16z-corpus (A2b council) BULLISH sectors that clear BOTH a 1mo AND 3mo momentum gate
(dual+ trend confirmation, #27); equal-weight liquid baskets; hold BTC when none qualify; beta-neutralize
in bear regimes (#20). Monthly, $10mm.

Why this construction (from the 15-finding research arc):
  - corpus signal is a cross-sectionally informative SECTOR selector (A2b IC t=2.42), not a token picker
  - momentum works at monthly cadence and TIMES the signal (complementary, not redundant — alpha 42% beats
    pure-momentum 34% and pure-signal 22%)
  - monotonic threshold response + both-half OOS stability ⇒ structural, not overfit
Profile (2022-12..2026-03, 39 mo): $10mm->$167mm, Sharpe 1.41 (BTC 1.09), Jensen alpha +73%/yr, β0.81,
DD-36%. WFV OOS +382% vs BTC +61% (selects 3M filter 27/27). HONEST: alpha t~1.7, not formally significant
at n=39; ~half P&L is beta, alpha rests on ~2 early theme hits (AI23, privacy25). Small-AUM (fades >$50mm)."""
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
MOM_GATE = 0.10      # trailing-1mo basket momentum bar
MOM_GATE_3M = 0.10  # trailing-3mo trend confirmation (dual+, kills the whipsaw — finding #27)
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
        m1 = _basket_trailing(b, prices, t, 30)
        m3 = _basket_trailing(b, prices, t, 91)
        if m1 is not None and m1 >= MOM_GATE and m3 is not None and m3 >= MOM_GATE_3M:
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
    print(f"  (top-{TOP_K} A2b-bullish ∩ 1M≥{MOM_GATE:.0%} & 3M≥{MOM_GATE_3M:.0%}, BTC fallback, bear-hedge, monthly)")
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
