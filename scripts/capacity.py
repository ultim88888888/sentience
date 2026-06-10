"""Capacity-realism backtest (honest deflation test). The deliverable trades thin sector baskets; the
token attribution showed ZEC (only ~$13.6mm OI) drove 37% of P&L. Spec constraint: $10mm AUM, max 5% of a
name's OI/day via TWAP. With ~5 trading days to build before monthly rebalance → max position ≈ 25% of OI.

This caps each token's dollar weight at (cap_days × 5% × OI) / AUM, redistributes the un-fillable excess to
BTC (liquid), and re-runs the refined strategy. If returns survive, the edge is real at size; if they
collapse, the backtest was capacity-fiction."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import scripts.alpha_hunt3 as ah
import scripts.new_ideas as ni
from signals.backtest import realize_period, beta_neutralize, metrics
from signals.informativeness import forward_return

A2B = ni.A2B
AUM = 10_000_000
dates = [d for d in ah.MD if d in ah.NXT]


def cap_weights(t, w, *, oi_pct_day=0.05, days=5):
    """Cap each token weight to (days*oi_pct_day*OI)/AUM; excess → BTC. w sums to ~1 (gross)."""
    oin = ah.oi_at(ah.oi, t)
    capped, spill = {}, 0.0
    for s, wt in w.items():
        if s == "BTC":
            capped["BTC"] = capped.get("BTC", 0.0) + wt
            continue
        max_dollar = days * oi_pct_day * (oin.get(s) or 0)
        max_w = max_dollar / AUM
        if abs(wt) <= max_w:
            capped[s] = capped.get(s, 0.0) + wt
        else:
            fill = np.sign(wt) * max_w
            capped[s] = capped.get(s, 0.0) + fill
            spill += abs(wt) - abs(fill)
    if spill > 1e-9:
        capped["BTC"] = capped.get("BTC", 0.0) + spill  # park un-fillable in liquid BTC
    return capped


def run(name, *, capped, hedge=True, oi_pct_day=0.05, days=5):
    out = []
    for t in dates:
        secs = ah.topk(A2B, t, 3, mom_thr=0.10)
        w = ah.ew(t, secs) if secs else {"BTC": 1.0}
        if hedge and ni.trailing_btc(t) < -0.10:
            w = beta_neutralize(w, ah.BETAS)
        if capped:
            w = cap_weights(t, w, oi_pct_day=oi_pct_day, days=days)
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, ah.prices, ah.funding, t, ah.NXT[t], cost_bps=10)})
    r = pd.DataFrame(out)
    m = metrics(r.set_index("as_of")["ret"], periods_per_year=12)
    mm = r.merge(ah.btc, on="as_of").dropna()
    X = np.column_stack([np.ones(len(mm)), mm["ret_y"].values]); c, *_ = np.linalg.lstsq(X, mm["ret_x"].values, rcond=None)
    res = mm["ret_x"].values - X @ c; s2 = (res @ res) / (len(mm) - 2); ta = c[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    print(f"  {name:<34} tot {m['total']:>+7.0%} | Sharpe {m['sharpe']:>4.2f} | DD {m['max_dd']:>4.0%} | alpha {c[0]*12:>+5.0%} t={ta:.2f}")
    return m


def main():
    bm = metrics(ah.btc.set_index("as_of")["ret"], periods_per_year=12)
    print(f"BTC: Sharpe {bm['sharpe']:.2f} tot {bm['total']:+.0%}\n")
    print("=== refined strategy: uncapped vs capacity-capped (5%/OI/day TWAP @ $10mm) ===")
    run("uncapped (current headline)", capped=False)
    run("capped 5%/day × 5d (=25% OI)", capped=True, days=5)
    run("capped 5%/day × 3d (=15% OI)", capped=True, days=3)
    run("capped 5%/day × 1d (=5% OI, harsh)", capped=True, days=1)
    print("\n=== capacity at larger AUM (does it scale?) — 25% OI cap ===")
    for aum in [10e6, 50e6, 100e6]:
        global AUM
        AUM = aum
        m = run(f"  AUM ${aum/1e6:.0f}mm", capped=True, days=5)
    AUM = 10e6


if __name__ == "__main__":
    main()
