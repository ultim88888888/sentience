"""Refine the momentum gate to cut the 2024 whipsaw (Jax: flat-while-BTC-rises period).
Diagnosis: the 1M-momentum gate kept buying themes on 1-month bounces within larger reversals
(ai-crypto re-entered 5×, bleeding). Test anti-whipsaw variants, full-sample + 2024 window + WFV.

Variants:
  base      : 1M basket momentum >= 10% (current)
  dual      : 1M >= 10% AND 3M > 0 (theme in sustained uptrend, not a dead-cat bounce)
  dual+     : 1M >= 10% AND 3M >= 10% (stronger trend confirmation)
  persist   : base, but sector must have qualified the PRIOR month too (no one-month head-fakes)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import scripts.alpha_hunt3 as ah, scripts.new_ideas as ni
from signals.backtest import realize_period, beta_neutralize, metrics
from signals.informativeness import forward_return
from signals.market import sector_basket, OI_FLOOR_USD
from signals.digest import _basket_trailing

A2B = ni.A2B
dates = [d for d in ah.MD if d in ah.NXT]


def mom_n(t, sec, days):
    b = sector_basket(ah.sm, ah.oi_at(ah.oi, t), sec, oi_floor=OI_FLOOR_USD)
    return _basket_trailing(b, ah.prices, t, days) if b else None


def qualifies(t, sec, variant):
    m1 = mom_n(t, sec, 30)
    if m1 is None or m1 < 0.10:
        return False
    if variant == "base":
        return True
    if variant == "dual":
        m3 = mom_n(t, sec, 91); return m3 is not None and m3 > 0
    if variant == "dual+":
        m3 = mom_n(t, sec, 91); return m3 is not None and m3 >= 0.10
    return True


def picks(t, variant):
    sc = {s: v for s, v in ah.sconv(A2B, t).items() if v > 0 and s in ah.tradeable(t)}
    qual = {s: v for s, v in sc.items() if qualifies(t, s, variant)}
    return [s for s, _ in sorted(qual.items(), key=lambda kv: -kv[1])[:3]]


def run(variant):
    prev = set()
    rows = []
    for t in dates:
        p = picks(t, variant)
        if variant == "persist":
            p = [s for s in p if s in prev] or p  # require prior-month qualify; if none persisted, keep current
            prev = set(picks(t, "base"))
        w = ah.ew(t, p) if p else {"BTC": 1.0}
        if ni.trailing_btc(t) < -0.10:
            w = beta_neutralize(w, ah.BETAS)
        rows.append({"m": t.strftime("%Y-%m"), "ret": realize_period(w, ah.prices, ah.funding, t, ah.NXT[t], cost_bps=10),
                     "btc": forward_return(ah.prices, "BTC", t, ah.NXT[t]) or 0.0})
    return pd.DataFrame(rows)


def jt(r):
    mm = r.merge(r[["m"]], on="m")  # noop
    X = np.column_stack([np.ones(len(r)), r["btc"].values]); c, *_ = np.linalg.lstsq(X, r["ret"].values, rcond=None)
    res = r["ret"].values - X @ c; s2 = (res @ res) / (len(r) - 2); ta = c[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return c[0] * 12, ta


def tot(x): return (1 + x).prod() - 1


def main():
    print("=== whipsaw-refinement variants ===")
    print(f"{'variant':<10}{'FULL tot':>10}{'Sharpe':>8}{'alpha':>7}{'t':>6}{'  2024-25H1 excess':>20}")
    for v in ["base", "dual", "dual+", "persist"]:
        r = run(v)
        m = metrics(r.set_index("m")["ret"], periods_per_year=12)
        a, ta = jt(r)
        win = r[(r.m >= "2024-01") & (r.m <= "2025-08")]
        ex = tot(win["ret"]) - tot(win["btc"])
        print(f"{v:<10}{m['total']:>+9.0%}{m['sharpe']:>8.2f}{a:>+6.0%}{ta:>6.2f}{ex:>+19.0%}")


if __name__ == "__main__":
    main()
