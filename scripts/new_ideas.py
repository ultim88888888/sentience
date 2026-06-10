"""New-idea round (Jax: sizing, allocation, hedge, conviction ideas). Baseline = deliverable (top-3 A2b
bullish ∩ mom≥10%, BTC fallback, EW). Test each idea head-to-head, full-sample for exploration; WFV the
winners separately. All vs BTC.

Ideas:
  H1 regime-conditional hedge — beta-neutralize ONLY when trailing-3mo BTC < −10% (from finding #19).
  S1 vol-target sizing — scale gross to target 15%/mo realized vol (de-risk in chop), cap 1.0, rest→cash.
  S2 conviction-scaled gross — scale gross by avg conviction of picks (lean in when council is confident).
  A1 conviction-tilt — within top-3, weight by conviction^1 instead of EW.
  C1 conviction-floor — only take picks with conviction≥70 (drop weak calls).
  COMBO — regime-hedge + conviction-tilt.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
import scripts.alpha_hunt3 as ah
from signals.backtest import realize_period, beta_neutralize, metrics
from signals.informativeness import forward_return

A2B = ah.load("data/signal/a2b_council/signal_panel.parquet")
dates = [d for d in ah.MD if d in ah.NXT]
btc = ah.btc


def trailing_btc(t, days=91):
    s = ah.prices["BTC"].dropna()
    a = s[s.index <= t - pd.Timedelta(days=days)]
    b = s[s.index <= t]
    return (b.iloc[-1] / a.iloc[-1] - 1) if len(a) and len(b) else 0.0


def realized_vol(t, days=60):
    s = ah.prices["BTC"].dropna()
    s = s[s.index <= t].tail(days).pct_change().dropna()
    return float(s.std() * np.sqrt(21)) if len(s) > 5 else 0.20  # ~monthly vol


def picks_conv(t, k=3, gate=0.10, floor=0):
    sc = {s: v for s, v in ah.sconv(A2B, t).items() if v > 0 and s in ah.tradeable(t) and v >= floor}
    sc = {s: v for s, v in sc.items() if (ah.mom(t, s) or -9) >= gate}
    return sorted(sc.items(), key=lambda kv: -kv[1])[:k]


def weights(t, *, tilt=False, floor=0):
    pk = picks_conv(t, floor=floor)
    if not pk:
        return {"BTC": 1.0}, 50.0
    oin = ah.oi_at(ah.oi, t)
    raw = {}
    for sec, conv in pk:
        b = ah.sector_basket(ah.sm, oin, sec, oi_floor=ah.OI_FLOOR_USD)
        wsec = conv if tilt else 1.0
        for s in b:
            raw[s] = raw.get(s, 0.0) + wsec / len(b)
    g = sum(abs(x) for x in raw.values())
    return ({s: x / g for s, x in raw.items()} if g else {"BTC": 1.0},
            float(np.mean([c for _, c in pk])))


def run(name, wfn):
    out = []
    for t in dates:
        w, _ = wfn(t)
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, ah.prices, ah.funding, t, ah.NXT[t], cost_bps=10)})
    r = pd.DataFrame(out)
    m = metrics(r.set_index("as_of")["ret"], periods_per_year=12)
    mm = r.merge(btc, on="as_of").dropna()
    X = np.column_stack([np.ones(len(mm)), mm["ret_y"].values]); coef, *_ = np.linalg.lstsq(X, mm["ret_x"].values, rcond=None)
    resid = mm["ret_x"].values - X @ coef; s2 = (resid @ resid) / (len(mm) - 2)
    ta = coef[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    print(f"  {name:<32} tot {m['total']:>+7.0%} | Sharpe {m['sharpe']:>4.2f} | DD {m['max_dd']:>4.0%} | "
          f"alpha {coef[0]*12:>+5.0%}/yr t={ta:.2f} | beta {coef[1]:.2f}")
    return m["sharpe"]


def main():
    bm = metrics(btc.set_index("as_of")["ret"], periods_per_year=12)
    print(f"BTC: Sharpe {bm['sharpe']:.2f} tot {bm['total']:+.0%}\n")
    print("=== baseline + new ideas (full-sample exploration) ===")
    run("baseline top-3+mom EW", lambda t: weights(t))

    def h1(t):
        w, c = weights(t)
        return (beta_neutralize(w, ah.BETAS) if trailing_btc(t) < -0.10 else w), c
    run("H1 regime-hedge (bear-only)", h1)

    def s1(t):
        w, c = weights(t)
        scale = min(1.0, 0.15 / max(realized_vol(t), 0.05))
        return ({k: v * scale for k, v in w.items()} if "BTC" not in w or len(w) > 1 else w), c
    run("S1 vol-target 15%", s1)

    def s2(t):
        w, c = weights(t)
        scale = min(1.2, c / 70.0)  # lean in when conviction high
        return {k: v * scale for k, v in w.items()}, c
    run("S2 conviction-scaled gross", s2)

    run("A1 conviction-tilt weights", lambda t: weights(t, tilt=True))
    run("C1 conviction-floor>=70", lambda t: weights(t, floor=70))

    def combo(t):
        w, c = weights(t, tilt=True)
        return (beta_neutralize(w, ah.BETAS) if trailing_btc(t) < -0.10 else w), c
    run("COMBO regime-hedge+conv-tilt", combo)


if __name__ == "__main__":
    main()
