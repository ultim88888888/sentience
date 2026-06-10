"""Regime + period analysis (Jax: which periods does it win/lose; does hedge help in bear/sideways?).

Tag each month by BTC's TRAILING 3mo return (no lookahead): bull >+15%, bear <−10%, sideways between.
Report strategy vs BTC conditional on regime; best/worst months with picks; and test whether a
beta-neutral hedge HELPS specifically in bear/sideways months (Jax's intuition — hedge fails in the bull
but should help when the market isn't ripping)."""
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


def trailing_btc(t, days=91):
    s = ah.prices["BTC"].dropna()
    s0 = s[s.index <= t - pd.Timedelta(days=days)]
    s1 = s[s.index <= t]
    return (s1.iloc[-1] / s0.iloc[-1] - 1) if len(s0) and len(s1) else 0.0


def regime(t):
    r = trailing_btc(t)
    return "bull" if r > 0.15 else ("bear" if r < -0.10 else "sideways")


def strat_w(t, hedge=False):
    secs = ah.topk(A2B, t, 3, mom_thr=0.10)
    w = ah.ew(t, secs) if secs else {"BTC": 1.0}
    return beta_neutralize(w, ah.BETAS) if hedge else w


def main():
    rows = []
    for t in dates:
        w = strat_w(t)
        wh = strat_w(t, hedge=True)
        rows.append({
            "t": t, "regime": regime(t),
            "strat": realize_period(w, ah.prices, ah.funding, t, ah.NXT[t], cost_bps=10),
            "strat_hedged": realize_period(wh, ah.prices, ah.funding, t, ah.NXT[t], cost_bps=10),
            "btc": forward_return(ah.prices, "BTC", t, ah.NXT[t]) or 0.0,
            "picks": ",".join(ah.topk(A2B, t, 3, mom_thr=0.10)) or "BTC",
        })
    df = pd.DataFrame(rows)
    df["excess"] = df["strat"] - df["btc"]

    print("=== performance by REGIME (trailing-3mo BTC: bull>+15%, bear<−10%, else sideways) ===")
    print(f"{'regime':<10}{'n':>3}{'strat avg':>11}{'BTC avg':>10}{'excess':>9}{'hedged avg':>12}{'hedge Δ':>9}")
    for reg in ["bull", "sideways", "bear"]:
        g = df[df.regime == reg]
        if not len(g):
            continue
        hd = g["strat_hedged"].mean() - g["strat"].mean()
        print(f"{reg:<10}{len(g):>3}{g['strat'].mean():>+10.1%}{g['btc'].mean():>+9.1%}"
              f"{g['excess'].mean():>+8.1%}{g['strat_hedged'].mean():>+11.1%}{hd:>+8.1%}")

    print("\n=== BEST 5 months (strat) ===")
    for _, r in df.nlargest(5, "strat").iterrows():
        print(f"  {r['t'].date()} [{r['regime']:<8}] strat {r['strat']:+.0%} btc {r['btc']:+.0%}  picks: {r['picks']}")
    print("=== WORST 5 months (strat) ===")
    for _, r in df.nsmallest(5, "strat").iterrows():
        print(f"  {r['t'].date()} [{r['regime']:<8}] strat {r['strat']:+.0%} btc {r['btc']:+.0%}  picks: {r['picks']}")

    print("\n=== hedge verdict by regime (does beta-neutral HELP when not bull?) ===")
    for reg in ["bull", "sideways", "bear"]:
        g = df[df.regime == reg]
        if not len(g):
            continue
        m = metrics(g.set_index(g["t"].astype(str))["strat"], periods_per_year=12)
        mh = metrics(g.set_index(g["t"].astype(str))["strat_hedged"], periods_per_year=12)
        print(f"  {reg:<10} unhedged Sharpe {m['sharpe']:>5.2f} | hedged {mh['sharpe']:>5.2f} "
              f"-> hedge {'HELPS' if mh['sharpe']>m['sharpe'] else 'hurts'}")


if __name__ == "__main__":
    main()
