"""Relative-momentum gate (Jax: absolute momentum only fires in uptrends → sat in BTC through 2022).
Replace 'sector basket up >= X% absolute' with 'sector basket OUTPERFORMING BTC by >= X%' — regime-neutral
(in a bear, some sectors fall less than BTC). Test BOTH in-sample (2022-12..2026-03) AND the independent
2021-2022 OOS regime. The real question: does relative momentum extract alpha in the bear where absolute
couldn't, or does it just re-select the same froth signal that inverted (#29)?"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
import scripts.alpha_hunt3 as ah, scripts.new_ideas as ni
from signals.reconcile import apply_map
from signals.market import sector_basket, OI_FLOOR_USD
from signals.digest import _basket_trailing
from signals.backtest import realize_period, beta_neutralize, metrics
from signals.informativeness import oi_at, forward_return
from signals.run import rebalance_dates

IS_PANEL = ni.A2B
OOS_PANEL = apply_map(pd.read_parquet("data/signal/pre2022/a2b_council/signal_panel.parquet"), ah.ID, ah.TC)


def btc_trailing(t, days):
    s = ah.prices["BTC"].dropna(); s = s[s.index <= pd.Timestamp(t)]
    s0 = ah.prices["BTC"].dropna(); s0 = s0[s0.index <= pd.Timestamp(t) - pd.Timedelta(days=days)]
    return (s.iloc[-1] / s0.iloc[-1] - 1) if len(s) and len(s0) else None


def rel_mom(t, sec, days):
    b = sector_basket(ah.sm, oi_at(ah.oi, t), sec, oi_floor=OI_FLOOR_USD)
    if not b:
        return None
    bm = _basket_trailing(b, ah.prices, t, days); bt = btc_trailing(t, days)
    return (bm - bt) if (bm is not None and bt is not None) else None


def sconv(panel, t):
    dts = sorted(pd.Timestamp(x) for x in panel.as_of.unique()); asof = [d for d in dts if d <= t]
    if not asof:
        return {}
    live = panel[(pd.to_datetime(panel.as_of) == asof[-1]) & (panel.item_type == "sector") & (panel.stance == "bullish")]
    return {r["item"]: r["conviction"] for _, r in live.iterrows()}


def run(panel, dates, *, gate_kind, thr1, thr3, hedge=True):
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    rows = []
    for t in dates:
        if t not in nxt:
            continue
        oin = oi_at(ah.oi, t); cand = []
        for sec, conv in sconv(panel, t).items():
            b = sector_basket(ah.sm, oin, sec, oi_floor=OI_FLOOR_USD)
            if not b:
                continue
            if gate_kind == "abs":
                m1 = _basket_trailing(b, ah.prices, t, 30); m3 = _basket_trailing(b, ah.prices, t, 91)
            else:  # relative vs BTC
                m1 = rel_mom(t, sec, 30); m3 = rel_mom(t, sec, 91)
            if m1 is not None and m1 >= thr1 and m3 is not None and m3 >= thr3:
                cand.append((conv, sec, b))
        cand.sort(key=lambda x: -x[0]); cand = cand[:3]
        w = {"BTC": 1.0} if not cand else None
        if cand:
            raw = {}
            for conv, sec, b in cand:
                for s in b:
                    raw[s] = raw.get(s, 0.0) + 1.0 / len(b)
            g = sum(abs(x) for x in raw.values()); w = {s: x / g for s, x in raw.items()}
        if hedge and ni.trailing_btc(t) < -0.10:
            w = beta_neutralize(w, ah.BETAS)
        rows.append({"as_of": t.isoformat(), "ret": realize_period(w, ah.prices, ah.funding, t, nxt[t], cost_bps=10),
                     "btc": forward_return(ah.prices, "BTC", t, nxt[t]) or 0.0, "active": len(cand)})
    return pd.DataFrame(rows)


def rep(label, df):
    m = metrics(df.set_index("as_of")["ret"], periods_per_year=12)
    mb = metrics(df.set_index("as_of")["btc"], periods_per_year=12)
    tot = lambda x: (1 + x).prod() - 1
    fb = (df["active"] == 0).mean()
    print(f"  {label:<28} strat {tot(df.ret):>+6.0%} (Sh {m['sharpe']:>5.2f}) | BTC {tot(df.btc):>+6.0%} "
          f"| excess {tot(df.ret)-tot(df.btc):>+6.0%} | BTC-fallback {fb:.0%} | beat {(df.ret>df.btc).sum()}/{len(df)}")


def main():
    ISD = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")]
    OOSD = [pd.Timestamp(d) for d in rebalance_dates(date(2021, 9, 30), date(2022, 11, 30), "monthly")]
    print("=== IN-SAMPLE 2022-12..2026-03 ===")
    rep("absolute gate (deliverable)", run(IS_PANEL, ISD, gate_kind="abs", thr1=0.10, thr3=0.10))
    rep("relative gate (>BTC, thr=0)", run(IS_PANEL, ISD, gate_kind="rel", thr1=0.0, thr3=0.0))
    rep("relative gate (>BTC +5%)", run(IS_PANEL, ISD, gate_kind="rel", thr1=0.05, thr3=0.05))
    rep("relative gate (>BTC +10%)", run(IS_PANEL, ISD, gate_kind="rel", thr1=0.10, thr3=0.10))
    print("\n=== OUT-OF-SAMPLE 2021-09..2022-11 (the bear — where absolute just sat in BTC) ===")
    rep("absolute gate", run(OOS_PANEL, OOSD, gate_kind="abs", thr1=0.10, thr3=0.10))
    rep("relative gate (>BTC, thr=0)", run(OOS_PANEL, OOSD, gate_kind="rel", thr1=0.0, thr3=0.0))
    rep("relative gate (>BTC +5%)", run(OOS_PANEL, OOSD, gate_kind="rel", thr1=0.05, thr3=0.05))
    rep("relative gate (>BTC +10%)", run(OOS_PANEL, OOSD, gate_kind="rel", thr1=0.10, thr3=0.10))


if __name__ == "__main__":
    main()
