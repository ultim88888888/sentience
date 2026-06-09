"""A2b new-knob exploration (Jax: create/tune knobs, try to find something).
Beyond the cap sweep already run, test LONG-side knobs that target the cross-sectional ranking:
  - rank-top-K: long only the top-K council sectors by signed conviction (vs trading all bullish)
  - conviction floor: drop sectors below a conviction threshold
  - equal-weight vs conviction-weight at sector level
Honest bar: beat BTC buy-hold Sharpe (0.93). Report the landscape; do not cherry-pick."""
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


def rank_topk_long(top_k=None, conv_floor=0, weight="cw"):
    """Per period: rank bullish sectors with a tradeable basket by conviction; take top-K above floor;
    long equal-weight (ew) or conviction-weight (cw) across their baskets. Returns per-period return series."""
    nxt = {d: DATES[i + 1] for i, d in enumerate(DATES[:-1])}
    out = []
    for t in DATES:
        if t not in nxt:
            continue
        live = a2b[(pd.to_datetime(a2b["as_of"]) == t) & (a2b["lifecycle_state"] != "EXITED")]
        oin = oi_at(oi, t)
        cand = []
        for _, r in live.iterrows():
            if r["item_type"] != "sector" or r["stance"] != "bullish" or r["conviction"] < conv_floor:
                continue
            basket = sector_basket(sm, oin, r["item"], oi_floor=OI_FLOOR_USD)
            if basket:
                cand.append((r["conviction"], r["item"], basket))
        cand.sort(key=lambda x: -x[0])
        if top_k:
            cand = cand[:top_k]
        if not cand:
            out.append({"as_of": t.isoformat(), "ret": 0.0})
            continue
        raw = {}
        for conv, sec, basket in cand:
            w_sec = (conv / 100.0) if weight == "cw" else 1.0
            for s in basket:
                raw[s] = raw.get(s, 0.0) + w_sec / len(basket)
        g = sum(abs(v) for v in raw.values())
        w = {s: v / g for s, v in raw.items()} if g else {}
        out.append({"as_of": t.isoformat(), "ret": realize_period(w, prices, funding, t, nxt[t], cost_bps=10.0)})
    return pd.DataFrame(out)


def alpha_beta(r):
    m = r.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 3:
        return float("nan"), float("nan")
    beta, a = np.polyfit(m["ret_b"].values, m["ret_s"].values, 1)
    return float(a) * 4, float(beta)


print(f"BTC bench: Sharpe={btcm['sharpe']:.2f} total={btcm['total']:.0%}  <- beat this\n")
print(f"{'K':>4} {'floor':>5} {'wt':>3} {'sharpe':>7} {'total':>8} {'maxDD':>7} {'alphaA':>8} {'beta':>6}")
best = []
for K in [2, 3, 4, 5, 6, None]:
    for floor in [0, 50, 60, 70]:
        for wt in ["cw", "ew"]:
            r = rank_topk_long(top_k=K, conv_floor=floor, weight=wt)
            m = metrics(r.set_index("as_of")["ret"])
            a, bt = alpha_beta(r)
            best.append((m["sharpe"], K, floor, wt, m, a, bt))
            print(f"{str(K):>4} {floor:>5} {wt:>3} {m['sharpe']:>7.2f} {m['total']:>8.0%} {m['max_dd']:>7.0%} {a:>8.1%} {bt:>6.2f}")
best.sort(key=lambda x: -x[0])
print(f"\nBEST by Sharpe: K={best[0][1]} floor={best[0][2]} wt={best[0][3]} -> Sharpe {best[0][0]:.2f} "
      f"(BTC {btcm['sharpe']:.2f}); beats BTC: {best[0][0] > btcm['sharpe']}")
