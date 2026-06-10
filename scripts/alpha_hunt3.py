"""Alpha hunt R3 — is the beta-neutral alpha STATISTICALLY REAL, and can we sharpen it?

R2 lead: the corpus edge is a beta-neutral sector-selection alpha (~15-35%/yr), not a Sharpe-beating long
book. R3:
  1. Properly beta-neutralize A2b top-3 (BTC overlay → net β≈0) and put a t-STAT on the alpha (Jensen
     regression, monthly). Point alphas are meaningless at n=39 without significance.
  2. Momentum-timer with BTC-fallback (fixes R2's empty-book compounding drag).
  3. LS construction sweep (k, conviction-weighting) with alpha t-stats.
Verdict gate: alpha t>2 = real; else honest "directional only"."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at, forward_return
from signals.market import sector_basket, compute_beta, OI_FLOOR_USD
from signals.backtest import realize_period, benchmark_returns, metrics, beta_neutralize
from signals.digest import _basket_trailing
from signals.reconcile import apply_map
from signals.run import rebalance_dates

prices = load_close_panel("data/market_data", "ohlcv")
funding = load_close_panel("data/market_data", "funding")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
ALL = sorted(set(sm.values()))
ID = json.load(open("data/signal/reconciled/reconciliation_map.json"))
TC = {d["id"]: d["new_type"] for d in json.load(open("data/signal/reconciled/type_corrections.json"))}
MD = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")]
NXT = {d: MD[i + 1] for i, d in enumerate(MD[:-1])}
btc = benchmark_returns(prices, MD, mode="btc")
b90 = compute_beta(prices.pct_change(), window=90, market="BTC")
BETAS = {c: (float(b90[c].dropna().iloc[-1]) if len(b90[c].dropna()) else 1.0) for c in b90.columns}
BETAS["BTC"] = 1.0


def load(p): return apply_map(pd.read_parquet(p), ID, TC)
def tradeable(t): return [s for s in ALL if sector_basket(sm, oi_at(oi, t), s, oi_floor=OI_FLOOR_USD)]


def sconv(panel, t):
    dts = sorted(pd.Timestamp(x) for x in panel["as_of"].unique())
    asof = [d for d in dts if d <= t]
    if not asof:
        return {}
    live = panel[(pd.to_datetime(panel["as_of"]) == asof[-1]) & (panel["item_type"] == "sector")]
    out = {}
    for _, r in live.iterrows():
        s = {"bullish": 1, "bearish": -1}.get(r["stance"], 0)
        if s: out[r["item"]] = out.get(r["item"], 0.0) + s * r["conviction"]
    return out


def mom(t, sec):
    b = sector_basket(sm, oi_at(oi, t), sec, oi_floor=OI_FLOOR_USD)
    return _basket_trailing(b, prices, t, 30) if b else None


def topk(panel, t, k, mom_thr=None):
    sc = {s: v for s, v in sconv(panel, t).items() if v > 0 and s in tradeable(t)}
    if mom_thr is not None:
        sc = {s: v for s, v in sc.items() if (mom(t, s) or -9) >= mom_thr}
    return [s for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]


def ew(t, secs):
    oin = oi_at(oi, t); raw = {}
    for sec in secs:
        b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
        for s in b: raw[s] = raw.get(s, 0.0) + 1.0 / len(b)
    g = sum(abs(x) for x in raw.values()); return {s: x / g for s, x in raw.items()} if g else {}


def run(wfn):
    out = []
    for t in MD:
        if t not in NXT: continue
        out.append({"as_of": t.isoformat(), "ret": realize_period(wfn(t), prices, funding, t, NXT[t], cost_bps=10)})
    return pd.DataFrame(out)


def jensen_t(r):
    """Jensen alpha + t-stat from monthly regression ret ~ a + b*btc. alpha annualized."""
    m = r.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 5: return float("nan"), float("nan"), float("nan")
    x, y = m["ret_b"].values, m["ret_s"].values
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = (resid @ resid) / (len(x) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    a, b = coef
    ta = a / np.sqrt(cov[0, 0])
    return float(a) * 12, float(ta), float(b)


def rep(name, r):
    m = metrics(r.set_index("as_of")["ret"], periods_per_year=12)
    aA, ta, beta = jensen_t(r)
    flag = "  <<< alpha t>2" if (ta == ta and abs(ta) > 2) else ""
    print(f"{name:<38} Sh={m['sharpe']:>5.2f} tot={m['total']:>7.0%} DD={m['max_dd']:>5.0%} "
          f"alphaA={aA:>6.1%} t={ta:>5.2f} beta={beta:>5.2f}{flag}")
    return ta


def main():
    A2B = load("data/signal/a2b_council/signal_panel.parquet")
    bm = metrics(btc.set_index("as_of")["ret"], periods_per_year=12)
    print(f"BTC monthly Sharpe={bm['sharpe']:.2f}\n")

    print("=== A2b top-3: raw vs beta-neutralized (BTC overlay), with alpha t-stat ===")
    rep("A2b top-3 long-only", run(lambda t: ew(t, topk(A2B, t, 3))))
    rep("A2b top-3 beta-neutral (β→0)", run(lambda t: beta_neutralize(ew(t, topk(A2B, t, 3)), BETAS)))

    print("\n=== momentum-timer + BTC-fallback (hold BTC when no sector clears mom) ===")
    for thr in [0.0, 0.05, 0.10]:
        def wf(t, thr=thr):
            secs = topk(A2B, t, 3, mom_thr=thr)
            return ew(t, secs) if secs else {"BTC": 1.0}
        rep(f"A2b top-3 mom>={thr:.0%} +BTC-fallback", run(wf))
        def wfn(t, thr=thr):  # beta-neutral version
            secs = topk(A2B, t, 3, mom_thr=thr)
            return beta_neutralize(ew(t, secs), BETAS) if secs else {}
        rep(f"  ^ beta-neutral (cash when empty)", run(wfn))

    print("\n=== LS sweep: long top-k / short bottom-k by conviction, alpha t-stat ===")
    for k in [2, 3, 5]:
        def wf(t, k=k):
            sc = {s: v for s, v in sconv(A2B, t).items() if s in tradeable(t)}
            longs = [s for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]
            shorts = [s for s, _ in sorted(sc.items(), key=lambda kv: kv[1])[:k]]
            oin = oi_at(oi, t); raw = {}
            for sec in longs:
                b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
                for s in b: raw[s] = raw.get(s, 0.0) + 0.5 / len(b)
            for sec in shorts:
                b = sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD)
                for s in b: raw[s] = raw.get(s, 0.0) - 0.5 / len(b)
            g = sum(abs(x) for x in raw.values()); return {s: x / g for s, x in raw.items()} if g else {}
        rep(f"LS top{k}/bottom{k}", run(wf))


if __name__ == "__main__":
    main()
