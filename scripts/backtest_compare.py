"""Backtest comparison driver — A1 / A2a / A2a-diff / A2b in the winning construction
(long-only sector, momentum, cap=5, conviction-weighted, unhedged), plus the A2b council
hedge-decision test. Replaces the earlier ad-hoc in-session runs with one committed,
reproducible script. Reports Sharpe, total return, max-DD, and Jensen beta-adjusted alpha
(regress per-period strategy return on BTC; intercept = alpha/qtr, slope = realized beta)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel
from signals.market import compute_beta
from signals.backtest import walk_forward, benchmark_returns, metrics
from signals.reconcile import apply_map
from signals.run import rebalance_dates

MKT = "data/market_data"
REC = "data/signal/reconciled"
DATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "quarterly")]

PANELS = {
    "A1": f"{REC}/a1_reconciled.parquet",
    "A2a": f"{REC}/a2a_reconciled.parquet",
    "A2a-diff": f"{REC}/a2a_diffweighted.parquet",
    # A2b is added below only if its (reconciled) panel exists.
}
A2B_RAW = "data/signal/a2b_council/signal_panel.parquet"
A2B_HEDGES = "data/signal/a2b_council/hedge_decisions.json"

# Winning construction (from the characterization).
CFG = dict(strategy="long_only", bet_sign="momentum", cap=5,
           conviction_weighted=True, beta_neutral=False)


def static_betas(prices: pd.DataFrame) -> dict:
    """One ex-ante beta per symbol vs BTC = most-recent rolling-90d beta (fallback 1.0)."""
    rets = prices.pct_change()
    b = compute_beta(rets, window=90, market="BTC")
    out = {}
    for c in b.columns:
        s = b[c].dropna()
        out[c] = float(s.iloc[-1]) if len(s) else 1.0
    out["BTC"] = 1.0
    return out


def jensen(strat: pd.DataFrame, bench: pd.DataFrame) -> dict:
    """Beta-adjusted alpha: regress per-period strat ret on BTC ret. alpha=intercept (×4 = ann)."""
    m = strat.merge(bench, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 3:
        return {"alpha_q": float("nan"), "beta": float("nan"), "n": len(m)}
    x, y = m["ret_b"].values, m["ret_s"].values
    beta, alpha = np.polyfit(x, y, 1)
    return {"alpha_q": round(float(alpha), 4), "alpha_ann": round(float(alpha) * 4, 4),
            "beta": round(float(beta), 3), "n": len(m)}


def load_a2b():
    """Reconcile the A2b panel through the shared map so its vocab aligns with A1/A2a."""
    if not Path(A2B_RAW).exists():
        return None, None
    id_map = json.load(open(f"{REC}/reconciliation_map.json"))
    tc = json.load(open(f"{REC}/type_corrections.json"))
    type_map = {d["id"]: d["new_type"] for d in tc}
    df = apply_map(pd.read_parquet(A2B_RAW), id_map, type_map)
    hedges = json.load(open(A2B_HEDGES)) if Path(A2B_HEDGES).exists() else {}
    return df, hedges


def main():
    prices = load_close_panel(MKT, "ohlcv")
    funding = load_close_panel(MKT, "funding")
    oi = load_close_panel(MKT, "oi")
    sector_map = json.load(open(f"{MKT}/sector_map.json"))
    betas = static_betas(prices)
    btc = benchmark_returns(prices, DATES, mode="btc")

    panels = dict(PANELS)
    a2b_df, a2b_hedges = load_a2b()

    print("=== Winning construction: long-only sector, momentum, cap=5, conv-wtd, UNHEDGED ===")
    print(f"{'panel':<10} {'sharpe':>7} {'total':>8} {'maxDD':>8} {'alpha_q':>8} {'beta':>6} {'n':>3}")
    btc_m = metrics(btc.set_index('as_of')['ret'])
    print(f"{'BTC(bench)':<10} {btc_m['sharpe']:>7.2f} {btc_m['total']:>8.2%} {btc_m['max_dd']:>8.2%} {'—':>8} {'1.00':>6} {btc_m['n']:>3}")
    for label, path in panels.items():
        df = pd.read_parquet(path)
        res = walk_forward(df, prices, funding, oi, sector_map, DATES, betas=betas, hedge_mode="none", **CFG)
        if res.empty:
            print(f"{label:<10}  (no periods)"); continue
        mt = metrics(res.set_index("as_of")["ret"])
        j = jensen(res, btc)
        print(f"{label:<10} {mt['sharpe']:>7.2f} {mt['total']:>8.2%} {mt['max_dd']:>8.2%} {j['alpha_q']:>8} {j['beta']:>6} {mt['n']:>3}")

    if a2b_df is not None:
        res = walk_forward(a2b_df, prices, funding, oi, sector_map, DATES, betas=betas, hedge_mode="none", **CFG)
        mt = metrics(res.set_index("as_of")["ret"]); j = jensen(res, btc)
        print(f"{'A2b':<10} {mt['sharpe']:>7.2f} {mt['total']:>8.2%} {mt['max_dd']:>8.2%} {j['alpha_q']:>8} {j['beta']:>6} {mt['n']:>3}")

        print("\n=== A2b council HEDGE-DECISION test (Jax's gut: council risk call as trigger) ===")
        nh = walk_forward(a2b_df, prices, funding, oi, sector_map, DATES, betas=betas, hedge_mode="none", **CFG)
        hh = walk_forward(a2b_df, prices, funding, oi, sector_map, DATES, betas=betas,
                          hedge_mode="council", risk_by_date=a2b_hedges, **CFG)
        n_hedge = sum(1 for v in a2b_hedges.values() if v == "hedge")
        print(f"council said 'hedge' in {n_hedge}/{len(a2b_hedges)} periods")
        for lbl, r in [("A2b no-hedge", nh), ("A2b council-hedge", hh)]:
            mt = metrics(r.set_index("as_of")["ret"]); j = jensen(r, btc)
            print(f"  {lbl:<20} sharpe={mt['sharpe']:.2f} total={mt['total']:.2%} maxDD={mt['max_dd']:.2%} alpha_q={j['alpha_q']} beta={j['beta']}")
    else:
        print("\n[A2b panel not ready yet — rerun when data/signal/a2b_council/signal_panel.parquet exists]")


if __name__ == "__main__":
    main()
