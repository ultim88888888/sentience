"""Full strategy sweep on A2b — the apex signal (per-period IC t=2.42). The earlier 'winning
construction' (long-only sector) was found on the WEAKER signals; A2b's edge is cross-sectional IC,
whose natural home is MARKET-NEUTRAL L/S (strips the beta that swamps long-only). Sweep the grid and
report the landscape sorted by Sharpe and by beta-adjusted alpha."""
from __future__ import annotations
import json
from itertools import product
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel
from signals.market import compute_beta
from signals.backtest import walk_forward, benchmark_returns, metrics
from signals.reconcile import apply_map
from signals.run import rebalance_dates

DATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "quarterly")]
prices = load_close_panel("data/market_data", "ohlcv")
funding = load_close_panel("data/market_data", "funding")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
b = compute_beta(prices.pct_change(), window=90, market="BTC")
betas = {c: (float(b[c].dropna().iloc[-1]) if len(b[c].dropna()) else 1.0) for c in b.columns}
betas["BTC"] = 1.0
id_map = json.load(open("data/signal/reconciled/reconciliation_map.json"))
tc = json.load(open("data/signal/reconciled/type_corrections.json"))
a2b = apply_map(pd.read_parquet("data/signal/a2b_council/signal_panel.parquet"),
                id_map, {d["id"]: d["new_type"] for d in tc})
hedges = json.load(open("data/signal/a2b_council/hedge_decisions.json"))
btc = benchmark_returns(prices, DATES, mode="btc")


def jensen(strat):
    m = strat.merge(btc, on="as_of", suffixes=("_s", "_b")).dropna()
    if len(m) < 3:
        return float("nan"), float("nan")
    beta, alpha = np.polyfit(m["ret_b"].values, m["ret_s"].values, 1)
    return float(alpha) * 4, float(beta)  # annualized alpha, realized beta


# Which hedge modes make sense per strategy:
#  - L/S strategies (sector_ls, intra_sector, token_ls, token_vs_sector, both): none / neutral(=L/S already)
#  - long_only: none / neutral(beta-hedge to 0) / council(conditional)
STRATS = ["sector_ls", "intra_sector", "token_ls", "token_vs_sector", "both", "long_only"]
rows = []
for strat, cw, cap in product(STRATS, [True, False], [3, 5, 8, None]):
    hedge_modes = ["none"]
    if strat == "sector_ls":
        hedge_modes = ["none", "neutral"]   # neutral = beta-overlay the L/S book to net-0
    if strat == "long_only":
        hedge_modes = ["none", "neutral", "council"]
    for hm in hedge_modes:
        kw = dict(strategy=strat, bet_sign="momentum", cap=cap, conviction_weighted=cw,
                  beta_neutral=(hm == "neutral"), betas=betas, hedge_mode=hm)
        if hm == "council":
            kw["risk_by_date"] = hedges
        r = walk_forward(a2b, prices, funding, oi, sm, DATES, **kw)
        if r.empty:
            continue
        m = metrics(r.set_index("as_of")["ret"])
        a, bt = jensen(r)
        rows.append({"strat": strat, "hedge": hm, "cw": cw, "cap": str(cap),
                     "sharpe": m["sharpe"], "total": m["total"], "maxDD": m["max_dd"],
                     "alpha_ann": a, "beta": bt, "n": m["n"]})

df = pd.DataFrame(rows)
btcm = metrics(btc.set_index("as_of")["ret"])
print(f"BTC bench: sharpe={btcm['sharpe']:.2f} total={btcm['total']:.0%}  (beat this)\n")


def show(d, by, title):
    print(f"=== {title} ===")
    print(f"{'strat':<16}{'hedge':<9}{'cw':<6}{'cap':<5}{'sharpe':>7}{'total':>8}{'maxDD':>7}{'alphaA':>8}{'beta':>6}")
    for _, r in d.sort_values(by, ascending=False).head(12).iterrows():
        print(f"{r['strat']:<16}{r['hedge']:<9}{str(r['cw']):<6}{r['cap']:<5}"
              f"{r['sharpe']:>7.2f}{r['total']:>8.0%}{r['maxDD']:>7.0%}{r['alpha_ann']:>8.1%}{r['beta']:>6.2f}")
    print()


show(df, "sharpe", "TOP 12 by Sharpe")
show(df, "alpha_ann", "TOP 12 by beta-adjusted alpha (annualized)")
# Spotlight: market-neutral sector L/S — the natural home for a cross-sectional IC signal
mn = df[(df["strat"] == "sector_ls") & (df["hedge"] == "neutral")]
print("=== Market-neutral sector L/S (the cross-sectional-IC home) ===")
print(mn[["cw", "cap", "sharpe", "total", "maxDD", "alpha_ann", "beta"]].to_string(index=False))
