"""Per-period IC significance — the honest stat the memo lacked.

Pooled Spearman over all (item,period) rows overstates N: rows within a period
share a common return shock, so they are not independent. The standard quant
approach is the IC time series: compute the cross-sectional Spearman IC *within
each period*, then t-test that series of per-period ICs (n = #periods). That
t-stat is the defensible significance number.

We report it for each panel (A1 / A2a / A2a-diff) and, for A2a-diff, split the
universe by coverage to test the core finding (thin-coverage minority calls carry
the alpha, broad-consensus calls do not)."""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from signals.config import STANCE_SIGN
from signals.market import sector_basket, OI_FLOOR_USD
from signals.informativeness import load_close_panel, forward_return, basket_forward_return, oi_at
from signals.run import rebalance_dates
from datetime import date

PANELS = {
    "A1": "data/signal/reconciled/a1_reconciled.parquet",
    "A2a": "data/signal/reconciled/a2a_reconciled.parquet",
    "A2a-diff": "data/signal/reconciled/a2a_diffweighted.parquet",
}
SECTOR_MAP = json.load(open("data/market_data/sector_map.json"))
DATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "quarterly")]
PRICES = load_close_panel("data/market_data", "ohlcv")
OI = load_close_panel("data/market_data", "oi")


def eval_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, as_of): signed conviction, coverage, forward return."""
    nxt = {d: DATES[i + 1] for i, d in enumerate(DATES[:-1])}
    rows = []
    for _, r in panel.iterrows():
        t = pd.Timestamp(r["as_of"])
        if t not in nxt or r.get("lifecycle_state") == "EXITED":
            continue
        t_next = nxt[t]
        if r["item_type"] == "token":
            fr = forward_return(PRICES, r["item"], t, t_next)
        else:
            fr = basket_forward_return(r["item"], t, t_next, SECTOR_MAP, PRICES,
                                       oi_at(OI, t), oi_floor=OI_FLOOR_USD)
        if fr is None:
            continue
        sign = STANCE_SIGN.get(r["stance"], 0)
        rows.append({"as_of": r["as_of"], "item_type": r["item_type"],
                     "conviction_signed": sign * r["conviction"],
                     "coverage_frac": r.get("coverage_frac", np.nan), "fwd_return": fr})
    return pd.DataFrame(rows)


def period_ic_series(table: pd.DataFrame, signal_col="conviction_signed", min_n=4):
    """Cross-sectional Spearman IC within each period; return the series (one IC per period)."""
    ics = []
    for t, g in table.groupby("as_of"):
        g = g[[signal_col, "fwd_return"]].dropna()
        if len(g) < min_n or g[signal_col].nunique() < 2:
            continue
        ic = g[signal_col].corr(g["fwd_return"], method="spearman")
        if pd.notna(ic):
            ics.append(ic)
    return np.array(ics)


def stats(ics: np.ndarray) -> dict:
    n = len(ics)
    if n < 2:
        return {"n_periods": n, "mean_ic": float(ics[0]) if n else float("nan"),
                "t": float("nan"), "p_two": float("nan"), "pct_pos": float("nan")}
    m, sd = ics.mean(), ics.std(ddof=1)
    se = sd / np.sqrt(n)
    t = m / se if se > 0 else float("nan")
    # two-sided p via normal approx (small n => indicative, not exact)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) if np.isfinite(t) else float("nan")
    return {"n_periods": n, "mean_ic": round(float(m), 4), "sd_ic": round(float(sd), 4),
            "t": round(float(t), 2), "p_two": round(float(p), 3),
            "pct_pos": round(float((ics > 0).mean()), 2)}


def main():
    print("=== Per-period IC significance (IC time-series t-test, n=#periods) ===\n")
    for label, path in PANELS.items():
        tbl = eval_rows(pd.read_parquet(path))
        ics = period_ic_series(tbl)
        print(f"[{label}] signed-conviction IC  rows={len(tbl)}  {stats(ics)}")
    print()
    # Core finding: A2a-diff split by coverage
    tbl = eval_rows(pd.read_parquet(PANELS["A2a-diff"]))
    tbl = tbl.dropna(subset=["coverage_frac"])
    thin = tbl[tbl["coverage_frac"] <= 0.2]   # <=2 of 10 members
    broad = tbl[tbl["coverage_frac"] >= 0.4]   # >=4 of 10
    print("=== A2a-diff core finding: thin (minority) vs broad (consensus) ===")
    print(f"[thin  cov<=0.2] rows={len(thin)}  {stats(period_ic_series(thin))}")
    print(f"[broad cov>=0.4] rows={len(broad)}  {stats(period_ic_series(broad))}")
    # pooled (overstated-N) for contrast
    print("\n=== pooled Spearman (overstates N — for contrast only) ===")
    for label, path in PANELS.items():
        t = eval_rows(pd.read_parquet(path))
        tt = t[["conviction_signed", "fwd_return"]].dropna()
        ic = tt["conviction_signed"].corr(tt["fwd_return"], method="spearman")
        print(f"[{label}] pooled IC={ic:.4f}  n_rows={len(tt)}")


if __name__ == "__main__":
    main()
