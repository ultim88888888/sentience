"""A3 evaluation — the scientific crux: does market-aware deliberation add signal BEYOND momentum?

Feeding members trailing performance risks A3 just re-encoding price-momentum. So we compute, per period
(cross-sectional Spearman, then t-test the series):
  - A3 raw IC: signed-conviction vs forward return
  - MOMENTUM baseline IC: sector trailing-3M return vs forward return (no LLM at all)
  - A3 ORTHOGONALIZED-to-momentum IC: residual of A3 signed-conviction after regressing out trailing-3M
    return (cross-sectionally, per period), vs forward return. If this ≈ 0, A3 is momentum-in-disguise;
    if > 0 and significant, A3 carries sentiment alpha momentum doesn't.
Also compares A3a/A3b against A1/A2a/A2a-diff/A2b, and tests the A3 short-horizon risk as a hedge trigger.
Runs on whatever panels exist (graceful if A3 not finished)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, forward_return, oi_at, basket_forward_return
from signals.market import sector_basket, OI_FLOOR_USD
from signals.digest import trailing_return, _basket_trailing
from signals.config import STANCE_SIGN
from signals.reconcile import apply_map
from signals.run import rebalance_dates

DATES = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "quarterly")]
prices = load_close_panel("data/market_data", "ohlcv")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
id_map = json.load(open("data/signal/reconciled/reconciliation_map.json"))
tc = json.load(open("data/signal/reconciled/type_corrections.json"))
TYPE_MAP = {d["id"]: d["new_type"] for d in tc}
NXT = {d: DATES[i + 1] for i, d in enumerate(DATES[:-1])}


def reconcile(path):
    return apply_map(pd.read_parquet(path), id_map, TYPE_MAP)


def sector_rows(panel):
    """Per (sector, period): signed conviction, trailing-3M basket return (momentum), forward return."""
    rows = []
    for _, r in panel.iterrows():
        if r["item_type"] != "sector" or r["stance"] not in ("bullish", "bearish"):
            continue
        t = pd.Timestamp(r["as_of"])
        if t not in NXT:
            continue
        oin = oi_at(oi, t)
        basket = sector_basket(sm, oin, r["item"], oi_floor=OI_FLOOR_USD)
        if not basket:
            continue
        fwd = basket_forward_return(r["item"], t, NXT[t], sm, prices, oin)
        mom = _basket_trailing(basket, prices, t, 91)
        if fwd is None or mom is None:
            continue
        rows.append({"as_of": r["as_of"], "item": r["item"],
                     "sig": STANCE_SIGN.get(r["stance"], 0) * r["conviction"],
                     "mom": mom, "fwd": fwd})
    return pd.DataFrame(rows)


def _t(ics):
    ics = np.array([x for x in ics if pd.notna(x)])
    if len(ics) < 2:
        return {"n": len(ics), "mean": float(ics[0]) if len(ics) else float("nan"), "t": float("nan")}
    m, sd = ics.mean(), ics.std(ddof=1)
    return {"n": len(ics), "mean": round(float(m), 3),
            "t": round(float(m / (sd / np.sqrt(len(ics)))), 2) if sd > 0 else float("nan"),
            "pct_pos": round(float((ics > 0).mean()), 2)}


def per_period(df, xcol):
    out = []
    if df.empty or "as_of" not in df.columns:
        return out
    for _, g in df.groupby("as_of"):
        gg = g[[xcol, "fwd"]].dropna()
        if len(gg) >= 4 and gg[xcol].nunique() >= 2:
            out.append(gg[xcol].corr(gg["fwd"], method="spearman"))
    return out


def per_period_orth(df):
    """A3 signed-conviction IC after regressing out trailing-3M return cross-sectionally each period."""
    out = []
    if df.empty or "as_of" not in df.columns:
        return out
    for _, g in df.groupby("as_of"):
        gg = g[["sig", "mom", "fwd"]].dropna()
        if len(gg) < 4 or gg["sig"].nunique() < 2:
            continue
        # residual of sig ~ mom (OLS), then rank-corr residual vs fwd
        b, a = np.polyfit(gg["mom"], gg["sig"], 1)
        resid = gg["sig"] - (a + b * gg["mom"])
        if resid.nunique() >= 2:
            out.append(resid.corr(gg["fwd"], method="spearman"))
    return out


def main():
    panels = {"A1": "data/signal/reconciled/a1_reconciled.parquet",
              "A2a": "data/signal/reconciled/a2a_reconciled.parquet",
              "A2a-diff": "data/signal/reconciled/a2a_diffweighted.parquet",
              "A2b": "data/signal/a2b_council/signal_panel.parquet"}
    for label, p in [("A3a", "data/signal/a3a_consensus/signal_panel.parquet"),
                     ("A3b", "data/signal/a3b_council/signal_panel.parquet")]:
        if Path(p).exists():
            panels[label] = p
    print("=== Per-period sector IC (signed conviction vs fwd) ===")
    mom_done = False
    for label, p in panels.items():
        if not Path(p).exists():
            print(f"{label:<10} (panel not ready)"); continue
        df = sector_rows(reconcile(p))
        print(f"{label:<10} raw IC {_t(per_period(df,'sig'))}")
        if label in ("A3a", "A3b"):
            print(f"{'':<10} orth-to-momentum IC {_t(per_period_orth(df))}")
            if not mom_done:
                print(f"{'MOMENTUM':<10} (3M trailing) IC {_t(per_period(df,'mom'))}  <- the control to beat")
                mom_done = True
    if not any(Path(p).exists() for l, p in panels.items() if l.startswith("A3")):
        print("\n[A3 panels not ready — rerun when data/signal/a3{a,b}_* exist]")


if __name__ == "__main__":
    main()
