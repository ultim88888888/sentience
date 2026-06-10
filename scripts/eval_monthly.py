"""Monthly headline eval — does market-aware deliberation pay at MONTHLY cadence (Jax's core thesis)?
Per-period IC (all-item + sector) with momentum control, on the monthly panels, vs their quarterly counterparts.
Flexible: pass interval + panels."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at, forward_return, basket_forward_return
from signals.market import sector_basket, OI_FLOOR_USD
from signals.digest import _basket_trailing
from signals.config import STANCE_SIGN
from signals.reconcile import apply_map
from signals.run import rebalance_dates

prices = load_close_panel("data/market_data", "ohlcv")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
ID = json.load(open("data/signal/reconciled/reconciliation_map.json"))
TC = {d["id"]: d["new_type"] for d in json.load(open("data/signal/reconciled/type_corrections.json"))}


def _t(ics):
    ics = np.array([x for x in ics if pd.notna(x)])
    if len(ics) < 2:
        return {"n": len(ics), "mean": float("nan"), "t": float("nan"), "pos": float("nan")}
    m, sd = ics.mean(), ics.std(ddof=1)
    return {"n": len(ics), "mean": round(float(m), 3),
            "t": round(float(m / (sd / np.sqrt(len(ics)))), 2) if sd > 0 else float("nan"),
            "pos": round(float((ics > 0).mean()), 2)}


def all_item_rows(panel, nxt):
    rows = []
    for _, r in panel.iterrows():
        t = pd.Timestamp(r["as_of"])
        if t not in nxt or r.get("lifecycle_state", "NEW") == "EXITED":
            continue
        oin = oi_at(oi, t)
        if r["item_type"] == "token":
            fr = forward_return(prices, r["item"], t, nxt[t])
        else:
            fr = basket_forward_return(r["item"], t, nxt[t], sm, prices, oin)
        if fr is None:
            continue
        rows.append({"as_of": r["as_of"], "sig": STANCE_SIGN.get(r["stance"], 0) * r["conviction"], "fwd": fr})
    return pd.DataFrame(rows)


def sector_rows(panel, nxt):
    rows = []
    for _, r in panel.iterrows():
        if r["item_type"] != "sector" or r["stance"] not in ("bullish", "bearish"):
            continue
        t = pd.Timestamp(r["as_of"])
        if t not in nxt:
            continue
        oin = oi_at(oi, t)
        b = sector_basket(sm, oin, r["item"], oi_floor=OI_FLOOR_USD)
        if not b:
            continue
        fr = basket_forward_return(r["item"], t, nxt[t], sm, prices, oin)
        mom = _basket_trailing(b, prices, t, 30)
        if fr is None or mom is None:
            continue
        rows.append({"as_of": r["as_of"], "sig": STANCE_SIGN.get(r["stance"], 0) * r["conviction"],
                     "mom": mom, "fwd": fr})
    return pd.DataFrame(rows)


def per_period(df, x="sig"):
    out = []
    if df.empty:
        return out
    for _, g in df.groupby("as_of"):
        gg = g[[x, "fwd"]].dropna()
        if len(gg) >= 4 and gg[x].nunique() >= 2:
            out.append(gg[x].corr(gg["fwd"], method="spearman"))
    return out


def per_period_orth(df):
    out = []
    if df.empty:
        return out
    for _, g in df.groupby("as_of"):
        gg = g[["sig", "mom", "fwd"]].dropna()
        if len(gg) < 4 or gg["sig"].nunique() < 2:
            continue
        b, a = np.polyfit(gg["mom"], gg["sig"], 1)
        resid = gg["sig"] - (a + b * gg["mom"])
        if resid.nunique() >= 2:
            out.append(resid.corr(gg["fwd"], method="spearman"))
    return out


def load(p):
    return apply_map(pd.read_parquet(p), ID, TC)


def eval_panels(panels, interval):
    dates = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), interval)]
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    print(f"\n========== {interval.upper()} ==========")
    print(f"{'panel':<26}{'all-item IC':>22}{'sector IC':>20}{'sector ⊥mom':>20}")
    mom_printed = False
    for label, p in panels.items():
        if not os.path.exists(p):
            print(f"{label:<26} (no panel)"); continue
        panel = load(p)
        ai = _t(per_period(all_item_rows(panel, nxt)))
        sr = sector_rows(panel, nxt)
        sec = _t(per_period(sr)); orth = _t(per_period_orth(sr))
        print(f"{label:<26} t={ai['t']:>5} m={ai['mean']:+.3f} p{ai['pos']}   "
              f"t={sec['t']:>5} m={sec['mean']:+.3f}   t={orth['t']:>5} m={orth['mean']:+.3f}")
        if not mom_printed and not sr.empty:
            mdf = sr[["as_of", "mom", "fwd"]].rename(columns={"mom": "sig"})
            mom = _t(per_period(mdf))
            print(f"{'  MOMENTUM(1mo) baseline':<26} {'':<22}t={mom['t']:>5} m={mom['mean']:+.3f}")
            mom_printed = True


if __name__ == "__main__":
    eval_panels({
        "A1 monthly": "data/signal/a1_monthly/signal_panel.parquet",
        "A4a-m SOUL consensus": "data/signal/a4a_consensus_m/signal_panel.parquet",
        "A4b-m SOUL council": "data/signal/a4b_council_m/signal_panel.parquet",
    }, "monthly")
    eval_panels({
        "A2b-q static council": "data/signal/a2b_council/signal_panel.parquet",
        "A3b-q view council": "data/signal/a3b_council/signal_panel.parquet",
        "A4b-q SOUL council": "data/signal/a4b_council/signal_panel.parquet",
    }, "quarterly")
