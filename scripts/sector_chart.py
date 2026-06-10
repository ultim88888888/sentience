"""Monthly sector + BTC realized-returns HEATMAP, tagged with consensus (A2b council) bullish views.

Rows = sectors (+ BTC), cols = months. Cell color = realized forward 1mo return (red→green). A bold box
marks every (sector, month) the a16z consensus (A2b) was BULLISH on — so you can SEE whether the consensus
tags landed on green (winners) or red (losers). The visual answer to 'rankings vs our picks'."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at, forward_return, basket_forward_return
from signals.market import sector_basket, OI_FLOOR_USD
from signals.reconcile import apply_map
from signals.run import rebalance_dates

prices = load_close_panel("data/market_data", "ohlcv")
oi = load_close_panel("data/market_data", "oi")
sm = json.load(open("data/market_data/sector_map.json"))
ALL = sorted(set(sm.values()))
ID = json.load(open("data/signal/reconciled/reconciliation_map.json"))
TC = {d["id"]: d["new_type"] for d in json.load(open("data/signal/reconciled/type_corrections.json"))}


def build(panel_path, label, out_png, interval="monthly"):
    dates = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), interval)]
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    panel = apply_map(pd.read_parquet(panel_path), ID, TC)

    def bullish(t):
        dts = sorted(pd.Timestamp(x) for x in panel["as_of"].unique())
        asof = [d for d in dts if d <= t]
        if not asof:
            return set()
        live = panel[(pd.to_datetime(panel["as_of"]) == asof[-1]) & (panel["item_type"] == "sector")
                     & (panel["stance"] == "bullish")]
        return set(live["item"])

    cols = [d for d in dates if d in nxt]
    rows = ALL + ["BTC"]
    M = np.full((len(rows), len(cols)), np.nan)
    tag = np.zeros_like(M, dtype=bool)
    for j, t in enumerate(cols):
        oin = oi_at(oi, t)
        bull = bullish(t)
        for i, sec in enumerate(ALL):
            if not sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD):
                continue
            r = basket_forward_return(sec, t, nxt[t], sm, prices, oin)
            if r is not None:
                M[i, j] = r
                tag[i, j] = sec in bull
        M[len(rows) - 1, j] = forward_return(prices, "BTC", t, nxt[t])

    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.34), len(rows) * 0.34 + 1.5))
    vmax = np.nanpercentile(np.abs(M), 90)
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    for j in range(len(cols)):
        for i in range(len(rows)):
            if tag[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.2))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([t.strftime("%y-%m") for t in cols], rotation=90, fontsize=6)
    ax.set_title(f"Monthly forward sector returns (red=down, green=up) — "
                 f"BLACK BOX = {label} consensus BULLISH that month\n"
                 f"(BTC bottom row; box-on-green = consensus picked a winner)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("realized fwd 1mo return", fontsize=8)
    # tag hit-rate annotation
    tagged = M[tag]
    btc_row = M[len(rows) - 1]
    hit = np.nanmean(tagged > np.nanmean(btc_row)) if len(tagged) else float("nan")
    fig.text(0.01, 0.01, f"tagged cells: {(~np.isnan(tagged)).sum()} | "
             f"% of consensus-bullish that beat BTC that month: {hit:.0%}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}  ({(~np.isnan(tagged)).sum()} consensus tags, {hit:.0%} beat-BTC)")


if __name__ == "__main__":
    build("data/signal/a2b_council/signal_panel.parquet", "A2b council",
          "data/signal/sector_returns_heatmap_a2b.png")
