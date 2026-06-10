"""Monthly sector + BTC performance-ranking report, and signal-pick overlay (Jax's ask).

For each monthly rebalance date T:
  - compute the REALIZED forward 1-month return of every liquid sector basket and of BTC
  - rank sectors by that forward return (1 = best performer this month)
Then overlay a signal's top-K bullish sector picks at T and report WHERE those picks landed in the realized
ranking — i.e. did our signal pick the months' actual winners? Produces:
  (a) a per-month table (markdown) of sector rankings with BTC's position marked,
  (b) a pick-skill scorecard: avg realized rank of our picks vs random (uniform), hit-rate in top tercile.
Deterministic — pure market data + the signal panel. No lookahead (forward return is the eval target)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
from signals.informativeness import load_close_panel, oi_at, basket_forward_return
from signals.market import sector_basket, OI_FLOOR_USD
from signals.reconcile import apply_map
from signals.run import rebalance_dates

MKT = "data/market_data"
prices = load_close_panel(MKT, "ohlcv")
oi = load_close_panel(MKT, "oi")
sm = json.load(open(f"{MKT}/sector_map.json"))
ALL_SECTORS = sorted(set(sm.values()))


def forward_btc(t, t_next):
    from signals.informativeness import forward_return
    return forward_return(prices, "BTC", t, t_next)


def month_ranking(t, t_next):
    """Realized forward returns of every tradeable sector + BTC at T; ranked best→worst.
    Returns (ranked list of (sector, ret), btc_ret, btc_rank_among_sectors)."""
    oin = oi_at(oi, t)
    rows = []
    for sec in ALL_SECTORS:
        if not sector_basket(sm, oin, sec, oi_floor=OI_FLOOR_USD):
            continue
        r = basket_forward_return(sec, t, t_next, sm, prices, oin)
        if r is not None:
            rows.append((sec, r))
    rows.sort(key=lambda x: -x[1])
    btc = forward_btc(t, t_next)
    btc_rank = 1 + sum(1 for _, r in rows if (btc is not None and r > btc))
    return rows, btc, btc_rank


def signal_picks(panel, t, k=3):
    """Top-k bullish sectors by conviction at T from a signal panel (the ones we'd go long)."""
    live = panel[(pd.to_datetime(panel["as_of"]) == pd.Timestamp(t)) &
                 (panel.get("lifecycle_state", "NEW") != "EXITED")]
    sec = live[(live["item_type"] == "sector") & (live["stance"] == "bullish")]
    sec = sec.sort_values("conviction", ascending=False)
    return list(sec["item"].head(k))


def run(interval="monthly", panel_path=None, panel_label="", k=3, out_md=None):
    dates = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), interval)]
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    panel = None
    if panel_path and Path(panel_path).exists():
        id_map = json.load(open("data/signal/reconciled/reconciliation_map.json"))
        tc = json.load(open("data/signal/reconciled/type_corrections.json"))
        panel = apply_map(pd.read_parquet(panel_path), id_map, {d["id"]: d["new_type"] for d in tc})

    lines = [f"# Monthly sector + BTC ranking report  (signal overlay: {panel_label or 'none'})", ""]
    pick_ranks, n_tradeable, hits_top3 = [], [], 0
    for t in dates:
        if t not in nxt:
            continue
        ranked, btc, btc_rank = month_ranking(t, nxt[t])
        if not ranked:
            continue
        rank_of = {sec: i + 1 for i, (sec, _) in enumerate(ranked)}
        N = len(ranked)
        picks = signal_picks(panel, t, k) if panel is not None else []
        picks_in = [p for p in picks if p in rank_of]
        prs = [rank_of[p] for p in picks_in]
        pick_ranks += [(r, N) for r in prs]
        n_tradeable.append(N)
        hits_top3 += sum(1 for r in prs if r <= max(1, N // 3))
        top = ", ".join(f"{s}({r:+.0%})" for s, r in ranked[:3])
        bot = ", ".join(f"{s}({r:+.0%})" for s, r in ranked[-2:])
        pick_str = ", ".join(f"{p}=#{rank_of.get(p,'NA')}" for p in picks) if panel is not None else ""
        lines.append(f"**{t.date()}** (N={N}, BTC {btc:+.0%} #{btc_rank}/{N}) | "
                     f"top: {top} | bottom: {bot}" + (f" | **picks: {pick_str}**" if pick_str else ""))

    if panel is not None and pick_ranks:
        # skill: normalized rank (0=best,1=worst); random expectation = 0.5
        norm = [(r - 1) / (N - 1) for r, N in pick_ranks if N > 1]
        avg_norm = float(np.mean(norm))
        hit_rate = hits_top3 / len(pick_ranks)
        lines += ["", "## Pick-skill scorecard",
                  f"- picks evaluated: {len(pick_ranks)} over {len(n_tradeable)} months "
                  f"(avg {np.mean(n_tradeable):.0f} tradeable sectors/month)",
                  f"- **avg normalized realized rank of our picks: {avg_norm:.3f}** "
                  f"(0=always the month's best sector, 0.5=random, 1=worst)",
                  f"- **top-tercile hit rate: {hit_rate:.0%}** (random ≈ 33%)",
                  f"- verdict: {'SKILL — picks beat random' if avg_norm < 0.46 else ('flat/none' if avg_norm < 0.54 else 'ANTI-skill')}"]

    md = "\n".join(lines)
    if out_md:
        Path(out_md).write_text(md)
        print(f"wrote {out_md} ({len(lines)} lines)")
    print("\n".join(lines[-8:]) if panel is not None else "\n".join(lines[:12]))
    return md


if __name__ == "__main__":
    # default: monthly report with A2b (the static winner) overlay
    run("monthly", panel_path="data/signal/a2b_council/signal_panel.parquet", panel_label="A2b static council",
        out_md="data/signal/sector_report_a2b.md")
