"""Token-level attribution (Jax: sector AND token performance vs our picks). The strategy trades sector
BASKETS; this decomposes realized P&L down to the individual TOKENS held, to show how concentrated the
returns really are (the ZEC-lottery question) and which names carry the strategy."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import scripts.alpha_hunt3 as ah
from signals.informativeness import forward_return

A2B = ah.load("data/signal/a2b_council/signal_panel.parquet")
dates = [d for d in ah.MD if d in ah.NXT]


def main():
    contrib = {}   # token -> summed weight*return contribution
    held_months = {}
    sector_of = {}
    for t in dates:
        secs = ah.topk(A2B, t, 3, mom_thr=0.10)
        oin = ah.oi_at(ah.oi, t)
        if not secs:
            tok = "BTC"
            r = forward_return(ah.prices, "BTC", t, ah.NXT[t]) or 0.0
            contrib[tok] = contrib.get(tok, 0.0) + r
            held_months[tok] = held_months.get(tok, 0) + 1
            continue
        raw = {}
        for sec in secs:
            b = ah.sector_basket(ah.sm, oin, sec, oi_floor=ah.OI_FLOOR_USD)
            for s in b:
                raw[s] = raw.get(s, 0.0) + 1.0 / len(b)
                sector_of[s] = sec
        g = sum(abs(x) for x in raw.values())
        for s, w in raw.items():
            r = forward_return(ah.prices, s, t, ah.NXT[t])
            if r is None:
                continue
            contrib[s] = contrib.get(s, 0.0) + (w / g) * r
            held_months[s] = held_months.get(s, 0) + 1

    tot = sum(contrib.values())
    rows = sorted(contrib.items(), key=lambda kv: -kv[1])
    print(f"=== token-level P&L attribution (sum of monthly weight×return contributions) ===")
    print(f"total summed contribution: {tot:+.2f}  (≈ arithmetic, not compounded)\n")
    print(f"{'token':<10}{'sector':<22}{'months held':>12}{'contrib':>10}{'% of total':>11}")
    for s, c in rows[:15]:
        print(f"{s:<10}{sector_of.get(s,'-'):<22}{held_months.get(s,0):>12}{c:>+10.2f}{c/tot*100:>10.0f}%")
    print("  ...")
    for s, c in rows[-3:]:
        print(f"{s:<10}{sector_of.get(s,'-'):<22}{held_months.get(s,0):>12}{c:>+10.2f}{c/tot*100:>10.0f}%")
    # concentration
    top3 = sum(c for _, c in rows[:3])
    print(f"\nCONCENTRATION: top-3 tokens = {top3/tot*100:.0f}% of total P&L; "
          f"top-1 ({rows[0][0]}) = {rows[0][1]/tot*100:.0f}%")


if __name__ == "__main__":
    main()
