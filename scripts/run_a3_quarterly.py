"""A3 quarterly run: market-aware member calls (digest + framework) -> A3a consensus + A3b council.
Reuses the existing A2 quarterly per-member corpus views as each member's framework. Resumable.
Runs concurrently with the still-grinding monthly A2 (independent data dirs)."""
import json
from datetime import date
from pathlib import Path
from signals.informativeness import load_close_panel
from signals.run import rebalance_dates
from signals.a3 import run_a3_members
from signals.consensus import run_a2a_consensus
from signals.council import run_a2b

MEMBERS = [("daren-matsuoka", "Daren Matsuoka"), ("justin-thaler", "Justin Thaler"),
           ("tim-roughgarden", "Tim Roughgarden"), ("ali-yahya", "Ali Yahya"),
           ("chris-dixon", "Chris Dixon"), ("eddy-lazzarin", "Eddy Lazzarin"),
           ("guy-wuollet", "Guy Wuollet"), ("jason-rosenthal", "Jason Rosenthal"),
           ("miles-jennings", "Miles Jennings"), ("scott-kominers", "Scott Kominers")]

if __name__ == "__main__":
    dates = rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "quarterly")
    prices = load_close_panel("data/market_data", "ohlcv")
    oi = load_close_panel("data/market_data", "oi")
    sm = json.load(open("data/market_data/sector_map.json"))
    base = Path("data/signal")
    a3_root = base / "a3_members_q"

    print(f"[a3] building {len(dates)} quarterly digests + member calls (10 members)...", flush=True)
    run_a3_members(dates, interval_days=91, a2_members_root=base / "members", out_root=a3_root,
                   sector_map=sm, prices=prices, oi_panel=oi, members=MEMBERS, audit=True, news=True,
                   model="opus")
    print("[a3] member views done; running A3a consensus...", flush=True)
    run_a2a_consensus(dates, members_root=a3_root, out_dir=base / "a3a_consensus",
                      registry_path=base / "registry.json")
    print("[a3] running A3b council...", flush=True)
    run_a2b(dates, members_root=a3_root, out_dir=base / "a3b_council",
            registry_path=base / "registry.json")
    print("A3 QUARTERLY DONE", flush=True)
