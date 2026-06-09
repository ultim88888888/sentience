"""A4 quarterly: TRUE doppelganger (soul + memory + digest) member calls -> A4a consensus + A4b council.
Concurrency-pooled (2-3) with retry+backoff. Resumable. Run AFTER the A3 rerun to avoid rate contention."""
import json
from datetime import date
from pathlib import Path
from signals.informativeness import load_close_panel
from signals.run import rebalance_dates
from signals.a4 import run_a4_members
from signals.consensus import run_a2a_consensus
from signals.council import run_a2b

MEMBERS = [("daren-matsuoka","Daren Matsuoka"),("justin-thaler","Justin Thaler"),
           ("tim-roughgarden","Tim Roughgarden"),("ali-yahya","Ali Yahya"),
           ("chris-dixon","Chris Dixon"),("eddy-lazzarin","Eddy Lazzarin"),
           ("guy-wuollet","Guy Wuollet"),("jason-rosenthal","Jason Rosenthal"),
           ("miles-jennings","Miles Jennings"),("scott-kominers","Scott Kominers")]

if __name__ == "__main__":
    dates = rebalance_dates(date(2022,12,31), date(2026,3,31), "quarterly")
    prices = load_close_panel("data/market_data","ohlcv"); oi = load_close_panel("data/market_data","oi")
    sm = json.load(open("data/market_data/sector_map.json"))
    base = Path("data/signal"); a4 = base/"a4_members_q"
    print(f"[a4] {len(dates)} periods x {len(MEMBERS)} members (soul+memory+digest), concurrency=2", flush=True)
    run_a4_members(dates, interval_days=91, out_root=a4, sector_map=sm, prices=prices, oi_panel=oi,
                   members=MEMBERS, audit=True, news=True, max_workers=2, model="opus")
    print("[a4] member views done; A4a consensus...", flush=True)
    run_a2a_consensus(dates, members_root=a4, out_dir=base/"a4a_consensus", registry_path=base/"registry.json")
    print("[a4] A4b council...", flush=True)
    run_a2b(dates, members_root=a4, out_dir=base/"a4b_council", registry_path=base/"registry.json")
    print("A4 QUARTERLY DONE", flush=True)
