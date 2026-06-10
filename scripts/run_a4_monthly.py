"""Monthly A4: the real test of the market-aware thesis (Jax: value compounds at monthly cadence).
TRUE doppelganger (soul + time-gated memory + monthly digest), 40 monthly periods x 10 members.
Independent of the (stuck) monthly A2 — A4 uses souls directly, not A2 views. Resumable, concurrency-2."""
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
    dates = rebalance_dates(date(2022,12,31), date(2026,3,31), "monthly")
    prices = load_close_panel("data/market_data","ohlcv"); oi = load_close_panel("data/market_data","oi")
    sm = json.load(open("data/market_data/sector_map.json"))
    base = Path("data/signal"); a4 = base/"a4_members_m"
    print(f"[a4-monthly] {len(dates)} periods x {len(MEMBERS)} members (soul+memory+30d digest)", flush=True)
    run_a4_members(dates, interval_days=30, out_root=a4, sector_map=sm, prices=prices, oi_panel=oi,
                   members=MEMBERS, audit=True, news=True, max_workers=5, model="opus")
    print("[a4-monthly] member views done; A4a-monthly consensus...", flush=True)
    run_a2a_consensus(dates, members_root=a4, out_dir=base/"a4a_consensus_m", registry_path=base/"registry.json")
    print("[a4-monthly] A4b-monthly council...", flush=True)
    run_a2b(dates, members_root=a4, out_dir=base/"a4b_council_m", registry_path=base/"registry.json")
    print("A4 MONTHLY DONE", flush=True)
