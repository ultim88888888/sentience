"""Run ONLY A4b-monthly council, concurrently with the A4a consensus already running in run_a4_monthly.
Separate registry copy avoids a write race on the shared registry.json."""
from datetime import date
from pathlib import Path
from signals.run import rebalance_dates
from signals.council import run_a2b
dates = rebalance_dates(date(2022,12,31), date(2026,3,31), "monthly")
base = Path("data/signal")
print(f"[a4b-monthly] council over {len(dates)} periods (concurrent with A4a)", flush=True)
run_a2b(dates, members_root=base/"a4_members_m", out_dir=base/"a4b_council_m",
        registry_path=base/"registry_a4b_m.json")
print("A4B MONTHLY DONE", flush=True)
