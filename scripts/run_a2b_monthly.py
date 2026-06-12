"""Monthly A2b council — now that the monthly A2 member views are backfilled (9.5/10 members x 40 months),
run the council deliberation monthly. Then we can settle: does a MONTHLY corpus refresh beat the
quarterly-signal + monthly-timing deliverable? Resumable."""
from datetime import date
from pathlib import Path
from signals.run import rebalance_dates
from signals.council import run_a2b
if __name__ == "__main__":
    dates = rebalance_dates(date(2022,12,31), date(2026,3,31), "monthly")
    base = Path("data/signal/members_monthly")
    print(f"[a2b-monthly] council over {len(dates)} monthly periods", flush=True)
    run_a2b(dates, members_root=base/"members", out_dir=Path("data/signal/a2b_council_m"),
            registry_path=base/"registry.json")
    print("A2B MONTHLY COUNCIL DONE", flush=True)
