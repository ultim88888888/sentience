from datetime import date
from pathlib import Path
from signals.run import rebalance_dates
from signals.consensus import run_a2a_consensus
if __name__ == "__main__":
    dates = rebalance_dates(date(2022,12,31), date(2026,3,31), "quarterly")
    df = run_a2a_consensus(dates, members_root=Path("data/signal/members"),
                           out_dir=Path("data/signal/a2a_consensus"),
                           registry_path=Path("data/signal/registry.json"))
    print(f"A2A CONSENSUS DONE — {len(df)} rows, {df['as_of'].nunique()} quarters")
