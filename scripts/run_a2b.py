from datetime import date
from pathlib import Path
from signals.run import rebalance_dates
from signals.council import run_a2b
if __name__=="__main__":
    dates=rebalance_dates(date(2022,12,31),date(2026,3,31),"quarterly")
    df=run_a2b(dates, members_root=Path("data/signal/members"),
               out_dir=Path("data/signal/a2b_council"), registry_path=Path("data/signal/registry.json"))
    print("A2B DONE rows:",len(df),flush=True)
