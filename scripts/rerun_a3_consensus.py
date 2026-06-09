"""Re-run only the A3a consensus + A3b council aggregation (member views intact; only this step failed
during the overnight rate-limit window). Serialized to avoid re-tripping the limit."""
from datetime import date
from pathlib import Path
from signals.run import rebalance_dates
from signals.consensus import run_a2a_consensus
from signals.council import run_a2b
dates = rebalance_dates(date(2022,12,31), date(2026,3,31), "quarterly")
base = Path("data/signal"); a3 = base/"a3_members_q"
print("[rerun] A3a consensus...", flush=True)
run_a2a_consensus(dates, members_root=a3, out_dir=base/"a3a_consensus", registry_path=base/"registry.json")
print("[rerun] A3b council...", flush=True)
run_a2b(dates, members_root=a3, out_dir=base/"a3b_council", registry_path=base/"registry.json")
print("A3 CONSENSUS RERUN DONE", flush=True)
