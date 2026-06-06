"""doppelganger.run_full — run the full quarterly walk-forward for all subjects.

Resumable: skips any (date, variant) whose view JSON already exists, and skips
(continues past) transient `claude -p` failures — so re-running fills any gaps.
Designed to run unattended on an always-on box:

    tmux new -s wf 'source .venv/bin/activate && caffeinate -i python -m doppelganger.run_full'

Reattach: `tmux attach -t wf`. Detach: Ctrl-b then d.
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from doppelganger import config
from doppelganger.walkforward import quarter_ends, run_walkforward

SUBJECTS = ["eddy-lazzarin", "ali-yahya"]
START = date(2022, 12, 31)


def main() -> None:
    for slug in SUBJECTS:
        ev = pd.read_parquet(config.OUT_DIR / slug / "evidence.parquet")
        end = pd.to_datetime(ev["timestamp"], utc=True).max().date()
        dates = quarter_ends(START, end)
        print(f"=== {slug}: {len(dates)} quarters {START} -> {end} === {time.strftime('%H:%M:%S')}", flush=True)
        rows = run_walkforward(slug, dates)
        print(f"=== {slug} DONE: {len(rows)} rows === {time.strftime('%H:%M:%S')}", flush=True)
    print("ALL DONE", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
