"""doppelganger.judge — held-out prediction judge.

Compares a view@T against what the subject ACTUALLY said in (T, T+horizon],
scoring each claim confirmed/contradicted/absent. One claude -p call, cached.
"""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.llm import run_claude


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def post_t_evidence(slug: str, t0: date, horizon_months: int = 6, *,
                    evidence_path: Path | None = None) -> str:
    path = evidence_path or (config.OUT_DIR / slug / "evidence.parquet")
    ev = pd.read_parquet(path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    end = _add_months(t0, horizon_months)
    d = ev["timestamp"].dt.date
    win = ev[(d > t0) & (d <= end)].sort_values("timestamp")
    return "\n".join(
        f"[{pd.Timestamp(r['timestamp']).date().isoformat()}] ({r['source_type']}) {r['text']}"
        for _, r in win.iterrows()
    )
