"""doppelganger.walkforward — drive respond() across a quarterly schedule.

Produces matched full + ablation views per step, audits each, and records a
resumable trajectory + coverage map. The raw material Unit 5b scores.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.respond import respond
from doppelganger.soul_audit import audit_answer

_QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def quarter_ends(start: date, end: date) -> list[date]:
    """Quarter-end dates (Mar 31 / Jun 30 / Sep 30 / Dec 31) within [start, end], inclusive."""
    out: list[date] = []
    for yr in range(start.year, end.year + 1):
        for m, d in _QUARTER_ENDS:
            qd = date(yr, m, d)
            if start <= qd <= end:
                out.append(qd)
    return out
