"""doppelganger.memory — time-gated view of a subject's evidence for the doppelganger.

Feed-all: the full <=T corpus fits in context, so v1 does NO retrieval — it filters
to <= t0 behind a leakage firewall and hands over everything chronologically. The
`query` parameter is accepted but ignored: the seam where retrieval drops in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config


@dataclass
class MemoryView:
    items: pd.DataFrame       # the <= t0 evidence, sorted chronologically
    text: str                 # formatted block the doppelganger reads
    n_items: int
    max_date: date | None     # leakage guarantee: must be <= t0 (None if empty)


def _format(items: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, r in items.iterrows():
        d = pd.Timestamp(r["timestamp"]).date().isoformat()
        ctx = r.get("context")
        ctx_s = f" (context: {ctx})" if isinstance(ctx, str) and ctx else ""
        lines.append(f"[{d}] ({r['source_type']}){ctx_s} {r['text']}")
    return "\n".join(lines)


def load_memory(
    slug: str, t0: date, *,
    evidence_path: Path | None = None,
    query: str | None = None,    # accepted, IGNORED in v1 — the retrieval seam
) -> MemoryView:
    path = evidence_path or (config.OUT_DIR / slug / "evidence.parquet")
    ev = pd.read_parquet(path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    # FIREWALL: <= t0, applied first, before anything else.
    ev = ev[ev["timestamp"].dt.date <= t0].sort_values("timestamp").reset_index(drop=True)
    max_date = pd.Timestamp(ev["timestamp"].max()).date() if len(ev) else None
    return MemoryView(items=ev, text=_format(ev), n_items=len(ev), max_date=max_date)
