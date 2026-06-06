"""doppelganger.soul — build the frozen-at-T0 soul card via a single claude -p pass.

The LLM call is isolated in `_run_claude` so the rest is deterministically testable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.identity import build_identity
from doppelganger.ingest import build_evidence_stream
from doppelganger.schema import IdentityProfile


def load_soul_inputs(
    slug: str, t0: date, *,
    evidence_path: Path | None = None,
    identity_path: Path | None = None,
    team_path: Path | None = None,
    tracked_people_path: Path | None = None,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
) -> tuple[IdentityProfile, pd.DataFrame]:
    """Return (identity truncated to <=t0, evidence DataFrame filtered to <=t0, sorted)."""
    identity = build_identity(
        slug, linkedin_path=identity_path, team_path=team_path,
        tracked_people_path=tracked_people_path,
    ).as_of(t0)

    if evidence_path is not None and Path(evidence_path).exists():
        ev = pd.read_parquet(evidence_path)
    else:
        items = build_evidence_stream(
            slug, twitter_path=twitter_path, articles_path=articles_path,
            podcast_path=podcast_path, tracked_people_path=tracked_people_path,
        )
        from dataclasses import asdict
        ev = pd.DataFrame([asdict(e) for e in items])

    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    ev = ev[ev["timestamp"].dt.date <= t0].sort_values("timestamp").reset_index(drop=True)
    return identity, ev
