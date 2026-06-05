"""doppelganger.ingest — orchestrate adapters + identity into artifacts.

Outputs (per subject): data/doppelganger/<slug>/evidence.parquet + identity.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.adapters.podcast import load_podcast
from doppelganger.adapters.research import load_research
from doppelganger.adapters.twitter import load_twitter
from doppelganger.identity import build_identity
from doppelganger.registry import resolve_subject
from doppelganger.schema import EvidenceItem

EVIDENCE_COLS = ["id", "subject", "timestamp", "source_type", "text", "speaker_slug",
                 "attribution_confidence", "thread_id", "context", "context_missing", "engagement"]


def build_evidence_stream(
    slug: str, *,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
    tracked_people_path: Path | None = None,
) -> list[EvidenceItem]:
    ref = resolve_subject(slug, tracked_people_path=tracked_people_path)
    items: list[EvidenceItem] = []

    tw = twitter_path or (config.TWITTER_DIR / f"{ref.x_handle}.parquet")
    if Path(tw).exists():
        items += load_twitter(Path(tw), slug)

    art = articles_path or config.RESEARCH_ARTICLES
    if Path(art).exists():
        items += load_research(Path(art), slug)

    pod = podcast_path or config.ATTRIBUTED_TRANSCRIPTS
    if Path(pod).exists():
        items += load_podcast(Path(pod), slug)

    items.sort(key=lambda e: e.timestamp)
    return items


def _evidence_df(items: list[EvidenceItem]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(e) for e in items])
    if df.empty:
        return pd.DataFrame(columns=EVIDENCE_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[EVIDENCE_COLS]


def _identity_json(profile) -> str:
    def _enc(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        raise TypeError(type(o))
    return json.dumps(asdict(profile), default=_enc, indent=2)


def ingest(
    slug: str, *,
    out_dir: Path | None = None,
    linkedin_path: Path | None = None,
    team_path: Path | None = None,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
    tracked_people_path: Path | None = None,
) -> dict[str, Path]:
    items = build_evidence_stream(
        slug, twitter_path=twitter_path, articles_path=articles_path,
        podcast_path=podcast_path, tracked_people_path=tracked_people_path,
    )
    profile = build_identity(
        slug, linkedin_path=linkedin_path, team_path=team_path,
        tracked_people_path=tracked_people_path,
    )

    base = Path(out_dir or config.OUT_DIR) / slug
    base.mkdir(parents=True, exist_ok=True)
    ev_path, id_path = base / "evidence.parquet", base / "identity.json"
    _evidence_df(items).to_parquet(ev_path)
    id_path.write_text(_identity_json(profile))
    return {"evidence": ev_path, "identity": id_path}
