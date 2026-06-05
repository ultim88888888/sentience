"""TDD tests for doppelganger.ingest orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doppelganger.ingest import build_evidence_stream, ingest

FIX = Path("tests/fixtures/doppelganger")
EVIDENCE_COLS = ["id", "subject", "timestamp", "source_type", "text", "speaker_slug",
                 "attribution_confidence", "thread_id", "context", "context_missing", "engagement"]


def _sources():
    return dict(
        twitter_path=FIX / "twitter" / "testy.parquet",
        articles_path=FIX / "articles.parquet",
        podcast_path=FIX / "attributed_transcripts.jsonl",
        tracked_people_path=FIX / "tracked_people.yaml",
    )


def test_evidence_stream_merged_and_sorted():
    items = build_evidence_stream("testy-mctest", **_sources())
    # sorted ascending by timestamp
    ts = [e.timestamp for e in items]
    assert ts == sorted(ts)
    # contains items from all three sources
    types = {e.source_type for e in items}
    assert {"x_original", "research", "research_firm", "podcast"} <= types


def test_ingest_writes_artifacts(tmp_path):
    out = ingest(
        "testy-mctest", out_dir=tmp_path,
        linkedin_path=FIX / "linkedin" / "testy-1.json", team_path=FIX / "team.parquet",
        **_sources(),
    )
    ev = pd.read_parquet(out["evidence"])
    assert list(ev.columns) == EVIDENCE_COLS
    assert len(ev) > 0
    ident = json.loads(Path(out["identity"]).read_text())
    assert ident["name"] == "Testy McTest" and "experience" in ident
