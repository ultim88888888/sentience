"""TDD tests for doppelganger.soul."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.soul import load_soul_inputs

FIX = Path("tests/fixtures/doppelganger")


def test_load_soul_inputs_gates_by_t0():
    identity, evidence = load_soul_inputs(
        "testy-mctest", date(2022, 8, 31),
        evidence_path=None,  # use the built fixture stream below
        identity_path=FIX / "linkedin" / "testy-1.json",
        team_path=FIX / "team.parquet",
        tracked_people_path=FIX / "tracked_people.yaml",
        twitter_path=FIX / "twitter" / "testy.parquet",
        articles_path=FIX / "articles.parquet",
        podcast_path=FIX / "attributed_transcripts.jsonl",
    )
    # identity is truncated to <= 2022-08-31 (Engineer only; GP/CTO are 2023+/2026)
    assert [e.title for e in identity.experience] == ["Engineer"]
    # evidence is filtered to <= 2022-08-31: the 2022-09-01 quote (id "6") is excluded
    assert (evidence["timestamp"].dt.date <= date(2022, 8, 31)).all()
    assert "6" not in set(evidence["id"])
    # sorted ascending
    assert list(evidence["timestamp"]) == sorted(evidence["timestamp"])
