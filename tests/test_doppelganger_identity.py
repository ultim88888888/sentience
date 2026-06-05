"""TDD tests for doppelganger.identity + registry."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.identity import parse_li_date, build_identity
from doppelganger.registry import resolve_subject

FIX = Path("tests/fixtures/doppelganger")


def test_parse_li_date():
    assert parse_li_date("May 2026") == date(2026, 5, 1)
    assert parse_li_date("2008") == date(2008, 1, 1)
    assert parse_li_date(None) is None
    assert parse_li_date("") is None


def test_resolve_subject_reads_registry():
    s = resolve_subject("testy-mctest", tracked_people_path=FIX / "tracked_people.yaml")
    assert s.slug == "testy-mctest"
    assert s.x_handle == "testy"
    assert s.linkedin_file == "testy-1.json"   # derived from linkedin_url trailing segment


def test_build_identity_merges_linkedin_and_bio():
    prof = build_identity(
        "testy-mctest",
        linkedin_path=FIX / "linkedin" / "testy-1.json",
        team_path=FIX / "team.parquet",
        tracked_people_path=FIX / "tracked_people.yaml",
    )
    assert prof.name == "Testy McTest"
    assert prof.headline == "Investing in things."          # from LinkedIn
    assert "GP at Acme" in (prof.bio or "")                  # a16z bio merged in
    assert len(prof.experience) == 3 and prof.experience[0].title == "General Partner"
    assert prof.experience[0].start == date(2026, 5, 1)     # "May 2026" parsed
    assert prof.socials.get("x_url") == "https://twitter.com/@testy"
    # time-gate: as-of EOY 2022 drops the 2023/2026 roles
    at = prof.as_of(date(2022, 12, 31))
    assert [e.title for e in at.experience] == ["Engineer"]
