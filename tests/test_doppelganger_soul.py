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


from doppelganger.soul import build_extraction_prompt
from doppelganger.schema import IdentityProfile, Experience


def test_build_extraction_prompt_structure():
    import pandas as pd
    identity = IdentityProfile(
        slug="testy-mctest", name="Testy McTest", headline="Investing.", bio="A GP.",
        current_role="Engineer", experience=[Experience("Engineer", "Beta", None, None, None)],
        education=[], socials={},
    )
    ev = pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives.",
         "attribution_confidence": 1.0, "context": None},
    ])
    system, user = build_extraction_prompt(identity, ev)
    # system instructions name the required sections and the citation format
    for section in ["How He Thinks", "What He Believes", "What He Attends To",
                    "Open Contradictions", "How He Talks", "Bio Lens"]:
        assert section in system
    assert "[2022-06-01]" not in system          # the date format is described, not pre-filled
    assert '[<YYYY-MM-DD>]' in system or "YYYY-MM-DD" in system
    # user content carries the identity and every evidence item with its date + text
    assert "Testy McTest" in user and "A GP." in user
    assert "2022-06-01" in user and "Tokens align incentives." in user
    assert "x_original" in user
