"""TDD tests for doppelganger.schema."""
from __future__ import annotations

from datetime import date, datetime, timezone

from doppelganger.schema import EvidenceItem, Experience, Education, IdentityProfile


def test_evidence_item_defaults():
    e = EvidenceItem(
        id="1", subject="testy-mctest", timestamp=datetime(2022, 6, 1, tzinfo=timezone.utc),
        source_type="x_original", text="hi", speaker_slug="testy-mctest", attribution_confidence=1.0,
    )
    assert e.thread_id is None and e.context is None and e.context_missing is False and e.engagement is None


def test_identity_as_of_truncates_experience_and_education():
    prof = IdentityProfile(
        slug="testy-mctest", name="Testy", headline=None, bio=None, current_role=None,
        experience=[
            Experience("GP", "Acme", date(2026, 5, 1), None, None),
            Experience("CTO", "Acme", date(2023, 2, 1), date(2026, 5, 1), None),
            Experience("Engineer", "Beta", date(2016, 1, 1), date(2018, 1, 1), "Lead."),
        ],
        education=[Education("State U", "BA", "Philosophy", date(2006, 1, 1), date(2010, 1, 1))],
        socials={},
    )
    at = prof.as_of(date(2022, 12, 31))
    # GP (2026) and CTO (2023) are in the future relative to 2022 -> dropped
    assert [x.title for x in at.experience] == ["Engineer"]
    # current role at 2022-12-31: most recent experience started on/before then = Engineer ended 2018,
    # but nothing active; current_role is the latest-started role with start<=T -> "Engineer"
    assert at.current_role == "Engineer"
    assert [x.school for x in at.education] == ["State U"]


def test_identity_as_of_picks_active_role():
    prof = IdentityProfile(
        slug="s", name="N", headline=None, bio=None, current_role=None,
        experience=[
            Experience("CTO", "Acme", date(2023, 2, 1), date(2026, 5, 1), None),
            Experience("Engineer", "Beta", date(2016, 1, 1), date(2018, 1, 1), None),
        ],
        education=[],
        socials={},
    )
    at = prof.as_of(date(2024, 6, 1))
    # CTO active 2023-2026 covers 2024 -> current role
    assert at.current_role == "CTO"
