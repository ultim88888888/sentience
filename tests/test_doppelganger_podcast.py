"""TDD tests for doppelganger.adapters.podcast."""
from __future__ import annotations

from pathlib import Path

from doppelganger.adapters.podcast import load_podcast

FIX = Path("tests/fixtures/doppelganger/attributed_transcripts.jsonl")


def test_only_subject_turns_above_confidence():
    items = load_podcast(FIX, "testy-mctest")
    texts = [e.text for e in items]
    # both high-confidence subject turns kept; AUDIENCE turn and other-podcast excluded
    assert any("Who here may issue a token" in t for t in texts)
    assert any("balance in a database" in t for t in texts)
    assert all("not testy" not in t for t in texts)
    # low-confidence subject turn (0.5 < 0.8) dropped
    assert all("should be dropped" not in t for t in texts)


def test_preceding_question_attached_as_context():
    items = load_podcast(FIX, "testy-mctest")
    answer = next(e for e in items if "balance in a database" in e.text)
    assert answer.context == "What about points?"   # the AUDIENCE turn just before


def test_fields():
    items = load_podcast(FIX, "testy-mctest")
    e = items[0]
    assert e.source_type == "podcast" and e.speaker_slug == "testy-mctest"
    assert e.timestamp.year == 2022 and e.timestamp.tzinfo is not None
    assert e.id.startswith("200-0:")                # <object_id>:<idx>
