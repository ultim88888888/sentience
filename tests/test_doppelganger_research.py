"""TDD tests for doppelganger.adapters.research."""
from __future__ import annotations

from pathlib import Path

from doppelganger.adapters.research import load_research

FIX = Path("tests/fixtures/doppelganger/articles.parquet")


def _by_id(items):
    return {e.id: e for e in items}


def test_only_subject_posts():
    items = load_research(FIX, "testy-mctest")
    ids = {e.id for e in items}
    assert ids == {"100-0", "101-0"}     # 102-0 is someone else's


def test_solo_vs_firm_attribution():
    items = _by_id(load_research(FIX, "testy-mctest"))
    assert items["100-0"].source_type == "research" and items["100-0"].attribution_confidence == 1.0
    assert items["101-0"].source_type == "research_firm" and items["101-0"].attribution_confidence == 0.5


def test_timestamp_and_text():
    items = _by_id(load_research(FIX, "testy-mctest"))
    assert items["100-0"].timestamp.year == 2022 and items["100-0"].timestamp.tzinfo is not None
    assert "token design" in items["100-0"].text
