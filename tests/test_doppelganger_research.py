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


def test_nan_acf_falls_back_to_extracted_text(tmp_path):
    import numpy as np
    import pandas as pd

    p = tmp_path / "a.parquet"
    pd.DataFrame([{
        "object_id": "n-0", "title": "t", "post_date": "2022-01-01T00:00:00+00:00",
        "author_slugs": ["subj"], "formats": ["articles"],
        "acf_content": np.nan, "extracted_text": "valid body here",
    }]).to_parquet(p)
    items = load_research(p, "subj")
    assert len(items) == 1 and items[0].text == "valid body here"
