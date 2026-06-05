"""TDD tests for doppelganger.adapters.twitter."""
from __future__ import annotations

from pathlib import Path

from doppelganger.adapters.twitter import load_twitter

FIX = Path("tests/fixtures/doppelganger/twitter/testy.parquet")


def _by_id(items):
    return {e.id: e for e in items}


def test_drops_retweets_and_noise_replies():
    items = load_twitter(FIX, "testy-mctest")
    ids = {e.id for e in items}
    assert "7" not in ids        # retweet dropped
    assert "5" not in ids        # "@someone lol" noise reply dropped


def test_keeps_substantive_reply_to_other():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    assert "4" in items
    assert items["4"].source_type == "x_reply"


def test_self_thread_reassembled():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    # root "2" absorbs self-reply "3"; "3" is not its own item
    assert "3" not in items
    assert items["2"].source_type == "x_original"
    assert "rollups inherit security" in items["2"].text
    assert "Thread on L2s" in items["2"].text


def test_quote_context_missing_flag():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    assert items["6"].source_type == "x_quote"
    assert items["6"].context_missing is True   # quoted tweet 55555 not in corpus


def test_fields_populated():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    e = items["1"]
    assert e.subject == "testy-mctest" and e.speaker_slug == "testy-mctest"
    assert e.attribution_confidence == 1.0
    assert e.timestamp.tzinfo is not None        # tz-aware
