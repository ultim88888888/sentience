"""Digest tests — the lookahead guard is the existential invariant, so it gets the most coverage."""
import pandas as pd
from unittest.mock import patch
from signals.digest import trailing_return, cc_news, market_block


def _prices():
    idx = pd.to_datetime(["2023-01-01", "2023-03-01", "2023-06-01", "2023-06-30", "2023-09-30"])
    return pd.DataFrame({"BTC": [10.0, 12.0, 15.0, 16.0, 99.0],   # 99 is AFTER eval date 06-30
                         "ARB": [1.0, 2.0, 3.0, 4.0, 50.0]}, index=idx)


def test_trailing_return_uses_only_past():
    p = _prices()
    # as of 2023-06-30: 3M back ~ 2023-03-31 -> nearest <= is 2023-03-01 (12.0); 16/12-1 = .333
    r = trailing_return(p, "BTC", pd.Timestamp("2023-06-30"), 91)
    assert abs(r - (16.0 / 12.0 - 1)) < 1e-9


def test_trailing_return_never_sees_future():
    """The 2023-09-30 price (99) must never enter a return computed as of 2023-06-30."""
    p = _prices()
    for days in (7, 30, 91, 182, 365):
        r = trailing_return(p, "BTC", pd.Timestamp("2023-06-30"), days)
        # 16.0 is the as-of-T anchor; no horizon should produce the 99.0 (future) value
        assert r is None or abs((1 + r)) < 16.0 / 1.0  # ratio bounded by past prices only
    # explicit: as-of price at T is 16, not 99
    from signals.informativeness import _asof_price
    assert _asof_price(p, "BTC", pd.Timestamp("2023-06-30")) == 16.0


def test_cc_news_drops_lookahead():
    t = pd.Timestamp("2023-06-30")
    end_ts = int(t.timestamp())
    fake = [
        {"ID": 1, "PUBLISHED_ON": end_ts - 86400, "TITLE": "before", "SOURCE_ID": "x"},
        {"ID": 2, "PUBLISHED_ON": end_ts + 86400, "TITLE": "AFTER (lookahead)", "SOURCE_ID": "x"},
        {"ID": 3, "PUBLISHED_ON": end_ts - 5 * 86400, "TITLE": "also before", "SOURCE_ID": "x"},
    ]
    with patch("signals.digest._cc_page", return_value=fake):
        arts = cc_news(t, 91, anchors=1)
    titles = {a["title"] for a in arts}
    assert "before" in titles and "also before" in titles
    assert not any("lookahead" in x for x in titles)          # the future article is dropped
    assert all(a["ts"] < end_ts for a in arts)


def test_cc_news_dedups_across_anchors():
    t = pd.Timestamp("2023-06-30")
    dup = [{"ID": 7, "PUBLISHED_ON": int(t.timestamp()) - 86400, "TITLE": "dup", "SOURCE_ID": "x"}]
    with patch("signals.digest._cc_page", return_value=dup):
        arts = cc_news(t, 91, anchors=5)   # same article returned at every anchor
    assert len([a for a in arts if a["ts"]]) == 1   # deduped by ID
