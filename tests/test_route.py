import pandas as pd
from attribution.route import route_post, STRUCTURED, CONVERSATIONAL

def _row(fmt, n_auth=1, title="t"):
    return pd.Series({"formats": [fmt], "author_slugs": ["a"] * n_auth, "title": title})

def test_podcast_is_conversational():
    r = route_post(_row("podcasts"))
    assert r.mode == CONVERSATIONAL and r.has_audio is True

def test_video_is_structured():
    r = route_post(_row("videos"))
    assert r.mode == STRUCTURED and r.has_audio is False

def test_multi_author_video_flagged_as_panel_candidate():
    r = route_post(_row("videos", n_auth=3))
    assert r.mode == STRUCTURED and r.panel_candidate is True

def test_single_author_video_not_panel():
    assert route_post(_row("videos", n_auth=1)).panel_candidate is False
