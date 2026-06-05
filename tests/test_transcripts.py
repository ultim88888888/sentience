"""Routing logic for the transcript collector — pure, no network.

Covers the source-selection decision tree: video->youtube, podcast-with-youtube->youtube,
podcast-without-youtube->whisper(simplecast), and the unroutable flags.
"""
import pandas as pd

from scrapers.a16z_transcripts.sources import (NONE, WHISPER, YOUTUBE, route_corpus, route_row)

YT_EMBED = '<iframe src="https://www.youtube.com/embed/gOX30oemfLM?rel=0"></iframe>'
SIMC_EMBED = '<iframe src="https://player.simplecast.com/566bc850-8fcc-4a70-a64a-013b809fbde2"></iframe>'


def _row(object_id="1", title="t", formats=("videos",), html=""):
    return pd.Series({"object_id": object_id, "title": title,
                      "formats": list(formats), "raw_html": html})


def test_video_routes_to_youtube():
    r = route_row(_row(formats=["videos"], html=YT_EMBED))
    assert r.source == YOUTUBE and r.media_id == "gOX30oemfLM"


def test_podcast_with_youtube_routes_to_youtube():
    r = route_row(_row(formats=["podcasts"], html=YT_EMBED + SIMC_EMBED))
    assert r.source == YOUTUBE and r.media_id == "gOX30oemfLM"


def test_podcast_without_youtube_routes_to_whisper():
    r = route_row(_row(formats=["podcasts"], html=SIMC_EMBED))
    assert r.source == WHISPER
    assert r.media_id == "566bc850-8fcc-4a70-a64a-013b809fbde2"


def test_podcast_with_neither_is_unroutable():
    r = route_row(_row(formats=["podcasts"], html="<p>no media</p>"))
    assert r.source == NONE and r.media_id is None and r.note


def test_video_without_youtube_flagged():
    r = route_row(_row(formats=["videos"], html="<p>nothing</p>"))
    assert r.source == NONE and "video" in r.note


def test_articles_and_papers_are_skipped():
    assert route_row(_row(formats=["articles"], html=YT_EMBED)) is None
    assert route_row(_row(formats=["papers"])) is None
    assert route_row(_row(formats=[])) is None


def test_youtu_be_short_url_matches():
    r = route_row(_row(formats=["videos"], html='<a href="https://youtu.be/abcdefghijk">x</a>'))
    assert r.source == YOUTUBE and r.media_id == "abcdefghijk"


def test_route_corpus_filters_and_counts():
    df = pd.DataFrame([
        _row("1", formats=["videos"], html=YT_EMBED),
        _row("2", formats=["podcasts"], html=SIMC_EMBED),
        _row("3", formats=["articles"], html=YT_EMBED),  # dropped
    ])
    routes = route_corpus(df)
    assert len(routes) == 2
    assert {r.source for r in routes} == {YOUTUBE, WHISPER}
