from datetime import date
import pandas as pd
from signals.corpus import in_window, assemble_corpus

def test_in_window_respects_trailing_bound_and_t():
    t = date(2024, 6, 30)
    assert in_window("2024-01-01", t, 18)       # inside
    assert not in_window("2024-07-01", t, 18)   # after T (leakage)
    assert not in_window("2022-01-01", t, 18)   # before window start

def test_assemble_is_chronological_and_tagged(tmp_path, monkeypatch):
    tw = pd.DataFrame({
        "created_at": pd.to_datetime(["2024-05-01", "2024-03-01"], utc=True),
        "type": ["original", "original"],
        "text": [
            "late tweet: a substantive take on zk rollups and their role in scaling ethereum",
            "early tweet: a substantive take on modular blockchains and data availability layers",
        ],
        "url": ["u1", "u2"],
    })
    tw_path = tmp_path / "eddy.parquet"; tw.to_parquet(tw_path)
    arts = pd.DataFrame({"post_date": ["2024-04-01"], "extracted_text": ["an article body"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    text = assemble_corpus(
        t=date(2024, 6, 30), window_months=18,
        twitter_paths=[tw_path], articles=arts, distillates={},
    )
    # chronological: early tweet (Mar) before article (Apr) before late tweet (May)
    assert text.index("early tweet") < text.index("an article body") < text.index("late tweet")
    assert "(x)" in text and "(research)" in text

def test_distillate_passages_included_in_window(tmp_path):
    arts = pd.DataFrame({"post_date": ["2024-04-01"], "extracted_text": ["body"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    distillates = {"o1": [{"date": "2024-04-01", "passage": "zk is the endgame"}]}
    text = assemble_corpus(t=date(2024, 6, 30), window_months=18,
                           twitter_paths=[], articles=arts, distillates=distillates)
    assert "zk is the endgame" in text
    assert "(transcript)" in text

def test_post_t_evidence_excluded(tmp_path):
    arts = pd.DataFrame({"post_date": ["2025-01-01"], "extracted_text": ["future body"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    text = assemble_corpus(t=date(2024, 6, 30), window_months=18,
                           twitter_paths=[], articles=arts, distillates={})
    assert "future body" not in text

def test_low_substance_tweets_dropped(tmp_path):
    from signals.corpus import assemble_corpus, is_substantive_tweet
    assert is_substantive_tweet("This is a genuinely substantive take on zk rollups and scaling tradeoffs")
    assert not is_substantive_tweet("lol @bob https://t.co/x")
    tw = pd.DataFrame({
        "created_at": pd.to_datetime(["2024-05-01", "2024-05-02"], utc=True),
        "type": ["original", "original"],
        "text": ["@a @b https://t.co/x", "A real substantive thesis about modular blockchains and data availability"],
        "url": ["u1", "u2"],
    })
    p = tmp_path / "m.parquet"; tw.to_parquet(p)
    text = assemble_corpus(t=date(2024,6,30), window_months=18, twitter_paths=[p], articles=None, distillates={})
    assert "substantive thesis" in text
    assert "t.co" not in text  # the low-substance tweet was dropped


def test_article_distillates_replace_full_body(tmp_path):
    from datetime import date
    import pandas as pd
    from signals.corpus import assemble_corpus
    arts = pd.DataFrame({"post_date": ["2024-04-01"], "extracted_text": ["FULL BODY should not appear"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    art_dist = {"o1": [{"date": "2024-04-01", "passage": "distilled stance passage"}]}
    text = assemble_corpus(t=date(2024,6,30), window_months=18, twitter_paths=[],
                           articles=arts, distillates={}, article_distillates=art_dist)
    assert "distilled stance passage" in text
    assert "FULL BODY" not in text   # full body suppressed when distillates provided
    assert "(research)" in text
