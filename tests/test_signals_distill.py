import json
import pandas as pd
from unittest.mock import patch
from signals.distill import distill_one, build_distillate_cache

FAKE_LLM = json.dumps({"passages": [
    {"date": "2023-02-10", "passage": "I think zk rollups are the endgame for scaling."},
    {"date": "2023-02-10", "passage": "Restaking introduces real systemic risk."},
]})

def test_distill_one_returns_verbatim_passages():
    with patch("signals.distill.run_claude", return_value=FAKE_LLM):
        out = distill_one(object_id="x1", title="T", transcript="...long...",
                          post_date="2023-02-10")
    assert len(out) == 2
    assert out[0]["passage"].startswith("I think zk rollups")
    assert out[0]["date"] == "2023-02-10"

def test_build_cache_is_resumable(tmp_path):
    tx = pd.DataFrame({"object_id": ["x1", "x2"], "title": ["A", "B"],
                       "transcript": ["t1", "t2"], "status": ["ok", "ok"]})
    arts = pd.DataFrame({"object_id_join": ["x1", "x2"],
                         "post_date": ["2023-01-01", "2023-02-01"]})
    tx_path = tmp_path / "tx.parquet"; tx.to_parquet(tx_path)
    cache = tmp_path / "distillates.jsonl"
    # pre-seed cache with x1 already done
    cache.write_text(json.dumps({"object_id": "x1", "passages": []}) + "\n")
    calls = []
    def fake(system, user, **kw):
        calls.append(user)
        return FAKE_LLM
    with patch("signals.distill.run_claude", side_effect=fake):
        build_distillate_cache(tx_path, arts, cache_path=cache,
                               post_dates={"x1": "2023-01-01", "x2": "2023-02-01"})
    assert len(calls) == 1  # only x2 distilled; x1 skipped (resume)
    lines = [json.loads(l) for l in cache.read_text().splitlines()]
    assert {l["object_id"] for l in lines} == {"x1", "x2"}

def test_skips_empty_transcripts(tmp_path):
    tx = pd.DataFrame({"object_id": ["x3"], "title": ["C"], "transcript": [""],
                       "status": ["no_captions"]})
    tx_path = tmp_path / "tx.parquet"; tx.to_parquet(tx_path)
    cache = tmp_path / "d.jsonl"
    with patch("signals.distill.run_claude", return_value=FAKE_LLM) as m:
        build_distillate_cache(tx_path, None, cache_path=cache, post_dates={"x3": "2023-01-01"})
    m.assert_not_called()


def test_build_article_cache_filters_and_distills(tmp_path):
    import json, pandas as pd
    from unittest.mock import patch
    from signals.distill import build_article_distillate_cache
    arts = pd.DataFrame({
        "object_id": ["a1", "a2", "a3"],
        "title": ["A", "B", "C"],
        "post_date": ["2023-05-01", "2020-01-01", "2023-06-01"],  # a2 too old (filtered)
        "extracted_text": ["x"*600, "y"*600, "z"*100],            # a3 too short (filtered)
    })
    cache = tmp_path / "article_distillates.jsonl"
    calls = []
    def fake(system, user, **kw):
        calls.append(user)
        return json.dumps({"passages": [{"date": "2023-05-01", "passage": "verbatim view"}]})
    with patch("signals.distill.run_claude", side_effect=fake):
        build_article_distillate_cache(arts, cache_path=cache, since="2021-01-01", min_chars=500)
    rows = [json.loads(l) for l in cache.read_text().splitlines()]
    assert {r["object_id"] for r in rows} == {"a1"}   # a2 too old, a3 too short
    assert len(calls) == 1


def test_distill_tweet_batch_keeps_verbatim(monkeypatch):
    import json as _j
    from unittest.mock import patch
    from signals.distill import distill_tweet_batch
    resp = _j.dumps({"kept":[{"date":"2024-01-01","text":"zk rollups are the endgame"}]})
    with patch("signals.distill.run_claude", return_value=resp):
        out = distill_tweet_batch([("2024-01-01","zk rollups are the endgame"),("2024-01-02","gm")])
    assert out == [{"date":"2024-01-01","text":"zk rollups are the endgame"}]

def test_build_tweet_cache_resumable_and_batches(tmp_path):
    import json as _j, pandas as pd
    from unittest.mock import patch
    from signals.distill import build_tweet_distillate_cache, load_tweet_distillates
    tw = pd.DataFrame({"created_at": pd.to_datetime(["2023-01-01","2023-02-01"], utc=True),
                       "type":["original","original"],
                       "text":["A substantive thesis about modular blockchains and DA layers",
                               "Another real take on restaking risk and shared security tradeoffs"]})
    p = tmp_path/"eddylazzarin.parquet"; tw.to_parquet(p)
    cache = tmp_path/"tweet_distillates.jsonl"
    resp = _j.dumps({"kept":[{"date":"2023-01-01","text":"distilled view"}]})
    with patch("signals.distill.run_claude", return_value=resp) as m:
        build_tweet_distillate_cache([p], cache_path=cache, batch_chars=130000)
        n1 = m.call_count
        build_tweet_distillate_cache([p], cache_path=cache, batch_chars=130000)  # resume: no new calls
        assert m.call_count == n1
    flat = load_tweet_distillates(cache)
    assert ("2023-01-01","distilled view") in flat

def test_build_cache_skips_malformed_llm_response(tmp_path):
    import pandas as pd
    from unittest.mock import patch
    from signals.distill import build_distillate_cache
    tx = pd.DataFrame({"object_id": ["x1", "x2"], "title": ["A", "B"],
                       "transcript": ["t1", "t2"], "status": ["ok", "ok"]})
    txp = tmp_path / "tx.parquet"; tx.to_parquet(txp)
    cache = tmp_path / "d.jsonl"
    calls = {"n": 0}
    def fake(system, user, **kw):
        calls["n"] += 1
        return "{bad json no close" if calls["n"] == 1 else '{"passages": []}'
    with patch("signals.distill.run_claude", side_effect=fake):
        build_distillate_cache(txp, None, cache_path=cache,
                               post_dates={"x1": "2023-01-01", "x2": "2023-02-01"})
    import json
    rows = {json.loads(l)["object_id"] for l in cache.read_text().splitlines()}
    assert rows == {"x1", "x2"}   # x1 malformed -> skipped (empty), x2 fine; batch did NOT crash
