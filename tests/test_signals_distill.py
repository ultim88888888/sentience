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
