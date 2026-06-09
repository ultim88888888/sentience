import json
from datetime import date
from unittest.mock import patch
from signals.extract import parse_extraction, build_a1_prompt
from signals.schema import PeriodSignal

RAW = json.dumps({
    "sectors_excited": [{"name": "zk rollups", "why": "scaling endgame",
                         "conviction": 88, "horizon": "structural", "provenance": "grounded",
                         "age_note": None, "citations": [{"date": "2024-02-01", "quote": "zk is the endgame"}]}],
    "sectors_concerned": [{"name": "memecoins", "why": "froth", "conviction": 60,
                           "horizon": "tactical", "provenance": "grounded", "age_note": None,
                           "citations": [{"date": "2024-03-01", "quote": "memecoins are pure froth"}]}],
    "tokens_excited": [{"name": "HYPE", "why": "flow", "conviction": 70, "horizon": "tactical",
                        "parent_sector": "perp dex", "provenance": "grounded", "age_note": None,
                        "citations": []}],
    "tokens_concerned": [],
    "risk_regime": {"stance": "risk_on", "conviction": 65, "why": "liquidity", "provenance": "grounded"},
    "notes": "n",
})

def test_parse_maps_arrays_to_stance_and_clamps():
    p = parse_extraction(RAW, t=date(2024, 6, 30))
    assert isinstance(p, PeriodSignal)
    by = {i.item: i for i in p.items}
    assert by["zk rollups"].stance == "bullish"        # from sectors_excited
    assert by["memecoins"].stance == "bearish"         # from sectors_concerned
    assert by["zk rollups"].item_type == "sector"
    assert by["HYPE"].item_type == "token"
    assert by["HYPE"].parent_sector == "perp dex"      # raw, pre-canonicalization
    assert p.risk_regime.stance == "risk_on"
    assert p.as_of == "2024-06-30"

def test_parse_tolerates_fenced_json_and_missing_arrays():
    raw = "```json\n" + json.dumps({"sectors_excited": [], "risk_regime":
          {"stance": "neutral", "conviction": 50, "why": "", "provenance": "extrapolated"}}) + "\n```"
    p = parse_extraction(raw, t=date(2024, 6, 30))
    assert p.items == ()
    assert p.risk_regime.stance == "neutral"

def test_prompt_demands_recency_weighting_and_no_taxonomy():
    system, user = build_a1_prompt("[2024-01-01] (x) zk good", t=date(2024, 6, 30))
    assert "recent" in system.lower() or "recency" in system.lower()
    assert "2024-06-30" in system
    # must NOT leak the seed taxonomy into extraction (free-form naming)
    assert "perp-dex" not in system and "liquid-staking" not in system

def test_member_prompt_is_individual_not_consensus():
    from signals.extract import build_member_prompt
    system, _ = build_member_prompt("[2024-01-01] (x) zk good", date(2024,6,30), "Eddy Lazzarin")
    assert "Eddy Lazzarin" in system
    assert "individual" in system.lower()
    assert "consensus" not in system.lower() or "not a team consensus" in system.lower()


def test_parse_tolerates_malformed_entries():
    import json
    from datetime import date
    from signals.extract import parse_extraction
    raw = json.dumps({
        "sectors_excited": ["just a string", {"name": "zk", "conviction": 80, "horizon": "structural",
                            "provenance": "grounded", "citations": ["bad cite", {"date": "2024-01-01", "quote": "q"}]}],
        "risk_regime": "not a dict",
    })
    p = parse_extraction(raw, t=date(2024, 6, 30))
    assert [i.item for i in p.items] == ["zk"]          # bare string skipped, dict kept
    assert len(p.items[0].citations) == 1               # bad citation skipped
    assert p.risk_regime.stance == "no_view"            # non-dict risk -> default


def test_extract_member_chunks_oversized_corpus(tmp_path):
    from datetime import date
    import pandas as pd
    from unittest.mock import patch
    from signals.extract import extract_member
    tw = pd.DataFrame({
        "created_at": pd.to_datetime(["2023-01-01","2023-06-01","2023-12-01"], utc=True),
        "type": ["original","original","original"],
        "text": ["A substantive thesis about modular blockchains and data availability tradeoffs here",
                 "A real take on restaking systemic risk and shared security economics in this post",
                 "Strong view that zk rollups are the scaling endgame for ethereum going forward now"]})
    p = tmp_path/"m.parquet"; tw.to_parquet(p)
    partial = '{"sectors_excited":[{"name":"zk","why":"w","conviction":80,"horizon":"structural","provenance":"grounded","age_note":null,"citations":[]}],"risk_regime":{"stance":"risk_on","conviction":60,"why":"w","provenance":"grounded"}}'
    merged = '{"sectors_excited":[{"name":"zk","why":"merged","conviction":85,"horizon":"structural","provenance":"grounded","age_note":null,"citations":[]}],"risk_regime":{"stance":"risk_on","conviction":65,"why":"m","provenance":"grounded"}}'
    calls = []
    def fake(system, user, **kw):
        calls.append(system)
        return merged if "TIME-SLICE" in system else partial
    with patch("signals.extract.run_claude", side_effect=fake):
        out = extract_member(date(2024,6,30), "Scott", window_months=18, twitter_path=p, max_chars=120)
    assert len(calls) >= 3                       # >=2 chunk extracts + 1 merge
    assert any("TIME-SLICE" in c for c in calls) # merge happened
    assert out.approach == "A2a:Scott"
    assert out.items[0].item == "zk" and out.items[0].conviction == 85  # from merged
