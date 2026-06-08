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
