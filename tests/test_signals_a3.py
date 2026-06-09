"""A3 tests — schema reuse, risk-by-horizon capture, and the lookahead reason-audit."""
import json
from datetime import date
from unittest.mock import patch
from signals.schema import SignalItem, RiskRegime, PeriodSignal
from signals.a3 import member_call, audit_reasons, _render_framework

_OUT = json.dumps({
    "sectors_excited": [{"name": "zk", "why": "ZK proofs maturing per the digest", "conviction": 85,
                         "horizon": "structural"}],
    "sectors_concerned": [{"name": "cefi", "why": "Prime Trust failure this period", "conviction": 70,
                           "horizon": "tactical"}],
    "risk_by_horizon": {"short": {"stance": "risk_off", "why": "counterparty stress"},
                        "medium": {"stance": "neutral", "why": "mixed"},
                        "long": {"stance": "risk_on", "why": "secular"}},
    "notes": "reasoned"})


def _fw():
    return PeriodSignal(as_of="2023-06-30", approach="A2a:x",
                        items=(SignalItem(item="zk", item_type="sector", parent_sector=None,
                               stance="bullish", conviction=80, horizon="structural", rationale="r",
                               provenance="grounded", age_note=None, citations=()),),
                        risk_regime=RiskRegime("risk_on", 60, "w", "grounded"))


def test_member_call_captures_risk_by_horizon_and_short_drives_regime():
    with patch("signals.a3.run_claude", return_value=_OUT):
        v = member_call(date(2023, 6, 30), "Chris Dixon", _fw(), {"as_of": "2023-06-30", "market": "m", "news": "n"})
    assert v.risk_regime.stance == "risk_off"                 # short-term horizon drives actionable regime
    rbh = json.loads(v.notes)["risk_by_horizon"]
    assert rbh["long"]["stance"] == "risk_on" and rbh["short"]["stance"] == "risk_off"
    assert {i.item for i in v.items} == {"zk", "cefi"}


def test_audit_drops_lookahead_flagged():
    v = PeriodSignal(as_of="2023-06-30", approach="A3a:x", items=(
        SignalItem(item="zk", item_type="sector", parent_sector=None, stance="bullish", conviction=80,
                   horizon="structural", rationale="ZK maturing", provenance="grounded", age_note=None, citations=()),
        SignalItem(item="sol", item_type="token", parent_sector="pos-l1", stance="bullish", conviction=80,
                   horizon="structural", rationale="SOL would later 5x", provenance="grounded", age_note=None, citations=())),
        risk_regime=RiskRegime("neutral", 50, "", "grounded"))
    with patch("signals.a3.run_claude", return_value=json.dumps({"lookahead_indices": [1]})):
        out = audit_reasons(v, date(2023, 6, 30))
    assert {i.item for i in out.items} == {"zk"}              # the hindsight "would later 5x" call dropped


def test_render_framework_handles_empty():
    assert "no prior corpus stance" in _render_framework(None)


def test_extract_json_salvages_truncated():
    """A truncated LLM object (missing closing braces) must still parse — the A3 truncation failure mode."""
    from signals.extract import _extract_json
    truncated = '{"sectors_excited":[{"name":"zk","why":"maturing","conviction":80}],"risk_by_horizon":{"short":{"stance":"risk_off","why":"cut off here'
    obj = _extract_json(truncated)
    assert obj["sectors_excited"][0]["name"] == "zk"
    assert obj["risk_by_horizon"]["short"]["stance"] == "risk_off"


def test_extract_json_strips_trailing_comma():
    from signals.extract import _extract_json
    assert _extract_json('{"a":1,"b":[1,2,],}')["b"] == [1, 2]
