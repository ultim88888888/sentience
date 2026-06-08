from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal

def _item(**kw):
    base = dict(item="zk", item_type="sector", parent_sector=None, stance="bullish",
               conviction=80, horizon="structural", rationale="r",
               provenance="grounded", age_note=None,
               citations=(Citation("2023-01-01", "validity proofs are the endgame"),))
    base.update(kw)
    return SignalItem(**base)

def test_period_roundtrips_through_dict():
    p = PeriodSignal(as_of="2023-03-31", approach="A1", items=(_item(),),
                     risk_regime=RiskRegime("risk_on", 70, "why", "grounded"),
                     notes="n")
    d = p.to_dict()
    p2 = PeriodSignal.from_dict(d)
    assert p2 == p
    assert d["items"][0]["citations"][0]["quote"] == "validity proofs are the endgame"

def test_conviction_is_clamped_on_construction():
    assert _item(conviction=150).conviction == 100
    assert _item(conviction=-5).conviction == 0

def test_invalid_stance_rejected():
    import pytest
    with pytest.raises(ValueError):
        _item(stance="mega-bull")
